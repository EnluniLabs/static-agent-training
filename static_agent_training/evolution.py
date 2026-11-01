"""One population, fitness tournaments, inherited elites, and offspring mutation."""

from dataclasses import dataclass, field
import random
import math
import hashlib
import json

from .environment import NumberLine, POSITIONS, sample_starts
from .evaluation import evaluate
from .baseline import random_task_mean
from .genome import DevelopmentConfig, develop, inherit_lifetime_mappings
from .initialize import Config, initialize
from .learning import LearningConfig
from .mutation import MutationConfig, mutate
from .seeds import derive
from .tasks import sample_tasks


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    population: int = 200
    generations: int = 500
    episodes: int = 20
    assessment_episodes: int = 20
    elites: int = 5
    tournament_size: int = 5
    initial_pairs: int = 4
    chain_budget: int = 64
    encoding_seed: int = 0
    learning: bool = True
    inherit_lifetime: bool = False
    varying_goals: bool = True
    backend: str = "python"
    workers: int = 8
    batch_size: int = 256
    boundary_mode: str = "clamp"
    distance_distribution: str = "uniform"
    include_zero: bool = True
    distance_mean: float = 10.0
    distance_stddev: float = 5.0
    speciation: bool = False
    speciation_threshold: float = .6
    max_species: int = 16
    mutation: MutationConfig = field(default_factory=MutationConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    learning_config: LearningConfig = field(default_factory=LearningConfig)

    def __post_init__(self):
        for name in ('population', 'generations', 'episodes', 'assessment_episodes',
                     'tournament_size', 'initial_pairs', 'workers', 'batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.inherit_lifetime) is not bool:
            raise ValueError("inherit_lifetime must be a boolean")
        if type(self.speciation) is not bool:
            raise ValueError('speciation must be a boolean')
        if not math.isfinite(self.speciation_threshold) or not 0 <= self.speciation_threshold <= 1:
            raise ValueError('speciation threshold must be between zero and one')
        if type(self.max_species) is not int or self.max_species < 1:
            raise ValueError('max species must be a positive integer')
        if type(self.include_zero) is not bool:
            raise ValueError("include_zero must be a boolean")
        if self.boundary_mode not in ("clamp", "terminate"):
            raise ValueError("boundary mode must be clamp or terminate")
        if self.distance_distribution not in ('normal', 'uniform'):
            raise ValueError('distance distribution must be normal or uniform')
        if not math.isfinite(self.distance_mean) or not 1 <= self.distance_mean <= 40:
            raise ValueError('distance mean must be finite and between 1 and 40')
        if not math.isfinite(self.distance_stddev) or self.distance_stddev <= 0:
            raise ValueError('distance standard deviation must be finite and positive')
        if self.backend not in ('python', 'native', 'cuda'):
            raise ValueError('backend must be python, native, or cuda')
        if type(self.elites) is not int or not 0 <= self.elites < self.population:
            raise ValueError('elites must be between zero and population minus one')
        if type(self.chain_budget) is not int or self.chain_budget < 0:
            raise ValueError('chain_budget must be a nonnegative integer')

    @classmethod
    def from_data(cls, data):
        data = dict(data)
        data.setdefault("varying_goals", False)
        data.setdefault("distance_distribution", "uniform")
        data.setdefault("boundary_mode", "terminate")
        data.setdefault("include_zero", False)
        data['mutation'] = MutationConfig(**data['mutation'])
        data['development'] = DevelopmentConfig(**data['development'])
        data['learning_config'] = LearningConfig(**data['learning_config'])
        return cls(**data)


def partitions(seed, *, include_zero=False):
    positions = list(POSITIONS) if include_zero else [x for x in POSITIONS if x]
    random.Random(derive(seed, 'partitions')).shuffle(positions)
    count=len(positions)-10
    return dict(training=sorted(positions[:count]), assessment=sorted(positions[count:]))


def initialize_population(config, progress=None):
    vocabulary = NumberLine(encoding_seed=config.encoding_seed, goal_conditioned=config.varying_goals)
    observations = [s.token for s in vocabulary.observation_tokens]
    outputs = [o.token for o in vocabulary.output_tokens]
    population = []
    for index in range(config.population):
        generated = initialize(Config(seed=derive(config.seed, 'initial', index), count=config.initial_pairs,
                                      max_depth=config.mutation.max_depth, max_nodes=config.mutation.max_nodes,
                                      combinator_probability=config.mutation.combinator_probability))
        pairs = [(p['left']['ast'], p['right']['ast']) for p in generated['initial_pairs']]
        population.append(develop(pairs, observations, outputs, budgets=config.chain_budget,
                                  development_seed=derive(config.seed, 'initial-development', index),
                                  mapping_seed=derive(config.seed, 'initial-mapping', index), config=config.development))
        if progress:
            progress(dict(phase='developing', completed=index + 1, total=config.population))
    return population


def generation_tasks(config, generation, split):
    if config.distance_distribution == 'normal':
        tasks = []
        for phase, count, seed in (
                ('training', config.episodes, derive(config.seed, 'training-tasks', generation)),
                ('assessment', config.assessment_episodes, derive(config.seed, 'assessment-tasks'))):
            tasks.append(sample_tasks(seed, count, starts=split[phase],
                                     goals=split[phase] if config.varying_goals else (0,),
                                     mean=config.distance_mean, stddev=config.distance_stddev))
        training, assessment = tasks
        return ([s for s,g in training], [s for s,g in assessment],
                [g for s,g in training] if config.varying_goals else None,
                [g for s,g in assessment] if config.varying_goals else None)
    def starts_for(seed,count,allowed):
        if not config.varying_goals:
            allowed=tuple(x for x in allowed if x!=0)
        if 0 not in allowed:
            return sample_starts(seed,count,allowed=allowed)
        rng=random.Random(seed)
        return tuple(rng.choice(allowed) for _ in range(count))
    training_starts = starts_for(derive(config.seed, 'training-starts', generation), config.episodes,
                                    allowed=tuple(split['training']))
    # Fixed common assessment episodes give the live diagnostic a stable comparison set.
    assessment_starts = starts_for(derive(config.seed, 'assessment-starts'), config.assessment_episodes,
                                      allowed=tuple(split['assessment']))
    training_goals = assessment_goals = None
    if config.varying_goals:
        train_rng = random.Random(derive(config.seed, 'training-goals', generation))
        assessment_rng = random.Random(derive(config.seed, 'assessment-goals'))
        training_goals = [train_rng.choice([g for g in split['training'] if g != start])
                          for start in training_starts]
        assessment_goals = [assessment_rng.choice([g for g in split['assessment'] if g != start])
                            for start in assessment_starts]
    return training_starts, assessment_starts, training_goals, assessment_goals


def evaluate_generation(population, config, generation, split, progress=None, *, return_lifetime=False):
    training_starts, assessment_starts, training_goals, assessment_goals = generation_tasks(config, generation, split)
    random_training = random_task_mean(training_starts, training_goals, boundary_mode=config.boundary_mode)
    random_assessment = random_task_mean(assessment_starts, assessment_goals, boundary_mode=config.boundary_mode)
    lifetime_population=[] if config.inherit_lifetime else None
    accelerated_results = None
    if config.backend != 'python':
        from .accelerated import evaluate_batch
        accelerated_results = []
        for first in range(0, len(population), config.batch_size):
            batch = population[first:first + config.batch_size]
            evaluated=evaluate_batch(
                batch, training_starts, assessment_starts,
                seeds=[derive(config.seed, 'evaluation', generation, i) for i in range(first, first + len(batch))],
                encoding_seed=config.encoding_seed, learning=config.learning,
                learning_config=config.learning_config, training_goals=training_goals,
                assessment_goals=assessment_goals, backend=config.backend, workers=config.workers, boundary_mode=config.boundary_mode,return_weights=config.inherit_lifetime)
            if config.inherit_lifetime:
                batch_results,weights=evaluated
                for lane,individual in enumerate(batch):
                    inherited=individual.inherited;ids=inherited.pair_ids
                    embedding={s:{q:float(weights[0][lane,i,j]) for j,q in enumerate(ids)}
                               for i,s in enumerate(inherited.observations)}
                    output={q:{o:float(weights[1][lane,i,j]) for j,o in enumerate(inherited.outputs)}
                            for i,q in enumerate(ids)}
                    lifetime_population.append(inherit_lifetime_mappings(individual,embedding,output))
                accelerated_results.extend(batch_results)
            else:
                accelerated_results.extend(evaluated)
            if progress:
                values = [r['training']['fitness'] for r in accelerated_results]
                progress(dict(phase='evaluating', generation=generation, completed=len(values),
                              total=len(population), mean=sum(values)/len(values), best=max(values)))
    results, champion_detail, champion = [], None, 0
    for index, individual in enumerate(population):
        if accelerated_results is None:
            result, detail = evaluate(individual, training_starts, assessment_starts,
                                      seed=derive(config.seed, 'evaluation', generation, index),
                                      encoding_seed=config.encoding_seed, learning=config.learning,
                                      learning_config=config.learning_config,
                                      training_goals=training_goals, assessment_goals=assessment_goals, boundary_mode=config.boundary_mode)
        else:
            result, detail = accelerated_results[index], None
        if config.inherit_lifetime and accelerated_results is None:
            lifetime_population.append(inherit_lifetime_mappings(individual,detail['learned_embedding'],detail['learned_output']))
        result['index'] = index
        results.append(result)
        if index == 0 or result['training']['fitness'] > results[champion]['training']['fitness']:
            champion, champion_detail = index, detail
        if progress and accelerated_results is None:
            scores = [r['training']['fitness'] for r in results]
            progress(dict(phase='evaluating', generation=generation, completed=index + 1,
                          total=len(population), mean=sum(scores) / len(scores), best=max(scores)))
    if accelerated_results is not None:
        from .accelerated import assert_equivalent
        reference, champion_detail = evaluate(
            population[champion], training_starts, assessment_starts,
            seed=derive(config.seed, 'evaluation', generation, champion),
            encoding_seed=config.encoding_seed, learning=config.learning,
            learning_config=config.learning_config, training_goals=training_goals,
            assessment_goals=assessment_goals, boundary_mode=config.boundary_mode)
        assert_equivalent({k: v for k, v in results[champion].items() if k != 'index'}, reference)
        # Preserve a full, independently checked champion trace for export and replay.
        reference['index'] = champion
        results[champion] = reference
    scores = [r['training']['fitness'] for r in results]
    assessed = [r['assessment']['fitness'] for r in results]
    summary = dict(generation=generation, mean=sum(scores) / len(scores), best=max(scores),
                   assessment_mean=sum(assessed) / len(assessed),
                   random_training=random_training, random_assessment=random_assessment,
                   champion_assessment=results[champion]['assessment']['fitness'],
                   unique_genotypes=len({r['genotype_id'] for r in results}),
                   mean_repertoire=sum(r['repertoire_size'] for r in results) / len(results),
                   expectations=sum(r['training']['expectations'] for r in results),
                   tested=sum(r['training']['tested'] for r in results),
                   conflicts=sum(r['training']['conflicts'] for r in results))
    tested = summary['tested']
    summary['prediction_accuracy'] = sum(r['training']['correct'] for r in results) / tested if tested else None
    from .speciation import describe
    summary['unique_structures'] = len({describe(p) for p in population})
    summary['task_id'] = hashlib.sha256(json.dumps([training_starts,assessment_starts,training_goals,assessment_goals],separators=(',',':')).encode()).hexdigest()
    summary['population_id'] = hashlib.sha256(json.dumps([r['genotype_id'] for r in results],separators=(',',':')).encode()).hexdigest()
    sampling = dict(distribution=config.distance_distribution,
                    mean=config.distance_mean if config.distance_distribution == 'normal' else None,
                    stddev=config.distance_stddev if config.distance_distribution == 'normal' else None,
                    training_distances=[abs(s-g) for s,g in zip(training_starts, training_goals or [0]*len(training_starts))],
                    assessment_distances=[abs(s-g) for s,g in zip(assessment_starts, assessment_goals or [0]*len(assessment_starts))])
    record = dict(task_sampling=sampling, summary=summary, training_starts=training_starts, assessment_starts=assessment_starts,
                  training_goals=training_goals, assessment_goals=assessment_goals,
                  individuals=results, champion_index=champion)
    return (record,champion_detail,lifetime_population) if return_lifetime else (record,champion_detail)


def reproduce(population, results, config, generation, progress=None, *, speciation_state=None, return_speciation=False, lifetime_population=None):
    if config.inherit_lifetime:
        if lifetime_population is None or len(lifetime_population)!=len(population):
            raise ValueError('lifetime inheritance requires end-of-training mappings for every parent')
        population=lifetime_population
    if config.speciation:
        from .speciation import reproduce as reproduce_species
        children,ancestry,report,state = reproduce_species(population,results,config,generation,speciation_state,progress)
        return (children,ancestry,report,state) if return_speciation else (children,ancestry)
    scores = [r['training']['fitness'] for r in results]
    order = sorted(range(len(population)), key=lambda i: (-scores[i], i))
    children, ancestry = [], []
    for parent in order[:config.elites]:
        children.append(population[parent])
        ancestry.append(dict(kind='elite', parent_index=parent, genotype_id=population[parent].genotype_id))
    rng = random.Random(derive(config.seed, 'selection', generation))
    while len(children) < config.population:
        competitors = rng.sample(range(len(population)), min(config.tournament_size, len(population)))
        parent = max(competitors, key=lambda i: scores[i])
        child, log = mutate(population[parent], seed=derive(config.seed, 'mutation', generation, len(children)),
                            config=config.mutation)
        children.append(child)
        ancestry.append(dict(kind='offspring', parent_index=parent, competitors=competitors, mutation=log))
        if progress:
            progress(dict(phase='reproducing', generation=generation, completed=len(children), total=config.population))
    if return_speciation:
        return children, ancestry, dict(enabled=False,count=1,groups=[dict(id=0,size=len(population))]), None
    return children, ancestry
