"""One population, fitness tournaments, inherited elites, and offspring mutation."""

from dataclasses import dataclass, field, replace
import random
import math
import hashlib
import json
import warnings

from .environments import environment_spec, categorical_world
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
    reproduction_workers: int = 0
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
    model: str = 'categorical'
    agent_y: bool = False
    receptors: int = 256
    embedding_selection: str = "argmax"
    sensory_partitions: int = 1
    sensory_interface: str = 'legacy-bytes'
    retina_width: int = 16
    retina_height: int = 16
    action_policy: str = "agreement"
    environment: str = "original"
    initial_population: str | None = None
    population_save_count: int = 10
    initialization_seed: int | None = None

    def __post_init__(self):
        environment_spec(self.environment)
        if type(self.reproduction_workers) is not int or self.reproduction_workers < 0:
            raise ValueError("reproduction_workers must be a nonnegative integer (0 selects automatic)")
        if self.initialization_seed is not None and (type(self.initialization_seed) is not int or self.initialization_seed < 0):
            raise ValueError('initialization seed must be a nonnegative integer')
        if type(self.agent_y) is not bool:
            raise ValueError('agent_y must be boolean')
        if self.agent_y and (self.model != 'senses' or self.embedding_selection != 'argmax'):
            raise ValueError('Agent Y requires senses and argmax embedding')
        if self.model not in ('categorical', 'senses'):
            raise ValueError('model must be categorical or senses')
        if self.embedding_selection not in ("sample", "argmax"):
            raise ValueError("embedding_selection must be sample or argmax")
        if type(self.receptors) is not int or self.receptors < 16 or self.receptors % 16:
            raise ValueError('receptors must be a positive multiple of 16')
        if type(self.sensory_partitions) is not int or not 1 <= self.sensory_partitions <= self.receptors:
            raise ValueError('sensory_partitions must be a positive integer no greater than receptors')
        if self.sensory_interface not in ('legacy-bytes', 'retina', 'signals'):
            raise ValueError('sensory_interface must be legacy-bytes, retina, or signals')
        if any(type(v) is not int or v < 1 for v in (self.retina_width, self.retina_height)):
            raise ValueError('receptor layout dimensions must be positive integers')
        if self.sensory_interface in ('retina', 'signals'):
            if self.model != 'senses':
                raise ValueError('signal surfaces require a fixed-senses model')
            if self.receptors != self.retina_width * self.retina_height:
                raise ValueError('receptor resolution must equal layout width times height')
        if self.action_policy not in ('agreement', 'vote'):
            raise ValueError('action_policy must be agreement or vote')
        if self.sensory_partitions > 1:
            if self.model != 'senses':
                raise ValueError('categorical models cannot use multiple sensory partitions')
            if self.embedding_selection != 'argmax':
                raise ValueError('multiple sensory partitions require argmax embedding selection')
        for name in ('population', 'generations', 'episodes', 'assessment_episodes',
                     'tournament_size', 'initial_pairs', 'workers', 'batch_size', 'population_save_count'):
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
        if self.backend not in ('python', 'native'):
            raise ValueError('backend must be python or native')
        if type(self.elites) is not int or not 0 <= self.elites < self.population:
            raise ValueError('elites must be between zero and population minus one')
        if type(self.chain_budget) is not int or self.chain_budget < 0:
            raise ValueError('chain_budget must be a nonnegative integer')

    @classmethod
    def from_data(cls, data):
        data = dict(data)
        if data.get('backend') == 'cuda':
            warnings.warn('CUDA support was removed; this saved configuration will use native CPU evaluation.',
                          RuntimeWarning, stacklevel=2)
            data['backend'] = 'native'
        data.setdefault("sensory_interface", "legacy-bytes")
        data.setdefault("retina_width", 16)
        data.setdefault("retina_height", 16)
        data.setdefault("sensory_partitions", 1)
        data.setdefault("action_policy", "agreement")
        data.setdefault("embedding_selection", "sample")
        data.setdefault("varying_goals", False)
        data.setdefault("distance_distribution", "uniform")
        data.setdefault("boundary_mode", "terminate")
        data.setdefault("include_zero", False)
        data['mutation'] = MutationConfig(**data['mutation'])
        data['development'] = DevelopmentConfig(**data['development'])
        data['learning_config'] = LearningConfig(**data['learning_config'])
        return cls(**data)


def partitions(seed, *, include_zero=False, environment="original"):
    spec = environment_spec(environment)
    if spec.kind=='lights-out':
        from .lights_out import solutions
        positions=sorted(set(solutions(environment))-{0})
    else:
        positions = list(spec.positions) if include_zero else [x for x in spec.positions if x != spec.center]
    random.Random(derive(seed, 'partitions')).shuffle(positions)
    count=len(positions)-(max(1,len(positions)//5) if spec.kind=='lights-out' else 10)
    return dict(training=sorted(positions[:count]), assessment=sorted(positions[count:]))


def initialize_population(config, progress=None):
    if config.initial_population:
        from .populations import initialize_saved
        return initialize_saved(config)
    initialization_seed = config.seed if config.initialization_seed is None else config.initialization_seed
    senses = None
    if config.model == 'senses':
        from .experimental.apparatus import make_senses
        senses = make_senses(receptors=config.receptors, sensory_partitions=config.sensory_partitions,
                             sensory_interface=config.sensory_interface,
                             retina_width=config.retina_width, retina_height=config.retina_height)
        observations, outputs = (), ()
    else:
        vocabulary = categorical_world(config.environment,encoding_seed=config.encoding_seed,goal_conditioned=config.varying_goals)
        observations = [s.token for s in vocabulary.observation_tokens]
        outputs = [o.token for o in vocabulary.output_tokens]
    population = []
    for index in range(config.population):
        generated = initialize(Config(seed=derive(initialization_seed, 'initial', index), count=config.initial_pairs,
                                      max_depth=config.mutation.max_depth, max_nodes=config.mutation.max_nodes,
                                      combinator_probability=config.mutation.combinator_probability))
        pairs = [(p['left']['ast'], p['right']['ast']) for p in generated['initial_pairs']]
        population.append(develop(pairs, observations, outputs, budgets=config.chain_budget,
                                  development_seed=derive(initialization_seed, 'initial-development', index),
                                  mapping_seed=derive(initialization_seed, 'initial-mapping', index), config=config.development,
                                  senses=senses))
        if config.agent_y:
            from .experimental.agent_y import AgentY
            from .experimental.sensory_genome import SensoryMappings
            individual = population[-1]
            inherited = individual.inherited
            model = AgentY(individual.repertoire, senses=senses, parameters=inherited.to_parameters())
            population[-1] = replace(individual, inherited=SensoryMappings(
                inherited.pair_ids, senses, inherited.parameters, model.router_data()))
        if progress:
            progress(dict(phase='developing', completed=index + 1, total=config.population))
    return population


def structural_population_id(population):
    from dataclasses import asdict
    value = [dict(initial_pairs=p.initial_pairs, budgets=p.budgets, repertoire=p.repertoire.to_data(),
                  development_config=asdict(p.development_config)) for p in population]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def generation_tasks(config, generation, split):
    spec = environment_spec(config.environment)
    if spec.kind=='lights-out':
        from .lights_out import sample_tasks as puzzle_tasks
        training=puzzle_tasks(derive(config.seed,'training-tasks',generation),config.episodes,split['training'],config.environment)
        assessment=puzzle_tasks(derive(config.seed,'assessment-tasks'),config.assessment_episodes,split['assessment'],config.environment)
        return [s for s,g in training],[s for s,g in assessment],[0]*len(training),[0]*len(assessment)
    if config.distance_distribution == 'normal':
        tasks = []
        for phase, count, seed in (
                ('training', config.episodes, derive(config.seed, 'training-tasks', generation)),
                ('assessment', config.assessment_episodes, derive(config.seed, 'assessment-tasks'))):
            tasks.append(sample_tasks(seed, count, starts=split[phase],
                                     goals=split[phase] if config.varying_goals else (spec.center,),
                                     mean=config.distance_mean, stddev=config.distance_stddev, **spec.world_kwargs))
        training, assessment = tasks
        return ([s for s,g in training], [s for s,g in assessment],
                [g for s,g in training] if config.varying_goals else None,
                [g for s,g in assessment] if config.varying_goals else None)
    def starts_for(seed,count,allowed):
        if not config.varying_goals:
            allowed=tuple(x for x in allowed if x!=spec.center)
        if 0 not in allowed:
            return sample_starts(seed,count,allowed=allowed, **spec.world_kwargs)
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
    if config.model == 'senses':
        from .experimental.sensory_evolution import evaluate_generation as evaluate_sensory_generation
        return evaluate_sensory_generation(population, config, generation, split, progress, return_lifetime=return_lifetime)
    training_starts, assessment_starts, training_goals, assessment_goals = generation_tasks(config, generation, split)
    random_training = random_task_mean(training_starts, training_goals, boundary_mode=config.boundary_mode, **environment_spec(config.environment).baseline_kwargs)
    random_assessment = random_task_mean(assessment_starts, assessment_goals, boundary_mode=config.boundary_mode, **environment_spec(config.environment).baseline_kwargs)
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
                assessment_goals=assessment_goals, backend=config.backend, workers=config.workers, boundary_mode=config.boundary_mode,environment=config.environment,return_weights=config.inherit_lifetime)
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
                                      training_goals=training_goals, assessment_goals=assessment_goals, boundary_mode=config.boundary_mode, environment=config.environment)
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
            assessment_goals=assessment_goals, boundary_mode=config.boundary_mode, environment=config.environment)
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
                   mean_unique_initial_pairs=sum(len(set(p.initial_pairs)) for p in population) / len(population),
                   expectations=sum(r['training']['expectations'] for r in results),
                   tested=sum(r['training']['tested'] for r in results),
                   conflicts=sum(r['training']['conflicts'] for r in results))
    tested = summary['tested']
    summary['prediction_accuracy'] = sum(r['training']['correct'] for r in results) / tested if tested else None
    from .speciation import describe
    summary['unique_structures'] = len({describe(p) for p in population})
    summary['task_id'] = hashlib.sha256(json.dumps([training_starts,assessment_starts,training_goals,assessment_goals],separators=(',',':')).encode()).hexdigest()
    summary['population_id'] = hashlib.sha256(json.dumps([r['genotype_id'] for r in results],separators=(',',':')).encode()).hexdigest()
    from .structural_diagnostics import population_structure
    summary.update(population_structure(population, champion))
    summary['structural_population_id'] = structural_population_id(population)
    sampling = dict(distribution=config.distance_distribution,
                    mean=config.distance_mean if config.distance_distribution == 'normal' else None,
                    stddev=config.distance_stddev if config.distance_distribution == 'normal' else None,
                    training_distances=[abs(s-g) for s,g in zip(training_starts, training_goals or [0]*len(training_starts))],
                    assessment_distances=[abs(s-g) for s,g in zip(assessment_starts, assessment_goals or [0]*len(assessment_starts))])
    record = dict(task_sampling=sampling, summary=summary, training_starts=training_starts, assessment_starts=assessment_starts,
                  training_goals=training_goals, assessment_goals=assessment_goals,
                  individuals=results, champion_index=champion)
    if environment_spec(config.environment).kind=='lights-out':
        from .lights_out import solutions
        record['task_sampling']=dict(distribution='uniform-solvable-boards',
            training_solution_lengths=[len(solutions(config.environment)[s]) for s in record['training_starts']],
            assessment_solution_lengths=[len(solutions(config.environment)[s]) for s in record['assessment_starts']])
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
    schedule = []
    for index in range(len(children), config.population):
        competitors = rng.sample(range(len(population)), min(config.tournament_size, len(population)))
        parent = max(competitors, key=lambda i: scores[i])
        schedule.append(dict(index=index, parent_index=parent, competitors=competitors))
    from .offspring import construct_offspring
    for job, (child, log) in zip(schedule, construct_offspring(population, schedule, config, generation)):
        children.append(child)
        ancestry.append(dict(kind='offspring', parent_index=job['parent_index'], competitors=job['competitors'], mutation=log))
        if progress:
            progress(dict(phase='reproducing', generation=generation, completed=len(children), total=config.population))
    if return_speciation:
        return children, ancestry, dict(enabled=False,count=1,groups=[dict(id=0,size=len(population))]), None
    return children, ancestry
