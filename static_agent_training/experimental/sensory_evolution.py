"""Sensory lifetime evaluation inside the shared evolutionary training pipeline."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import random

from ..environments import environment_spec, stimulus_world, categorical_world
from ..baseline import random_task_mean
from ..seeds import derive
from ..speciation import describe
from .bit_model import BitModel, run_episode
from .number_line import StimulusNumberLine
from .senses import SensoryAdapter
from .sensory_genome import SensoryMappings


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def block(model, tasks, *, learn, encoding_seed, boundary_mode="clamp", goal_conditioned=True, environment="original"):
    world = SensoryAdapter(stimulus_world(environment,encoding_seed=encoding_seed,boundary_mode=boundary_mode,goal_conditioned=goal_conditioned), model.senses)
    episodes, counts = [], Counter()
    for start, goal in tasks:
        offset = len(model.events)
        final = run_episode(model, world, (start, goal), learn=learn)[-1]
        events = model.events[offset:]
        feedback = [e for e in events if e['kind'] == 'FEEDBACK']
        empirical = [e for e in events if e['kind'] == 'EMPIRICAL_UPDATE']
        updates = Counter(e['kind'] for e in events)
        counts.update(updates)
        episodes.append(dict(start=start, goal=goal, success=final.success, reason=final.reason,
                             outputs=(updates['ACTION_SIGNAL'] + updates['ACTION']) + updates['EXPECTATION'],
                             actions=(updates['BATCH_ACTION'] + updates['ACTION']),
                             expectations=updates['EXPECTATION'], tested=len(feedback),
                             correct=sum(e['correct'] for e in feedback),
                             loss_sum=sum(e['prediction_nll'] for e in feedback),
                             empirical_loss_sum=sum(e['loss'] for e in empirical), expired=updates['EXPIRED']))
    tested = sum(e['tested'] for e in episodes)
    correct = sum(e['correct'] for e in episodes)
    return dict(fitness=sum(e['success'] for e in episodes) / len(episodes), episodes=episodes,
                outputs=sum(e['outputs'] for e in episodes), actions=sum(e['actions'] for e in episodes),
                expectations=counts['EXPECTATION'], tested=tested, correct=correct,
                prediction_accuracy=correct / tested if tested else None,
                prediction_nll=sum(e['loss_sum'] for e in episodes) / tested if tested else None,
                **(dict(partition_groups=[e['groups'] for e in model.events if e['kind']=='PARTITION_SELECT'])
                   if hasattr(model, 'router_weights') else {}),
                expired=counts['EXPIRED'], empirical_updates=counts['EMPIRICAL_UPDATE'], conflicts=counts['CONFLICT'])


def evaluate(individual, training_tasks, assessment_tasks, *, seed, config, sensory_partitions=None, action_policy=None, capture_detail=True):
    sensory_partitions = config.sensory_partitions if sensory_partitions is None else sensory_partitions
    action_policy = config.action_policy if action_policy is None else action_policy
    inherited = individual.inherited
    if not isinstance(inherited, SensoryMappings):
        raise ValueError('sensory training requires sensory inherited parameters')
    model = inherited.model(individual.repertoire, seed=seed, learning_config=config.learning_config, embedding_selection=config.embedding_selection,
                        sensory_partitions=sensory_partitions, action_policy=action_policy)
    model.embedding_rng = random.Random(derive(seed, 'training', 'embedding'))
    model.output_rng = random.Random(derive(seed, 'training', 'output'))
    training = block(model, training_tasks, learn=config.learning, encoding_seed=config.encoding_seed,
                     boundary_mode=config.boundary_mode, goal_conditioned=config.varying_goals, environment=config.environment)
    if not capture_detail:
        model.events.clear()
    learned = model.to_data()
    assessor = BitModel.from_data(learned)
    assessor.events.clear()
    assessor.embedding_rng = random.Random(derive(seed, 'assessment', 'embedding'))
    assessor.output_rng = random.Random(derive(seed, 'assessment', 'output'))
    assessment = block(assessor, assessment_tasks, learn=False, encoding_seed=config.encoding_seed,
                     boundary_mode=config.boundary_mode, goal_conditioned=config.varying_goals, environment=config.environment)
    result = dict(genotype_id=individual.genotype_id, seed=seed, training=training, assessment=assessment,
                  repertoire_size=len(individual.repertoire.pairs), successor_count=len(individual.repertoire.successors),
                  model='senses', embedding_selection=config.embedding_selection,
                  sensory_partitions=sensory_partitions, action_policy=action_policy)
    if config.agent_y:
        result['agent_y'] = True
    detail = dict(learned_model=learned, assessment_events=assessor.events) if capture_detail else None
    reproductive = replace(individual, inherited=SensoryMappings(inherited.pair_ids, inherited.senses,
                                                                 model.parameters(), model.router_data() if config.agent_y else None)) if config.inherit_lifetime else None
    return result, detail, reproductive


def evaluate_generation(population, config, generation, split, progress=None, *, return_lifetime=False):
    from ..evolution import generation_tasks, structural_population_id
    train_starts, test_starts, train_goals, test_goals = generation_tasks(config, generation, split)
    training_tasks = list(zip(train_starts, train_goals or [environment_spec(config.environment).center] * len(train_starts)))
    assessment_tasks = list(zip(test_starts, test_goals or [environment_spec(config.environment).center] * len(test_starts)))
    results, champion, champion_detail = [], 0, None
    lifetime_population = [] if config.inherit_lifetime else None
    accelerated = None
    if config.agent_y and config.backend == 'native':
        from .agent_y_accelerated import evaluate_batch as y_batch
        accelerated = []
        for first in range(0, len(population), config.batch_size):
            batch = population[first:first + config.batch_size]
            rows, reproductive = y_batch(batch, train_starts, test_starts, train_goals, test_goals,
                seeds=[derive(config.seed, 'evaluation', generation, i) for i in range(first,first+len(batch))], config=config)
            accelerated.extend(rows)
            if config.inherit_lifetime:
                lifetime_population.extend(reproductive)
            if progress:
                scores = [r['training']['fitness'] for r in accelerated]
                progress(dict(phase='evaluating',generation=generation,completed=len(scores),
                    total=len(population),mean=sum(scores)/len(scores),best=max(scores)))
    elif config.backend != 'python':
        from .sensory_accelerated import evaluate_batch, learned_mappings
        accelerated = []
        for first in range(0, len(population), config.batch_size):
            batch = population[first:first + config.batch_size]
            evaluated = evaluate_batch(batch, train_starts, test_starts,
                training_goals=train_goals, assessment_goals=test_goals,
                seeds=[derive(config.seed, 'evaluation', generation, i) for i in range(first, first+len(batch))],
                encoding_seed=config.encoding_seed, learning=config.learning, learning_config=config.learning_config,
                backend=config.backend, workers=config.workers, boundary_mode=config.boundary_mode,
                goal_conditioned=config.varying_goals, environment=config.environment, embedding_selection=config.embedding_selection,
                sensory_partitions=config.sensory_partitions, action_policy=config.action_policy, return_weights=config.inherit_lifetime)
            if config.inherit_lifetime:
                rows, weights = evaluated
                lifetime_population.extend(replace(p, inherited=learned_mappings(p, weights, i)) for i,p in enumerate(batch))
            else:
                rows = evaluated
            accelerated.extend(rows)
            if progress:
                values = [r['training']['fitness'] for r in accelerated]
                progress(dict(phase='evaluating', generation=generation, completed=len(values),
                              total=len(population), mean=sum(values)/len(values), best=max(values)))
    parallel = None
    if config.agent_y and config.backend == 'python' and config.workers > 1 and len(population) > 1:
        from .sensory_parallel import evaluate_population
        parallel = evaluate_population(population, config, generation, training_tasks, assessment_tasks)
    for index, individual in enumerate(population):
        if parallel is not None:
            result, reproductive = next(parallel)
            detail = None
            if config.inherit_lifetime:
                lifetime_population.append(reproductive)
        elif accelerated is None:
            result, detail, reproductive = evaluate(individual, training_tasks, assessment_tasks,
                        seed=derive(config.seed, 'evaluation', generation, index), config=config)
            if config.inherit_lifetime:
                lifetime_population.append(reproductive)
        else:
            result, detail = accelerated[index], None
        result['index'] = index
        results.append(result)
        if index == 0 or result['training']['fitness'] > results[champion]['training']['fitness']:
            champion, champion_detail = index, detail
        if progress and accelerated is None:
            values = [r['training']['fitness'] for r in results]
            progress(dict(phase='evaluating', generation=generation, completed=index + 1,
                          total=len(population), mean=sum(values) / len(values), best=max(values)))
    if parallel is not None:
        # Exhaust the generator so its process pool closes before reproduction.
        next(parallel, None)
    if accelerated is not None or parallel is not None:
        from ..accelerated import assert_equivalent
        reference, champion_detail, _ = evaluate(population[champion], training_tasks, assessment_tasks,
            seed=derive(config.seed, 'evaluation', generation, champion), config=config)
        assert_equivalent({k:v for k,v in results[champion].items() if k!='index'}, reference)
        reference['index'] = champion
        results[champion] = reference
    tested = sum(r['training']['tested'] for r in results)
    scores = [r['training']['fitness'] for r in results]
    summary = dict(generation=generation, sensory_partitions=config.sensory_partitions, action_policy=config.action_policy, model='senses', embedding_selection=config.embedding_selection, mean=sum(scores) / len(scores), best=max(scores),
                   assessment_mean=sum(r['assessment']['fitness'] for r in results) / len(results),
                   champion_assessment=results[champion]['assessment']['fitness'],
                   random_training=random_task_mean(train_starts, train_goals, boundary_mode=config.boundary_mode, **environment_spec(config.environment).baseline_kwargs),
                   random_assessment=random_task_mean(test_starts, test_goals, boundary_mode=config.boundary_mode, **environment_spec(config.environment).baseline_kwargs),
                   unique_genotypes=len({r['genotype_id'] for r in results}),
                   unique_structures=len({describe(p) for p in population}),
                   mean_repertoire=sum(r['repertoire_size'] for r in results) / len(results),
                   mean_unique_initial_pairs=sum(len(set(p.initial_pairs)) for p in population) / len(population),
                   expectations=sum(r['training']['expectations'] for r in results), tested=tested,
                   conflicts=sum(r['training']['conflicts'] for r in results),
                   prediction_accuracy=sum(r['training']['correct'] for r in results) / tested if tested else None,
                   task_id=hashlib.sha256(json.dumps([train_starts, test_starts, train_goals, test_goals],separators=(',', ':')).encode()).hexdigest(),
                   population_id=hashlib.sha256(json.dumps([r['genotype_id'] for r in results],separators=(',', ':')).encode()).hexdigest())
    from ..structural_diagnostics import population_structure
    summary.update(population_structure(population, champion))
    summary['structural_population_id'] = structural_population_id(population)
    record = dict(summary=summary, individuals=results, champion_index=champion,
                  training_starts=train_starts, assessment_starts=test_starts,
                  training_goals=train_goals, assessment_goals=test_goals,
                  task_sampling=dict(distribution='uniform-solvable-boards' if environment_spec(config.environment).kind=='lights-out' else config.distance_distribution,
                       training_distances=[abs(s-g) for s,g in training_tasks],
                       assessment_distances=[abs(s-g) for s,g in assessment_tasks]))
    if environment_spec(config.environment).kind=='lights-out':
        from ..lights_out import solutions
        record['task_sampling']=dict(distribution='uniform-solvable-boards',
            training_solution_lengths=[len(solutions(config.environment)[s]) for s in record['training_starts']],
            assessment_solution_lengths=[len(solutions(config.environment)[s]) for s in record['assessment_starts']])
    return (record, champion_detail, lifetime_population) if return_lifetime else (record, champion_detail)


def benchmark(directory, bundle, tasks, *, seed):
    from collections import Counter
    from pathlib import Path
    from ..benchmark import interval
    from ..environment import NumberLine, Output
    from ..experiment import write_json
    from ..genome import Species
    model = BitModel.from_data(bundle['learned_model'])
    model.embedding_rng.seed(derive(seed, 'benchmark-animal-embedding'))
    model.output_rng.seed(derive(seed, 'benchmark-animal-output'))
    spec = environment_spec(bundle.get('environment', 'original'))
    boundary = bundle.get('boundary_mode', 'clamp')
    world = SensoryAdapter(stimulus_world(bundle.get('environment','original'),encoding_seed=bundle['encoding_seed'],boundary_mode=boundary,goal_conditioned=bundle.get('varying_goals',True)), model.senses)
    control = categorical_world(bundle.get('environment','original'),encoding_seed=bundle['encoding_seed'],goal_conditioned=True,boundary_mode=boundary)
    moves = tuple(control.diagnostic_state()['move_decoding'])
    rng = random.Random(derive(seed, 'benchmark-random-moves'))
    before = model.parameters()
    outcomes = []
    for start, goal in tasks:
        final = run_episode(model, world, (start, goal), learn=False)[-1]
        model.events.clear()
        transition = control.reset(start=start, goal=goal)
        while not transition.done:
            transition = control.emit(Output(rng.choice(moves)), source='random')
        outcomes.append(dict(start=start, goal=goal, animal_success=final.success, random_success=transition.success,
                             animal_reason=final.reason, random_reason=transition.reason))
    if model.parameters() != before:
        raise AssertionError('frozen benchmark changed sensory parameters')
    result = dict(schema_version='crt-benchmark-v1', model='senses', embedding_selection=model.embedding_selection,seed=seed, episodes=len(tasks),
                  generation=bundle['generation'], genotype_id=Species.from_data(bundle['individual']).genotype_id,
                  environment=bundle.get('environment','original'), boundary_mode=boundary, exact_random_success=random_task_mean([s for s,g in tasks], [g for s,g in tasks], boundary_mode=boundary, **spec.baseline_kwargs),
                  outcomes=outcomes, task_source='fresh shared held-out draws; no benchmark learning',
                  random_policy='uniform over nine puzzle buttons; 12 presses' if spec.kind=='lights-out' else 'uniform over all nine movement buttons, including zero; 12 moves')
    for name in ('animal', 'random'):
        successes = sum(row[name + '_success'] for row in outcomes)
        result[name] = dict(successes=successes, success_rate=successes / len(tasks), interval95=interval(successes, len(tasks)),
                            reasons=dict(Counter(row[name + '_reason'] for row in outcomes)))
    differences = [int(row['animal_success']) - int(row['random_success']) for row in outcomes]
    mean = sum(differences) / len(tasks)
    variance = sum((value - mean)**2 for value in differences) / (len(tasks) * (len(tasks) - 1)) if len(tasks) > 1 else 0
    result['paired_difference'] = dict(percentage_points=100 * mean,
                                       interval95=[100 * (mean - 1.96 * variance**.5), 100 * (mean + 1.96 * variance**.5)])
    write_json(Path(directory) / 'demo/benchmark.json', result)
    return result
