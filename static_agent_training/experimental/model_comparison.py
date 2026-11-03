"""Matched fixed-repertoire lifetime learning: categorical versus fixed senses."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

from ..agent import Agent, run_episode as old_episode
from ..baseline import random_task_mean
from ..benchmark import interval
from ..comparison import paired_summary
from ..engine import run
from ..environment import NumberLine
from ..evolution import partitions
from ..experiment import write_json
from ..initialize import Config, initialize
from ..mappings import InheritedMappings
from ..repertoire import Repertoire
from ..seeds import derive
from .bit_model import BitModel, run_episode as sensory_episode
from .number_line import StimulusNumberLine
from .senses import FixedSenses, SensoryAdapter


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sample_tasks(seed, positions, count):
    rng = random.Random(seed)
    pairs = [(start, goal) for start in positions for goal in positions if start != goal]
    return [rng.choice(pairs) for _ in range(count)]


def baseline_snapshot(agent):
    maps = agent.mappings
    return dict(schema_version='crt-tabular-reference-v1', repertoire=agent.repertoire.to_data(),
                mappings=dict(observations=maps.observations, outputs=maps.outputs, pair_ids=maps.pair_ids,
                              embedding=maps.embedding, output=maps.output),
                embedding_rng=agent._embedding_rng.getstate(), output_rng=agent._output_rng.getstate(),
                revised=agent.learner.revised, training_events=agent.events)


def block(model, tasks, *, sensory, learn, progress=None):
    counts, reasons = Counter(), Counter()
    states, edges = set(), set()
    tested = correct = 0
    outcomes = []
    world = SensoryAdapter(StimulusNumberLine(), model.senses) if sensory else NumberLine(goal_conditioned=True)
    started = time.monotonic()
    for index, (start, goal) in enumerate(tasks):
        offset = len(model.events)
        if sensory:
            final = sensory_episode(model, world, (start, goal), learn=learn)[-1]
        else:
            transitions = old_episode(model, world, start=start, goal=goal, learn=learn)
            final = transitions[-1]
            feedback = [f for step in transitions for f in step.feedback]
            tested += len(feedback)
            correct += sum(f.correct for f in feedback)
        events = model.events[offset:]
        counts.update(e['kind'] for e in events)
        if sensory:
            feedback = [e for e in events if e['kind'] == 'FEEDBACK']
            tested += len(feedback)
            correct += sum(e['correct'] for e in feedback)
        previous_pair = None
        for event in events:
            if event['kind'] in ('EMBED', 'OUTPUT', 'ACTION', 'EXPECTATION') and event.get('pair') is not None:
                states.add(event['pair'])
            if event['kind'] == 'EDGE':
                edges.add((event['source'], event['target']))
            if not sensory and event['kind'] == 'EMBED':
                previous_pair = event['pair']
            elif not sensory and event['kind'] == 'OUTPUT':
                if previous_pair is not None:
                    edges.add((previous_pair, event['pair']))
                previous_pair = event['pair']
        outcomes.append(bool(final.success))
        reasons[final.reason] += 1
        if not learn:
            # Frozen assessors don't need cross-episode event provenance. Never
            # clear the learning agent's log while revision indices refer to it.
            model.events.clear()
        if progress and (index + 1) % 100 == 0:
            progress(index + 1)
    successes = sum(outcomes)
    return dict(episodes=len(tasks), successes=successes, success_rate=successes / len(tasks),
                interval95=interval(successes, len(tasks)), reasons=dict(reasons),
                tested_predictions=tested, exact_prediction_accuracy=correct / tested if tested else None,
                event_counts=dict(counts), visited_nodes=len(states), exercised_edges=len(edges),
                empirical_updates=counts['EMPIRICAL_UPDATE'], structural_updates=counts['CONFLICT'],
                seconds=time.monotonic() - started, outcomes=outcomes)


def run_pair(directory, seed, animal, plan, task_lists):
    directory = Path(directory)
    generation_seed = derive(seed, 'interface-repertoire', animal)
    generated = initialize(Config(seed=generation_seed, count=4))
    pairs = [(p['left']['ast'], p['right']['ast']) for p in generated['initial_pairs']]
    development = run(pairs, 64, seed=derive(seed, 'interface-development', animal))
    repertoire = Repertoire.from_data(development['repertoire'])
    structure_hash = digest(repertoire.to_data())
    vocabulary = NumberLine(goal_conditioned=True)
    maps = InheritedMappings.initialize(repertoire, [v.token for v in vocabulary.observation_tokens],
                                       [v.token for v in vocabulary.output_tokens],
                                       seed=derive(seed, 'interface-parameters', animal))
    old = Agent(repertoire, maps.lifetime(),
                embedding_seed=derive(seed, 'interface-training-embedding', animal),
                output_seed=derive(seed, 'interface-training-output', animal))
    sensory = BitModel(repertoire, senses=FixedSenses(receptors=plan['receptors']),
                       seed=derive(seed, 'interface-parameters', animal))
    sensory.embedding_rng.seed(derive(seed, 'interface-training-embedding', animal))
    sensory.output_rng.seed(derive(seed, 'interface-training-output', animal))
    write_json(directory / 'structure.json.gz', dict(generated=generated, development=development,
                                                     structure_hash=structure_hash))
    write_json(directory / 'old-initial.json.gz', baseline_snapshot(old))
    write_json(directory / 'senses-initial.json.gz', sensory.to_data())
    old_training = block(old, task_lists['training'], sensory=False, learn=True)
    sensory_training = block(sensory, task_lists['training'], sensory=True, learn=True)
    write_json(directory / 'old-trained.json.gz', baseline_snapshot(old))
    write_json(directory / 'senses-trained.json.gz', sensory.to_data())
    learned = InheritedMappings(maps.observations, maps.outputs, maps.pair_ids,
                                old.mappings.embedding, old.mappings.output)
    assessor = Agent(repertoire, learned.lifetime(), learning=False,
                     embedding_seed=derive(seed, 'interface-test-embedding', animal),
                     output_seed=derive(seed, 'interface-test-output', animal))
    # Old training logs are saved before releasing their potentially large rows.
    del old
    sensory.events.clear()
    sensory.embedding_rng.seed(derive(seed, 'interface-test-embedding', animal))
    sensory.output_rng.seed(derive(seed, 'interface-test-output', animal))
    frozen = sensory.parameters()
    old_result = block(assessor, task_lists['assessment'], sensory=False, learn=False)
    sensory_result = block(sensory, task_lists['assessment'], sensory=True, learn=False)
    if sensory.parameters() != frozen or digest(repertoire.to_data()) != structure_hash:
        raise AssertionError('frozen test changed parameters or structure')
    if (assessor.mappings.embedding != {s: dict(row) for s, row in learned.embedding.items()} or
            assessor.mappings.output != {s: dict(row) for s, row in learned.output.items()}):
        raise AssertionError('frozen baseline changed parameters')
    result = dict(seed=seed, animal=animal, structure_hash=structure_hash,
                  repertoire_nodes=len(repertoire.pairs), repertoire_edges=len(repertoire.successors),
                  task_hash=digest(task_lists),
                  parameter_counts=dict(old=len(repertoire.pairs) * 132,
                                        senses=len(repertoire.pairs) * (sensory.spec.features + 1 +
                                                sensory.spec.actions + sensory.spec.channels + sensory.spec.bits)),
                  old=dict(training=old_training, assessment=old_result),
                  senses=dict(training=sensory_training, assessment=sensory_result),
                  difference_percentage_points=100 * (sensory_result['success_rate'] - old_result['success_rate']))
    write_json(directory / 'result.json', result)
    return result


def report_for(directory, plan, status, pid):
    results = []
    for seed in plan['seeds']:
        for animal in range(plan['population']):
            path = directory / f'seed-{seed}' / f'animal-{animal:04d}' / 'result.json'
            if path.is_file():
                results.append(json.loads(path.read_text()))
    seed_results = []
    for seed in plan['seeds']:
        rows = [r for r in results if r['seed'] == seed]
        if len(rows) == plan['population']:
            means = {arm: statistics.mean(r[arm]['assessment']['success_rate'] for r in rows)
                     for arm in ('old', 'senses')}
            seed_results.append(dict(seed=seed, **means,
                                     difference_percentage_points=100 * (means['senses'] - means['old'])))
    aggregate = {arm: statistics.mean(r[arm]['assessment']['success_rate'] for r in results)
                 for arm in ('old', 'senses')} if results else None
    return dict(schema_version='crt-interface-comparison-v1', status=status, pid=pid,
                updated=time.time(), name=directory.name, plan=plan,
                completed=len(results), target=len(plan['seeds']) * plan['population'],
                population_mean=aggregate, per_seed=seed_results,
                paired_effect=paired_summary([r['difference_percentage_points'] / 100 for r in seed_results]),
                primary_endpoint='mean frozen held-out task success; no evolution or winner selection',
                limitations=['Different parameter counts, encodings, prediction targets and output normalization.',
                             'Fresh independently developed repertoires; neither arm uses evolved or learned mappings.',
                             'Prediction accuracies have different targets and are not directly comparable.',
                             'Within-animal intervals are descriptive; effect uncertainty uses completed independent seed means.'],
                pairs=[dict(seed=r['seed'], animal=r['animal'], structure_hash=r['structure_hash'],
                            repertoire_nodes=r['repertoire_nodes'], task_hash=r['task_hash'],
                            old=r['old']['assessment']['success_rate'], senses=r['senses']['assessment']['success_rate'],
                            difference_percentage_points=r['difference_percentage_points']) for r in results])


def run_study(directory, plan=None, *, resume=False):
    directory = Path(directory)
    if resume:
        plan = json.loads((directory / 'plan.json').read_text())
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError('output is not empty; choose a new output or use --resume')
    if plan is None:
        raise ValueError('a new study requires a plan')
    for key in ('population', 'episodes', 'assessment_episodes'):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f'{key} must be positive')
    if not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']) or any(type(s) is not int for s in plan['seeds']):
        raise ValueError('seeds must be nonempty distinct integers')
    FixedSenses(receptors=plan['receptors'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('study already has an active writer') from error
        if not resume:
            write_json(directory / 'plan.json', plan)
        def publish(status):
            report = report_for(directory, plan, status, os.getpid())
            write_json(directory / 'report.json', report)
            return report
        publish('running')
        try:
            for seed in plan['seeds']:
                split = partitions(seed, include_zero=True)
                task_lists = dict(training=sample_tasks(derive(seed, 'interface-training-tasks'), split['training'], plan['episodes']),
                                  assessment=sample_tasks(derive(seed, 'interface-assessment-tasks'), split['assessment'], plan['assessment_episodes']))
                write_json(directory / 'tasks' / f'{seed}.json', dict(partitions=split, **task_lists,
                    exact_random_success=random_task_mean([s for s, g in task_lists['assessment']],
                                                         [g for s, g in task_lists['assessment']])))
                for animal in range(plan['population']):
                    pair_dir = directory / f'seed-{seed}' / f'animal-{animal:04d}'
                    if (pair_dir / 'result.json').is_file():
                        continue
                    print(f'seed {seed} | animal {animal + 1}/{plan["population"]} | evaluating both models', flush=True)
                    result = run_pair(pair_dir, seed, animal, plan, task_lists)
                    publish('running')
                    print(f"seed {seed} | animal {animal + 1} | old {result['old']['assessment']['success_rate']:.2%} | senses {result['senses']['assessment']['success_rate']:.2%}", flush=True)
        except BaseException as error:
            report = publish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
            report['error'] = f'{type(error).__name__}: {error}'
            write_json(directory / 'report.json', report)
            raise
        return publish('completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=int, default=16, help='independent paired animals per seed, no evolution')
    parser.add_argument('--seeds', type=int, nargs='+', default=[48, 49, 50])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--assessment-episodes', type=int, default=2000)
    parser.add_argument('--receptors', type=int, default=256)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--memory-limit-mb', type=int, default=4096)
    args = parser.parse_args()
    if args.output and args.resume:
        parser.error('choose --output or --resume')
    if args.resume and any(value.split('=', 1)[0] in
            ('--population', '--seeds', '--episodes', '--assessment-episodes', '--receptors') for value in sys.argv[1:]):
        parser.error('resume uses the saved plan; only --memory-limit-mb may override')
    directory = args.resume or args.output or Path('runs/interfaces') / datetime.now().strftime('old-vs-senses-%Y%m%d-%H%M%S-%f')
    plan = dict(population=args.population, seeds=args.seeds, episodes=args.episodes,
                assessment_episodes=args.assessment_episodes, receptors=args.receptors,
                boundary_mode='clamp', bounds=[-20, 20], action_limit=12, internal_limit=16,
                parameter_initialization_scale=.1, speciation=False, evolution=False)
    plan['learning'] = dict(prediction_rate=.05, embedding_rate=.05, compatibility_threshold=.1)
    plan['world_serialization'] = 'old-categorical-positions-and-goals; senses-utf8-json-position-goal'
    package = Path(__file__).resolve().parents[1]
    source_files = ['agent.py', 'learning.py', 'mappings.py', 'environment.py', 'engine.py',
                    'initialize.py', 'reduction.py', 'experimental/bit_model.py', 'experimental/senses.py',
                    'experimental/number_line.py', 'experimental/model_comparison.py']
    plan['source_sha256'] = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in source_files}
    print(f'Interface comparison: {directory}', flush=True)
    from ..diagnostics import diagnostics
    with diagnostics(directory, args.memory_limit_mb):
        result = run_study(directory, plan, resume=bool(args.resume))
    print(json.dumps(dict(population_mean=result['population_mean'], paired_effect=result['paired_effect'])), flush=True)


if __name__ == '__main__':
    main()
