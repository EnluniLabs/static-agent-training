"""Matched switchboard evolution and a shared, frozen champion evaluation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

from .candidates import load_saved_candidate
from .evolution import TrainingConfig, generation_tasks, partitions, initialize_population, structural_population_id
from .experiment import run_experiment, write_json
from .experimental.sensory_evolution import benchmark, fingerprint
from .seeds import derive


MODES = {'single': (1, 'agreement'), 'parallel3agreement': (3, 'agreement'),
         'parallel3vote': (3, 'vote')}
LIMITATIONS = (
    'Episodes, action limits and evolutionary budgets are matched; computation and number of '
    'learning updates are not matched. Three chains can emit more outputs, expectations and '
    'updates. Final tasks are fresh draws with replacement from the saved assessment start pool, '
    'disjoint from the training start pool. Prior assessment starts may repeat; rollout visitation '
    'can overlap. These are not unseen states. Benchmark scores do not select champions. '
    'A tiny smoke run supports plumbing checks only, '
    'not an advantage claim. Champion selection uses training fitness only.'
)


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _replay_metrics(bundle, tasks, seed):
    """Replay the same frozen stream; EMBED events count activated chains."""
    from .environments import stimulus_world
    from .experimental.bit_model import BitModel, run_episode
    from .experimental.senses import SensoryAdapter
    model = BitModel.from_data(bundle['learned_model'])
    model.embedding_rng.seed(derive(seed, 'benchmark-animal-embedding'))
    model.output_rng.seed(derive(seed, 'benchmark-animal-output'))
    world = SensoryAdapter(stimulus_world('switchboard', encoding_seed=bundle['encoding_seed'],
                           boundary_mode=bundle['boundary_mode'], goal_conditioned=False), model.senses)
    before = model.parameters()
    counts = Counter()
    regions = set()
    conflicts = 0
    max_regions = 0
    for task in tasks:
        model.events.clear()
        run_episode(model, world, task, learn=False)
        counts.update(e['kind'] for e in model.events)
        conflicts += sum(bool(e.get('conflict')) for e in model.events if e['kind'] == 'BATCH_ACTION')
        active = {e.get('region', 0) for e in model.events if e['kind'] == 'EMBED'}
        regions.update(active)
        max_regions = max(max_regions, len(active))
    if model.parameters() != before:
        raise AssertionError('frozen replay changed parameters')
    return dict(action_conflicts=conflicts, outputs=counts['ACTION'] + counts['ACTION_SIGNAL'] + counts['EXPECTATION'],
                expectations=counts['EXPECTATION'], empirical_updates=counts['EMPIRICAL_UPDATE'],
                active_chain_metrics=dict(chains_started=counts['EMBED'], chain_edges=counts['EDGE'],
                    chains_terminated=counts['TERMINATED'], distinct_regions=len(regions),
                    max_distinct_regions_per_episode=max_regions), source='shared frozen benchmark replay')


def run_benchmark(output, *, population=16, generations=10, episodes=50,
                  assessment_episodes=100, benchmark_episodes=100, seeds=(48,),
                  backend='native', workers=2):
    for name, value in locals().copy().items():
        if name in ('population', 'generations', 'episodes', 'assessment_episodes', 'benchmark_episodes', 'workers'):
            _positive(name, value)
    seeds = list(seeds)
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    if backend not in ('native', 'python'):
        raise ValueError('backend must be native or python')
    output = Path(output).resolve()
    paths = {(seed, mode): output.with_name(f'{output.name}-seed{seed}-{mode}')
             for seed in seeds for mode in MODES}
    for path in [output, *paths.values()]:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f'refusing to overwrite nonempty output: {path}')
    # Validate every config and reserve tasks before writing any artifacts.
    configs, task_sets = {}, {}
    for seed in seeds:
        for mode, (parts, policy) in MODES.items():
            configs[seed, mode] = TrainingConfig(model='senses', environment='switchboard', receptors=144,
                embedding_selection='argmax', sensory_partitions=parts, action_policy=policy,
                seed=seed, initialization_seed=seed, encoding_seed=seed, varying_goals=False,
                population=population, generations=generations, episodes=episodes,
                assessment_episodes=assessment_episodes, elites=min(5, population - 1),
                backend=backend, workers=workers)
        config = configs[seed, 'single']
        split = partitions(seed, include_zero=False, environment='switchboard')
        assessment_used = set()
        for generation in range(generations):
            train, assess, _, _ = generation_tasks(config, generation, split)
            assessment_used.update(assess)
        pool = sorted(set(split['assessment']) - {0})
        if set(pool) & set(split['training']):
            raise ValueError('training and assessment start pools overlap')
        if not pool:
            raise ValueError('assessment start pool is empty')
        task_seed = derive(seed, 'switchboard-final-tasks')
        rng = random.Random(task_seed)
        tasks = [(rng.choice(pool), 0) for _ in range(benchmark_episodes)]
        task_sets[seed] = dict(seed=task_seed, pool=pool, tasks=tasks, task_id=fingerprint(tasks),
            sampling='uniform with replacement from saved assessment start pool',
            prior_assessment_unique_starts=len(assessment_used),
            prior_assessment_overlap_unique_starts=len({s for s, _ in tasks} & assessment_used),
            prior_assessment_overlap_episodes=sum(s in assessment_used for s, _ in tasks))
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version='switchboard-benchmark-v1', seeds=seeds, limitations=LIMITATIONS,
                  smoke_only=population < 16 or generations < 10 or episodes < 50 or assessment_episodes < 100 or benchmark_episodes < 100,
                  report_path=str(output / 'report.json'), arms=[], pairs=[], heldout=task_sets)
    for seed in seeds:
        rows = []
        for mode in MODES:
            directory = paths[seed, mode]
            config = configs[seed, mode]
            initial = initialize_population(config)
            expected_initial_id = fingerprint([p.genotype_id for p in initial])
            signature = dict(structure=structural_population_id(initial),
                             parameters=fingerprint([p.inherited.to_parameters() for p in initial]))
            del initial
            run_experiment(config, directory)
            records = [json.loads(p.read_text()) for p in sorted((directory / 'generations').glob('*.json'))]
            if records[0]['summary']['population_id'] != expected_initial_id:
                raise ValueError('saved population/task fingerprints do not match initialization')
            totals = {}
            for phase in ('training', 'assessment'):
                totals[phase] = {key: sum(i[phase][key] for r in records for i in r['individuals'])
                                 for key in ('outputs', 'actions', 'expectations', 'empirical_updates')}
            row = dict(seed=seed, mode=mode, directory=str(directory), config=asdict(config),
                       initialization_signature=signature,
                       curves=[r['summary'] for r in records], work_totals=totals)
            rows.append(row)
        initial_ids = [r['curves'][0]['population_id'] for r in rows]
        task_ids = [[s['task_id'] for s in r['curves']] for r in rows]
        signatures = [r['initialization_signature'] for r in rows]
        if any(s != signatures[0] for s in signatures) or any(ids != task_ids[0] for ids in task_ids):
            raise ValueError('saved population/task fingerprints do not match across modes')
        report['pairs'].append(dict(seed=seed, initialization_verified=True, tasks_verified=True,
                                    initial_population_id=initial_ids[0],
                                    initial_genotype_population_ids={r['mode']:i for r,i in zip(rows,initial_ids)},
                                    initialization_signature=signatures[0],task_ids=task_ids[0]))
        tasks = task_sets[seed]['tasks']
        evaluation_seed = derive(seed, 'switchboard-frozen-evaluation')
        for row in rows:
            bundle = load_saved_candidate(row['directory'])
            row['heldout'] = benchmark(row['directory'], bundle, tasks, seed=evaluation_seed)
            row['heldout']['task_source'] = task_sets[seed]['sampling']
            write_json(Path(row['directory']) / 'demo/benchmark.json', row['heldout'])
            actual = [(o['start'], o['goal']) for o in row['heldout']['outcomes']]
            if actual != tasks:
                raise AssertionError('benchmark did not use shared tasks')
            row['champion_metrics'] = _replay_metrics(bundle, tasks, evaluation_seed)
            row['benchmark_task_id'] = fingerprint(actual)
            report['arms'].append(row)
    write_json(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name, default in [('population', 16), ('generations', 10), ('episodes', 50),
                          ('assessment-episodes', 100), ('benchmark-episodes', 100), ('workers', 2)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seeds', type=int, nargs='+', default=[48])
    parser.add_argument('--backend', choices=['native', 'python'], default='native')
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print('SMOKE ONLY' if report['smoke_only'] else 'Matched evolutionary comparison')
    print(LIMITATIONS)
    for arm in report['arms']:
        print(f"seed {arm['seed']} {arm['mode']}: eval {arm['heldout']['animal']['success_rate']:.1%}, "
              f"action conflicts {arm['champion_metrics']['action_conflicts']}; work {arm['work_totals']}; "
              f"artifacts {arm['directory']}")
    print(f"Report: {report['report_path']}")


if __name__ == '__main__':
    main()
