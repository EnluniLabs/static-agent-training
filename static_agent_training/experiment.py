"""Run and resume a single-population evolutionary experiment."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import json
import os
import shlex
from pathlib import Path

from .evolution import TrainingConfig, evaluate_generation, initialize_population, partitions, reproduce
from .genome import Species
from .terminal import Dashboard


CHECKPOINT = 'checkpoint.json.gz'


def write_json(path, data):
    """Replace a complete file atomically; interrupted writes preserve its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if str(path).endswith('.gz'):
        with gzip.open(temporary, 'wt', encoding='utf-8') as handle:
            json.dump(data, handle, sort_keys=True, separators=(',', ':'))
    else:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(data, handle, sort_keys=True, separators=(',', ':'))
            handle.write('\n')
    os.replace(temporary, path)


def load_checkpoint(directory):
    with gzip.open(Path(directory) / CHECKPOINT, 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    if data['schema_version'] != 'crt-training-v1':
        raise ValueError('unsupported checkpoint schema')
    return data


def run_experiment(config, directory, *, resume=False, until_generation=None, progress=None):
    """Checkpoint the next inherited population after every completed generation.

    All random streams derive from saved seed and generation/individual indices,
    so replay needs no mutable global RNG state. Interrupted generations rerun.
    `until_generation` allows a bounded segment without changing experiment config.
    """
    directory = Path(directory)
    if resume:
        state = load_checkpoint(directory)
        saved = TrainingConfig.from_data(state['config'])
        if replace(config, generations=saved.generations) != saved:
            raise ValueError('resume must retain the saved experiment configuration')
        if config.generations < state['next_generation']:
            raise ValueError('requested generations precede the checkpoint')
        population = [Species.from_data(data) for data in state['population']]
        history, split, first = state['history'], state['partitions'], state['next_generation']
        if len(population) != config.population or len(history) != first:
            raise ValueError('inconsistent checkpoint population or history')
        if state['config'] != asdict(config):
            state['config'] = asdict(config)
            write_json(directory / CHECKPOINT, state)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('output directory is not empty; use --resume or a new directory')
        directory.mkdir(parents=True, exist_ok=True)
        population = initialize_population(config, progress)
        history, split, first = [], partitions(config.seed, include_zero=config.include_zero and config.varying_goals), 0
        state = dict(schema_version='crt-training-v1', config=asdict(config), next_generation=0,
                     history=history, partitions=split, population=[p.to_data() for p in population])
        write_json(directory / CHECKPOINT, state)
    speciation_state = state.get('speciation_state')
    write_json(directory / 'config.json', dict(config=asdict(config), partitions=split))
    if progress:
        progress(dict(phase='ready', generation=first, history=history))
    last = config.generations if until_generation is None else min(config.generations, until_generation)
    for generation in range(first, last):
        record, detail = evaluate_generation(population, config, generation, split, progress)
        champion = population[record['champion_index']]
        next_population, ancestry, species_report, speciation_state = reproduce(
            population, record['individuals'], config, generation, progress,
            speciation_state=speciation_state, return_speciation=True)
        record['speciation'] = species_report
        record['summary']['species_count'] = species_report['count']
        record['summary']['species_sizes'] = [g['size'] for g in species_report['groups']]
        record['reproduction'] = ancestry
        history.append(record['summary'])
        write_json(directory / 'generations' / f'{generation:05d}.json', record)
        write_json(directory / 'traces' / f'{generation:05d}-champion.json.gz',
                   dict(generation=generation, individual=champion.to_data(),
                        result=record['individuals'][record['champion_index']], **detail))
        state = dict(schema_version='crt-training-v1', config=asdict(config), next_generation=generation + 1,
                     history=history, partitions=split, population=[p.to_data() for p in next_population], speciation_state=speciation_state)
        write_json(directory / CHECKPOINT, state)
        population = next_population
        if progress:
            progress(dict(phase='generation-complete', generation=generation, summary=record['summary'], history=history))
    if progress:
        progress(dict(phase='finished', history=history))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, help='full saved training configuration for a new run')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--population', type=int)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--assessment-episodes', type=int)
    parser.add_argument('--elites', type=int)
    parser.add_argument('--tournament-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--backend', choices=('python', 'native', 'cuda'))
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--distance-distribution', choices=('normal', 'uniform'))
    parser.add_argument('--distance-mean', type=float)
    parser.add_argument('--boundary-mode', choices=('clamp', 'terminate'))
    parser.add_argument('--include-zero', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--distance-stddev', type=float)
    parser.add_argument('--speciation', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--speciation-threshold', type=float)
    parser.add_argument('--max-species', type=int)
    parser.add_argument('--no-learning', action='store_true', default=None)
    parser.add_argument('--fixed-goal', action='store_true', default=None, help='use the legacy zero-target task')
    parser.add_argument('--no-live', action='store_true', help='print generation summaries instead of redrawing a graph')
    args = parser.parse_args()
    try:
        if args.resume:
            if any(getattr(args, name) is not None for name in
                   ('output', 'config', 'population', 'episodes', 'assessment_episodes', 'elites',
                    'tournament_size', 'seed', 'no_learning', 'fixed_goal', 'backend', 'workers', 'batch_size', 'distance_distribution', 'distance_mean', 'distance_stddev', 'speciation', 'speciation_threshold', 'max_species', 'boundary_mode', 'include_zero')):
                parser.error('resume uses saved configuration; only --generations and --no-live may override')
            config = TrainingConfig.from_data(load_checkpoint(args.resume)['config'])
            if args.generations is not None:
                config = replace(config, generations=args.generations)
            directory = args.resume
        else:
            values = {name: getattr(args, name) for name in
                      ('population', 'generations', 'episodes', 'assessment_episodes', 'elites', 'tournament_size', 'seed', 'backend', 'workers', 'batch_size', 'distance_distribution', 'distance_mean', 'distance_stddev', 'speciation', 'speciation_threshold', 'max_species', 'boundary_mode', 'include_zero')
                      if getattr(args, name) is not None}
            if 'population' in values and 'elites' not in values:
                values['elites'] = min(5, values['population'] - 1)
            if args.fixed_goal:
                values['varying_goals'] = False
            if args.no_learning:
                values['learning'] = False
            config = replace(TrainingConfig.from_data(json.loads(args.config.read_text())), **values) if args.config else TrainingConfig(**values)
            directory = args.output or Path('runs') / datetime.now().strftime('run-%Y%m%d-%H%M%S')
        dashboard = Dashboard(config, live=not args.no_live)
        print(f'Run directory: {directory}', flush=True)
        try:
            run_experiment(config, directory, resume=bool(args.resume), progress=dashboard)
        except KeyboardInterrupt:
            dashboard(dict(phase='interrupted'))
            if (directory / CHECKPOINT).exists():
                print('Interrupted. Resume with: python3 -m static_agent_training.experiment --resume '
                      + shlex.quote(str(directory)))
            else:
                print('Interrupted before the first checkpoint; restart with a new output directory.')
            raise SystemExit(130)
        print(f'Checkpoint: {directory / CHECKPOINT}')
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
