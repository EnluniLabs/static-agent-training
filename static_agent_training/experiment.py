"""Run and resume a single-population evolutionary experiment."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gzip
from . import serialization as json
import os
import shlex
from pathlib import Path

from .evolution import TrainingConfig, evaluate_generation, initialize_population, partitions, reproduce
from .genome import Species
from .terminal import Dashboard
from .structure_profiles import STRUCTURE_ARGUMENTS, add_structure_arguments, apply_structure_arguments


CHECKPOINT = 'checkpoint.json.gz'


def _dump_json(data, handle, population):
    options = dict(sort_keys=True, separators=(',', ':'))
    if population is None:
        json.dump(data, handle, **options)
        return

    # Keep the population array in the existing schema without materializing
    # every agent's serialized data at once.
    handle.write('{')
    for index, key in enumerate(sorted((*data, 'population'))):
        if index:
            handle.write(',')
        handle.write(json.dumps(key) + ':')
        if key == 'population':
            handle.write('[')
            for agent_index, agent in enumerate(population):
                if agent_index:
                    handle.write(',')
                agent_data = (agent.to_data(encoded_development=True) if isinstance(agent, Species)
                              else agent.to_data())
                json.dump(agent_data, handle, **options)
            handle.write(']')
        else:
            json.dump(data[key], handle, **options)
    handle.write('}')


def write_json(path, data, *, population=None):
    """Replace a complete file atomically; interrupted writes preserve its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if str(path).endswith('.gz'):
        # Checkpoints and the candidate collection are replaced each generation;
        # prioritize save latency. Named exports retain archival compression.
        level = 1 if path.name in (CHECKPOINT, 'best-population.json.gz') else 9
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=level) as handle:
            _dump_json(data, handle, population)
    else:
        with temporary.open('w', encoding='utf-8') as handle:
            _dump_json(data, handle, population)
            handle.write('\n')
    # gzip must close before syncing so its trailer is included.
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_checkpoint(directory):
    with gzip.open(Path(directory) / CHECKPOINT, 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    if data['schema_version'] != 'crt-training-v1':
        raise ValueError('unsupported checkpoint schema')
    return data


def update_champion_collection(directory, count=10, record=None):
    """Index distinct saved genotypes using training fitness and earliest ties."""
    directory = Path(directory)
    if type(count) is not int or count < 1:
        raise ValueError('champion count must be a positive integer')
    index = directory / 'champions.json'
    entries = json.loads(index.read_text())['champions'] if record is not None and index.exists() else []
    def saved_records():
        for path in sorted((directory / 'generations').glob('*.json')):
            try:
                item = json.loads(path.read_text())
                if 'generation' in item.get('summary', {}):
                    yield item
            except (OSError, ValueError):
                continue
    records = [record] if record is not None else saved_records()
    for item in records:
        generation = item['summary']['generation']
        trace = f'traces/{generation:05d}-champion.json.gz'
        if not (directory / trace).is_file():
            continue
        if 'individuals' in item:
            result = item['individuals'][item['champion_index']]
        else:
            # Older runs may retain summary-only generation records.
            try:
                with gzip.open(directory / trace, 'rt', encoding='utf-8') as handle:
                    result = json.load(handle)['result']
            except (OSError, EOFError, ValueError, KeyError):
                continue
        entries.append(dict(generation=generation, individual_index=item.get('champion_index', result.get('index')),
                            genotype_id=result['genotype_id'], trace=trace,
                            training_success=result['training']['fitness'],
                            assessment_success=result['assessment']['fitness']))
    entries.sort(key=lambda row: (-row['training_success'], row['generation']))
    unique = {}
    for row in entries:
        unique.setdefault(row['genotype_id'], row)
    collection = dict(schema_version='crt-champion-collection-v1', count=count,
                      selection='training fitness descending; earliest generation breaks ties; distinct genotype IDs',
                      champions=list(unique.values())[:count])
    write_json(index, collection)
    return collection


def run_experiment(config, directory, *, resume=False, until_generation=None, progress=None, champion_count=10, return_state=True):
    """Checkpoint the next inherited population after every completed generation.

    All random streams derive from saved seed and generation/individual indices,
    so replay needs no mutable global RNG state. Interrupted generations rerun.
    `until_generation` allows a bounded segment without changing experiment config.
    Set `return_state=False` when the caller only needs the on-disk checkpoint.
    """
    directory = Path(directory)
    if type(champion_count) is not int or champion_count < 1:
        raise ValueError('champion count must be a positive integer')
    backend_migration = None
    if resume:
        state = load_checkpoint(directory)
        backend_migration = state.get('backend_migration')
        if state['config'].get('backend') == 'cuda':
            backend_migration = dict(previous_backend='cuda', current_backend='native',
                                     resumed_at_generation=state['next_generation'])
        saved = TrainingConfig.from_data(state['config'])
        comparison = replace(config, generations=saved.generations)
        if saved.agent_y and saved.backend == 'python' and config.backend == 'native':
            comparison = replace(comparison, backend='python')
            backend_migration = dict(previous_backend='python', current_backend='native',
                                     resumed_at_generation=state['next_generation'])
        if comparison != saved:
            raise ValueError('resume must retain the saved experiment configuration')
        if config.generations < state['next_generation']:
            raise ValueError('requested generations precede the checkpoint')
        population = [Species.from_data(data) for data in state['population']]
        history, split, first = state['history'], state['partitions'], state['next_generation']
        if len(population) != config.population or len(history) != first:
            raise ValueError('inconsistent checkpoint population or history')
        if state['config'] != asdict(config):
            state['config'] = asdict(config)
            if backend_migration:
                state['backend_migration'] = backend_migration
            write_json(directory / CHECKPOINT, state)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('output directory is not empty; use --resume or a new directory')
        directory.mkdir(parents=True, exist_ok=True)
        population = initialize_population(config, progress)
        history, split, first = [], partitions(config.seed, include_zero=config.include_zero and config.varying_goals, environment=config.environment), 0
        state = dict(schema_version='crt-training-v1', config=asdict(config), next_generation=0,
                     history=history, partitions=split)
        write_json(directory / CHECKPOINT, state, population=population)
    speciation_state = state.get('speciation_state')
    if config.initial_population and not resume:
        from .populations import read_population
        imported = read_population(config.initial_population)
        write_json(directory/'initial-population.json.gz',imported)
        write_json(directory/'candidate-import.json',dict(source=config.initial_population,
            source_environment=imported['config'].get('environment','original'),target_environment=config.environment,
            model=config.model,saved_candidates=len(imported['members']),population=config.population,
            mappings='fixed-sense mappings retained' if config.model=='senses' else 'matching categorical entries retained; unseen tokens initialized',
            extra_slots='deterministic mutated descendants'))
    metadata = dict(config=asdict(config), partitions=split)
    if backend_migration:
        metadata['backend_migration'] = backend_migration
    write_json(directory / 'config.json', metadata)
    update_champion_collection(directory, champion_count)
    if progress:
        progress(dict(phase='ready', generation=first, history=history))
    last = config.generations if until_generation is None else min(config.generations, until_generation)
    for generation in range(first, last):
        del state
        record, detail, lifetime_population = evaluate_generation(population, config, generation, split, progress, return_lifetime=True)
        from .populations import update_best
        update_best(directory,population,record['individuals'],config,generation)
        champion = population[record['champion_index']]
        next_population, ancestry, species_report, speciation_state = reproduce(
            population, record['individuals'], config, generation, progress,
            speciation_state=speciation_state, return_speciation=True, lifetime_population=lifetime_population)
        record['speciation'] = species_report
        record['summary']['species_count'] = species_report['count']
        record['summary']['species_sizes'] = [g['size'] for g in species_report['groups']]
        for event in ancestry:
            index=event['parent_index']
            event['lifetime_inherited']=config.inherit_lifetime
            event['evaluated_parent_id']=record['individuals'][index]['genotype_id']
            event['reproductive_parent_id']=(lifetime_population[index].genotype_id if config.inherit_lifetime
                                               else record['individuals'][index]['genotype_id'])
        record['inheritance']=dict(enabled=config.inherit_lifetime,source='end-of-training' if config.inherit_lifetime else 'inherited-genotype',
                                   changed_parents=sum(p.genotype_id!=r['genotype_id'] for p,r in zip(lifetime_population,record['individuals'])) if config.inherit_lifetime else 0)
        record['reproduction'] = ancestry
        history.append(record['summary'])
        write_json(directory / 'generations' / f'{generation:05d}.json', record)
        write_json(directory / 'traces' / f'{generation:05d}-champion.json.gz',
                   dict(generation=generation, individual=champion.to_data(encoded_development=True),
                        result=record['individuals'][record['champion_index']], **detail))
        update_champion_collection(directory, champion_count, record)
        state = dict(schema_version='crt-training-v1', config=asdict(config), next_generation=generation + 1,
                     history=history, partitions=split, speciation_state=speciation_state)
        if backend_migration:
            state['backend_migration'] = backend_migration
        write_json(directory / CHECKPOINT, state, population=next_population)
        population = next_population
        # Release episode logs and the evaluated agents before the next evaluation.
        if progress:
            progress(dict(phase='generation-complete', generation=generation, summary=record['summary'], history=history))
        del record, detail, lifetime_population, champion, next_population, ancestry
    if progress:
        progress(dict(phase='finished', history=history))
    if return_state:
        if 'population' not in state:
            state['population'] = [agent.to_data() for agent in population]
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, help='full saved training configuration for a new run')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--population', type=int)
    parser.add_argument('--initial-population', help='saved top-performer population to seed a new session')
    parser.add_argument('--population-save-count', type=int)
    parser.add_argument('--environment', choices=('original', 'shifted', 'lights-out', 'switches-2x2', 'lights-out-2x2', 'switchboard'))
    parser.add_argument('--agent-y', action='store_true', default=None)
    parser.add_argument('--model', choices=('categorical', 'senses'))
    parser.add_argument('--receptors', type=int)
    parser.add_argument('--sensory-interface', choices=('legacy-bytes','retina','signals'))
    parser.add_argument('--retina-width', '--receptor-width', type=int)
    parser.add_argument('--retina-height', '--receptor-height', type=int)
    parser.add_argument('--embedding-selection', choices=('sample','argmax'))
    parser.add_argument('--sensory-partitions', type=int, default=None)
    parser.add_argument('--action-policy', choices=('agreement','vote'), default=None)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--assessment-episodes', type=int)
    parser.add_argument('--champion-count', type=int, default=10, help='number of distinct generation champions indexed by training success')
    parser.add_argument('--elites', type=int)
    parser.add_argument('--tournament-size', type=int)
    parser.add_argument('--encoding-seed', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--backend', choices=('python', 'native'))
    parser.add_argument('--reproduction-workers', type=int, help='offspring processes: 0 automatic, 1 serial')
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
    parser.add_argument('--inherit-lifetime', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fixed-goal', action='store_true', default=None, help='use the legacy zero-target task')
    parser.add_argument('--memory-limit-mb', type=int, default=4096, help='process address-space ceiling in MiB')
    parser.add_argument('--no-live', action='store_true', help='print generation summaries instead of redrawing a graph')
    add_structure_arguments(parser)
    args = parser.parse_args()
    try:
        if args.resume:
            if any(getattr(args, name) is not None for name in
                   ('output', 'config', 'population', 'model', 'agent_y', 'receptors', 'sensory_interface', 'retina_width', 'retina_height', 'embedding_selection', 'sensory_partitions', 'action_policy', 'environment', 'initial_population', 'population_save_count', 'episodes', 'assessment_episodes', 'elites',
                    'tournament_size', 'seed', 'encoding_seed', 'no_learning', 'fixed_goal', 'workers', 'reproduction_workers', 'batch_size', 'distance_distribution', 'distance_mean', 'distance_stddev', 'speciation', 'speciation_threshold', 'max_species', 'boundary_mode', 'include_zero', 'inherit_lifetime') + STRUCTURE_ARGUMENTS):
                parser.error('resume uses saved configuration; only --generations and --no-live may override')
            config = TrainingConfig.from_data(load_checkpoint(args.resume)['config'])
            if args.backend is not None:
                if not config.agent_y or args.backend != 'native':
                    parser.error('only Agent Y supports explicit --backend native migration on resume')
                config = replace(config, backend='native')
            if args.generations is not None:
                config = replace(config, generations=args.generations)
            directory = args.resume
        else:
            values = {name: getattr(args, name) for name in
                      ('population', 'model', 'agent_y', 'receptors', 'sensory_interface', 'retina_width', 'retina_height', 'embedding_selection', 'sensory_partitions', 'action_policy', 'environment', 'initial_population', 'population_save_count', 'generations', 'episodes', 'assessment_episodes', 'elites', 'tournament_size', 'seed', 'encoding_seed', 'backend', 'workers', 'reproduction_workers', 'batch_size', 'distance_distribution', 'distance_mean', 'distance_stddev', 'speciation', 'speciation_threshold', 'max_species', 'boundary_mode', 'include_zero', 'inherit_lifetime')
                      if getattr(args, name) is not None}
            if 'population' in values and 'elites' not in values:
                values['elites'] = min(5, values['population'] - 1)
            if args.fixed_goal:
                values['varying_goals'] = False
            if args.no_learning:
                values['learning'] = False
            base = TrainingConfig.from_data(json.loads(args.config.read_text())) if args.config else TrainingConfig()
            interface = values.get('sensory_interface', base.sensory_interface)
            if interface in ('retina', 'signals') and 'receptors' not in values and any(
                    name in values for name in ('sensory_interface','retina_width','retina_height')):
                values['receptors'] = values.get('retina_width', base.retina_width) * values.get('retina_height', base.retina_height)
            config = replace(base, **values)
            config = apply_structure_arguments(config, args)
            directory = args.output or Path('runs') / datetime.now().strftime('run-%Y%m%d-%H%M%S')
        dashboard = Dashboard(config, live=not args.no_live)
        def report_progress(event):
            dashboard(event)
            # Small status only: no histories, agent state or episode logs.
            fields = ('phase', 'generation', 'completed', 'total', 'mean', 'best')
            write_json(Path(directory) / 'progress.json', {k:event[k] for k in fields if k in event})
        print(f'Run directory: {directory}', flush=True)
        try:
            from .diagnostics import diagnostics
            with diagnostics(directory, args.memory_limit_mb):
                run_experiment(config, directory, resume=bool(args.resume), progress=report_progress,
                               champion_count=args.champion_count, return_state=False)
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
