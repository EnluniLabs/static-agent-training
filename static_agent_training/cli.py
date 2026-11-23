"""One entry point for existing tools and experimental sensory models."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from .catalog import catalog, describe, read_json, resolve_run


HELP = '''Usage: ./project COMMAND [options]

Inspect and organize (no experiments launched):
  runs [--kind KIND] [--all] [--json]
  show RUN                  Configuration, reports and artifact locations
  champions RUN [--rebuild]  List top saved generation champions
  status [--watch] [--json]  Live baseline training status
  doctor                    Interpreter, optional dependencies and disk space
  clean [--dry-run]          Remove generated Python bytecode only
  help                      Show this command map

Run tools (only when explicitly invoked):
  agent-d [options]         Run Agent D evolution
  agent-z [options]         Run or assess Agent Z
  train [--model senses]     Evolutionary training for either model
  resume [RUN] [options]     Resume evolutionary training
  compare [--treatment model] Matched evolutionary comparisons
  compare-models [options]   Matched old versus fixed-senses lifetime benchmark
  senses [options]           Experimental fixed-senses lifetime/transfer run
  benchmark RUN [options]    Frozen trained champion versus random movement
  audit RUN [options]        Goal-input and movement diagnostics for either model
  speed [options]            Evaluator throughput benchmark
  assess [options]           Frozen saved-population assessment
  populations save|benchmark Save reusable populations or compare several sets
  stream [options]          Inspect byte streams or run a saved sensory agent
  switchboard-test [options] Matched single/parallel three-panel experiment
  trace-learning [options]   Record one learning event in both models
  demo [RUN] [options]       Model workspace (optional legacy baseline source)
  ui [options]              Model workspace, session control and playground
  check [test.module ...]    Run regression tests, not training experiments

RUN: a directory, an exact saved name, latest-training, latest-senses, or
latest-sensory-training, latest-model-comparison, latest-comparison, or
latest-interfaces. Each alias selects its stated run type.
All relative paths are resolved from the project root. Existing ./train,
./status and ./demo commands still work. Read USE.md for examples and limits.
'''


def execute(module, args):
    os.execv(sys.executable, [sys.executable, '-m', module, *map(str, args)])


def has_option(args, name):
    return any(arg == name or arg.startswith(name + '=') for arg in args)


def replace_option(args, flag, transform):
    args = list(args)
    for i, value in enumerate(args):
        if value == flag and i + 1 < len(args):
            args[i + 1] = str(transform(args[i + 1]))
        elif value.startswith(flag + '='):
            args[i] = flag + '=' + str(transform(value.split('=', 1)[1]))
    return args


def cleanup(root, *, dry_run=False):
    """Only bytecode under source/tests; never runs, native caches or .venv."""
    paths = []
    for base in (root / 'static_agent_training', root / 'tests'):
        paths.extend(sorted(base.rglob('__pycache__')))
        paths.extend(sorted(p for p in base.rglob('*.py[co]') if '__pycache__' not in p.parts))
    for path in paths:
        print(('Would remove ' if dry_run else 'Removing ') + str(path.relative_to(root)))
        if not dry_run:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    if not paths:
        print('No generated Python bytecode found.')


def main(root, argv=None):
    root = Path(root).resolve()
    args = list(sys.argv[1:] if argv is None else argv)
    os.chdir(root)
    if not args or args[0] in ('help', '-h', '--help'):
        print(HELP)
        return
    command, args = args[0], args[1:]
    try:
        if command in ('agent-d', 'agent-z'):
            execute('static_agent_training.experimental.' + command.replace('-', '_') + '_experiment', args)
            return
        if command == 'runs':
            parser = argparse.ArgumentParser(prog='./project runs')
            parser.add_argument('--kind', choices=('training', 'sensory-training', 'senses', 'comparison', 'model-comparison', 'interfaces', 'population-benchmark','starting-population-benchmark','assessment'))
            parser.add_argument('--limit', type=int, default=20)
            parser.add_argument('--all', action='store_true')
            parser.add_argument('--json', action='store_true')
            options = parser.parse_args(args)
            if options.limit < 1:
                parser.error('limit must be positive')
            rows = [r for r in catalog(root) if options.kind is None or r['kind'] == options.kind]
            count = len(rows)
            rows = rows if options.all else rows[:options.limit]
            if options.json:
                print(json.dumps(rows, indent=2))
            else:
                for row in rows:
                    print(f"{row['kind']:10} {row['state']:10} {row.get('progress') or '':12} {row['name']}")
                print(f'{len(rows)} of {count} saved runs. Use --all or --kind to change the listing.')
            return
        if command in ('show', 'champions'):
            parser = argparse.ArgumentParser(prog=f'./project {command}')
            parser.add_argument('run')
            if command == 'champions':
                parser.add_argument('--count', type=int, default=10)
                parser.add_argument('--rebuild', action='store_true')
            options = parser.parse_args(args)
            directory = resolve_run(root, options.run)
            if command == 'champions' and not (directory / 'config.json').is_file():
                raise ValueError('Champion collections require an evolutionary training run')
            if command == 'show':
                data = describe(directory)
                data['catalog'] = next((r for r in catalog(root) if Path(r['path']) == directory), None)
            else:
                if options.count < 1:
                    parser.error('count must be positive')
                if options.rebuild:
                    from .experiment import update_champion_collection
                    data = update_champion_collection(directory, options.count)
                else:
                    data = read_json(directory / 'champions.json')
                    if not data:
                        parser.error('no champion index; use --rebuild to create it from saved traces')
                    data['champions'] = data['champions'][:options.count]
            print(json.dumps(data, indent=2))
            return
        if command == 'clean':
            parser = argparse.ArgumentParser(prog='./project clean')
            parser.add_argument('--dry-run', action='store_true')
            options = parser.parse_args(args)
            cleanup(root, dry_run=options.dry_run)
            return
        if command == 'doctor':
            argparse.ArgumentParser(prog='./project doctor').parse_args(args)
            print(f'Project: {root}\nPython: {sys.executable}\nVersion: {sys.version.split()[0]}')
            for name in ('numpy',):
                print(f'Optional {name}: ' + ('installed' if importlib.util.find_spec(name) else 'not installed'))
            print(f'C++ compiler: {shutil.which("c++") or "not found"}')
            print(f'Free disk: {shutil.disk_usage(root).free / 2**30:.1f} GiB')
            return
        if command == 'check':
            if not args:
                args = ['discover', '-s', 'tests']
            execute('unittest', args)
            return
        if command in ('train', 'resume'):
            if any(arg in ('-h', '--help') for arg in args):
                execute('static_agent_training.experiment', ['--help'])
                return
            if command == 'resume' and args and not args[0].startswith('-'):
                path = resolve_run(root, args[0])
                if not (path / 'config.json').is_file():
                    raise ValueError('Resume requires an evolutionary training run')
                args[0] = str(path)
            from .launcher import main as train_main
            train_main(root, (['resume'] if command == 'resume' else []) + args)
            return
        modules = dict(status='status', compare='comparison', senses='experimental.bit_experiment',
                       benchmark='benchmark', audit='bias_audit', speed='speed', demo='play',
                       ui='play', assess='assessments', populations='populations', **{'stream': 'stream_inspect', 'switchboard-test': 'switchboard_benchmark', 'trace-learning': 'learning_trace', 'compare-models': 'experimental.model_comparison'})
        if command not in modules:
            raise ValueError(f'Unknown command: {command}. Use ./project help.')
        help_requested = any(a in ('-h', '--help') for a in args)
        if not help_requested:
            if command == 'benchmark' and args and not args[0].startswith('-'):
                path = resolve_run(root, args[0])
                if not (path / 'config.json').is_file():
                    raise ValueError('Benchmark requires an evolutionary training run')
                args[0] = str(path)
            if command == 'audit' and args and not args[0].startswith('-'):
                args[0] = str(resolve_run(root, args[0]))
            if command == 'demo' and args and not args[0].startswith('-'):
                # Retain the playground's exported-file support.
                if not Path(args[0]).is_file():
                    args[0] = str(resolve_run(root, args[0]))
            if command == 'senses':
                args = replace_option(args, '--candidate', lambda v: resolve_run(root, v, kind='training'))
                if not has_option(args, '--output'):
                    args += ['--output', str(root / 'runs' / datetime.now().strftime('senses-%Y%m%d-%H%M%S-%f'))]
            if command == 'compare':
                args = replace_option(args, '--resume', lambda v: resolve_run(root, v))
            if command == 'compare-models':
                args = replace_option(args, '--resume', lambda v: resolve_run(root, v, kind='interfaces'))
                if not has_option(args, '--output') and not has_option(args, '--resume'):
                    args += ['--output', str(root / 'runs' / 'interfaces' /
                                            datetime.now().strftime('old-vs-senses-%Y%m%d-%H%M%S-%f'))]
        execute('static_agent_training.' + modules[command], args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
