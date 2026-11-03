"""Validated local session launching; process control is limited to owned jobs."""
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

from .catalog import read_json
from .experiment import write_json


KINDS = {
    'training': ('Baseline evolution', 'static_agent_training.experiment'),
    'sensory-training': ('Fixed-senses evolution', 'static_agent_training.experiment'),
    'senses': ('Fixed-senses adaptation', 'static_agent_training.experimental.bit_experiment'),
    'interfaces': ('Old vs fixed senses', 'static_agent_training.experimental.model_comparison'),
    'comparison': ('Baseline treatment comparison', 'static_agent_training.comparison'),
    'model-comparison': ('Evolutionary model comparison', 'static_agent_training.comparison'),
}

TRAINING = ('training', 'sensory-training')
COMPARISONS = ('comparison', 'model-comparison')


def integer(config, key, default, low=1, high=1_000_000):
    value = config.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{key} must be an integer between {low} and {high}')
    return value


def boolean(config, key, default):
    value = config.get(key, default)
    if type(value) is not bool:
        raise ValueError(f'{key} must be true or false')
    return value


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]
    except (OSError, IndexError):
        return None


class SessionManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.registry = self.root / 'runs' / 'ui-sessions'
        self.lock = threading.RLock()
        self.children = {}

    def path(self, relative):
        if not isinstance(relative, str) or not relative:
            raise ValueError('Choose a saved run')
        parent = (self.root / 'runs').resolve()
        path = (parent / relative).resolve()
        if not path.is_relative_to(parent) or path == parent or not path.is_dir():
            raise ValueError('Unknown run')
        return path

    def records(self):
        with self.lock:
            records = []
            for path in self.registry.glob('*.json'):
                record = read_json(path)
                if record.get('schema_version') != 'crt-ui-session-v1':
                    continue
                self.refresh(record)
                records.append(record)
            return sorted(records, key=lambda row: row['created'], reverse=True)

    def refresh(self, record):
        process = self.children.get(record['id'])
        code = process.poll() if process is not None else None
        live = record.get('process_identity') is not None and process_identity(record['pid']) == record.get('process_identity')
        if live:
            return
        if record['state'] in ('running', 'stopping'):
            if code is not None:
                state = 'completed' if code == 0 else 'stopped' if record['state'] == 'stopping' or code in (130, -2) else 'failed'
            else:
                report = read_json(Path(record['directory']) / 'report.json')
                if record['kind'] == 'senses':
                    report = read_json(Path(record['directory']) / 'run.json')
                state = {'completed': 'completed', 'failed': 'failed'}.get(report.get('status'), 'stopped')
                if record['kind'] in TRAINING:
                    from .run_records import read_summary
                    files = sorted((Path(record['directory']) / 'generations').glob('*.json'))
                    config = read_json(Path(record['directory']) / 'config.json').get('config', {})
                    if files and read_summary(files[-1])['generation'] + 1 >= config.get('generations', float('inf')):
                        state = 'completed'
            record.update(state=state, returncode=code, finished=time.time())
            write_json(self.registry / f"{record['id']}.json", record)

    def command(self, kind, config, directory):
        if kind not in KINDS or not isinstance(config, dict):
            raise ValueError('Choose a supported session type and configuration')
        allowed = {'episodes', 'assessment_episodes', 'seed', 'notes'}
        if kind in (*TRAINING, *COMPARISONS, 'interfaces'):
            allowed |= {'population', 'memory_limit_mb'}
        if kind in (*TRAINING, *COMPARISONS):
            allowed |= {'generations', 'backend', 'workers', 'batch_size'}
        if kind in TRAINING:
            allowed |= {'learning', 'inherit_lifetime', 'speciation', 'champion_count'}
        if kind in ('senses', 'sensory-training', 'model-comparison'):
            allowed |= {'receptors'}
        if kind == 'senses':
            allowed |= {'receptors', 'candidate'}
        if kind == 'interfaces':
            allowed |= {'receptors', 'seeds'}
        if kind in COMPARISONS:
            allowed |= {'seeds', 'treatment', 'benchmark_episodes', 'model', 'receptors'}
        if set(config) - allowed:
            raise ValueError('Unsupported settings: ' + ', '.join(sorted(set(config) - allowed)))
        notes = config.get('notes', '')
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError('Notes must be text, at most 2000 characters')
        args = ['--output', str(directory), '--episodes', str(integer(config, 'episodes', 100)),
                '--assessment-episodes', str(integer(config, 'assessment_episodes', 2000 if kind == 'interfaces' else 200))]
        if kind in (*TRAINING, 'senses'):
            args += ['--seed', str(integer(config, 'seed', 48, 0, 2**32 - 1))]
        if kind in (*TRAINING, *COMPARISONS, 'interfaces'):
            args += ['--population', str(integer(config, 'population', 16 if kind == 'interfaces' else 32, 1 if kind == 'interfaces' else 2)),
                     '--memory-limit-mb', str(integer(config, 'memory_limit_mb', 4096, 256, 131072))]
        if kind in (*TRAINING, *COMPARISONS):
            backend = config.get('backend', 'native')
            if backend not in ('python', 'native', 'cuda'):
                raise ValueError('Unknown evaluator backend')
            args += ['--generations', str(integer(config, 'generations', 50)), '--backend', backend,
                     '--workers', str(integer(config, 'workers', 4, 1, 1024)),
                     '--batch-size', str(integer(config, 'batch_size', 256))]
        if kind in TRAINING:
            args += ['--model', 'senses' if kind == 'sensory-training' else 'categorical']
            args += ['--elites', str(min(5, config.get('population', 32) - 1)), '--no-live',
                     '--champion-count', str(integer(config, 'champion_count', 10))]
            if not boolean(config, 'learning', True):
                args.append('--no-learning')
            args.append('--inherit-lifetime' if boolean(config, 'inherit_lifetime', False) else '--no-inherit-lifetime')
            args.append('--speciation' if boolean(config, 'speciation', False) else '--no-speciation')
        if kind in ('senses', 'interfaces', 'sensory-training', *COMPARISONS):
            receptors = integer(config, 'receptors', 256, 16)
            if receptors % 16:
                raise ValueError('Receptor count must be a multiple of 16')
            args += ['--receptors', str(receptors)]
        if kind in ('interfaces', *COMPARISONS):
            seeds = config.get('seeds', [48, 49, 50])
            if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 32 or
                    any(type(s) is not int or not 0 <= s <= 2**32 - 1 for s in seeds) or len(set(seeds)) != len(seeds)):
                raise ValueError('Provide up to 32 distinct integer seeds')
            args += ['--seeds', *map(str, seeds)]
        if kind in COMPARISONS:
            model = config.get('model', 'categorical')
            if model not in ('categorical', 'senses'):
                raise ValueError('Unknown model')
            args += ['--model', model]
            treatment = 'model' if kind == 'model-comparison' else config.get('treatment', 'learning')
            if treatment not in ('learning', 'inheritance', 'speciation', 'model'):
                raise ValueError('Unknown treatment')
            args += ['--treatment', treatment, '--benchmark-episodes', str(integer(config, 'benchmark_episodes', 2000))]
            args += ['--runs-root', str(self.root / 'runs')]
        if kind == 'senses':
            if config.get('candidate'):
                candidate = self.path(config['candidate'])
                if not (candidate / 'config.json').is_file():
                    raise ValueError('Structure source must be a baseline training run')
                args += ['--candidate', str(candidate)]
            args += ['--notes', notes]
        return KINDS[kind][1], args

    def spawn(self, kind, directory, args, *, name, config, previous=None):
        python = self.root / '.venv' / 'bin' / 'python'
        python = str(python) if python.exists() else sys.executable
        command = [python, '-u', '-m', KINDS[kind][1], *args]
        identifier = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
        log = directory.parent / (directory.name + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parent.parent)
        environment['PYTHONPATH'] = source + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
        with log.open('ab', buffering=0) as output:
            process = subprocess.Popen(command, cwd=self.root, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        record = dict(schema_version='crt-ui-session-v1', id=identifier, kind=kind, name=name,
                      directory=str(directory), run=directory.relative_to(self.root / 'runs').as_posix(),
                      log=str(log), created=time.time(), pid=process.pid,
                      process_identity=process_identity(process.pid), state='running', command=command,
                      config=config, previous=previous)
        self.children[identifier] = process
        try:
            write_json(self.registry / f'{identifier}.json', record)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self.children.pop(identifier, None)
            raise
        return record

    def launch(self, data):
        if not isinstance(data, dict) or set(data) - {'kind', 'name', 'config'}:
            raise ValueError('Expected a session type, name and configuration')
        kind, name, config = data.get('kind'), data.get('name', ''), data.get('config', {})
        if not isinstance(name, str) or len(name) > 80:
            raise ValueError('Session name must be at most 80 characters')
        slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', name).strip('-') or 'session'
        parent = self.root / 'runs'
        if kind in ('interfaces', *COMPARISONS):
            parent /= 'interfaces' if kind == 'interfaces' else 'comparisons'
        directory = parent / (datetime.now().strftime('ui-%Y%m%d-%H%M%S-') + slug + '-' + uuid.uuid4().hex[:6])
        module, args = self.command(kind, config, directory)
        if kind == 'senses':
            args += ['--label', name or KINDS[kind][0]]
        with self.lock:
            return self.spawn(kind, directory, args, name=name or KINDS[kind][0], config=config)

    def get(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{8}', identifier):
            raise ValueError('Unknown managed session')
        record = read_json(self.registry / f'{identifier}.json')
        if not record:
            raise ValueError('Unknown managed session')
        self.refresh(record)
        return record

    def stop(self, identifier):
        with self.lock:
            record = self.get(identifier)
            if record['state'] != 'running':
                raise ValueError('This session is not running')
            if record.get('process_identity') is None or process_identity(record['pid']) != record['process_identity']:
                raise ValueError('The original process is no longer running')
            # Session processes are created in their own process group.
            os.killpg(record['pid'], signal.SIGINT)
            record['state'] = 'stopping'
            write_json(self.registry / f'{identifier}.json', record)
            return record

    def resume(self, identifier, generations=None):
        with self.lock:
            record = self.get(identifier)
            if record['state'] in ('running', 'stopping'):
                raise ValueError('This session is still active')
            if record['state'] == 'completed' and (record['kind'] not in TRAINING or generations is None):
                raise ValueError('A completed baseline session needs a larger generation target')
            if any(r['directory'] == record['directory'] and r['state'] in ('running', 'stopping') for r in self.records()):
                raise ValueError('This run already has an active session')
            kind, directory = record['kind'], Path(record['directory'])
            required = {'training': 'checkpoint.json.gz', 'sensory-training': 'checkpoint.json.gz',
                        'interfaces': 'plan.json', 'comparison': 'manifest.json', 'model-comparison': 'manifest.json'}
            if kind not in required:
                raise ValueError('Fixed-senses adaptation does not yet support resume')
            if not (directory / required[kind]).is_file():
                raise ValueError('This session has not saved a resumable checkpoint yet')
            args = ['--resume', str(directory)]
            if kind in TRAINING:
                args.append('--no-live')
                args += ['--champion-count', str(record['config'].get('champion_count', 10))]
                if generations is not None:
                    config = {'generations': generations}
                    target = integer(config, 'generations', 50)
                    old_target = read_json(directory / 'config.json')['config']['generations']
                    if target < old_target:
                        raise ValueError('Generation target cannot be reduced')
                    if record['state'] == 'completed' and target <= old_target:
                        raise ValueError('Choose a larger generation target to extend completed training')
                    args += ['--generations', str(target)]
            elif generations is not None:
                raise ValueError('Generation overrides apply only to baseline training')
            args += ['--memory-limit-mb', str(record['config'].get('memory_limit_mb', 4096))]
            return self.spawn(kind, directory, args, name=record['name'], config=record['config'], previous=identifier)

    def resume_run(self, relative, generations=None):
        """Adopt a saved resumable run, without taking ownership of a live process."""
        with self.lock:
            directory = self.path(relative)
            owned = next((r for r in self.records() if r['directory'] == str(directory)), None)
            if owned:
                return self.resume(owned['id'], generations)
            if (directory / 'config.json').is_file():
                from .status import training_processes
                if str(directory) in training_processes():
                    raise ValueError('This baseline run is already running outside this workspace')
                if not (directory / 'checkpoint.json.gz').is_file():
                    raise ValueError('No saved checkpoint to resume')
                config = read_json(directory / 'config.json')['config']
                kind = 'sensory-training' if config.get('model') == 'senses' else 'training'
                old_target = config['generations']
                if generations is not None:
                    target = integer({'generations': generations}, 'generations', old_target)
                    if target < old_target:
                        raise ValueError('Generation target cannot be reduced')
                from .run_records import read_summary
                files = sorted((directory / 'generations').glob('*.json'))
                if files and read_summary(files[-1])['generation'] + 1 >= old_target and (generations is None or generations <= old_target):
                    raise ValueError('Choose a larger generation target to extend completed training')
                args = ['--resume', str(directory), '--no-live', '--champion-count',
                        str(read_json(directory / 'champions.json').get('count', 10))]
                if generations is not None:
                    args += ['--generations', str(generations)]
            else:
                from .catalog import sensory_state
                report = read_json(directory / 'report.json')
                if report.get('schema_version') == 'crt-interface-comparison-v1':
                    kind = 'interfaces'
                    config = read_json(directory / 'plan.json')
                elif (directory / 'manifest.json').is_file():
                    kind = 'model-comparison' if report.get('treatment') == 'model' else 'comparison'
                    config = read_json(directory / 'manifest.json').get('config', {})
                    # Check the comparison parent as well as any active arms.
                    from .status import training_processes
                    manifest = read_json(directory / 'manifest.json')
                    if any(arm.get('directory') in training_processes() for arm in manifest.get('arms', [])):
                        raise ValueError('This comparison already has live training arms')
                else:
                    raise ValueError('This session type cannot be resumed')
                module = KINDS[kind][1]
                if sensory_state(directory, report, module=module) == 'running':
                    raise ValueError('This session is already running outside the workspace')
                if report.get('status') == 'completed':
                    raise ValueError('This comparison has already completed')
                if generations is not None:
                    raise ValueError('Generation overrides apply only to baseline training')
                args = ['--resume', str(directory)]
            args += ['--memory-limit-mb', '4096']
            return self.spawn(kind, directory, args, name=directory.name, config=config, previous='external')
