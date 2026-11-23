"""Validated local session launching; process control is limited to owned jobs."""
from datetime import datetime
import os
import math
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
from .structure_profiles import STRUCTURE_PRESETS


KINDS = {
    'training': ('Baseline evolution', 'static_agent_training.experiment'),
    'sensory-training': ('Fixed-senses evolution', 'static_agent_training.experiment'),
    'senses': ('Fixed-senses adaptation', 'static_agent_training.experimental.bit_experiment'),
    'interfaces': ('Old vs fixed senses', 'static_agent_training.experimental.model_comparison'),
    'comparison': ('Baseline treatment comparison', 'static_agent_training.comparison'),
    'starting-population-benchmark': ('Evolutionary starting-population benchmark', 'static_agent_training.comparison'),
    'assessment': ('Frozen population assessment','static_agent_training.assessments'),
    'population-benchmark': ('Saved population benchmark', 'static_agent_training.populations'),
    'model-comparison': ('Evolutionary model comparison', 'static_agent_training.comparison'),
}

TRAINING = ('training', 'sensory-training')
COMPARISONS = ('comparison', 'model-comparison', 'starting-population-benchmark')


STRUCTURE_INTEGERS = ('initial_pairs', 'chain_budget', 'seed_term_max_nodes',
    'seed_term_max_depth', 'development_max_nodes', 'development_max_transitions', 'new_pair_budget')
STRUCTURE_PROBABILITIES = ('combinator_probability', 'subtree_probability',
    'term_probability', 'insertion_probability', 'deletion_probability')
STRUCTURE_KEYS = set(STRUCTURE_PRESETS['baseline']) | {'structure_preset'}


def structure_arguments(config):
    # The backend resolves scientific presets. Absent settings stay absent.
    args = []
    if 'structure_preset' in config:
        preset = config['structure_preset']
        if not isinstance(preset, str) or preset not in STRUCTURE_PRESETS:
            raise ValueError('Unknown structure preset')
        args += ['--structure-preset', preset]
    for key in STRUCTURE_INTEGERS:
        if key in config:
            minimum = 0 if key in ('chain_budget', 'new_pair_budget') else 2 if key in ('seed_term_max_nodes', 'seed_term_max_depth') else 1
            args += ['--' + key.replace('_', '-'), str(integer(config, key, None, minimum, 2**31 - 1))]
    for key in STRUCTURE_PROBABILITIES:
        if key in config:
            value = config[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must be a finite number between 0 and 1')
            args += ['--' + key.replace('_', '-'), str(value)]
    return args


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

    def population_path(self, relative):
        parent = (self.root/'runs/populations').resolve()
        if not isinstance(relative,str):
            raise ValueError('Choose a saved population')
        path = (self.root/'runs'/relative).resolve()
        if not path.is_relative_to(parent) or not path.is_file() or not path.name.endswith('.json.gz'):
            raise ValueError('Unknown saved population')
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
        if kind == 'assessment':
            if set(config)-{'populations','environment','episodes','seed','workers','boundary_mode','notes'}:
                raise ValueError('Frozen assessments only accept assessment settings')
            selected=config.get('populations',[])
            if not isinstance(selected,list) or not 1<=len(selected)<=32 or any(not isinstance(x,str) for x in selected) or len(set(selected))!=len(selected):
                raise ValueError('Choose one or more distinct saved populations, at most 32')
            paths=[self.population_path(value) for value in selected]
            environment=config.get('environment','original')
            from .assessments import validate_inputs
            validate_inputs(paths,environment)
            boundary=config.get('boundary_mode','clamp')
            if boundary not in ('clamp','terminate'):
                raise ValueError('Unknown boundary mode')
            return KINDS[kind][1],['--output',str(directory),'--populations',*[str(p) for p in paths],
                '--environment',environment,'--episodes',str(integer(config,'episodes',200)),
                '--seed',str(integer(config,'seed',48,0,2**32-1)),'--workers',str(integer(config,'workers',4,1,1024)),
                '--boundary-mode',boundary]
        if kind == 'population-benchmark':
            from .environments import environment_spec
            selected = config.get('populations',[])
            if set(config)-{'populations','environment','episodes','assessment_episodes','seed','backend','workers','notes'}:
                raise ValueError('Unsupported population benchmark settings')
            if not isinstance(selected,list) or not 2 <= len(selected) <= 32 or len(set(selected))!=len(selected):
                raise ValueError('Choose at least two distinct saved populations')
            environment = config.get('environment','original'); environment_spec(environment)
            backend = config.get('backend','native')
            if backend not in ('native',):
                raise ValueError('Population benchmark requires native CPU evaluation')
            return KINDS[kind][1], ['benchmark','--output',str(directory),'--populations',
                *[str(self.population_path(value)) for value in selected],'--environment',environment,'--backend',backend,
                '--episodes',str(integer(config,'episodes',100)),'--assessment-episodes',str(integer(config,'assessment_episodes',200)),
                '--seed',str(integer(config,'seed',48,0,2**32-1)),'--workers',str(integer(config,'workers',4,1,1024))]
        config = dict(config)
        if kind in ('sensory-training','starting-population-benchmark'):
            from .populations import read_population, sensory_apparatus
            keys = ('initial_population',) if kind == 'sensory-training' else ('population_a','population_b')
            for key in keys:
                if not config.get(key):
                    continue
                saved = read_population(self.population_path(config[key]))
                if saved['config'].get('model','categorical') != 'senses':
                    continue
                for name,value in sensory_apparatus(saved['config']).items():
                    if name in config and config[name] != value:
                        raise ValueError(f'Saved population {name} conflicts with manual override')
                    config[name] = value
        allowed = {'episodes', 'assessment_episodes', 'seed', 'notes'}
        if kind in (*TRAINING, *COMPARISONS, 'interfaces'):
            allowed |= {'population', 'memory_limit_mb'}
        if kind in (*TRAINING, *COMPARISONS):
            allowed |= {'generations', 'backend', 'workers', 'reproduction_workers', 'batch_size', 'environment'} | STRUCTURE_KEYS
        if kind in (*TRAINING,'starting-population-benchmark'):
            allowed |= {'learning', 'inherit_lifetime', 'speciation', 'champion_count', 'initial_population', 'population_save_count'}
        if kind in ('senses', 'sensory-training', 'model-comparison'):
            allowed |= {'receptors', 'embedding_selection'}
        if kind == 'sensory-training':
            allowed |= {'agent_y'}
        if kind in ('senses', 'sensory-training', *COMPARISONS):
            allowed |= {'sensory_partitions', 'action_policy'}
        if kind in ('sensory-training', *COMPARISONS):
            allowed |= {'sensory_interface','retina_width','retina_height'}
        if kind == 'senses':
            allowed |= {'receptors', 'candidate', 'embedding_selection'}
        if kind == 'interfaces':
            allowed |= {'receptors', 'seeds', 'embedding_selection'}
        if kind in COMPARISONS:
            allowed |= {'seeds', 'treatment', 'benchmark_episodes', 'model', 'receptors', 'embedding_selection'}
        if kind == 'starting-population-benchmark':
            allowed |= {'population_a','population_b'}
        if config.get('agent_y'):
            if kind != 'sensory-training' or not boolean(config, 'agent_y', False):
                raise ValueError('Agent Y requires sensory evolutionary training')
        if set(config) - allowed:
            raise ValueError('Unsupported settings: ' + ', '.join(sorted(set(config) - allowed)))
        notes = config.get('notes', '')
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError('Notes must be text, at most 2000 characters')
        args = ['--output', str(directory), '--episodes', str(integer(config, 'episodes', 100)),
                '--assessment-episodes', str(integer(config, 'assessment_episodes', 2000 if kind == 'interfaces' else 200))]
        if kind in (*TRAINING, *COMPARISONS):
            args += structure_arguments(config)
        if kind in (*TRAINING, 'senses'):
            args += ['--seed', str(integer(config, 'seed', 48, 0, 2**32 - 1))]
        if kind in (*TRAINING, *COMPARISONS, 'interfaces'):
            args += ['--population', str(integer(config, 'population', 16 if kind == 'interfaces' else 32, 1 if kind == 'interfaces' else 2)),
                     '--memory-limit-mb', str(integer(config, 'memory_limit_mb', 4096, 256, 131072))]
        if kind in (*TRAINING, *COMPARISONS):
            from .environments import environment_spec
            environment = config.get('environment','original'); environment_spec(environment)
            args += ['--environment',environment]
            backend = config.get('backend', 'native')
            if backend not in ('python', 'native'):
                raise ValueError('Unknown evaluator backend')
            args += ['--generations', str(integer(config, 'generations', 50)), '--backend', backend,
                     '--workers', str(integer(config, 'workers', 4, 1, 1024)),
                     '--reproduction-workers', str(integer(config, 'reproduction_workers', 0, 0, 1024)),
                     '--batch-size', str(integer(config, 'batch_size', 256))]
        if kind in TRAINING:
            args += ['--population-save-count',str(integer(config,'population_save_count',10))]
            if config.get('initial_population'):
                from .populations import read_population, validate_compatibility, sensory_apparatus
                from .evolution import TrainingConfig
                source = self.population_path(config['initial_population'])
                saved = read_population(source)
                imported = saved['config']
                validate_compatibility(saved,TrainingConfig(model='senses' if kind=='sensory-training' else 'categorical', agent_y=config.get('agent_y',False),
                    environment=environment,**sensory_apparatus(config),
                    varying_goals=imported.get('varying_goals',False),encoding_seed=imported.get('encoding_seed',0)),allow_new_environment=True)
                args += ['--initial-population',str(source),'--encoding-seed',str(imported.get('encoding_seed',0))]
                if not imported.get('varying_goals',False):args.append('--fixed-goal')
            args += ['--model', 'senses' if kind == 'sensory-training' else 'categorical']
            if boolean(config, 'agent_y', False):
                args.append('--agent-y')
            args += ['--elites', str(min(5, config.get('population', 32) - 1)), '--no-live',
                     '--champion-count', str(integer(config, 'champion_count', 10))]
            if not boolean(config, 'learning', True):
                args.append('--no-learning')
            args.append('--inherit-lifetime' if boolean(config, 'inherit_lifetime', False) else '--no-inherit-lifetime')
            args.append('--speciation' if boolean(config, 'speciation', False) else '--no-speciation')
        if kind in ('senses', 'interfaces', 'sensory-training', *COMPARISONS):
            selection=config.get('embedding_selection','argmax')
            if selection not in ('sample','argmax'):raise ValueError('Embedding selection must be sample or argmax')
            args += ['--embedding-selection',selection]
            interface = config.get('sensory_interface','legacy-bytes')
            if interface not in ('legacy-bytes','retina','signals'):
                raise ValueError('Unknown sensory interface')
            width = integer(config,'retina_width',16)
            height = integer(config,'retina_height',16)
            receptors = integer(config, 'receptors', width*height if interface in ('retina','signals') else 256, 1 if interface in ('retina','signals') else 16)
            if interface in ('retina', 'signals') and receptors != width*height:
                raise ValueError('Receptor resolution must equal layout width × height')
            if interface == 'legacy-bytes' and receptors % 16:
                raise ValueError('Receptor count must be a multiple of 16')
            args += ['--receptors', str(receptors)]
            if kind in ('sensory-training', *COMPARISONS):
                args += ['--sensory-interface',interface,'--retina-width',str(width),'--retina-height',str(height)]
        if kind in ('senses', 'sensory-training', *COMPARISONS):
            sensory_partitions = integer(config, 'sensory_partitions', 1, 1, receptors)
            action_policy = config.get('action_policy', 'agreement')
            if action_policy not in ('agreement', 'vote'):
                raise ValueError('Action policy must be agreement or vote')
            if sensory_partitions > 1:
                if kind in COMPARISONS and config.get('model', 'categorical') != 'senses':
                    raise ValueError('Categorical models cannot use multiple sensory partitions')
                if selection != 'argmax':
                    raise ValueError('Multiple sensory partitions require argmax embedding selection')
            if kind in ('senses', 'sensory-training') or 'sensory_partitions' in config or 'action_policy' in config:
                args += ['--sensory-partitions', str(sensory_partitions), '--action-policy', action_policy]
        if kind in ('interfaces', *COMPARISONS):
            seeds = [integer(config,'seed',48,0,2**32-1)] if kind=='starting-population-benchmark' else config.get('seeds', [48, 49, 50])
            if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 32 or
                    any(type(s) is not int or not 0 <= s <= 2**32 - 1 for s in seeds) or len(set(seeds)) != len(seeds)):
                raise ValueError('Provide up to 32 distinct integer seeds')
            args += ['--seeds', *map(str, seeds)]
        if kind in COMPARISONS:
            model = config.get('model', 'categorical')
            if model not in ('categorical', 'senses'):
                raise ValueError('Unknown model')
            args += ['--model', model]
            if kind=='starting-population-benchmark':
                from .populations import read_population, validate_compatibility, sensory_apparatus
                from .evolution import TrainingConfig
                sources={}
                for arm in ('a','b'):
                    selected=config.get('population_'+arm)
                    if selected:
                        path=self.population_path(selected); saved=read_population(path)
                        sources[arm]=(path,saved)
                modes={(saved['config'].get('encoding_seed',0),saved['config'].get('varying_goals',False)) for _,saved in sources.values()}
                if len(modes)>1:
                    raise ValueError('Both starting populations must use the same action encoding and goal-input mode')
                encoding,goals=next(iter(modes)) if modes else (0,True)
                shared=TrainingConfig(model=model,environment=environment,**sensory_apparatus(config),encoding_seed=encoding,varying_goals=goals)
                for arm,(path,saved) in sources.items():
                    validate_compatibility(saved,shared,allow_new_environment=True)
                    args+=['--population-'+arm,str(path)]
                args+=['--encoding-seed',str(encoding),'--population-save-count',str(integer(config,'population_save_count',10)),
                       '--champion-count',str(integer(config,'champion_count',10))]
                if not goals:args.append('--fixed-goal')
                if not boolean(config,'learning',True):args.append('--no-learning')
                args.append('--inherit-lifetime' if boolean(config,'inherit_lifetime',False) else '--no-inherit-lifetime')
                args.append('--speciation' if boolean(config,'speciation',False) else '--no-speciation')
            treatment = 'starting-population' if kind=='starting-population-benchmark' else 'model' if kind == 'model-comparison' else config.get('treatment', 'learning')
            if treatment not in ('learning', 'inheritance', 'speciation', 'model','starting-population'):
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

    def resume(self, identifier, generations=None, backend=None):
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
                        'interfaces': 'plan.json', 'comparison': 'manifest.json', 'model-comparison': 'manifest.json', 'starting-population-benchmark':'manifest.json'}
            if kind not in required:
                raise ValueError('Fixed-senses adaptation does not yet support resume')
            if not (directory / required[kind]).is_file():
                raise ValueError('This session has not saved a resumable checkpoint yet')
            args = ['--resume', str(directory)]
            if backend is not None:
                if backend != 'native' or not record['config'].get('agent_y'):
                    raise ValueError('Only Agent Y can migrate to native on resume')
                args += ['--backend', 'native']
                record = dict(record, config=dict(record['config'], backend='native'))
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

    def resume_run(self, relative, generations=None, backend=None):
        """Adopt a saved resumable run, without taking ownership of a live process."""
        with self.lock:
            directory = self.path(relative)
            owned = next((r for r in self.records() if r['directory'] == str(directory)), None)
            if owned:
                return self.resume(owned['id'], generations, backend)
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
                if backend is not None:
                    if backend != 'native' or not config.get('agent_y'):
                        raise ValueError('Only Agent Y can migrate to native on resume')
                    args += ['--backend', 'native']
                    config = dict(config, backend='native')
                if generations is not None:
                    args += ['--generations', str(generations)]
            else:
                from .catalog import sensory_state
                report = read_json(directory / 'report.json')
                if report.get('schema_version') == 'crt-interface-comparison-v1':
                    kind = 'interfaces'
                    config = read_json(directory / 'plan.json')
                elif (directory / 'manifest.json').is_file():
                    kind = 'starting-population-benchmark' if report.get('treatment')=='starting-population' else 'model-comparison' if report.get('treatment') == 'model' else 'comparison'
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
