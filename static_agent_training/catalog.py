"""Read-only discovery of baseline, sensory and paired-comparison artifacts."""
from . import serialization as json
from pathlib import Path

from .status import collect_runs


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def sensory_state(directory, metadata, module='static_agent_training.experimental.bit_experiment'):
    """Do not mistake a saved running flag for a process that is still alive."""
    state = metadata.get('status', 'unknown')
    if state != 'running':
        return state
    try:
        pid = metadata['pid']
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if module in args:
            flags = [args[i + 1] for i, flag in enumerate(args[:-1]) if flag in ('--output', '--resume', '--workspace-directory')]
            flags += [arg.split('=', 1)[1] for arg in args if arg.startswith(('--output=', '--resume=', '--workspace-directory='))]
            cwd = Path(f'/proc/{pid}/cwd').resolve()
            if any((cwd / value).resolve() == directory.resolve() for value in flags):
                return 'running'
    except (OSError, KeyError, ValueError, UnicodeError):
        pass
    return 'stopped'


def catalog(root, *, processes=None):
    root = Path(root)
    rows = []
    for run in collect_runs(root, processes=processes):
        directory = root / 'runs' / run['name']
        rows.append(dict(kind='sensory-training' if run.get('model') == 'senses' else 'training', name=run['name'], path=str(directory.resolve()),
                         state=run['state'], updated=run['updated'],
                         progress=f"{run['completed']}/{run['target']}",
                         summary=run['summary'], pid=run['pid'],
                         elapsed_seconds=run['elapsed_seconds'],
                         remaining_seconds=run['remaining_seconds']))
    sensory_directories = set()
    for path in (root / 'runs').glob('*/run.json'):
        metadata = read_json(path)
        if metadata.get('schema_version') != 'crt-sensory-run-v1':
            continue
        sensory_directories.add(path.parent)
        rows.append(dict(kind='senses', name=path.parent.name, path=str(path.parent.resolve()),
                         state=sensory_state(path.parent, metadata), updated=path.stat().st_mtime,
                         progress=None, summary={key: metadata.get(key) for key in
                         ('seed', 'receptors', 'label', 'notes', 'error')}, pid=metadata.get('pid')))
    reports = (sorted((root / 'runs').glob('*/report.json')) +
               sorted((root / 'runs' / 'comparisons').glob('*/report.json')) +
               sorted((root / 'runs' / 'interfaces').glob('*/report.json')))
    for path in reports:
        report = read_json(path)
        schema = report.get('schema_version', '')
        progress = None
        if schema == 'crt-agent-z-run-v1':
            kind = 'agent-z-training'
            state = sensory_state(path.parent, report, module='static_agent_training.experimental.agent_z_experiment')
            files = sorted((path.parent / 'summaries').glob('*.json'))
            summary = read_json(files[-1]) if files else {}
            progress = f"{summary.get('generation', 0)}/{report['config']['generations']}"
        elif schema == 'crt-frozen-assessment-v1':
            kind='assessment'
            state=sensory_state(path.parent,report,module='static_agent_training.assessments')
            progress=f"{report.get('progress',{}).get('completed_animals',0)}/{report.get('progress',{}).get('total_animals',0)} candidates"
            summary={'populations':len(report.get('populations',[])),'environment':report.get('config',{}).get('environment','original')}
        elif schema == 'crt-population-benchmark-v1':
            kind='population-benchmark'
            state=sensory_state(path.parent,report,module='static_agent_training.populations')
            progress=str(len(report.get('populations',[])))+' populations'
            summary={'populations':len(report.get('populations',[])),'environment':report.get('config',{}).get('environment','original')}
        elif schema == 'crt-interface-comparison-v1':
            kind = 'interfaces'
            state = sensory_state(path.parent, report, module='static_agent_training.experimental.model_comparison')
            progress = f"{report.get('completed', 0)}/{report.get('target', 0)}"
            summary = {key: report.get(key) for key in ('population_mean', 'paired_effect', 'plan')}
        elif schema.startswith('crt-bit-experiment-'):
            if path.parent in sensory_directories:
                continue
            kind, state = 'senses', 'completed'
            summary = dict(seed=report.get('seed'), senses=report.get('senses'),
                           scope=report.get('scope'), results={
                               name: result.get('frozen_evaluation', {}).get('expanded', {}).get('success_rate')
                               for name, result in report.get('results', {}).items()})
        elif schema in ('crt-treatment-comparison-v1', 'crt-speciation-comparison-v1'):
            kind, state = ('starting-population-benchmark' if report.get('treatment')=='starting-population' else 'model-comparison' if report.get('treatment') == 'model' else 'comparison'), report.get('status', 'unknown')
            if state == 'running':
                state = 'recorded-running'  # A historical report is not a live-process check.
            summary = {key: report.get(key) for key in ('treatment', 'seeds', 'paired_effect')}
            arms = report.get('arms', [])
            if arms:
                progress = f"{sum(a.get('completed', 0) for a in arms)}/{sum(a.get('target', 0) for a in arms)}"
        else:
            continue
        rows.append(dict(kind=kind, name=path.parent.name, path=str(path.parent.resolve()),
                         state=state, updated=path.stat().st_mtime, progress=progress, summary=summary))
    from .workspace_agent_z import legacy_result
    for directory in (root / 'runs').glob('agent-z-*'):
        if (directory / 'report.json').is_file():
            continue
        result = legacy_result(directory)
        if result:
            path = directory / 'result.json.gz'
            rows.append(dict(kind='agent-z-training', name=directory.name, path=str(directory.resolve()),
                             state='completed', updated=path.stat().st_mtime,
                             progress=f"{len(result['history'])}/{result['settings']['generations']}",
                             summary=result['history'][-1]))
    return sorted(rows, key=lambda r: r['updated'], reverse=True)


def resolve_run(root, value, *, kind=None, rows=None):
    """Resolve an existing path, exact catalog name, or explicitly typed latest."""
    root = Path(root)
    rows = catalog(root) if rows is None else rows
    aliases = {'latest-training': 'training', 'latest-senses': 'senses',
               'latest-comparison': 'comparison', 'latest-interfaces': 'interfaces',
               'latest-sensory-training': 'sensory-training', 'latest-model-comparison': 'model-comparison'}
    if value in aliases:
        requested = aliases[value]
        if kind is not None and requested != kind:
            raise ValueError(f'{value} is not a {kind} run')
        candidates = [row for row in rows if row['kind'] == requested]
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        candidates = [row for row in rows if row['name'] == value or Path(row['path']) == path.resolve()]
        if kind:
            candidates = [row for row in candidates if row['kind'] == kind]
        if not candidates and path.is_dir() and (kind is None or
                 (kind == 'training' and (path / 'config.json').is_file() and read_json(path / 'config.json').get('config', {}).get('model') != 'senses') or
                 (kind == 'sensory-training' and read_json(path / 'config.json').get('config', {}).get('model') == 'senses') or
                 (kind == 'comparison' and (path / 'manifest.json').is_file())):
            return path.resolve()
    if not candidates:
        raise ValueError(f'No matching {kind or "saved"} run: {value}. Use ./project runs.')
    if len(candidates) != 1 and value not in aliases:
        raise ValueError(f'Ambiguous run name: {value}; use its full path')
    return Path(candidates[0]['path'])


def describe(directory):
    directory = Path(directory)
    data = dict(path=str(directory), files=sorted(p.name for p in directory.iterdir() if p.is_file()))
    metadata = read_json(directory / 'run.json')
    if metadata:
        data['run'] = metadata
    config = read_json(directory / 'config.json')
    if config:
        data['config'] = config.get('config', config)
        data['partitions'] = config.get('partitions')
    report = read_json(directory / 'report.json')
    if report:
        data['report'] = {key: value for key, value in report.items()
                          if key not in ('task_lists', 'structure_source', 'arms', 'pairs')}
        if str(report.get('schema_version', '')).startswith('crt-bit-experiment-'):
            data['report']['results'] = {
                name: dict(adaptation_success=result.get('adaptation', {}).get('success_rate'),
                           frozen_success_by_region={region: metrics.get('success_rate') for region, metrics in
                               result.get('frozen_evaluation', {}).items()},
                           adaptation_updates={key: result.get('adaptation', {}).get('event_counts', {}).get(key, 0)
                                               for key in ('EMPIRICAL_UPDATE', 'CONFLICT')})
                for name, result in report.get('results', {}).items()}
        if 'arms' in report:
            data['report']['arms'] = [{key: arm.get(key) for key in
                ('run_name', 'state', 'completed', 'target', 'seed', 'enabled', 'latest', 'final_benchmark')} for arm in report['arms']]
    data['champions'] = read_json(directory / 'champions.json').get('champions', [])
    for name, path in [('log', directory.parent / (directory.name + '.log')),
                       ('diagnostics', directory.parent / (directory.name + '-diagnostics'))]:
        if path.exists():
            data[name] = str(path)
    return data
