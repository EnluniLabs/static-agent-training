"""Local workspace data, graph series and protected session control APIs."""
from functools import lru_cache
import json
from pathlib import Path
import secrets
import threading
import time

from .catalog import catalog, describe, read_json
from .run_records import read_summary
from .workspace_models import WorkspaceModels
from .workspace_sessions import SessionManager


WEB = Path(__file__).resolve().parent / 'web'


@lru_cache(maxsize=128)
def read_tail(filename, stamp, size):
    with open(filename, 'rb') as handle:
        handle.seek(max(0, size - 65536))
        return handle.read(65536).decode('utf-8', errors='replace')


class Workspace:
    def __init__(self, root, initial=None):
        self.root = Path(root).resolve()
        self.token = secrets.token_urlsafe(32)
        self.sessions = SessionManager(root)
        self.models = WorkspaceModels(root, self.sessions, initial or {})
        self.lock = threading.RLock()
        self.cached = (0, [])

    def invalidate(self):
        self.cached = (0, [])

    def runs(self):
        with self.lock:
            now = time.monotonic()
            if now - self.cached[0] < 2:
                return self.cached[1]
            rows = catalog(self.root)
            jobs = self.sessions.records()
            for row in rows:
                row['run'] = Path(row['path']).relative_to(self.root / 'runs').as_posix()
                owned = next((job for job in jobs if job['directory'] == row['path']), None)
                if owned:
                    row['session'] = owned
                    if owned['state'] in ('running', 'stopping', 'failed', 'stopped'):
                        row['state'] = owned['state']
                if row['kind'] == 'comparison' and row['state'] == 'recorded-running':
                    row['state'] = 'recorded running'
            known = {row['path'] for row in rows}
            for job in jobs:
                if job['directory'] not in known:
                    rows.append(dict(name=job['name'], path=job['directory'], run=job['run'], kind=job['kind'],
                                     state=job['state'], updated=job['created'], progress=None, summary=None, session=job))
                    known.add(job['directory'])
            for row in rows:
                if row['kind'] == 'sensory-training':
                    saved = read_json(Path(row['path']) / 'config.json').get('config', {})
                    settings = saved or row.get('session', {}).get('config', {})
                    if settings.get('agent_y'):
                        row['model_family'] = 'agent-y'
                    elif settings.get('sensory_partitions', 1) > 1 or settings.get('sensory_interface') == 'signals':
                        row['model_family'] = 'agent-x'
            rows.sort(key=lambda row: (row['state'] in ('running', 'stopping', 'finishing'), row['updated']), reverse=True)
            self.cached = (now, rows)
            return rows

    def detail(self, relative):
        directory = self.sessions.path(relative)
        result = describe(directory)
        metadata = result.pop('run', None)
        if isinstance(metadata, dict):
            result['run_metadata'] = metadata
        result['run'] = relative
        result['series'] = []
        result['evaluation_progress'] = read_json(directory / 'progress.json')
        if (directory / 'config.json').is_file():
            result['kind'] = 'sensory-training' if result.get('config', {}).get('model') == 'senses' else 'training'
            result['chart'] = dict(x='Generation', series=[['mean', 'Mean training'], ['assessment_mean', 'Mean held-out'], ['best', 'Best training'], ['random_assessment', 'Random']])
            for path in sorted((directory / 'generations').glob('*.json')):
                try:
                    summary = read_summary(path)
                    result['series'].append(dict(x=summary['generation'] + 1, **summary))
                except (OSError, ValueError, KeyError):
                    continue
        report = read_json(directory / 'report.json')
        schema = report.get('schema_version', '')
        if schema == 'crt-frozen-assessment-v1':
            result['kind']='assessment'
            result['config']=report['config']
            result['chart']=dict(x='Frozen population',series=[['success','Mean success'],['random','Random movement']])
            for i,row in enumerate(report.get('populations',[]),1):
                if row.get('mean_success') is not None:
                    result['series'].append(dict(x=i,label=row['name'],success=row['mean_success'],random=report['random_success']))
            result['limitations']=['Frozen weights as stored in the saved candidate population. No warmup or learning.',
                'Common assessment tasks may overlap tasks encountered in source training.']
        elif schema == 'crt-population-benchmark-v1':
            result['kind']='population-benchmark'
            result['chart']=dict(x='Saved population',series=[['training','Lifetime training'],['assessment','Frozen assessment'],['random','Random']])
            for i,row in enumerate(report.get('populations',[]),1):
                result['series'].append(dict(x=i,label=row['name'],training=row['training_mean'],assessment=row['assessment_mean'],random=report['random_assessment']))
            result['limitations']=['Fixed structures, common lifetime-learning budgets and frozen assessment; no evolutionary training.',
                'Benchmarks start from inherited genomes, not saved end-of-lifetime weights. Training-source scores may have different tasks.']
        elif schema == 'crt-interface-comparison-v1':
            result['kind'] = 'interfaces'
            result['chart'] = dict(x='Completed animal pairs', series=[['old', 'Old categorical'], ['senses', 'Fixed senses'], ['random', 'Random']])
            sums = dict(old=0, senses=0, random=0)
            baselines = {int(path.stem): read_json(path).get('exact_random_success')
                         for path in (directory / 'tasks').glob('*.json') if path.stem.lstrip('-').isdigit()}
            for index, pair in enumerate(report.get('pairs', []), 1):
                sums['old'] += pair['old']
                sums['senses'] += pair['senses']
                baseline = baselines.get(pair['seed'])
                sums['random'] += baseline or 0
                result['series'].append(dict(x=index, old=sums['old'] / index, senses=sums['senses'] / index,
                                             random=sums['random'] / index if baseline is not None else None))
            result['results'] = report.get('per_seed', [])
            result['limitations'] = report.get('limitations', [])
        elif schema.startswith('crt-bit-experiment-'):
            result['kind'] = 'senses'
            result['chart'] = dict(x='Frozen evaluation condition', series=[['old', 'Original-region tasks'], ['expanded', 'Expanded world'], ['new', 'New-region tasks'], ['crossing', 'Crossings']])
            for index, (name, data) in enumerate(report.get('results', {}).items(), 1):
                result['series'].append(dict(x=index, label=name, **{region: scores.get('success_rate') for region, scores in data.get('frozen_evaluation', {}).items()}))
            result['limitations'] = ['Fixed-structure lifetime learning; no evolution.',
                                    'Old-region scores use expanded-world walls and do not isolate original-world retention.',
                                    'The finite sensory converter can lose distinctions on long inputs.']
        elif schema in ('crt-treatment-comparison-v1', 'crt-speciation-comparison-v1'):
            result['kind'] = 'starting-population-benchmark' if report.get('treatment')=='starting-population' else 'model-comparison' if report.get('treatment') == 'model' else 'comparison'
            result['chart'] = dict(x='Completed seed pair', series=[['control', report.get('control_label', 'Control')], ['treatment', report.get('treatment_label', 'Treatment')]])
            for index, pair in enumerate(report.get('pairs', []), 1):
                result['series'].append(dict(x=index, label=str(pair['seed']), control=pair.get('control'),
                                             treatment=pair.get(report.get('treatment', 'speciation'))))
            if report.get('treatment')=='starting-population':
                result['chart']=dict(x='Generation',series=[['a_training','A · population training'],['b_training','B · population training'],['a_assessment','A · population held-out'],['b_assessment','B · population held-out']])
                curves={}
                for arm in report.get('arms',[]):
                    prefix='b' if arm['enabled'] else 'a'
                    for point in arm.get('curve',[]):
                        row=curves.setdefault(point['generation'],{})
                        row.setdefault(prefix+'_training',[]).append(point['mean'])
                        row.setdefault(prefix+'_assessment',[]).append(point['assessment_mean'])
                result['series']=[dict(x=g+1,**{key:sum(values)/len(values) for key,values in row.items()}) for g,row in sorted(curves.items())]
                result['config']=report['config']
            result['results'] = report.get('pairs', [])
            result['limitations'] = report.get('limitations', [])
        result['artifacts'] = [name for name in result['files'] if name.endswith(('.json', '.json.gz'))]
        result['session'] = next((r for r in self.sessions.records() if r['directory'] == str(directory)), None)
        return result

    def log(self, relative, source='output'):
        directory = self.sessions.path(relative)
        if source == 'output':
            path = directory.parent / (directory.name + '.log')
        elif source in ('crash', 'resources'):
            path = directory.parent / (directory.name + '-diagnostics') / ('crash.log' if source == 'crash' else 'resources.jsonl')
        else:
            raise ValueError('Unknown log source')
        if not path.is_file():
            return dict(text='No log saved for this source yet.', path=str(path))
        stat = path.stat()
        return dict(text=read_tail(str(path), stat.st_mtime_ns, stat.st_size), path=str(path))

    def artifact(self, relative, filename):
        directory = self.sessions.path(relative)
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(('.json', '.json.gz')):
            raise ValueError('Choose a saved JSON artifact')
        path = (directory / filename).resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError('Artifact not found')
        return path

    def post(self, route, data):
        if route in ('/api/delete-session','/api/delete-population'):
            from .workspace_storage import WorkspaceStorage
            return WorkspaceStorage(self).delete('session' if route.endswith('session') else 'population',data.get('run') if route.endswith('session') else data.get('population'))
        if route == '/api/save-population':
            from .populations import save_population
            from datetime import datetime
            import uuid
            directory = self.sessions.path(data.get('run'))
            output = self.root/'runs/populations'/(datetime.now().strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:8]+'.json.gz')
            result = save_population(directory,output,count=data.get('count',10),name=data.get('name') or directory.name)
            self.invalidate()
            return result
        if route == '/api/sessions':
            result = self.sessions.launch(data)
        elif route == '/api/stop':
            result = self.sessions.stop(data.get('id'))
        elif route == '/api/resume':
            result = self.sessions.resume(data.get('id'), data.get('generations'))
        elif route == '/api/resume-run':
            result = self.sessions.resume_run(data.get('run'), data.get('generations'), data.get('backend'))
        elif route == '/api/reset-lifetime':
            return self.models.reset_lifetime(data)
        elif route == '/api/stream-preview':
            return self.models.stream_preview(data)
        elif route == '/api/brain-scan':
            return self.models.brain_scan(data)
        elif route == '/api/attempt':
            return self.models.attempt(data)
        else:
            raise ValueError('Unknown workspace action')
        self.invalidate()
        return result
