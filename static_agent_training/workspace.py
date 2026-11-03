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
        if schema == 'crt-interface-comparison-v1':
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
            result['kind'] = 'model-comparison' if report.get('treatment') == 'model' else 'comparison'
            result['chart'] = dict(x='Completed seed pair', series=[['control', report.get('control_label', 'Control')], ['treatment', report.get('treatment_label', 'Treatment')]])
            for index, pair in enumerate(report.get('pairs', []), 1):
                result['series'].append(dict(x=index, label=str(pair['seed']), control=pair.get('control'),
                                             treatment=pair.get(report.get('treatment', 'speciation'))))
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
        if route == '/api/sessions':
            result = self.sessions.launch(data)
        elif route == '/api/stop':
            result = self.sessions.stop(data.get('id'))
        elif route == '/api/resume':
            result = self.sessions.resume(data.get('id'), data.get('generations'))
        elif route == '/api/resume-run':
            result = self.sessions.resume_run(data.get('run'), data.get('generations'))
        elif route == '/api/attempt':
            return self.models.attempt(data)
        else:
            raise ValueError('Unknown workspace action')
        self.invalidate()
        return result
