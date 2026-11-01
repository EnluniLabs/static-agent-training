"""Read-only discovery and loading of each run's best saved training champion."""
from collections import OrderedDict
import gzip
import json
from pathlib import Path
import threading

from .baseline import random_task_mean
from .genome import Species
from .status import collect_runs
from .run_records import read_summary


def best_saved(directory):
    """Ignore a generation until both its record and champion trace exist."""
    directory = Path(directory)
    best = None
    for path in sorted((directory / 'generations').glob('*.json')):
        try:
            summary = read_summary(path)
            generation = summary['generation']
            trace = directory / 'traces' / f'{generation:05d}-champion.json.gz'
            if trace.is_file() and (best is None or summary['best'] > best['best']):
                best = summary
        except (OSError, ValueError, KeyError):
            continue
    return best


def attach_benchmark(bundle, path):
    if path.is_file():
        data = json.loads(path.read_text())
        if (data.get('genotype_id') == Species.from_data(bundle['individual']).genotype_id and
                data.get('generation') == bundle['generation'] and
                data.get('boundary_mode', 'terminate') == bundle.get('boundary_mode', 'terminate')):
            bundle['benchmark'] = {k: data[k] for k in ('episodes', 'animal', 'random')}
    return bundle


def load_saved_candidate(directory, generation=None):
    directory = Path(directory)
    if generation is None:
        best = best_saved(directory)
        if best is None:
            raise ValueError("This run has no saved candidate yet")
        generation = best['generation']
    trace_path = directory / 'traces' / f'{generation:05d}-champion.json.gz'
    benchmark_path = directory / 'demo' / 'benchmark.json'
    with gzip.open(trace_path, 'rt') as handle:
        trace = json.load(handle)
    config = json.loads((directory / 'config.json').read_text())['config']
    result = trace['result']
    episodes = result['assessment']['episodes']
    bundle = dict(schema_version='crt-demo-v1', generation=generation+1,
                  training_success=result['training']['fitness'],
                  assessment_success=result['assessment']['fitness'],
                  individual=trace['individual'], learned_embedding=trace['learned_embedding'],
                  learned_output=trace['learned_output'], encoding_seed=config['encoding_seed'],
                  varying_goals=config.get('varying_goals', False), boundary_mode=config.get('boundary_mode','terminate'), episodes=episodes)
    bundle['random_assessment_success'] = random_task_mean(
        [e['start'] for e in episodes], [e.get('goal', 0) for e in episodes], boundary_mode=bundle['boundary_mode'])
    attach_benchmark(bundle, benchmark_path)
    return bundle


class CandidateStore:
    def __init__(self, root, initial):
        self.root = Path(root)
        self.initial = initial
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def directory(self, run_id):
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            raise ValueError('Unknown run')
        parent = (self.root / 'runs').resolve()
        directory = (parent / run_id).resolve()
        if directory.parent != parent or not (directory / 'config.json').is_file():
            raise ValueError('Unknown run')
        return directory

    def list(self):
        options = []
        for row in collect_runs(self.root):
            best = best_saved(self.root / 'runs' / row['name'])
            options.append(dict(run_id=row['name'], name=row['name'], state=row['state'],
                                population=row['population'], backend=row['backend'],
                                available=best is not None,
                                generation=best['generation']+1 if best else None,
                                training_success=best['best'] if best else None))
        default = self.initial.get('run_id', '_loaded')
        if default not in {o['run_id'] for o in options}:
            default = '_loaded'
            if self.initial:
                options.append(dict(run_id=default, name=self.initial.get('run_name', 'Loaded animal'),
                                    state='loaded', population=None, backend=None, available=True,
                                    generation=self.initial['generation'],
                                    training_success=self.initial['training_success']))
        return dict(default_run=default, runs=options)

    def latest(self, run_id):
        if run_id == '_loaded':
            if not self.initial:
                raise ValueError('No loaded animal')
            return self.initial, 'loaded', self.initial.get('run_name', 'Loaded animal')
        directory = self.directory(run_id)
        best = best_saved(directory)
        if best is None:
            raise ValueError('This run has no saved candidate yet; wait for a completed generation')
        candidate_id = f"{run_id}:{best['generation']+1}"
        return self.load(candidate_id), candidate_id, run_id

    def load(self, candidate_id):
        if candidate_id == 'loaded':
            if not self.initial:
                raise ValueError('No loaded animal')
            return self.initial
        if not isinstance(candidate_id, str):
            raise ValueError('Invalid candidate')
        run_id, separator, number = candidate_id.rpartition(':')
        if not separator or not number.isdigit() or int(number) < 1:
            raise ValueError('Invalid candidate')
        directory = self.directory(run_id)
        generation = int(number)-1
        trace_path = directory / 'traces' / f'{generation:05d}-champion.json.gz'
        benchmark_path = directory / 'demo' / 'benchmark.json'
        try:
            stamp = trace_path.stat().st_mtime_ns
            benchmark_stamp = benchmark_path.stat().st_mtime_ns if benchmark_path.exists() else None
            key = (candidate_id, stamp, benchmark_stamp)
            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
                bundle = load_saved_candidate(directory, generation)
                self.cache[key] = bundle
                while len(self.cache) > 16:
                    self.cache.popitem(last=False)
                return bundle
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f'Cannot load saved candidate: {error}') from error

    def info(self, run_id):
        bundle, candidate_id, name = self.latest(run_id)
        return dict(candidate_id=candidate_id, run_id=run_id, run_name=name,
                    boundary_mode=bundle.get('boundary_mode','terminate'),
                    generation=bundle['generation'], varying_goals=bundle.get('varying_goals', False),
                    training_success=bundle['training_success'], assessment_success=bundle['assessment_success'],
                    assessment_episodes=len(bundle['episodes']),
                    random_assessment_success=bundle.get('random_assessment_success'),
                    benchmark=bundle.get('benchmark'))
