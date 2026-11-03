"""Snapshot discovery and read-only attempts for both model families."""
import gzip
import json
from pathlib import Path
import random
from functools import lru_cache

from .agent import Agent, run_episode
from .baseline import random_success
from .candidates import CandidateStore, best_saved
from .catalog import read_json
from .demo import RecordingWorld
from .environment import MOVES
from .experimental.bit_model import BitModel
from .experimental.number_line import StimulusNumberLine
from .experimental.senses import SensoryAdapter
from .experimental.signals import Action, Prediction
from .mappings import InheritedMappings
from .repertoire import Repertoire
from .seeds import derive


@lru_cache(maxsize=20000)
def bounded_random(start, goal, moves=12, lower=-30, upper=30):
    if start == goal:
        return 1.
    if moves == 0:
        return 0.
    return sum(bounded_random(max(lower, min(upper, start + move)), goal, moves - 1, lower, upper)
               for move in MOVES) / len(MOVES)


class WorkspaceModels:
    def __init__(self, root, sessions, initial):
        self.root, self.sessions = Path(root), sessions
        self.candidates = CandidateStore(root, initial)

    def snapshots(self, relative, model):
        if model not in ('baseline', 'senses'):
            raise ValueError('Choose baseline or fixed senses')
        directory = self.sessions.path(relative)
        options = []
        if (directory / 'config.json').is_file():
            config = read_json(directory / 'config.json').get('config', {})
            expected = 'senses' if model == 'senses' else 'categorical'
            if config.get('model', 'categorical') != expected:
                return []
            entries = read_json(directory / 'champions.json').get('champions', [])
            if not entries:
                best = best_saved(directory)
                entries = [dict(generation=best['generation'], training_success=best['best'],
                                assessment_success=best['champion_assessment'])] if best else []
            for row in entries:
                generation = row['generation'] + 1
                options.append(dict(id=f"candidate:{directory.name}:{generation}", model=model,
                                    label=f"Generation {generation} · training champion", stage='trained',
                                    training_success=row.get('training_success'), assessment_success=row.get('assessment_success'),
                                    varying_goals=config.get('varying_goals', False), boundary_mode=config.get('boundary_mode', 'terminate')))
        if model == 'senses':
            for name in ('trained', 'prior-adapting', 'prior-frozen', 'fresh-adapting', 'initial'):
                path = directory / f'{name}.json.gz'
                if path.is_file():
                    options.append(dict(id='snapshot:' + path.relative_to(self.root / 'runs').as_posix(),
                                        model=model, label=name.replace('-', ' ').title(), stage=name))
        for result_path in sorted(directory.glob('seed-*/animal-*/result.json')):
            result = read_json(result_path)
            name = 'old' if model == 'baseline' else 'senses'
            for stage in ('trained', 'initial'):
                path = result_path.parent / f'{name}-{stage}.json.gz'
                if path.is_file():
                    metrics = result.get(name, {})
                    options.append(dict(id='snapshot:' + path.relative_to(self.root / 'runs').as_posix(),
                                        model=model, label=f"Seed {result.get('seed')} · animal {result.get('animal', 0) + 1} · {stage}",
                                        stage=stage, training_success=metrics.get('training', {}).get('success_rate') if stage == 'trained' else None,
                                        assessment_success=metrics.get('assessment', {}).get('success_rate') if stage == 'trained' else None))
        return options

    def load(self, identifier):
        if not isinstance(identifier, str):
            raise ValueError('Choose a saved model snapshot')
        if identifier.startswith('candidate:'):
            bundle = self.candidates.load(identifier[len('candidate:'):])
            if bundle.get('model') == 'senses':
                snapshot = dict(bundle['learned_model'])
                snapshot['playback_world'] = {k:bundle[k] for k in ('encoding_seed','boundary_mode','varying_goals')}
                return 'senses', snapshot, None
            return 'candidate', bundle, None
        if not identifier.startswith('snapshot:'):
            raise ValueError('Unknown snapshot')
        path = ((self.root / 'runs') / identifier[len('snapshot:'):]).resolve()
        if (not path.is_relative_to((self.root / 'runs').resolve()) or path.suffixes[-2:] != ['.json', '.gz']
                or not path.is_file()):
            raise ValueError('Unknown snapshot')
        allowed = {'initial', 'trained', 'prior-adapting', 'prior-frozen', 'fresh-adapting',
                   'old-initial', 'old-trained', 'senses-initial', 'senses-trained'}
        if path.name[:-8] not in allowed:
            raise ValueError('This file is not a supported model snapshot')
        with gzip.open(path, 'rt') as handle:
            data = json.load(handle)
        schema = data.get('schema_version')
        if schema == 'crt-bit-model-v1':
            # Playback is frozen; avoid duplicating historical learning traces.
            data.update(events=[], revised={})
            return 'senses', data, path
        if schema == 'crt-tabular-reference-v1':
            return 'baseline', data, path
        raise ValueError('Unsupported model format')

    def attempt(self, data):
        start, goal, seed = (data.get(k) for k in ('start', 'goal', 'seed'))
        if any(type(v) is not int for v in (start, goal, seed)) or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Start and goal must be integers; seed must be between 0 and 4294967295')
        kind, snapshot, _ = self.load(data.get('snapshot'))
        world_name = data.get('world', 'original')
        policy = data.get('policy', 'animal')
        if world_name not in ('original', 'expanded') or policy not in ('animal', 'random'):
            raise ValueError('Unknown world or policy')
        lower, upper = (-30, 30) if world_name == 'expanded' else (-20, 20)
        if not lower <= start <= upper or not lower <= goal <= upper:
            raise ValueError('Start and goal must be inside the selected world')
        if kind != 'senses' and world_name != 'original':
            raise ValueError('The categorical model has no mappings outside [-20,20]')
        if kind == 'candidate':
            from .play import attempt
            result = attempt(snapshot, start, goal, seed, policy)
            result['generation'] = snapshot['generation']
            return dict(result, model='baseline')
        if kind == 'baseline':
            maps = InheritedMappings.from_data(snapshot['mappings'])
            agent = Agent(Repertoire.from_data(snapshot['repertoire']), maps.lifetime(), learning=False,
                          embedding_seed=derive(seed, 'interactive', 'embedding'),
                          output_seed=derive(seed, 'interactive', 'output'))
            world = RecordingWorld(goal_conditioned=True)
            if start == goal:
                frames = [dict(position=start, actions=0, label='Already at goal', done=True, success=True,
                               reason='target', correct=0, tested=0)]
            elif policy == 'random':
                from .environment import Output
                rng = random.Random(derive(seed, 'interactive', 'random'))
                transition = world.reset(start=start, goal=goal)
                moves = tuple(world.diagnostic_state()['move_decoding'])
                while not transition.done:
                    transition = world.emit(Output(rng.choice(moves)), source='random')
                frames = world.frames
            else:
                run_episode(agent, world, start=start, goal=goal, learn=False)
                frames = world.frames
        else:
            from .experimental.sensory_playback import attempt as sensory_attempt
            bundle = dict(learned_model=snapshot, **snapshot.get('playback_world', {}))
            result = sensory_attempt(bundle, start, goal, seed, policy, lower=lower, upper=upper)
            if lower == -30:
                result['random_expected_success'] = bounded_random(start, goal)
            return result
        return dict(start=start, goal=goal, seed=seed, policy=policy, model=kind, frames=frames,
                    lower=lower, upper=upper, boundary_mode='clamp',
                    random_expected_success=random_success(start, goal) if lower == -20 else bounded_random(start, goal))
