"""Snapshot discovery and isolated playground lifetimes for both model families."""
import gzip
import json
from pathlib import Path
import random
import threading
from collections import Counter
from functools import lru_cache

from .agent import Agent, run_episode
from .baseline import random_success
from .candidates import CandidateStore, best_saved
from .catalog import read_json
from .demo import RecordingWorld, recording_world
from .environment import MOVES
from .environments import environment_spec
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
        self.lifetimes = {}
        self.lifetime_lock = threading.RLock()

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
                                    sensory_partitions=config.get('sensory_partitions',1),action_policy=config.get('action_policy','agreement'),
                                    environment=config.get('environment','original'), varying_goals=config.get('varying_goals', False), boundary_mode=config.get('boundary_mode', 'terminate')))
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
        if model == 'senses':
            from .brain_scan import sensory_geometry
            for option in options:
                _, snapshot, _ = self.load(option['id'])
                saved = BitModel.from_data(snapshot)
                if saved.senses is not None:
                    geometry = sensory_geometry(saved.senses, saved.sensory_partitions)
                    option.update(sensory_geometry=geometry, sensory_interface=geometry['interface'],
                                  retina_width=geometry['width'], retina_height=geometry['height'],
                                  sensory_partitions=geometry['partitions'])
        return options

    def load(self, identifier):
        if not isinstance(identifier, str):
            raise ValueError('Choose a saved model snapshot')
        if identifier.startswith('candidate:'):
            bundle = self.candidates.load(identifier[len('candidate:'):])
            if bundle.get('model') == 'senses':
                snapshot = dict(bundle['learned_model'])
                snapshot['playback_world'] = {k:bundle.get(k,'original' if k=='environment' else None) for k in ('encoding_seed','boundary_mode','varying_goals','environment')}
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
            # Keep revision eligibility for optional learning; omit historical traces.
            data.update(events=[])
            return 'senses', data, path
        if schema == 'crt-tabular-reference-v1':
            return 'baseline', data, path
        raise ValueError('Unsupported model format')

    @staticmethod
    def lifetime_key(data):
        key = data.get('lifetime_id')
        if not isinstance(key, str) or not 16 <= len(key) <= 80 or not all(c.isalnum() or c in '-_' for c in key):
            raise ValueError('Choose a valid playground lifetime identifier')
        return key

    def brain_scan(self, data):
        from .brain_scan import mapping_graph
        from .genome import Species
        identifier = data.get('snapshot')
        key = self.lifetime_key(data)
        with self.lifetime_lock:
            session = self.lifetimes.get(key)
            if session is not None and session['snapshot'] == identifier:
                model = session['model']
                graph = (mapping_graph(model.repertoire, parameters=model.parameters(), sensory_partitions=model.sensory_partitions, senses=model.senses)
                         if isinstance(model, BitModel) else
                         mapping_graph(model.repertoire, mappings=model.mappings))
                return dict(graph, source='Current playground lifetime', attempts=session['attempts'],
                            sensory_partitions=getattr(model,'sensory_partitions',1),
                            action_policy=getattr(model,'action_policy','agreement'))
            kind, snapshot, _ = self.load(identifier)
            if kind == 'senses':
                model = BitModel.from_data(snapshot)
                graph = mapping_graph(model.repertoire, parameters=model.parameters(), sensory_partitions=model.sensory_partitions, senses=model.senses)
            elif kind == 'candidate':
                species = Species.from_data(snapshot['individual'])
                maps = species.inherited.lifetime()
                maps.embedding = snapshot['learned_embedding']
                maps.output = snapshot['learned_output']
                graph = mapping_graph(species.repertoire, mappings=maps)
            else:
                graph = mapping_graph(Repertoire.from_data(snapshot['repertoire']),
                                      mappings=InheritedMappings.from_data(snapshot['mappings']))
            return dict(graph, source='Saved snapshot', attempts=0,
                        sensory_partitions=snapshot.get('sensory_partitions',1) if kind=='senses' else 1,
                        action_policy=snapshot.get('action_policy','agreement') if kind=='senses' else 'agreement')

    def stream_preview(self, data):
        """Run opaque bytes through an isolated, frozen copy of saved senses."""
        import base64
        import binascii
        from .brain_scan import mapping_graph
        from .experimental.stream_input import run_stream

        encoded = data.get('base64_data')
        if not isinstance(encoded, str):
            raise ValueError('base64_data must be valid Base64')
        if len(encoded) > 4 * ((256 * 1024 + 2) // 3):
            raise ValueError('File exceeds the 256 KiB limit')
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError('base64_data must be valid Base64') from error
        if len(payload) > 256 * 1024:
            raise ValueError('File exceeds the 256 KiB limit')
        limit, seed = data.get('max_frames', 16), data.get('seed', 42)
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError('max_frames must be a positive integer no greater than 1024')
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError('seed must be between 0 and 4294967295')
        key = self.lifetime_key(data)
        with self.lifetime_lock:
            session = self.lifetimes.get(key)
            if session is not None and session['snapshot'] == data.get('snapshot'):
                original = session['model']
                if not isinstance(original, BitModel):
                    raise ValueError('Universal stream preview requires a fixed senses model; categorical models require their saved symbols')
                model = BitModel.from_data(original.to_data())
                origin = 'Current playground lifetime'
            else:
                kind, snapshot, _ = self.load(data.get('snapshot'))
                if kind != 'senses':
                    raise ValueError('Universal stream preview requires a fixed senses model; categorical models require their saved symbols')
                model = BitModel.from_data(BitModel.from_data(snapshot).to_data())
                origin = 'Saved snapshot'
        if model.senses is None:
            raise ValueError('This model has no saved fixed senses')
        from .brain_scan import sensory_geometry
        geometry = sensory_geometry(model.senses, model.sensory_partitions)
        for field, saved in (('receptors', geometry['resolution']), ('resolution', geometry['resolution']),
                             ('partitions', geometry['partitions']), ('sensory_partitions', geometry['partitions']),
                             ('width', geometry['width']), ('height', geometry['height']),
                             ('retina_width', geometry['width']), ('retina_height', geometry['height']),
                             ('sensory_interface', geometry['interface'])):
            if field in data and data[field] != saved:
                raise ValueError(f'{field} conflicts with saved sensory geometry')
        model.events.clear()
        model.embedding_rng.seed(derive(seed, 'stream-preview', 'embedding'))
        model.output_rng.seed(derive(seed, 'stream-preview', 'output'))
        graph = mapping_graph(model.repertoire, parameters=model.parameters(),
                              sensory_partitions=model.sensory_partitions, senses=model.senses)
        preview = run_stream(model, payload, window_bytes=data.get('window_bytes'),
                             frame_limit=limit, learn=False)
        return dict(preview, source='Frozen stream copy', origin=origin, graph=graph,
                    geometry_policy='Saved model sensory geometry; arbitrary byte source',
                    sensory_partitions=model.sensory_partitions, action_policy=model.action_policy,
                    sensory_geometry=graph['sensory_geometry'],
                    partition_rectangles=getattr(model.senses, 'partition_rectangles', None),
                    regions=(model.senses.partition_indices(model.sensory_partitions)
                             if hasattr(model.senses, 'partition_indices') else
                             model.senses.regions(model.sensory_partitions)),
                    metadata=dict(bytes=len(payload), bytes_consumed=preview['bytes_consumed'],
                                  window_bytes=(getattr(model.senses, 'raw_window_bytes', model.senses.receptors // 16) if data.get('window_bytes') is None
                                                else data['window_bytes']),
                                  windows=preview['frame_count'], embedded_frames=len(preview['frames']),
                                  stop_reason=preview['stop_reason'], seed=seed))

    def reset_lifetime(self, data):
        key = self.lifetime_key(data)
        with self.lifetime_lock:
            self.lifetimes.pop(key, None)
        return dict(reset=True)

    def lifetime_attempt(self, data, kind, snapshot, lower, upper):
        environment=data.get('world',snapshot.get('playback_world',snapshot).get('environment','original'))
        puzzle=environment_spec(environment).kind=='lights-out'
        """Keep mutable model copies in memory; never write a saved candidate."""
        key = self.lifetime_key(data)
        identifier = data['snapshot']
        session = self.lifetimes.get(key)
        if session is not None and session['snapshot'] != identifier:
            raise ValueError('Reset the playground lifetime before changing snapshots')
        if session is None:
            if len(self.lifetimes) >= 64:
                raise ValueError('Too many playground lifetimes; reset unused lifetimes first')
            if kind == 'senses':
                model = BitModel.from_data(snapshot)
                config = snapshot.get('playback_world', {})
                model.events.clear()
            else:
                from .genome import Species
                if kind == 'candidate':
                    species = Species.from_data(snapshot['individual'])
                    maps = species.inherited.lifetime()
                    maps.embedding = {s:dict(row) for s,row in snapshot['learned_embedding'].items()}
                    maps.output = {q:dict(row) for q,row in snapshot['learned_output'].items()}
                    repertoire = species.repertoire
                    config = snapshot
                else:
                    maps = InheritedMappings.from_data(snapshot['mappings']).lifetime()
                    repertoire = Repertoire.from_data(snapshot['repertoire'])
                    config = dict(varying_goals=True)
                from .learning import LearningConfig
                model = Agent(repertoire, maps, learning_config=LearningConfig(**snapshot.get('learning_config', {})))
                model.learner.revised = dict(snapshot.get('revised', {}))
            session = dict(snapshot=identifier, model=model, config=config, attempts=0,
                           empirical_updates=0, conflicts=0)
            self.lifetimes[key] = session
        model, config = session['model'], session['config']
        start, goal, seed = (data[k] for k in ('start','goal','seed'))
        conditioned = config.get('varying_goals', True)
        if (puzzle and goal!=0) or (not puzzle and not conditioned and goal != environment_spec(config.get('environment','original')).center):
            raise ValueError('This saved animal has no goal input; choose goal zero')
        learn = data.get('learning', False)
        offset = len(model.events)
        if kind == 'senses':
            from .experimental.sensory_playback import record_episode
            model.embedding_rng.seed(derive(seed, 'interactive', 'embedding'))
            model.output_rng.seed(derive(seed, 'interactive', 'output'))
            boundary = config.get('boundary_mode', 'clamp')
            frames = record_episode(model, start, goal, encoding_seed=config.get('encoding_seed', 0),
                boundary_mode=boundary, goal_conditioned=conditioned, lower=lower, upper=upper, learn=learn,environment=environment)
        else:
            model._embedding_rng.seed(derive(seed, 'interactive', 'embedding'))
            model._output_rng.seed(derive(seed, 'interactive', 'output'))
            boundary = 'clamp'
            world = recording_world(environment,encoding_seed=config.get('encoding_seed', 0), goal_conditioned=conditioned,
                                   boundary_mode=boundary, lower=lower, upper=upper)
            if start == goal:
                frames = [dict(position=start,actions=0,label='Already at goal',done=True,success=True,
                               reason='target',correct=0,tested=0)]
            else:
                run_episode(model, world, start=start, goal=goal, learn=learn)
                frames = world.frames
        counts = Counter(e['kind'] for e in model.events[offset:])
        session['attempts'] += 1
        session['empirical_updates'] += counts['EMPIRICAL_UPDATE']
        session['conflicts'] += counts['CONFLICT']
        return dict(start=start, goal=goal, seed=seed, policy='animal', model='senses' if kind=='senses' else 'baseline',
            frames=frames, lower=lower, upper=upper, environment=environment,boundary_mode=boundary,
            sensory_partitions=getattr(model,'sensory_partitions',1),action_policy=getattr(model,'action_policy','agreement'),
            sensory_interface=getattr(getattr(model, 'senses', None), 'interface', 'legacy-bytes'),
            retina_width=getattr(getattr(model, 'senses', None), 'width', None),
            retina_height=getattr(getattr(model, 'senses', None), 'height', None),
            random_expected_success=random_success(start,goal,boundary_mode=boundary,**({'environment':environment} if puzzle else dict(lower=lower,upper=upper))),
            learning=dict(enabled=learn, attempts=session['attempts'], empirical_updates=counts['EMPIRICAL_UPDATE'],
                          structural_corrections=counts['CONFLICT'], total_empirical_updates=session['empirical_updates'],
                          total_structural_corrections=session['conflicts']))

    def attempt(self, data):
        start, goal, seed = (data.get(k) for k in ('start', 'goal', 'seed'))
        if any(type(v) is not int for v in (start, goal, seed)) or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Start and goal must be integers; seed must be between 0 and 4294967295')
        if type(data.get('learning', False)) is not bool:
            raise ValueError('Learning must be a boolean')
        kind, snapshot, _ = self.load(data.get('snapshot'))
        saved_environment = snapshot.get('playback_world',snapshot).get('environment','original')
        world_name = data.get('world', saved_environment)
        policy = data.get('policy', 'animal')
        if world_name not in ('original', 'shifted', 'expanded','switchboard','lights-out','switches-2x2','lights-out-2x2') or policy not in ('animal', 'random'):
            raise ValueError('Unknown world or policy')
        spec = environment_spec(world_name if world_name != 'expanded' else 'original')
        lower, upper = (-30, 30) if world_name == 'expanded' else (spec.lower, spec.upper)
        if not lower <= start <= upper or not lower <= goal <= upper:
            raise ValueError('Start and goal must be inside the selected world')
        if spec.kind=='lights-out' and goal!=0:raise ValueError('Lights Out goal must be 0')
        if kind != 'senses' and world_name != saved_environment:
            raise ValueError('The categorical model requires its saved training environment')
        key = data.get('lifetime_id')
        with self.lifetime_lock:
            if policy == 'animal' and (data.get('learning', False) or isinstance(key,str) and key in self.lifetimes):
                return self.lifetime_attempt(data, kind, snapshot, lower, upper)
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
            world = recording_world(world_name,goal_conditioned=True,lower=lower,upper=upper)
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
            bundle['environment']=world_name if world_name!='expanded' else 'original'
            result = sensory_attempt(bundle, start, goal, seed, policy, lower=lower, upper=upper)
            return result
        return dict(start=start, goal=goal, seed=seed, policy=policy, model=kind, frames=frames,
                    lower=lower, upper=upper,environment=world_name, boundary_mode='clamp',
                    random_expected_success=random_success(start,goal,**({'environment':world_name} if spec.kind=='lights-out' else dict(lower=lower,upper=upper))))
