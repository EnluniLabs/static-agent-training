"""Agent Z artifacts and native playback for the workspace."""
from functools import lru_cache
from pathlib import Path
import random

from .experimental.agent_z import AgentZ, Chain
from .experimental.agent_q import AgentQ, MODES
from .experimental.agent_q_accelerated import NativeQ
from .experimental.agent_z_experiment import read_z_artifact
from .experimental.agent_z_kernels import NativeKernels, sigmoid


@lru_cache(maxsize=8)
def completed_result(path, stamp):
    return read_z_artifact(path)


def legacy_result(directory):
    path = directory / 'result.json.gz'
    if not directory.name.startswith('agent-z-') or not path.is_file():
        return {}
    data = completed_result(path, path.stat().st_mtime_ns)
    return data if data.get('champion', {}).get('schema_version') == AgentZ.schema else {}


def snapshots(directory, root, family='agent-z'):
    from .catalog import read_json
    report=read_json(directory/'report.json')
    recorded=report.get('schema_version')
    if (recorded in ('crt-agent-z-run-v1','crt-agent-q-run-v1') and
            recorded != f'crt-{family}-run-v1') or (family=='agent-q' and recorded!='crt-agent-q-run-v1'):
        return []
    paths = sorted((directory / 'snapshots').glob('*.json.gz'), reverse=True)
    options = []
    if not paths and (data := legacy_result(directory)):
        paths = [directory / 'result.json.gz']
    for path in paths:
        if path.name == 'result.json.gz':
            settings = data['settings']
            row = data['history'][-1]
            generation = row['generation']
        else:
            from .catalog import read_json
            settings = read_json(directory / 'report.json').get('config', {})
            generation = int(path.name.split('.')[0])
            row = read_json(directory / 'summaries' / f'{generation:05d}.json')
        options.append(dict(id=family[-1]+'-snapshot:' + path.relative_to(root / 'runs').as_posix(),
                            model=family, label=f'Generation {generation} · training champion',
                            stage='trained', sensory_interface='universal-byte',
                            sensory_partitions=settings.get('regions', 2), environment='original',
                            boundary_mode='clamp', varying_goals=True,
                            training_success=row.get('best'), assessment_success=row.get('held_out')))
    return options


def load(identifier, root):
    runs = (root / 'runs').resolve()
    prefix, relative = identifier.split(':',1)
    family, cls = ('agent-q',AgentQ) if prefix=='q-snapshot' else ('agent-z',AgentZ)
    path = (runs / relative).resolve()
    if not path.is_relative_to(runs) or not path.is_file():
        raise ValueError('Unknown Agent Z snapshot')
    if path.name == 'result.json.gz':
        data = legacy_result(path.parent).get('champion')
    elif path.parent.name == 'snapshots' and path.name.removesuffix('.json.gz').isdigit():
        data = read_z_artifact(path)
    else:
        raise ValueError('Choose an Agent Z snapshot')
    if not isinstance(data, dict) or data.get('schema_version') != cls.schema:
        raise ValueError('Unsupported Agent Z snapshot')
    return family, data, None


def graph(model):
    nodes, edges = [], []
    q = isinstance(model,AgentQ)
    for bit in range(8 if q else model.width):
        nodes.extend([dict(id=f'input:{bit}', label=f'Input bit {bit}', kind='input',
                           region=bit % model.regions)])
        if not q:
            nodes.append(dict(id=f'output:{bit}', label=f'Output bit {bit}', kind='output'))
        inputs = sorted(range(len(model.ids)), key=lambda i: -abs(model.embedding[i][bit + 1]))[:4]
        for i in inputs:
            edges.append(dict(source=f'input:{bit}', target=model.ids[i], kind='embedding',
                              value=model.embedding[i][bit + 1], measure='weight'))
    if q:
        nodes.extend(dict(id=f'output:{i}',label=f'Physical action {signal.hex()}',kind='output')
                     for i,signal in enumerate(model.actions))
    for i, pair in enumerate(model.ids):
        nodes.append(dict(id=pair, label=f'State {i + 1}', kind='state',
                          pair_id=pair, bias=model.embedding[i][0]))
        outputs = (model.policy(Chain(0,pair)) if pair in model.structure.successors else []) if q else [
            sigmoid(value) for value in model.continuation[i]]
        for bit,value in enumerate(outputs):
            edges.append(dict(source=pair, target=f'output:{bit}', kind='output',
                              value=value, measure='probability'))
    for pair, target in model.structure.successors.items():
        edges.append(dict(source=pair, target=target, value=1, kind='successor', measure='structural'))
    return dict(nodes=nodes, edges=edges, connection_limit=8,
                sensory_partitions=model.regions, action_policy='derived policy · region 0' if q else 'external signal interpretation')


def attempt(model, start, goal, seed, learning, mode='both'):
    q = isinstance(model,AgentQ)
    if q:
        result = model.kernels.evaluate(model,[(start,goal)],seed=seed,
                                        learning=mode if learning else 'off',trace=True,return_weights=learning)
    else:
        result = model.kernels.evaluate(model, [(start, goal)], random.Random(seed), round_limit=64,
                                        empirical_rate=.05 if learning else 0,
                                        structural_rate=.05 if learning else 0,
                                        playback=True, return_weights=learning)
    if learning:
        model.embedding = result['embedding']
        if q:
            model.expectations = result['expectations']
        else:
            model.continuation = result['continuation']
    frames = [dict(position=start, actions=0, label='Initial observation', done=False,
                   tested=0, correct=0)]
    empirical = structural = 0
    for events, world in zip(result['trace'], result['world_frames']):
        measured = [event for event in events if event['observation'] is not None]
        if q and learning:
            empirical += len(measured) if MODES[mode][0] else 0
            structural += len(measured) if MODES[mode][1] else 0
        elif learning:
            empirical += len(measured)
            structural += sum(e['compatibility'] is not None and e['compatibility'] < .1
                              for e in measured)
        frames.append(dict(world, label=f'Round {len(frames)} · ' +
                           ', '.join(e['signal'] for e in events), emissions=events,
                           active_pairs=[e['chain'] for e in events],
                           current_pairs=[dict(region=e['chain']['region'], pair=e['successor'])
                                          for e in events if e['successor'] is not None],
                           tested=len(measured), correct=sum(e['mismatched_bits'] == 0 for e in measured),
                           done=False))
    frames[-1].update(done=True, success=bool(result['successes']),
                      reason='target' if result['successes'] else
                      'action_limit' if frames[-1]['actions'] >= 12 else
                      'round_limit' if result['rounds'] >= 64 else 'chains_ended')
    return frames, empirical, structural


def build(snapshot):
    if snapshot.get('schema_version') == AgentQ.schema:
        return AgentQ.from_data(snapshot,kernels=NativeQ())
    return AgentZ.from_data(snapshot, kernels=NativeKernels())


def save_population(directory, output, *, count, name):
    from .experiment import write_json

    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError('Candidate count must be between 1 and 4096')
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError('Candidate set name must contain at most 80 characters')
    source = directory / 'best-population.json.gz'
    if source.is_file():
        data = read_z_artifact(source)
    elif result := legacy_result(directory):
        data = dict(schema_version='crt-agent-z-population-v1',
                    config=dict(model='agent-z', environment='original',
                                sensory_partitions=result['settings'].get('regions', 2)),
                    source_type='champion', members=[dict(individual=result['champion'],
                        generation=result['history'][-1]['generation'] - 1)])
    else:
        raise ValueError('Complete at least one generation before saving candidates')
    data.update(name=name.strip(), source=str(directory), members=data['members'][:count])
    write_json(output, data)
    return dict(name=data['name'], count=len(data['members']), id='populations/' + output.name)


def preview(model, payload, *, limit, seed, window_bytes, origin):
    if window_bytes not in (None, 1):
        raise ValueError('Agent Z stream preview uses one byte per input frame')
    window = payload[:limit]
    result = model.kernels.evaluate(model, [(0, 0)] * len(window), random.Random(seed),
                                    round_limit=1, empirical_rate=0, structural_rate=0,
                                    trace=True, stream=window)
    frames = [dict(sensory=f'{byte:02x}', input_window_size=1,
                   active_pairs=[event['chain'] for event in events],
                   current_pairs=[dict(region=e['chain']['region'], pair=e['successor']) for e in events],
                   actions=[], predictions=[], emissions=events)
              for byte, events in zip(window, result['trace'])]
    return dict(frames=frames, source='Frozen stream copy', origin=origin, graph=graph(model),
                sensory_partitions=model.regions, action_policy='passive preview',
                geometry_policy='One input byte per frame; emitted signals are recorded',
                metadata=dict(bytes=len(payload), bytes_consumed=len(window), window_bytes=1,
                              windows=len(window), embedded_frames=len(window), seed=seed,
                              stop_reason='frame_limit' if len(window) < len(payload) else 'end_of_stream'))
