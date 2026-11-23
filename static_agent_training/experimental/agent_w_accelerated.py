"""Native CPU lifetime evaluation for Agent W's number-line adapter."""
import ctypes as c
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import random
import subprocess
import tempfile

from .agent_w import ActionFeedback
from .agent_w_environment import NumberLine
from .agent_z import Chain


class Diagnostics(c.Structure):
    _fields_ = [
        ('action_counts', c.c_uint64 * 9), ('starts', c.c_uint64 * 41), ('solved', c.c_uint64 * 41),
        ('pair_starts', c.c_uint64 * (41 * 41)), ('pair_solved', c.c_uint64 * (41 * 41)),
        ('ends', c.c_uint64 * 4), ('compatibility_bins', c.c_uint64 * 20),
        ('unique_first', c.c_uint64), ('tied_first', c.c_uint64), ('action_sign', c.c_uint64 * 3),
        ('compatibility_sum', c.c_double), ('action_entropy_sum', c.c_double),
    ]


class Trace(c.Structure):
    _fields_ = [(name, c.c_int) for name in
                ('episode', 'round', 'region', 'pair', 'action', 'feedback', 'successor')] + [
                    (name, c.c_double) for name in ('compatibility', 'probability', 'entropy')]


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).resolve().parents[1] / 'native' / 'evaluate_agent_w.cpp'
    digest = hashlib.sha256(source.read_bytes() + source.with_name('evaluate_agent_z.cpp').read_bytes())
    directory = Path.home() / '.cache' / 'static-agent-training'
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f'agent-w-{digest.hexdigest()[:24]}.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            target = Path(temp) / library.name
            try:
                subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared',
                                '-fPIC', str(source), '-o', str(target)],
                               check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f'Agent W native build failed: {getattr(error, "stderr", error)}') from error
            target.replace(library)
    call = c.CDLL(str(library)).w_evaluate
    integer = c.POINTER(c.c_int)
    decimal = c.POINTER(c.c_double)
    call.argtypes = ([c.c_int] * 7 + [integer] * 3 + [c.POINTER(c.c_uint32)] +
                     [decimal] * 2 + [c.c_double] * 2 + [integer, c.POINTER(Trace), c.POINTER(c.c_uint32),
                     c.POINTER(Diagnostics), c.POINTER(c.c_uint8), c.POINTER(c.c_uint8)])
    call.restype = c.c_int
    return call


def evaluate(agent, tasks, *, seed=7, action_rate=.05, structural_rate=.05,
             round_limit=64, encoding_seed=0, action_limit=12, trace=False, environment='original',
             rng_state=None, return_rng=False, diagnostics=False, task_diagnostics=False):
    """Run a native lifetime on a copy of inherited parameters."""
    if any(not math.isfinite(rate) or rate < 0 for rate in (action_rate, structural_rate)):
        raise ValueError('learning rates must be finite and nonnegative')
    if type(round_limit) is not int or type(action_limit) is not int or min(round_limit, action_limit) < 1:
        raise ValueError('round and action limits must be positive integers')
    tasks = tuple(tuple(task) for task in tasks)
    if environment not in ('original', 'shifted', 'remapped-number-line', 'lights-out-2x2'):
        raise ValueError('unsupported Agent W/D assessment environment')
    puzzle = environment == 'lights-out-2x2'
    sensor_rotate = 2 if environment == 'remapped-number-line' else 0
    diagnostics = diagnostics or task_diagnostics
    lower, upper = (0, 15) if puzzle else (30, 70) if environment == 'shifted' else (-20, 20)
    if any(len(task) != 2 or any(type(v) is not int or not lower <= v <= upper for v in task)
           or (puzzle and task[1] != 0) for task in tasks):
        raise ValueError('task is outside the selected environment')
    if environment == 'shifted':
        tasks = tuple((start - 50, goal - 50) for start, goal in tasks)
    world = NumberLine(encoding_seed=encoding_seed, action_limit=action_limit)
    if agent.actions != world.action_signals:
        raise ValueError('native Agent W requires the number-line action alphabet')
    pairs, episodes, action_count = len(agent.ids), len(tasks), len(agent.actions)
    capacity = episodes * min(action_limit, round_limit * agent.regions)
    if max(capacity, round_limit, action_limit, pairs * 9, episodes * 2) > 2**31 - 1:
        raise ValueError('native Agent W evaluation exceeds index range')
    successor = (c.c_int * pairs)(
        *(agent.index.get(agent.structure.successors.get(pair), -1) for pair in agent.ids))
    inputs = (c.c_int * (episodes * 2))(*(value for task in tasks for value in task))
    if puzzle:
        from ..lights_out import action_cells, press_mask
        cells = action_cells(environment)
        random.Random(f'moves:{encoding_seed}').shuffle(cells)
        transformations = [press_mask(cell, environment) for cell in cells]
    else:
        transformations = world._native_moves()
    moves = (c.c_int * action_count)(*transformations)
    if rng_state is None:
        rng_state = random.Random(seed).getstate()[1]
    if (len(rng_state) != 625 or not 0 <= rng_state[-1] <= 624 or
            any(type(item) is not int or not 0 <= item <= 0xffffffff for item in rng_state)):
        raise ValueError('invalid native Agent W RNG state')
    state = (c.c_uint32 * 625)(*rng_state)
    embedding = (c.c_double * (pairs * 9))(*(v for row in agent.embedding for v in row))
    continuation = (c.c_double * (pairs * action_count))(*(v for row in agent.continuation for v in row))
    stats = (c.c_int * 3)()
    records = (Trace * capacity)() if trace else None
    final_state = (c.c_uint32 * 625)() if return_rng else None
    counts = Diagnostics() if diagnostics else None
    changed_actions = (c.c_uint8 * pairs)() if diagnostics else None
    changed_embedding = (c.c_uint8 * pairs)() if diagnostics else None
    if _kernel()(pairs, agent.regions, episodes, round_limit, action_limit, int(puzzle), sensor_rotate, successor, inputs, moves,
                 state, embedding, continuation, action_rate, structural_rate, stats, records,
                 final_state, c.byref(counts) if counts is not None else None,
                 changed_actions, changed_embedding):
        raise MemoryError('native Agent W could not allocate evaluation buffers')
    successes, rounds, actions = stats
    result = dict(successes=successes, episodes=episodes, rounds=rounds, actions=actions,
                  embedding=[embedding[i:i + 9] for i in range(0, len(embedding), 9)],
                  continuation=[continuation[i:i + action_count]
                                for i in range(0, len(continuation), action_count)])
    if return_rng:
        result['rng_state'] = list(final_state)
    if diagnostics:
        transform_counts = {}
        for transform, count in zip(transformations, counts.action_counts):
            transform_counts[str(transform)] = transform_counts.get(str(transform), 0) + count
        result['diagnostics'] = dict(
            action_counts=transform_counts, compatibility_histogram=list(counts.compatibility_bins),
            compatibility_sum=counts.compatibility_sum, unique_first=counts.unique_first,
            tied_first=counts.tied_first, action_entropy_sum=counts.action_entropy_sum,
            action_update_sign=dict(zip(('negative', 'zero', 'positive'), counts.action_sign)),
            start_outcomes={str(i + lower): [total, counts.solved[i]]
                            for i, total in enumerate(counts.starts) if total},
            termination=dict(zip(('goal', 'action_limit', 'chains_ended', 'round_limit'), counts.ends)),
            action_updated_states=[i for i, value in enumerate(changed_actions) if value],
            embedding_updated_states=[i for i, value in enumerate(changed_embedding) if value])
        if task_diagnostics:
            result['diagnostics']['task_outcomes'] = {
                f'{i // 41 + lower},{i % 41 + lower}': [total, counts.pair_solved[i]]
                for i, total in enumerate(counts.pair_starts) if total}
    if trace:
        result['trace'] = []
        for i in range(actions):
            entry = records[i]
            successor = agent.ids[entry.successor] if entry.successor >= 0 else None
            event = ActionFeedback(Chain(entry.region, agent.ids[entry.pair]),
                                   agent.actions[entry.action], bytes([entry.feedback]), successor,
                                   entry.compatibility if successor is not None else None,
                                   entry.probability, entry.entropy)
            result['trace'].append((entry.episode, entry.round, event))
    return result
