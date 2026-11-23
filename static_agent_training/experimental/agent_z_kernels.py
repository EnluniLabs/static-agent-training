"""Agent Z reference math and native CPU lifetime evaluation bindings."""
import ctypes
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile


def sigmoid(value):
    z = math.exp(-abs(value))
    return 1 / (1 + z) if value >= 0 else z / (1 + z)


class PythonKernels:
    @staticmethod
    def embedding(rows, indices, signal, region, regions):
        scores = [rows[i][0] + sum(rows[i][bit + 1] for bit, value in enumerate(signal)
                                  if value and bit % regions == region) for i in indices]
        peak = max(scores)
        values = [math.exp(score - peak) for score in scores]
        total = sum(values)
        return [value / total for value in values]

    @staticmethod
    def gradient(probabilities, targets, rate):
        return [rate * (target - probability) for probability, target in zip(probabilities, targets)]

    @staticmethod
    def probabilities(logits):
        return [sigmoid(value) for value in logits]


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).resolve().parents[1] / 'native' / 'evaluate_agent_z.cpp'
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    cache = Path.home() / '.cache' / 'static-agent-training'
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f'agent-z-{key}.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            target = Path(directory) / library.name
            try:
                subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared',
                                '-fPIC', str(source), '-o', str(target)],
                               check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f'Agent Z native build failed: {getattr(error, "stderr", error)}') from error
            target.replace(library)
    return ctypes.CDLL(str(library))


class TraceRecord(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in
                ('round', 'region', 'pair', 'successor', 'signal', 'observation', 'mismatches', 'position', 'actions')]
    _fields_.append(('compatibility', ctypes.c_double))


class NativeKernels:
    def __init__(self):
        self.library = _library()
        vector = ctypes.POINTER(ctypes.c_double)
        self.library.z_probabilities.argtypes = [ctypes.c_int, vector, vector]
        self.library.z_probabilities.restype = None
        self.library.z_embedding.argtypes = [ctypes.c_int] * 4 + [vector, vector, vector]
        self.library.z_embedding.restype = None
        self.library.z_gradient.argtypes = [ctypes.c_int, vector, vector, ctypes.c_double, vector]
        self.library.z_gradient.restype = None
        index = ctypes.POINTER(ctypes.c_int)
        self.library.z_evaluate.argtypes = (
            [ctypes.c_int] * 4 + [index] * 3 + [ctypes.POINTER(ctypes.c_uint32)] +
            [vector] * 2 + [ctypes.c_double] * 3 + [index, ctypes.POINTER(TraceRecord), index])
        self.library.z_evaluate.restype = ctypes.c_int

    def evaluate(self, agent, tasks, rng, *, round_limit, empirical_rate, structural_rate,
                 trace=False, return_weights=False, playback=False, stream=None, metrics=False):
        from .agent_z_environment import NumberLine

        if agent.width != 8:
            raise ValueError('the native number line requires one-byte signals')
        tasks = tuple(tasks)
        if any(len(task) != 2 or any(type(v) is not int or not -20 <= v <= 20 for v in task)
               for task in tasks):
            raise ValueError('expected integer start and goal between -20 and 20')
        pairs, episodes = len(agent.ids), len(tasks)
        trace_capacity = episodes * max(0, round_limit) * agent.regions
        if max(trace_capacity, episodes, round_limit, pairs * 9) > 2**31 - 1:
            raise ValueError('Agent Z evaluation exceeds the native index range')
        successors = (ctypes.c_int * pairs)(
            *(agent.index.get(agent.structure.successors.get(pair), -1) for pair in agent.ids))
        task_buffer = (ctypes.c_int * (episodes * 2))(*(v for task in tasks for v in task))
        actions = NumberLine().actions
        moves = (ctypes.c_int * len(actions))(*(actions[code] for code in sorted(actions)))
        state = (ctypes.c_uint32 * 625)(*rng.getstate()[1])
        embedding = (ctypes.c_double * (pairs * 9))(*(v for row in agent.embedding for v in row))
        continuation = (ctypes.c_double * (pairs * 8))(*(v for row in agent.continuation for v in row))
        if stream is not None and (len(stream) != episodes or
                                   any(type(byte) is not int or not 0 <= byte <= 255 for byte in stream)):
            raise ValueError('Expected one byte per stream frame')
        stream_buffer = (ctypes.c_int * episodes)(*stream) if stream is not None else None
        stats = (ctypes.c_int * 6)()
        records = (TraceRecord * trace_capacity)() if trace or playback else None
        if self.library.z_evaluate(
                pairs, agent.regions, episodes, max(0, round_limit), successors, task_buffer, moves,
                state, embedding, continuation, empirical_rate, structural_rate, .1, stats, records, stream_buffer):
            raise MemoryError('Agent Z could not allocate native evaluation buffers')
        successes, rounds, emitted, actions, tested, correct = stats
        result = dict(successes=successes, episodes=episodes, rounds=rounds)
        if metrics:
            result.update(actions=actions, tested=tested, correct=correct)
        if trace or playback:
            result['trace'] = [[] for _ in range(rounds)]
            for i in range(emitted):
                event = records[i]
                result['trace'][event.round].append(dict(
                    chain=dict(region=event.region, pair=agent.ids[event.pair]),
                    signal=f'{event.signal:02x}',
                    successor=agent.ids[event.successor] if event.successor >= 0 else None,
                    observation=f'{event.observation:02x}' if event.observation >= 0 else None,
                    mismatched_bits=event.mismatches if event.mismatches >= 0 else None,
                    compatibility=event.compatibility if event.compatibility >= 0 else None))
        if playback:
            frames = [None] * rounds
            for i in range(emitted):
                event = records[i]
                frames[event.round] = dict(position=event.position, actions=event.actions)
            result['world_frames'] = frames
        if return_weights:
            result.update(embedding=[embedding[i:i + 9] for i in range(0, len(embedding), 9)],
                          continuation=[continuation[i:i + 8] for i in range(0, len(continuation), 8)])
        return result

    def gradient(self, probabilities, targets, rate):
        count = len(probabilities)
        vector = ctypes.c_double * count
        output = vector()
        self.library.z_gradient(count, vector(*probabilities), vector(*targets), rate, output)
        return list(output)

    def embedding(self, rows, indices, signal, region, regions):
        count, bits = len(indices), len(signal)
        weights = (ctypes.c_double * (count * (bits + 1)))(
            *(value for index in indices for value in rows[index]))
        inputs = (ctypes.c_double * bits)(*signal)
        output = (ctypes.c_double * count)()
        self.library.z_embedding(count, bits, region, regions, weights, inputs, output)
        return list(output)

    def probabilities(self, logits):
        vector = ctypes.c_double * len(logits)
        output = vector()
        self.library.z_probabilities(len(logits), vector(*logits), output)
        return list(output)
