"""Agent Z numeric kernels. Round scheduling and environment stay in Python."""
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
