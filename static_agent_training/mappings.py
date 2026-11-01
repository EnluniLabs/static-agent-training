"""Tabular physical mappings over an already developed repertoire."""

from dataclasses import dataclass
import hashlib
import math
import random
from types import MappingProxyType


def softmax(logits):
    values = tuple(logits)
    if not values:
        return ()
    if not all(math.isfinite(x) for x in values):
        raise ValueError('logits must be finite')
    largest = max(values)
    weights = tuple(math.exp(x - largest) for x in values)
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


def sample(labels, probabilities, rng):
    if not labels:
        return None
    draw, cumulative = rng.random(), 0.0
    for label, probability in zip(labels, probabilities):
        cumulative += probability
        if draw < cumulative:
            return label
    return labels[-1]  # Rounding at the upper endpoint.


def _frozen_rows(rows):
    return MappingProxyType({key: MappingProxyType(dict(row)) for key, row in rows.items()})


@dataclass(frozen=True)
class InheritedMappings:
    observations: tuple
    outputs: tuple
    pair_ids: tuple
    embedding: object
    output: object

    def __post_init__(self):
        for name in ('observations', 'outputs', 'pair_ids'):
            labels = tuple(getattr(self, name))
            if len(set(labels)) != len(labels):
                raise ValueError('duplicate vocabulary entry')
            object.__setattr__(self, name, labels)
        if not self.observations or not self.outputs:
            raise ValueError('physical vocabularies must be nonempty')
        for rows, row_labels, columns in (
            (self.embedding, self.observations, self.pair_ids),
            (self.output, self.pair_ids, self.outputs),
        ):
            if set(rows) != set(row_labels):
                raise ValueError('matrix row mismatch')
            for row in rows.values():
                if set(row) != set(columns) or not all(math.isfinite(x) for x in row.values()):
                    raise ValueError('invalid matrix columns or logits')
        object.__setattr__(self, 'embedding', _frozen_rows(self.embedding))
        object.__setattr__(self, 'output', _frozen_rows(self.output))

    @classmethod
    def initialize(cls, repertoire, observations, outputs, *, seed=0, previous=None, scale=0.1):
        """Preserve inherited cells by pair ID during mutation redevelopment."""
        observations, outputs = tuple(observations), tuple(outputs)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('initial logit scale must be nonnegative and finite')
        if previous is not None and (previous.observations != observations or previous.outputs != outputs):
            raise ValueError('physical vocabulary must remain fixed')
        ids = tuple(repertoire.pairs)

        def value(kind, row, column):
            old = getattr(previous, kind) if previous is not None else {}
            if row in old and column in old[row]:
                return old[row][column]
            key = repr(('crt-mapping-v1', seed, kind, row, column)).encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))
            return rng.gauss(0, scale)

        embedding = {s: {q: value('embedding', s, q) for q in ids} for s in observations}
        output = {q: {o: value('output', q, o) for o in outputs} for q in ids}
        return cls(observations, outputs, ids, embedding, output)

    def lifetime(self):
        return LifetimeMappings(self)


class LifetimeMappings:
    """Independent mutable logits; no repertoire construction or reduction."""

    def __init__(self, inherited):
        self.observations = inherited.observations
        self.outputs = inherited.outputs
        self.pair_ids = inherited.pair_ids
        self.embedding = {s: dict(row) for s, row in inherited.embedding.items()}
        self.output = {q: dict(row) for q, row in inherited.output.items()}

    def embedding_probabilities(self, observation):
        return softmax(self.embedding[observation][q] for q in self.pair_ids)

    def output_probabilities(self, pair):
        return softmax(self.output[pair][o] for o in self.outputs)

    def embed(self, observation, rng):
        return sample(self.pair_ids, self.embedding_probabilities(observation), rng)

    def emit(self, pair, rng):
        return sample(self.outputs, self.output_probabilities(pair), rng)
