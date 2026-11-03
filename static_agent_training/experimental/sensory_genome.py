"""Immutable heritable sensory parameters for the existing Species/evolution loop."""
from dataclasses import dataclass
import math
import random
from types import MappingProxyType

from ..seeds import derive
from .bit_model import BitModel, validate_parameters
from .senses import FixedSenses


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(child) for child in value)
    return value


def thaw(value):
    if hasattr(value, 'items'):
        return {key: thaw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class SensoryMappings:
    pair_ids: tuple
    senses: FixedSenses
    parameters: object

    def __post_init__(self):
        object.__setattr__(self, 'pair_ids', tuple(self.pair_ids))
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError('duplicate repertoire IDs')
        validate_parameters(self.parameters, self.pair_ids, self.senses.spec)
        object.__setattr__(self, 'parameters', freeze(thaw(self.parameters)))

    @classmethod
    def initialize(cls, repertoire, *, senses, seed=0, previous=None):
        if previous is not None and previous.senses != senses:
            raise ValueError('sensory capacity cannot change within a lineage')
        ids = tuple(repertoire.pairs)
        old = previous.parameters if previous is not None else None
        positions = {q: i for i, q in enumerate(previous.pair_ids)} if previous is not None else {}
        def value(*coordinates):
            return random.Random(derive(seed, 'bit-model-v1', *coordinates)).gauss(0, .1)
        bias = [old['bias'][positions[q]] if q in positions else value('bias', q) for q in ids]
        weights = [[old['weights'][i][positions[q]] if q in positions else value('input', i, q)
                    for q in ids] for i in range(senses.spec.features)]
        gates = {q: thaw(old['gates'][q]) if q in positions else
                 [value('gate', q, i) for i in range(senses.spec.actions + senses.spec.channels)] for q in ids}
        prediction = {q: thaw(old['prediction'][q]) if q in positions else
                      [[value('prediction', q, c, i) for i in range(senses.spec.bits)]
                       for c in range(senses.spec.channels)] for q in ids}
        return cls(ids, senses, dict(bias=bias, weights=weights, gates=gates, prediction=prediction))

    def to_parameters(self):
        return thaw(self.parameters)

    def model(self, repertoire, *, seed=0, learning_config=None):
        if tuple(repertoire.pairs) != self.pair_ids:
            raise ValueError('sensory mappings and repertoire differ')
        return BitModel(repertoire, senses=self.senses, parameters=self.to_parameters(),
                        seed=seed, learning_config=learning_config)

    def perturb(self, *, seed, scale):
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('mutation scale must be nonnegative and finite')
        rng = random.Random(seed)
        count = 0
        def noisy(value):
            nonlocal count
            if isinstance(value, dict):
                return {key: noisy(value[key]) for key in sorted(value)}
            if isinstance(value, list):
                return [noisy(child) for child in value]
            count += 1
            return value + rng.gauss(0, scale)
        return SensoryMappings(self.pair_ids, self.senses, noisy(self.to_parameters())), count

    def to_data(self):
        return dict(schema_version='crt-sensory-mappings-v1', pair_ids=self.pair_ids,
                    senses=self.senses.to_data(), parameters=self.to_parameters())

    @classmethod
    def from_data(cls, data):
        if data['schema_version'] != 'crt-sensory-mappings-v1':
            raise ValueError('unsupported sensory mappings')
        return cls(tuple(data['pair_ids']), FixedSenses.from_data(data['senses']), data['parameters'])
