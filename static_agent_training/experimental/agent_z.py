"""Agent Z: fixed lambda successors with universal binary continuations."""
from collections import defaultdict
from dataclasses import dataclass
import math
from types import MappingProxyType

from .. import engine
from ..initialize import canonical_bytes, is_closed, pair_id
from ..repertoire import freeze
from .agent_z_kernels import PythonKernels


@dataclass(frozen=True)
class Structure:
    pairs: object
    successors: object

    def __post_init__(self):
        pairs = {key: tuple(freeze(term) for term in pair) for key, pair in self.pairs.items()}
        if not pairs or any(len(pair) != 2 or not all(is_closed(t) for t in pair) or pair_id(*pair) != key
                            for key, pair in pairs.items()):
            raise ValueError('invalid Agent Z pairs')
        successors = dict(self.successors)
        for source, target in successors.items():
            if (source not in pairs or target not in pairs or
                    canonical_bytes(pairs[source][0]) != canonical_bytes(pairs[target][1])):
                raise ValueError('invalid function-first successor')
        object.__setattr__(self, 'pairs', MappingProxyType(dict(sorted(pairs.items()))))
        object.__setattr__(self, 'successors', MappingProxyType(successors))

    @classmethod
    def develop(cls, seeds, *, budget=32, seed=0, max_nodes=256, max_transitions=32):
        result = engine.run(seeds, budget, seed=seed, max_nodes=max_nodes,
                            max_transitions=max_transitions, function_first=True)
        pairs, successors = {}, {}
        for event in result['events']:
            if event['kind'] == 'PAIR':
                pairs[event['pair_id']] = event['pair']
            elif event['kind'] == 'NORMAL':
                target = event['successor_pair_id']
                pairs[target] = event['pair']
                successors[event['source_pair_id']] = target
        return cls(pairs, successors), tuple(group['status'] for group in result['groups'])


def bits(signal, width):
    if type(signal) is not bytes or len(signal) * 8 != width:
        raise ValueError('expected a fixed-width byte signal')
    return tuple((byte >> shift) & 1 for byte in signal for shift in range(7, -1, -1))


def signal_from_bits(values):
    return bytes(sum(values[i + offset] << (7 - offset) for offset in range(8))
                 for i in range(0, len(values), 8))


@dataclass(frozen=True)
class Chain:
    region: int
    pair: str


@dataclass(frozen=True)
class Feedback:
    observation: bytes | None
    recipients: frozenset
    done: bool = False
    success: bool = False


@dataclass(frozen=True)
class Emission:
    chain: Chain
    signal: bytes
    successor: str | None
    observation: bytes | None = None
    mismatched_bits: int | None = None
    compatibility: float | None = None


class AgentZ:
    schema = 'crt-agent-z-function-first-v1'

    def __init__(self, structure, embedding, continuation, *, regions=2, kernels=None):
        self.structure = structure
        self.ids = tuple(structure.pairs)
        self.index = {pair: i for i, pair in enumerate(self.ids)}
        self.embedding = [list(row) for row in embedding]
        self.continuation = [list(row) for row in continuation]
        self.width = len(self.continuation[0]) if self.continuation else 0
        if not self.width or self.width % 8 or not 1 <= regions <= self.width:
            raise ValueError('invalid signal width or region count')
        for rows, size in ((self.embedding, self.width + 1), (self.continuation, self.width)):
            if len(rows) != len(self.ids) or any(len(row) != size or not all(map(math.isfinite, row))
                                                 for row in rows):
                raise ValueError('invalid Agent Z weight shape or value')
        self.regions = regions
        self.kernels = kernels if kernels is not None else PythonKernels()

    def to_data(self):
        return dict(schema_version=self.schema, pairs=dict(self.structure.pairs),
                    successors=dict(self.structure.successors), regions=self.regions,
                    embedding=[row[:] for row in self.embedding],
                    continuation=[row[:] for row in self.continuation])

    @classmethod
    def from_data(cls, data, *, kernels=None):
        if data.get('schema_version') != cls.schema:
            raise ValueError('expected Agent Z data; regenerate historical seeds for Agent Z')
        return cls(Structure(data['pairs'], data['successors']), data['embedding'],
                   data['continuation'], regions=data['regions'], kernels=kernels)

    def _embedding(self, signal, region):
        return self.kernels.embedding(self.embedding, range(len(self.ids)), signal,
                                      region, self.regions)

    def activate(self, observation, rng):
        sensory_bits = bits(observation, self.width)
        return tuple(Chain(region, rng.choices(self.ids, weights=self._embedding(sensory_bits, region))[0])
                     for region in range(self.regions))

    def step(self, chains, adapter, rng, *, empirical_rate=0.05, structural_rate=0.05,
             threshold=0.1):
        """Sample, obtain shared feedback, calculate every gradient, then apply."""
        if (not all(math.isfinite(rate) and rate >= 0 for rate in (empirical_rate, structural_rate))
                or not 0 <= threshold <= 1):
            raise ValueError('invalid learning settings')
        chains = tuple(sorted(chains, key=lambda chain: chain.region))
        if len({chain.region for chain in chains}) != len(chains) or any(
                not 0 <= chain.region < self.regions for chain in chains):
            raise ValueError('expected at most one chain per region')
        sampled, probabilities = [], []
        for chain in chains:
            probabilities.append(self.kernels.probabilities(self.continuation[self.index[chain.pair]]))
            sampled.append(signal_from_bits(tuple(int(rng.random() < p) for p in probabilities[-1])))

        feedback = adapter.interpret(tuple(sampled))
        if ((feedback.observation is None) != (not feedback.recipients) or
                not feedback.recipients <= set(range(len(chains)))):
            raise ValueError('invalid environment feedback')
        observation = bits(feedback.observation, self.width) if feedback.recipients else None
        empirical, structural = defaultdict(list), defaultdict(list)
        emissions, successors = [], []
        for i, (chain, signal, distribution) in enumerate(zip(chains, sampled, probabilities)):
            successor = self.structure.successors.get(chain.pair)
            if successor is not None:
                successors.append(Chain(chain.region, successor))
            if i not in feedback.recipients:
                emissions.append(Emission(chain, signal, successor))
                continue
            expected = bits(signal, self.width)
            mismatches = sum(a != b for a, b in zip(expected, observation))
            source = self.index[chain.pair]
            if empirical_rate:
                for bit, delta in enumerate(self.kernels.gradient(distribution, observation, empirical_rate)):
                    empirical[source, bit].append(delta)
            compatibility = None
            if successor is not None:
                target = self.index[successor]
                embedded = self._embedding(expected, chain.region)
                compatibility = embedded[target]
                if structural_rate and compatibility < threshold:
                    active = [0, *(bit + 1 for bit, value in enumerate(expected)
                                   if value and bit % self.regions == chain.region)]
                    targets = [int(row == target) for row in range(len(embedded))]
                    for row, contribution in enumerate(self.kernels.gradient(embedded, targets, structural_rate)):
                        for column in active:
                            structural[row, column].append(contribution)
            emissions.append(Emission(chain, signal, successor, feedback.observation,
                                      mismatches, compatibility))
        for weights, updates in ((self.continuation, empirical), (self.embedding, structural)):
            for (row, column), contributions in updates.items():
                weights[row][column] += math.fsum(sorted(contributions))
        return tuple(successors), feedback, tuple(emissions)
