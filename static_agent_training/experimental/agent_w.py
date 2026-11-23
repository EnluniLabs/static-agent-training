"""Agent W: physical outcomes constrain action sampling through fixed successors.

The experimental action-logit update is rate * (compatibility - 1 / pair_count)
times the sampled action's log-softmax gradient. The uniform baseline is a
hypothesis about structural evidence. It supplies no goal reward or information
about the consequences of untried actions. The embedding update independently
minimizes negative log compatibility of the fixed successor on actual feedback.
"""
from collections import defaultdict
from dataclasses import dataclass
from math import exp, fsum, isfinite, log

from .agent_z import Chain, Structure, bits
from .agent_z_kernels import PythonKernels


def probabilities(logits):
    peak = max(logits)
    weights = [exp(value - peak) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


@dataclass(frozen=True)
class ActionFeedback:
    chain: Chain
    action: bytes
    feedback: bytes
    successor: str | None
    compatibility: float | None
    action_probability: float
    entropy: float


class AgentW:
    schema = 'crt-agent-w-function-first-v1'

    def __init__(self, structure, embedding, continuation, *, actions, regions=2, kernels=None):
        self.structure = structure
        self.ids = tuple(self.node_ids(structure))
        self.index = {pair: i for i, pair in enumerate(self.ids)}
        self.actions = tuple(actions)
        self.embedding = [list(row) for row in embedding]
        self.continuation = [list(row) for row in continuation]
        self.regions = regions
        self.width = 8
        self.kernels = kernels if kernels is not None else PythonKernels()
        if not self.actions or any(type(signal) is not bytes or not signal for signal in self.actions):
            raise ValueError('require opaque action signals')
        if len(set(self.actions)) != len(self.actions) or type(regions) is not int or not 1 <= regions <= 8:
            raise ValueError('invalid action alphabet or region count')
        for rows, width in ((self.embedding, 9), (self.continuation, len(self.actions))):
            if len(rows) != len(self.ids) or any(
                    len(row) != width or not all(map(isfinite, row)) for row in rows):
                raise ValueError('invalid Agent W weights')

    @staticmethod
    def node_ids(structure):
        return structure.pairs

    def evaluate(self, tasks, **options):
        """Run a complete number-line lifetime on the native CPU."""
        from .agent_w_accelerated import evaluate

        return evaluate(self, tasks, **options)

    def copy(self):
        return type(self)(self.structure, self.embedding, self.continuation, actions=self.actions,
                      regions=self.regions, kernels=self.kernels)

    def to_data(self):
        return dict(schema_version=self.schema, pairs=dict(self.structure.pairs),
                    successors=dict(self.structure.successors), regions=self.regions,
                    actions=[signal.hex() for signal in self.actions],
                    embedding=[row[:] for row in self.embedding],
                    continuation=[row[:] for row in self.continuation])

    @classmethod
    def from_data(cls, data, *, kernels=None):
        if data.get('schema_version') != cls.schema:
            raise ValueError('expected an Agent W artifact')
        return cls(Structure(data['pairs'], data['successors']), data['embedding'],
                   data['continuation'], actions=[bytes.fromhex(signal) for signal in data['actions']],
                   regions=data['regions'], kernels=kernels)

    def embed(self, signal, region):
        return self.kernels.embedding(self.embedding, range(len(self.ids)), bits(signal, self.width),
                                      region, self.regions)

    def activate(self, observation, rng):
        return tuple(Chain(region, rng.choices(self.ids, weights=self.embed(observation, region))[0])
                     for region in range(self.regions))

    def step(self, chains, world, rng, *, round_index=0, action_rate=.05, structural_rate=.05):
        if any(not isfinite(rate) or rate < 0 for rate in (action_rate, structural_rate)):
            raise ValueError('learning rates must be finite and nonnegative')
        if world.action_signals != self.actions:
            raise ValueError('adapter action alphabet differs from the saved model')
        by_region = {chain.region: chain for chain in chains}
        if len(by_region) != len(chains) or any(not 0 <= r < self.regions for r in by_region):
            raise ValueError('expected at most one chain per region')
        action_updates, structural_updates = defaultdict(list), defaultdict(list)
        advanced, events = [], []
        for region in world.order(tuple(sorted(by_region)), round_index):
            if world.done:
                break
            chain = by_region[region]
            source = self.index[chain.pair]
            distribution = probabilities(self.continuation[source])
            chosen = rng.choices(range(len(self.actions)), weights=distribution)[0]
            action = self.actions[chosen]
            feedback = world.execute(action)
            successor = self.structure.successors.get(chain.pair)
            compatibility = None
            if successor is not None:
                advanced.append(Chain(region, successor))
                target = self.index[successor]
                embedding = self.embed(feedback, region)
                compatibility = embedding[target]
                if action_rate:
                    # The uniform embedding baseline supplies a signed local coefficient.
                    scale = action_rate * (compatibility - 1 / len(self.ids))
                    for i, value in enumerate(distribution):
                        action_updates[source, i].append(scale * (int(chosen == i) - value))
                if structural_rate:
                    active = [0, *(i + 1 for i, value in enumerate(bits(feedback, self.width))
                                   if value and i % self.regions == region)]
                    for i, value in enumerate(embedding):
                        delta = structural_rate * (int(i == target) - value)
                        for column in active:
                            structural_updates[i, column].append(delta)
            entropy = -sum(p * log(p) for p in distribution if p > 0)
            events.append(ActionFeedback(chain, action, feedback, successor, compatibility,
                                         distribution[chosen], entropy))
        # Parameters stay unchanged until every executed action has been scored.
        for weights, updates in ((self.continuation, action_updates), (self.embedding, structural_updates)):
            for (row, column), deltas in updates.items():
                weights[row][column] += fsum(sorted(deltas))
        return tuple(sorted(advanced, key=lambda chain: chain.region)), tuple(events)
