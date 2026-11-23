"""Agent Q derives actions from predicted outcomes and fixed successors."""
from collections import defaultdict
from dataclasses import dataclass
from math import exp, fsum, isfinite, log, log1p
import random

from .agent_w_environment import NumberLine
from .agent_z import Chain, Structure, bits
from .agent_z_kernels import sigmoid


SIGNALS = tuple(bits(bytes([i]), 8) for i in range(256))
MODES = {'off': (0., 0.), 'empirical': (.05, 0.), 'structural': (0., .05), 'both': (.05, .05)}


@dataclass(frozen=True)
class Event:
    chain: Chain
    successor: str
    action: int
    expectation: int
    observation: int
    empirical_error: float
    structural_error: float


class AgentQ:
    schema = 'crt-agent-q-function-first-v1'

    def __init__(self, structure, embedding, expectations, *, regions=2, exploration=.05, kernels=None):
        self.structure = structure
        self.ids = tuple(structure.pairs)
        self.index = {pair: i for i, pair in enumerate(self.ids)}
        self.embedding = [list(row) for row in embedding]
        self.expectations = [[list(row) for row in pair] for pair in expectations]
        self.actions = NumberLine().action_signals
        self.regions, self.exploration, self.kernels = regions, exploration, kernels
        if type(regions) is not int or not 1 <= regions <= 8 or not 0 <= exploration <= 1:
            raise ValueError('invalid Agent Q region count or exploration rate')
        if (len(self.embedding) != len(self.ids) or len(self.expectations) != len(self.ids)
                or any(len(row) != 9 or not all(map(isfinite,row)) for row in self.embedding)
                or any(len(pair) != len(self.actions) or
                       any(len(row) != 8 or not all(map(isfinite,row)) for row in pair)
                       for pair in self.expectations)):
            raise ValueError('invalid Agent Q parameter shape or value')

    def to_data(self):
        return dict(schema_version=self.schema, pairs=dict(self.structure.pairs),
                    successors=dict(self.structure.successors), regions=self.regions,
                    exploration=self.exploration, embedding=[row[:] for row in self.embedding],
                    expectations=[[row[:] for row in pair] for pair in self.expectations])

    @classmethod
    def from_data(cls, data, *, kernels=None):
        if data.get('schema_version') != cls.schema:
            raise ValueError('expected an Agent Q artifact')
        return cls(Structure(data['pairs'], data['successors']), data['embedding'], data['expectations'],
                   regions=data['regions'], exploration=data['exploration'], kernels=kernels)

    def copy(self):
        return AgentQ(self.structure, self.embedding, self.expectations, regions=self.regions,
                      exploration=self.exploration,kernels=self.kernels)

    def _log_embedding(self, signal, region):
        active = [bit+1 for bit in range(region,8,self.regions) if SIGNALS[signal][bit]]
        h = [row[0] + sum(row[i] for i in active) for row in self.embedding]
        peak = max(h)
        shifted = [x-peak for x in h]
        norm = log(sum(exp(x) for x in shifted))
        return [x-norm for x in shifted]

    def embed(self, signal, region):
        return [exp(x) for x in self._log_embedding(signal,region)]

    def activate(self, observation, rng):
        return tuple(Chain(region,rng.choices(self.ids,weights=self.embed(observation[0],region))[0])
                     for region in range(self.regions))

    def action_compatibilities(self, chain):
        """Reference implementation enumerates all 256 full-byte expectations."""
        target = self.index[self.structure.successors[chain.pair]]
        compatible = [self.embed(signal,chain.region)[target] for signal in range(256)]
        result = []
        for row in self.expectations[self.index[chain.pair]]:
            p = [(sigmoid(-z),sigmoid(z)) for z in row]
            weights = [1.]
            for zero,one in p:
                weights = [part for weight in weights for part in (weight*zero,weight*one)]
            result.append(sum(a*b for a,b in zip(weights,compatible)))
        return result

    def policy(self, chain):
        k = self.action_compatibilities(chain)
        total = sum(k)
        uniform = 1/len(k)
        return [(1-self.exploration)*x/total+self.exploration*uniform for x in k] if total else [uniform]*len(k)

    def step(self, chains, world, rng, *, round_index=0, learning='both'):
        empirical_rate, structural_rate = MODES[learning]
        if self.actions != world.action_signals:
            raise ValueError('adapter action alphabet differs from the saved model')
        if len({c.region for c in chains}) != len(chains) or any(
                not 0 <= c.region < self.regions or c.pair not in self.index for c in chains):
            raise ValueError('expected one known anchor per active region')
        by_region = {chain.region:chain for chain in chains if chain.pair in self.structure.successors}
        order = world.order(tuple(sorted(by_region)),round_index)
        empirical, structural = defaultdict(list), defaultdict(list)
        events, advanced = [], []
        # Apply both correction tables only after all executed actions in the round.
        for region in order:
            if world.done:
                break
            chain = by_region[region]
            source = self.index[chain.pair]
            successor = self.structure.successors[chain.pair]
            target = self.index[successor]
            action = rng.choices(range(len(self.actions)),weights=self.policy(chain))[0]
            logits = self.expectations[source][action]
            p = [sigmoid(z) for z in logits]
            sample = 0
            for prob in p:
                sample = (sample << 1) | (rng.random() < prob)
            observation = world.execute(self.actions[action])[0]
            log_embedding = self._log_embedding(sample, region)
            loss = sum(max(-z if bit else z,0) + log1p(exp(-abs(z)))
                       for z,bit in zip(logits,SIGNALS[observation]))
            if empirical_rate:
                for i,(bit,prob) in enumerate(zip(SIGNALS[observation],p)):
                    empirical[source,action,i].append(empirical_rate*(bit-prob))
            if structural_rate:
                indices = [0,*(i+1 for i in range(region,8,self.regions) if SIGNALS[sample][i])]
                for row,logit in enumerate(log_embedding):
                    delta = structural_rate*(int(row==target)-exp(logit))
                    for i in indices:
                        structural[row,i].append(delta)
            events.append(Event(chain,successor,action,sample,observation,loss,-log_embedding[target]))
            advanced.append(Chain(region,successor))
        for (r,a,i),updates in empirical.items():
            self.expectations[r][a][i] += fsum(sorted(updates))
        for (r,i),updates in structural.items():
            self.embedding[r][i] += fsum(sorted(updates))
        return tuple(sorted(advanced,key=lambda c:c.region)),tuple(events)


def evaluate(agent,tasks,*,seed,learning='both',round_limit=64,trace=False,return_weights=False):
    """Frozen or adapting reference lifetime on a separate parameter copy."""
    model = agent.copy()
    world, rng = NumberLine(), random.Random(seed)
    successes = rounds = actions = tested = correct = 0
    trace_records, world_frames = [], []
    for task in tasks:
        chains = model.activate(world.reset(task),rng)
        for round_index in range(round_limit):
            if world.done:
                break
            chains, events = model.step(chains,world,rng,round_index=round_index,learning=learning)
            if not events:
                break
            actions += len(events)
            tested += len(events)
            correct += sum(e.expectation==e.observation for e in events)
            rounds += 1
            if trace:
                trace_records.append([event_data(e,model) for e in events])
                world_frames.append(dict(position=world._world.position, actions=12-world._world.remaining))
        successes += world.success
    result = dict(successes=successes,episodes=len(tasks),rounds=rounds,actions=actions,tested=tested,correct=correct)
    if trace:
        result.update(trace=trace_records,world_frames=world_frames)
    if return_weights:
        result.update(embedding=model.embedding,expectations=model.expectations)
    return result


def event_data(event,model):
    return dict(chain=dict(region=event.chain.region,pair=event.chain.pair),
                action=model.actions[event.action].hex(), signal=f'{event.expectation:02x}',
                observation=f'{event.observation:02x}', successor=event.successor,
                mismatched_bits=(event.expectation^event.observation).bit_count(),
                empirical_error=event.empirical_error,structural_error=event.structural_error,
                compatibility=exp(-event.structural_error))
