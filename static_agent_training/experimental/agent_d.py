"""Agent D: Agent W's evaluator over opaque, randomly connected states."""
from dataclasses import dataclass
from functools import cached_property
import hashlib
import re
from types import MappingProxyType

from ..speciation import Structure as Descriptor
from .agent_w import AgentW


def _new_id(rng, nodes):
    while True:
        identifier = f'{rng.getrandbits(256):064x}'
        if identifier not in nodes:
            return identifier


@dataclass(frozen=True)
class RandomGraph:
    nodes: tuple
    successors: object

    def __post_init__(self):
        nodes = tuple(self.nodes)
        if not nodes or any(not isinstance(node, str) or not re.fullmatch('[0-9a-f]{64}', node)
                            for node in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError('expected distinct opaque 256-bit node IDs')
        edges = dict(self.successors)
        known = set(nodes)
        if any(source not in known or target not in known for source, target in edges.items()):
            raise ValueError('successors must refer to existing nodes')
        object.__setattr__(self, 'nodes', tuple(sorted(nodes)))
        object.__setattr__(self, 'successors', MappingProxyType(edges))

    @classmethod
    def generate(cls, count, rng):
        if type(count) is not int or count < 1:
            raise ValueError('initial node count must be a positive integer')
        nodes = {}
        for _ in range(count):
            nodes[_new_id(rng, nodes)] = None
        choices = (*nodes, None)
        edges = {}
        for source in nodes:
            target = rng.choice(choices)
            if target is not None:
                edges[source] = target
        return cls(tuple(nodes), edges)

    def mutate(self, rng, config, max_nodes):
        nodes, edges = list(self.nodes), dict(self.successors)
        rewire, insert, delete = (rng.random() < p for p in (
            config.subtree_probability, config.insertion_probability, config.deletion_probability))
        if rewire:
            source, target = rng.choice(nodes), rng.choice((*nodes, None))
            if target is None:
                edges.pop(source, None)
            else:
                edges[source] = target
        if insert and len(nodes) < max_nodes:
            new = _new_id(rng, nodes)
            target = rng.choice((*nodes, new, None))
            if target is not None:
                edges[new] = target
            edges[rng.choice(nodes)] = new
            nodes.append(new)
        if delete and len(nodes) > 1:
            removed = rng.choice(nodes)
            nodes.remove(removed)
            edges = {source: target for source, target in edges.items()
                     if source != removed and target != removed}
        return self if nodes == list(self.nodes) and edges == self.successors else RandomGraph(tuple(nodes), edges)

    @cached_property
    def fingerprint(self):
        digest = hashlib.sha256(b'crt-agent-d-random-graph-v1')
        for node in self.nodes:
            digest.update(b'n' + bytes.fromhex(node))
        for source, target in sorted(self.successors.items()):
            digest.update(b'e' + bytes.fromhex(source) + bytes.fromhex(target))
        return digest.digest(), Descriptor(frozenset(), frozenset(self.nodes),
                                           frozenset(self.successors.items()))


class AgentD(AgentW):
    schema = 'crt-agent-d-random-graph-v1'

    @staticmethod
    def node_ids(structure):
        return structure.nodes

    def to_data(self):
        return dict(schema_version=self.schema, nodes=list(self.ids),
                    successors=dict(self.structure.successors), regions=self.regions,
                    actions=[signal.hex() for signal in self.actions],
                    embedding=[row[:] for row in self.embedding],
                    continuation=[row[:] for row in self.continuation])

    @classmethod
    def from_data(cls, data, *, kernels=None):
        if data.get('schema_version') != cls.schema:
            raise ValueError('expected an Agent D artifact')
        graph = RandomGraph(data['nodes'], data['successors'])
        return cls(graph, data['embedding'], data['continuation'],
                   actions=[bytes.fromhex(signal) for signal in data['actions']],
                   regions=data['regions'], kernels=kernels)
