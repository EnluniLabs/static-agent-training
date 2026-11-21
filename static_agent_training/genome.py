"""Inherited species state and its predeveloped, fixed repertoire."""

from dataclasses import asdict, dataclass, replace
import hashlib
from . import serialization as json

from .initialize import is_closed
from .mappings import InheritedMappings
from .repertoire import Repertoire, freeze


@dataclass(frozen=True)
class DevelopmentConfig:
    max_nodes: int = 256
    max_transitions: int = 1000

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_nodes, self.max_transitions)):
            raise ValueError('development limits must be positive integers')


@dataclass(frozen=True)
class Species:
    initial_pairs: tuple
    budgets: tuple
    inherited: InheritedMappings
    repertoire: Repertoire
    development_config: DevelopmentConfig
    development_json: str

    def __post_init__(self):
        pairs = tuple(tuple(freeze(t) for t in pair) for pair in self.initial_pairs)
        budgets = tuple(self.budgets)
        if not pairs or any(len(p) != 2 or not all(is_closed(t) for t in p) for p in pairs):
            raise ValueError('species requires closed starting pairs')
        if len(budgets) != len(pairs) or any(type(b) is not int or b < 0 for b in budgets):
            raise ValueError('species requires one nonnegative budget per starting pair')
        from .experimental.sensory_genome import SensoryMappings
        if not isinstance(self.inherited, (InheritedMappings, SensoryMappings)):
            raise TypeError('species must contain inherited mappings, never lifetime mappings')
        if set(self.inherited.pair_ids) != set(self.repertoire.pairs):
            raise ValueError('inherited mappings must match the repertoire')
        object.__setattr__(self, 'initial_pairs', pairs)
        object.__setattr__(self, 'budgets', budgets)

    @property
    def genotype_id(self):
        payload = dict(initial_pairs=self.initial_pairs, budgets=self.budgets,
                       mappings=self.inherited.to_data(), development_config=asdict(self.development_config))
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def to_data(self):
        return dict(schema_version='crt-species-v1', initial_pairs=self.initial_pairs,
                    budgets=self.budgets, inherited=self.inherited.to_data(),
                    repertoire=self.repertoire.to_data(),
                    development_config=asdict(self.development_config),
                    development=json.loads(self.development_json))

    @classmethod
    def from_data(cls, data):
        if data['schema_version'] != 'crt-species-v1':
            raise ValueError('unsupported species schema')
        return cls(tuple(data['initial_pairs']), tuple(data['budgets']),
                   mappings_from_data(data['inherited']),
                   Repertoire.from_data(data['repertoire']),
                   DevelopmentConfig(**data['development_config']),
                   json.dumps(data['development'], sort_keys=True))


def develop(initial_pairs, observations, outputs, *, budgets=64, development_seed=0,
            mapping_seed=0, config=None, senses=None):
    """Initial development; lifetime code never needs to call this function."""
    from .engine import run

    config = config or DevelopmentConfig()
    result = run(initial_pairs, budgets, seed=development_seed, **asdict(config))
    repertoire = Repertoire.from_data(result['repertoire'])
    if senses is None:
        inherited = InheritedMappings.initialize(repertoire, observations, outputs, seed=mapping_seed)
    else:
        from .experimental.sensory_genome import SensoryMappings
        inherited = SensoryMappings.initialize(repertoire, senses=senses, seed=mapping_seed)
    return Species(tuple(result['initial_pairs']), tuple(result['initial_budgets']), inherited,
                   repertoire, config, json.dumps(result, sort_keys=True))


def mappings_from_data(data):
    if data.get('schema_version') == 'crt-sensory-mappings-v1':
        from .experimental.sensory_genome import SensoryMappings
        return SensoryMappings.from_data(data)
    return InheritedMappings.from_data(data)


def reconcile_mappings(previous, repertoire, seed):
    if isinstance(previous, InheritedMappings):
        return InheritedMappings.initialize(repertoire, previous.observations, previous.outputs,
                                            previous=previous, seed=seed)
    from .experimental.sensory_genome import SensoryMappings
    return SensoryMappings.initialize(repertoire, senses=previous.senses, previous=previous, seed=seed)


def inherit_lifetime_mappings(parent, embedding, output):
    """Explicitly commit end-of-training logits as a new immutable genotype.

    Repertoire, seeds, budgets, and development provenance are retained. Mutable
    lifetime state and assessment data are not inherited.
    """
    inherited=parent.inherited
    mappings=InheritedMappings(inherited.observations,inherited.outputs,inherited.pair_ids,embedding,output)
    return replace(parent,inherited=mappings)
