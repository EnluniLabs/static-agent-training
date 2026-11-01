"""Fixed mathematical structure, with no evaluator dependency."""

from dataclasses import dataclass
from types import MappingProxyType

from .initialize import is_closed, pair_id


def freeze(term):
    """Validate and convert JSON ASTs to immutable tuple ASTs."""
    if not isinstance(term, (tuple, list)) or not term:
        raise ValueError('invalid AST')
    tag = term[0]
    if tag == 'var' and len(term) == 2 and type(term[1]) is int and term[1] >= 0:
        return (tag, term[1])
    if tag == 'abs' and len(term) == 2:
        return (tag, freeze(term[1]))
    if tag == 'app' and len(term) == 3:
        return (tag, freeze(term[1]), freeze(term[2]))
    raise ValueError('invalid AST')


@dataclass(frozen=True)
class Repertoire:
    pairs: object
    successors: object

    def __post_init__(self):
        pairs = {}
        for identity, pair in self.pairs.items():
            a, b = (freeze(t) for t in pair)
            if not is_closed(a) or not is_closed(b) or pair_id(a, b) != identity:
                raise ValueError('invalid repertoire pair or identity')
            pairs[identity] = (a, b)
        successors = dict(self.successors)
        for source, target in successors.items():
            if source not in pairs or target not in pairs:
                raise ValueError('successor references unknown pair')
            if pairs[source][1] != pairs[target][0]:
                raise ValueError('successor does not preserve overlapping term')
        object.__setattr__(self, 'pairs', MappingProxyType(dict(sorted(pairs.items()))))
        object.__setattr__(self, 'successors', MappingProxyType(successors))

    @classmethod
    def from_development(cls, result):
        """Export only completed NORMAL edges; keep seeds and terminal pairs.

        Development logs are trusted reduction certificates. Loading does not
        recheck beta reduction, and never imports or invokes the evaluator.
        """
        pairs, successors = {}, {}

        def add(pair):
            pair = tuple(freeze(t) for t in pair)
            identity = pair_id(*pair)
            if identity in pairs and pairs[identity] != pair:
                raise ValueError('pair hash collision')
            pairs[identity] = pair
            return identity

        for pair in result['initial_pairs']:
            add(pair)
        for event in result['events']:
            if event['kind'] == 'PAIR':
                add(event['pair'])
            elif event['kind'] == 'NORMAL':
                target = add(event['pair'])
                source = event['source_pair_id']
                if source in successors and successors[source] != target:
                    raise ValueError('conflicting successor certificates')
                successors[source] = target
        return cls(pairs, successors)

    def to_data(self):
        return dict(schema_version='crt-repertoire-v1', pairs=dict(self.pairs),
                    successors=dict(self.successors))

    @classmethod
    def from_data(cls, data):
        if data['schema_version'] != 'crt-repertoire-v1':
            raise ValueError('unsupported repertoire schema')
        return cls(data['pairs'], data['successors'])
