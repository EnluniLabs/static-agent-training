"""Offspring construction: inherited seed mutation and mapping mutation."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random

from .genome import Species
from .initialize import Config, depth, generate_term, is_closed, node_count, pair_id
from .mappings import InheritedMappings
from .repertoire import Repertoire


@dataclass(frozen=True)
class MutationConfig:
    weight_probability: float = 0.8
    subtree_probability: float = 0.15
    term_probability: float = 0.10
    insertion_probability: float = 0.05
    deletion_probability: float = 0.05
    weight_stddev: float = 0.1
    max_depth: int = 6
    max_nodes: int = 31
    combinator_probability: float = 0.5
    new_pair_budget: int = 64
    attempts: int = 32

    def __post_init__(self):
        probabilities = (self.weight_probability, self.subtree_probability,
                         self.term_probability, self.insertion_probability,
                         self.deletion_probability, self.combinator_probability)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError('probabilities must be finite and between zero and one')
        if not math.isfinite(self.weight_stddev) or self.weight_stddev < 0:
            raise ValueError('weight standard deviation must be finite and nonnegative')
        if type(self.new_pair_budget) is not int or self.new_pair_budget < 0:
            raise ValueError('new pair budget must be a nonnegative integer')
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError('attempts must be a positive integer')
        if type(self.max_depth) is not int or type(self.max_nodes) is not int:
            raise ValueError('term limits must be integers')
        self.generator.validate()

    @property
    def generator(self):
        return Config(max_depth=self.max_depth, max_nodes=self.max_nodes,
                      combinator_probability=self.combinator_probability)


def _seed(seed, purpose):
    return int.from_bytes(hashlib.sha256(f'crt-mutation-v1:{seed}:{purpose}'.encode()).digest(), 'big')


def _positions(term, path=(), binders=0):
    yield path, term, binders
    if term[0] != 'var':
        for child in range(1, len(term)):
            yield from _positions(term[child], path + (child,), binders + (term[0] == 'abs'))


def _replace(term, path, replacement):
    if not path:
        return replacement
    parts = list(term)
    parts[path[0]] = _replace(parts[path[0]], path[1:], replacement)
    return tuple(parts)


def _mergers(development):
    """Describe mergers by initial-pair identities, independent of list indices."""
    seeds = [pair_id(*p) for p in development['initial_pairs']]
    return {(e['pair_id'], tuple(sorted(seeds[i] for i in e['sources'])))
            for e in development['events'] if e['kind'] == 'MERGE'}


def mutate(parent, *, seed, config=None):
    """Return (offspring, log), never changing the parent or using lifetime weights.

    Independent operator draws; fixed order: subtree, term, insert, delete,
    redevelopment/reconciliation, then weight noise. No crossover.
    """
    if not isinstance(parent, Species):
        raise TypeError('mutation requires an inherited Species')
    config = config or MutationConfig()
    if any(depth(t) > config.max_depth or node_count(t) > config.max_nodes
           for p in parent.initial_pairs for t in p):
        raise ValueError('parent starting terms exceed mutation limits')
    purposes = ('operators', 'subtree', 'term', 'insert', 'delete', 'development', 'mapping', 'weights')
    seeds = {purpose: _seed(seed, purpose) for purpose in purposes}
    draw = random.Random(seeds['operators'])
    selected = {name: draw.random() < probability for name, probability in (
        ('weights', config.weight_probability), ('subtree', config.subtree_probability),
        ('term', config.term_probability), ('insert', config.insertion_probability),
        ('delete', config.deletion_probability))}
    pairs, budgets, operations = list(parent.initial_pairs), list(parent.budgets), []
    generator = config.generator
    for name in ('subtree', 'term'):
        if not selected[name]:
            continue
        rng = random.Random(seeds[name])
        index, component = rng.randrange(len(pairs)), rng.randrange(2)
        old = pairs[index][component]
        path, subtree, binders = (rng.choice(tuple(_positions(old))) if name == 'subtree'
                                  else ((), old, 0))
        limits = replace(generator, max_depth=config.max_depth - len(path),
                         max_nodes=config.max_nodes - node_count(old) + node_count(subtree))
        candidate = old
        for attempt in range(config.attempts):
            replacement = generate_term(rng, limits, binders=binders)
            candidate = _replace(old, path, replacement)
            if candidate != old:
                break
        assert is_closed(candidate) and depth(candidate) <= config.max_depth and node_count(candidate) <= config.max_nodes
        before = pair_id(*pairs[index])
        pair = list(pairs[index])
        pair[component] = candidate
        pairs[index] = tuple(pair)
        operations.append(dict(operator=name, index=index, component=component, path=path,
                               before=before, after=pair_id(*pairs[index]),
                               changed=candidate != old, attempts=attempt + 1))
    if selected['insert']:
        rng = random.Random(seeds['insert'])
        pair = (generate_term(rng, generator), generate_term(rng, generator))
        index = rng.randrange(len(pairs) + 1)
        pairs.insert(index, pair)
        budgets.insert(index, config.new_pair_budget)
        operations.append(dict(operator='insert', index=index, pair_id=pair_id(*pair),
                               budget=config.new_pair_budget, changed=True))
    if selected['delete']:
        if len(pairs) == 1:
            operations.append(dict(operator='delete', changed=False, reason='minimum_one_pair'))
        else:
            index = random.Random(seeds['delete']).randrange(len(pairs))
            pair, budget = pairs.pop(index), budgets.pop(index)
            operations.append(dict(operator='delete', index=index, pair_id=pair_id(*pair),
                                   budget=budget, changed=True))

    structural_changed = tuple(pairs) != parent.initial_pairs or tuple(budgets) != parent.budgets
    old_development = json.loads(parent.development_json)
    if structural_changed:
        from .engine import run
        development = run(pairs, budgets, seed=seeds['development'], **asdict(parent.development_config))
        repertoire = Repertoire.from_data(development['repertoire'])
        inherited = InheritedMappings.initialize(repertoire, parent.inherited.observations,
                                                 parent.inherited.outputs, seed=seeds['mapping'],
                                                 previous=parent.inherited)
        development_json = json.dumps(development, sort_keys=True)
    else:
        repertoire, inherited = parent.repertoire, parent.inherited
        development, development_json = old_development, parent.development_json

    weight_count = 0
    if selected['weights']:
        rng = random.Random(seeds['weights'])
        embedding = {s: dict(row) for s, row in inherited.embedding.items()}
        output = {q: dict(row) for q, row in inherited.output.items()}
        for matrix in (embedding, output):
            for row_key in sorted(matrix):
                row = matrix[row_key]
                for column in sorted(row):
                    row[column] += rng.gauss(0, config.weight_stddev)
                    weight_count += 1
        inherited = InheritedMappings(inherited.observations, inherited.outputs,
                                       inherited.pair_ids, embedding, output)
    child = Species(tuple(pairs), tuple(budgets), inherited, repertoire,
                    parent.development_config, development_json)
    before, after = set(parent.repertoire.pairs), set(repertoire.pairs)
    old_mergers, new_mergers = _mergers(old_development), _mergers(development)
    log = dict(schema_version='crt-mutation-v1', seed=seed, streams=seeds,
               config=asdict(config), parent_id=parent.genotype_id, offspring_id=child.genotype_id,
               selected=selected, operations=operations, structural_changed=structural_changed,
               redeveloped=structural_changed, added_pairs=sorted(after - before),
               removed_pairs=sorted(before - after), retained_pairs=sorted(before & after),
               lost_mergers=sorted(old_mergers - new_mergers),
               new_mergers=sorted(new_mergers - old_mergers), perturbed_logits=weight_count,
               development_contractions=development['accounting']['spent'] if structural_changed else 0)
    return child, log


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='saved Species JSON or offspring bundle')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    parent = Species.from_data(data.get('species', data))
    child, log = mutate(parent, seed=args.seed)
    args.output.write_text(json.dumps(dict(species=child.to_data(), mutation=log), indent=2) + '\n')


if __name__ == '__main__':
    main()
