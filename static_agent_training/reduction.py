"""Seeded stochastic full beta reduction of immutable De Bruijn terms."""

from dataclasses import dataclass
import random

from .initialize import is_closed, node_count


def _map_variables(term, context, variable):
    """Rebuild in postorder with binder context, without hashing ASTs."""
    pending = [(term, context, False)]
    values = []
    while pending:
        node, bound, visited = pending.pop()
        tag = node[0]
        if tag == 'var':
            values.append(variable(node, bound))
        elif visited:
            if tag == 'abs':
                values.append(('abs', values.pop()))
            else:
                right = values.pop()
                left = values.pop()
                values.append(('app', left, right))
        else:
            pending.append((node, bound, True))
            if tag == 'abs':
                pending.append((node[1], bound + 1, False))
            else:
                pending.append((node[2], bound, False))
                pending.append((node[1], bound, False))
    return values[0]


def shift(term, amount, cutoff=0):
    return _map_variables(term, cutoff, lambda node, bound:
                          ('var', node[1] + amount if node[1] >= bound else node[1]))


def substitute(term, index, replacement, depth=0):
    return _map_variables(term, depth, lambda node, bound:
                          shift(replacement, bound) if node[1] == index + bound else node)


def redexes(term, path=()):
    """Enumerate all redexes, including those beneath abstractions."""
    pending = [(term, path)]
    while pending:
        node, position = pending.pop()
        if node[0] == 'app' and node[1][0] == 'abs':
            yield position
        if node[0] != 'var':
            for child in range(len(node) - 1, 0, -1):
                pending.append((node[child], position + (child,)))


def contract(term, path):
    ancestors = []
    for child in path:
        ancestors.append((term, child))
        term = term[child]
    if term[0] != 'app' or term[1][0] != 'abs':
        raise ValueError('path does not select a beta redex')
    term = shift(substitute(term[1][1], 0, shift(term[2], 1)), -1)
    for parent, child in reversed(ancestors):
        parts = list(parent)
        parts[child] = term
        term = tuple(parts)
    return term


@dataclass(frozen=True)
class Reduction:
    status: str
    term: tuple
    steps: int
    remaining: int
    trace: tuple


def normalize(term, budget, *, seed=0, max_nodes=256):
    """Charge every contraction; exhaustion is unknown, never divergence proof."""
    if budget < 0 or max_nodes < 1:
        raise ValueError('budget must be nonnegative and max_nodes positive')
    if not is_closed(term):
        raise ValueError('expected a closed term')
    rng = random.Random(seed)
    trace = []
    while True:
        if node_count(term) > max_nodes:
            status = 'SIZE'
            break
        positions = tuple(redexes(term))
        if not positions:
            status = 'NORMAL'
            break
        if len(trace) == budget:
            status = 'BUDGET'
            break
        path = rng.choice(positions)
        term = contract(term, path)
        trace.append(path)
    return Reduction(status, term, len(trace), budget - len(trace), tuple(trace))
