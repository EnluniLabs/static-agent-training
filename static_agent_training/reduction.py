"""Seeded stochastic full beta reduction of immutable De Bruijn terms."""

from dataclasses import dataclass
import random

from .initialize import is_closed, node_count


def shift(term, amount, cutoff=0):
    tag = term[0]
    if tag == 'var':
        return ('var', term[1] + amount if term[1] >= cutoff else term[1])
    if tag == 'abs':
        return ('abs', shift(term[1], amount, cutoff + 1))
    return ('app', shift(term[1], amount, cutoff), shift(term[2], amount, cutoff))


def substitute(term, index, replacement, depth=0):
    if term[0] == 'var':
        return shift(replacement, depth) if term[1] == index + depth else term
    if term[0] == 'abs':
        return ('abs', substitute(term[1], index, replacement, depth + 1))
    return ('app', substitute(term[1], index, replacement, depth),
            substitute(term[2], index, replacement, depth))


def redexes(term, path=()):
    """Enumerate all redexes, including those beneath abstractions."""
    if term[0] == 'app' and term[1][0] == 'abs':
        yield path
    if term[0] != 'var':
        for child in range(1, len(term)):
            yield from redexes(term[child], path + (child,))


def contract(term, path):
    if path:
        child, *rest = path
        parts = list(term)
        parts[child] = contract(term[child], tuple(rest))
        return tuple(parts)
    if term[0] != 'app' or term[1][0] != 'abs':
        raise ValueError('path does not select a beta redex')
    return shift(substitute(term[1][1], 0, shift(term[2], 1)), -1)


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
