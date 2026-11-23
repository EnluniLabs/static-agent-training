"""Deterministic interleaving of independently budgeted coherent executions.

One scheduler round permits at most one beta contraction per active execution.
Historical pair junctions forward later arrivals to their current execution.
Spent work stays charged; an exhausted execution can resume with pooled budget.
"""

import argparse
from dataclasses import dataclass, field
from . import serialization as json
from pathlib import Path
import random

from .initialize import is_closed, node_count, pair_id
from .reduction import contract, redexes
from .repertoire import Repertoire, freeze



@dataclass
class _Execution:
    index: int
    pair: tuple
    remaining: int
    rng: random.Random
    sources: list
    expression: tuple | None = None
    registered: bool = False
    depth: int = 0
    spent: int = 0
    status: str = 'ACTIVE'
    parent: int | None = None
    path: list = field(default_factory=list)


def run(pairs, budgets, *, seed=0, max_nodes=256, max_transitions=1000, function_first=False):
    """Develop one or more pairs with pooled budgets at exact pair junctions.

    `budgets` is either a per-start integer or a sequence of integers. Cycles
    continue consuming budget; max_transitions also bounds zero-cost cycles.
    No physical mappings or cross-run state are involved.
    """
    pairs = [(freeze(a), freeze(b)) for a, b in pairs]
    if not pairs or any(not is_closed(t) for p in pairs for t in p):
        raise ValueError('provide at least one pair of closed terms')
    if type(budgets) is int:
        budgets = [budgets] * len(pairs)
    else:
        budgets = list(budgets)
    if len(budgets) != len(pairs) or any(type(b) is not int or b < 0 for b in budgets):
        raise ValueError('provide one nonnegative integer budget per pair')
    if type(max_nodes) is not int or max_nodes < 1 or type(max_transitions) is not int or max_transitions < 1:
        raise ValueError('limits must be positive integers')
    streams = random.Random(seed)
    executions = [_Execution(i, p, b, random.Random(streams.getrandbits(256)), [i])
                  for i, (p, b) in enumerate(zip(pairs, budgets))]
    events, successors, junctions = [], {}, {}

    def root(index):
        while executions[index].parent is not None:
            index = executions[index].parent
        return executions[index]

    def emit(kind, execution, **details):
        events.append(dict(kind=kind, round=round_index, execution=execution.index, **details))

    round_index = 0
    while any(e.parent is None and e.status == 'ACTIVE' for e in executions):
        for e in executions:
            if e.parent is not None or e.status != 'ACTIVE' or e.registered:
                continue
            pid = pair_id(*e.pair)
            previous = junctions.get(pid)
            if previous is not None:
                original_pair, owner = previous
                if original_pair != e.pair:
                    raise RuntimeError('pair hash collision')
                target = root(owner)
                if target.index != e.index:
                    emit('MERGE', e, target=target.index, pair_id=pid,
                         target_pair_id=pair_id(*target.pair),
                         target_status=target.status,
                         incoming_budget=e.remaining, target_budget=target.remaining,
                         pooled_budget=e.remaining + target.remaining,
                         sources=sorted(target.sources + e.sources))
                    target.remaining += e.remaining
                    target.spent += e.spent
                    target.sources = sorted(target.sources + e.sources)
                    e.remaining = e.spent = 0
                    e.parent = target.index
                    e.status = 'MERGED'
                    if target.status == 'BUDGET' and target.remaining:
                        target.status = 'ACTIVE'
                    continue
                emit('CYCLE', e, pair_id=pid)
            else:
                junctions[pid] = (e.pair, e.index)
            e.path.append(pid)
            e.registered = True
            emit('PAIR', e, pair_id=pid, pair=e.pair, remaining=e.remaining,
                 sources=list(e.sources))
        # Resolve every arrival before charging this round's computation.
        for e in executions:
            if e.parent is not None or e.status != 'ACTIVE':
                continue
            if e.depth >= max_transitions:
                e.status = 'DEPTH'
                continue
            if e.expression is None:
                f, x = e.pair if function_first else e.pair[::-1]
                e.expression = ('app', f, x)
            if node_count(e.expression) > max_nodes:
                e.status = 'SIZE'
                continue
            positions = tuple(redexes(e.expression))
            if positions:
                if e.remaining == 0:
                    e.status = 'BUDGET'
                    continue
                position = e.rng.choice(positions)
                e.expression = contract(e.expression, position)
                e.remaining -= 1
                e.spent += 1
                emit('CONTRACT', e, position=position, remaining=e.remaining)
                if node_count(e.expression) > max_nodes:
                    e.status = 'SIZE'
                    continue
                positions = tuple(redexes(e.expression))
            if not positions:
                pid = pair_id(*e.pair)
                successor = (e.expression, e.pair[0]) if function_first else (e.pair[1], e.expression)
                known = successors.setdefault(pid, successor)
                if known != successor:
                    raise RuntimeError('successful normal forms disagree')
                emit('NORMAL', e, source_pair_id=pid, pair=successor,
                     successor_pair_id=pair_id(*successor), remaining=e.remaining)
                e.pair = successor
                e.expression = None
                e.registered = False
                e.depth += 1
        round_index += 1

    groups = [dict(execution=e.index, sources=e.sources, status=e.status,
                   pair=e.pair, partial_term=e.expression, remaining=e.remaining,
                   spent=e.spent, transitions=e.depth)
              for e in executions if e.parent is None]
    initial = sum(budgets)
    spent = sum(g['spent'] for g in groups)
    remaining = sum(g['remaining'] for g in groups)
    assert initial == spent + remaining
    result = dict(schema_version='crt-engine-v1', seed=seed,
                config=dict(max_nodes=max_nodes, max_transitions=max_transitions),
                initial_pairs=pairs, initial_budgets=budgets, groups=groups,
                events=events, accounting=dict(initial=initial, spent=spent, remaining=remaining))
    if function_first:
        result['schema_version'] = 'crt-engine-function-first-v1'
    else:
        result['repertoire'] = Repertoire.from_development(result).to_data()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='initializer JSON file')
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-nodes', type=int, default=256)
    parser.add_argument('--max-transitions', type=int, default=1000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    pairs = [(p['left']['ast'], p['right']['ast']) for p in data['initial_pairs']]
    result = run(pairs, args.budget, seed=args.seed, max_nodes=args.max_nodes,
                 max_transitions=args.max_transitions)
    serialized = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end='')


if __name__ == '__main__':
    main()
