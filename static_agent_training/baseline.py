"""Exact uniform-random movement baseline for the number-line task."""
from functools import lru_cache

from .environment import MOVES, POSITIONS


@lru_cache(maxsize=None)
def random_success(start, goal, moves=12, *, boundary_mode="clamp", lower=-20, upper=20):
    """Exact probability under uniformly sampled moves, including the zero move."""
    if boundary_mode not in ("clamp", "terminate"):
        raise ValueError("unknown boundary mode")
    if not lower <= start <= upper:
        return 0.0
    if start == goal:
        return 1.0
    if moves == 0:
        return 0.0
    return sum(random_success(max(lower,min(upper,start+move)) if boundary_mode == "clamp" else start+move,
                              goal, moves-1, boundary_mode=boundary_mode, lower=lower, upper=upper) for move in MOVES)/len(MOVES)


def random_task_mean(starts, goals=None, *, boundary_mode="clamp", lower=-20, upper=20):
    goals = [(lower+upper)//2] * len(starts) if goals is None else goals
    if not starts or len(starts) != len(goals):
        raise ValueError('require equal nonempty task lists')
    return sum(random_success(s, g, boundary_mode=boundary_mode, lower=lower, upper=upper) for s, g in zip(starts, goals)) / len(starts)
