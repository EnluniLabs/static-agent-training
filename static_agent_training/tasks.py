"""Reproducible task sampling with an explicit distribution over absolute distance."""
import math
import random

from .environment import POSITIONS


def distance_weights(distances, mean=10.0, stddev=5.0):
    """Discrete Gaussian mass over feasible integer distances, renormalized.

    Weight distances themselves, not pairs: nearby distances must not gain extra
    mass merely because more start/target pairs happen to realize them.
    """
    if not math.isfinite(mean) or not 1 <= mean <= 40:
        raise ValueError('distance mean must be finite and between 1 and 40')
    if not math.isfinite(stddev) or stddev <= 0:
        raise ValueError('distance standard deviation must be finite and positive')
    if not distances:
        raise ValueError('no feasible nonzero distances')
    deviations = [abs(d-mean) for d in distances]
    closest = min(deviations)
    # Shift log weights by their maximum before exponentiation. This also handles
    # a narrow distribution whose center is absent from a held-out partition.
    weights = []
    for deviation in deviations:
        if deviation == closest:
            weights.append(1.0)
        else:
            a, b = (deviation-closest)/stddev, (deviation+closest)/stddev
            weights.append(math.exp(-0.5*a*b))
    return weights


def sample_tasks(seed, count, *, starts, goals, mean=10.0, stddev=5.0):
    if type(count) is not int or count < 1:
        raise ValueError('task count must be a positive integer')
    starts, goals = tuple(sorted(set(starts))), tuple(sorted(set(goals)))
    if not starts or not goals or any(type(x) is not int or x not in POSITIONS for x in (*starts, *goals)):
        raise ValueError('require valid nonempty start and goal positions')
    buckets = {}
    for start in starts:
        for goal in goals:
            if start != goal:
                buckets.setdefault(abs(goal-start), []).append((start, goal))
    distances = sorted(buckets)
    weights = distance_weights(distances, mean, stddev)
    rng = random.Random(seed)
    draws = rng.choices(distances, weights=weights, k=count)
    return [rng.choice(buckets[d]) for d in draws]
