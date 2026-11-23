"""Bounded spawned evaluation of independent Agent Y lifetimes."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
import os

from ..offspring import _pack_species, _unpack_species
from ..evolution import TrainingConfig
from ..seeds import derive

_context = None


def _initialize(config, training, assessment):
    global _context
    _context = TrainingConfig.from_data(config), training, assessment


def _evaluate(packet, seed):
    from .sensory_evolution import evaluate
    config, training, assessment = _context
    result, _, reproductive = evaluate(_unpack_species(packet), training, assessment,
        seed=seed, config=config, capture_detail=False)
    return result, _pack_species(reproductive) if reproductive is not None else None


def evaluate_population(population, config, generation, training, assessment):
    """Results retain original indices and seeds; pending agents <= worker count."""
    workers = min(config.workers, len(population), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize, initargs=(asdict(config), training, assessment)) as pool:
        pending = deque()
        next_index = 0
        def submit(index):
            return pool.submit(_evaluate, _pack_species(population[index]),
                               derive(config.seed, 'evaluation', generation, index))
        for _ in range(workers):
            pending.append(submit(next_index))
            next_index += 1
        while pending:
            result, reproductive = pending.popleft().result()
            if next_index < len(population):
                pending.append(submit(next_index))
                next_index += 1
            yield result, _unpack_species(reproductive) if reproductive is not None else None
