"""Bounded process-based offspring construction with ordered, seeded results."""
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
import os

from . import serialization as json
from .genome import Species, DevelopmentConfig, mappings_from_data
from .mutation import mutate
from .repertoire import Repertoire
from .seeds import derive


# Local to each worker, and bounded independently of the number of generations.
_parent_cache = OrderedDict()


def _pack_species(species):
    # Numeric mappings pickle cheaply. Encode ASTs iteratively so process transport
    # does not reintroduce Python's recursion limit for developed lambda terms.
    return (json.dumps(species.initial_pairs, separators=(',', ':')), species.budgets,
            species.inherited.to_data(), json.dumps(species.repertoire.to_data(), separators=(',', ':')),
            asdict(species.development_config), species.development_json, species.genotype_id)


def _unpack_species(packet):
    pairs, budgets, mappings, repertoire, config, development, identity = packet
    species = Species(tuple(json.loads(pairs)), budgets, mappings_from_data(mappings),
                      Repertoire.from_data(json.loads(repertoire)), DevelopmentConfig(**config), development)
    # The identity was computed by the same process pipeline before transport.
    species.__dict__['genotype_id'] = identity
    return species


def _mutate_packet(packet, seed, mutation):
    identity = (packet[-1], packet[-2])
    parent = _parent_cache.get(identity)
    if parent is None:
        parent = _unpack_species(packet)
        _parent_cache[identity] = parent
        if len(_parent_cache) > 2:
            _parent_cache.popitem(last=False)
    else:
        _parent_cache.move_to_end(identity)
    child, log = mutate(parent, seed=seed, config=mutation)
    return _pack_species(child), log


def reproduction_workers(population, schedule, config):
    requested = config.reproduction_workers
    available = min(len(schedule), getattr(os, 'process_cpu_count', os.cpu_count)() or 1)
    if requested:
        return max(1, min(requested, available))
    # Avoid process startup/transport on small generations. Four processes bounds
    # transient copies of potentially large ASTs and parameter matrices in auto mode.
    cells = sum(len(population[job['parent_index']].repertoire.pairs) *
                (2 * getattr(getattr(population[job['parent_index']].inherited, 'senses', None), 'receptors', 0) + 12)
                for job in schedule)
    return min(4, config.workers, available) if len(schedule) >= 8 and cells >= 700_000 else 1


def construct_offspring(population, schedule, config, generation):
    """Yield children in schedule order, with at most one pending job per worker."""
    workers = reproduction_workers(population, schedule, config)
    if workers <= 1:
        for job in schedule:
            yield mutate(population[job['parent_index']],
                         seed=derive(config.seed, 'mutation', generation, job['index']), config=config.mutation)
        return
    # Spawn is safe after native OpenMP evaluation; forking a multithreaded native
    # runtime could inherit inconsistent runtime locks.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        packets = OrderedDict()
        def submit(job):
            index = job['parent_index']
            packet = packets.get(index)
            if packet is None:
                packet = _pack_species(population[index])
                packets[index] = packet
                if len(packets) > 2:
                    packets.popitem(last=False)
            else:
                packets.move_to_end(index)
            return pool.submit(_mutate_packet, packet,
                               derive(config.seed, 'mutation', generation, job['index']), config.mutation)
        jobs = iter(schedule)
        pending = deque(submit(job) for _, job in zip(range(workers), jobs))
        while pending:
            packet, log = pending.popleft().result()
            child = _unpack_species(packet)
            del packet
            next_job = next(jobs, None)
            if next_job is not None:
                pending.append(submit(next_job))
            yield child, log
