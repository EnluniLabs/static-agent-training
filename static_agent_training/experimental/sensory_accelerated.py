"""Native CPU evaluation of fixed sensory mappings, with Python replay RNGs."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile

from ..accelerated import SOURCE as SHARED_SOURCE, _numpy, _random_states, _check_allocation, _unpack
from ..environments import environment_spec, stimulus_world
from .number_line import StimulusNumberLine
from .senses import stimulus_payload

SOURCE = SHARED_SOURCE.with_name('evaluate_senses.cpp')


@lru_cache(maxsize=16)
def _kernel(capacity, receptors, backend, environment="original"):
    source = SOURCE.read_text().replace('#include "evaluate.cpp"', SHARED_SOURCE.read_text())
    options = ['-std=c++17', f'-DMAX_Q={capacity}', f'-DMAX_R={receptors}']
    if environment_spec(environment).kind=='lights-out':options.extend(['-DLIGHTS_OUT',f'-DPUZZLE_STATES={len(environment_spec(environment).positions)}'])
    if backend != 'native':
        raise ValueError('backend must be native')
    key = hashlib.sha256((source + repr(options)).encode()).hexdigest()[:24]
    directory = Path.home() / '.cache/static-agent-training'
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f'senses-{key}.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            target = Path(tmp) / 'senses.so'
            try:
                subprocess.run(['g++', '-O3', '-shared', '-fPIC', '-fopenmp', '-ffp-contract=off',
                                *options, '-x', 'c++', '-', '-o', str(target)], input=source,
                               check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f'Native sensory evaluator compilation failed: {getattr(error, "stderr", str(error))}') from error
            target.replace(library)
    function = ctypes.CDLL(str(library)).evaluate_senses_cpu
    function.argtypes = ([ctypes.c_int]*4 + [ctypes.c_void_p]*13 + [ctypes.c_int]*2 +
                         [ctypes.c_double]*3 + [ctypes.c_int]*3 + [ctypes.c_void_p]*2 + [ctypes.c_int])
    function.restype = None
    return function


@lru_cache(maxsize=16)
def _frames(senses, goal_conditioned, environment="original"):
    spec = environment_spec(environment)
    np = _numpy()
    world = stimulus_world(environment,goal_conditioned=goal_conditioned)
    if spec.kind=='lights-out':
        frames = []
        for board in spec.positions:
            world.position, world.goal = board, 0
            payload = stimulus_payload(world, senses)
            frames.append(senses.spec.encode(senses.frame(payload)))
        return np.asarray(frames, dtype=np.uint8)
    frames = []
    for position in spec.positions:
        for goal in spec.positions:
            world.position, world.goal = position, goal
            frames.append(senses.spec.encode(senses.frame(stimulus_payload(world, senses))))
    return np.asarray(frames, dtype=np.uint8)


# Public source-frame cache uses the same source selection as the adapter.
native_frames = _frames

def _packed_sources(inherited):
    """Keep read-only sources with their immutable mappings, never a batch."""
    cached = getattr(inherited, '_packed_numpy_sources', None)
    if cached is None:
        np = _numpy()
        parameters = inherited.parameters
        ids = inherited.pair_ids
        values = (parameters['bias'], parameters['weights'],
                  tuple(parameters['gates'][q] for q in ids),
                  tuple(parameters['prediction'][q][0] for q in ids))
        # Immutable bytes backing also prevents callers re-enabling writes.
        def readonly(value):
            array = np.asarray(value, dtype=np.float64)
            return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
        cached = tuple(readonly(value) for value in values)
        object.__setattr__(inherited, '_packed_numpy_sources', cached)
    return cached


def _pack(population, encoding_seed, goal_conditioned, environment="original"):
    np = _numpy()
    senses = population[0].inherited.senses
    R, B = senses.receptors, len(population)
    Q = 1 << max(0, max(len(p.repertoire.pairs) for p in population)-1).bit_length()
    if Q > 4096 or R > 16384:
        raise ValueError('Accelerated sensory evaluation supports at most 4096 pairs and 16384 receptors')
    _check_allocation(B*Q*(2*R+12)*8 + (len(environment_spec(environment).positions) if environment_spec(environment).kind=='lights-out' else 1681)*(R+1))
    bias = np.zeros((B,Q)); weights = np.zeros((B,R+1,Q)); gates = np.zeros((B,Q,10)); prediction = np.zeros((B,Q,R))
    edges = np.full((B,Q), -1, dtype=np.int32); sizes = np.zeros(B, dtype=np.int32)
    for i, individual in enumerate(population):
        inherited = individual.inherited
        if inherited.senses != senses:
            raise ValueError('A sensory batch must share its fixed sensory capacity')
        ids = inherited.pair_ids; lookup = {q:j for j,q in enumerate(ids)}; n = len(ids); sizes[i] = n
        source_bias, source_weights, source_gates, source_prediction = _packed_sources(inherited)
        bias[i,:n] = source_bias; weights[i,:,:n] = source_weights
        gates[i,:n] = source_gates; prediction[i,:n] = source_prediction
        for j,q in enumerate(ids):
            edges[i,j] = lookup.get(individual.repertoire.successors.get(q), -1)
    moves = np.asarray(stimulus_world(environment,encoding_seed=encoding_seed).moves, dtype=np.int32)
    arrays = [bias, weights, gates, prediction, edges, sizes, _frames(senses, goal_conditioned, environment), moves]
    if getattr(senses, 'interface', None) in ('retina', 'signals'):
        arrays.append(_region_lookup(senses.partition_indices(), R))
    return arrays


def _region_lookup(regions, receptors):
    np = _numpy()
    mapping = np.full(receptors, -1, dtype=np.int32)
    for region, indices in enumerate(regions):
        for pixel in indices:
            if not 0 <= pixel < receptors or mapping[pixel] != -1:
                raise ValueError('partition indices must cover each receptor exactly once')
            mapping[pixel] = region
    if (mapping < 0).any():
        raise ValueError('partition indices must cover each receptor exactly once')
    return mapping


def _bound_partitions(senses, count):
    if getattr(senses, 'interface', None) in ('retina', 'signals'):
        senses.bind_partitions(count)
    return count


def _validate_parallel(embedding_selection, sensory_partitions, action_policy, receptors=None):
    if type(sensory_partitions) is not int or sensory_partitions < 1:
        raise ValueError("sensory_partitions must be a positive integer")
    if receptors is not None and sensory_partitions > receptors:
        raise ValueError("sensory partitions cannot exceed receptor count")
    if action_policy not in ("agreement", "vote"):
        raise ValueError("action_policy must be agreement or vote")
    if embedding_selection not in ("sample", "argmax"):
        raise ValueError("embedding_selection must be sample or argmax")
    if sensory_partitions > 1 and embedding_selection != "argmax":
        raise ValueError("multiple sensory partitions require argmax embedding selection")


def _phase(arrays, seeds, starts, goals, *, phase, learning, learning_config, backend, workers, boundary_mode, environment="original", embedding_selection="argmax", sensory_partitions=1, action_policy="agreement", region_map=None):
    B,Q,R = arrays[3].shape; E = len(starts)
    _validate_parallel(embedding_selection, sensory_partitions, action_policy, R)
    spec = environment_spec(environment)
    np = _numpy()
    parallel_bytes = min(B, workers)*sensory_partitions*(16*(R+12)+4) if sensory_partitions > 1 else 0
    _check_allocation(sum(a.nbytes for a in arrays)*3+B*E*12*8+parallel_bytes)
    metrics = np.zeros((B,E,12)); revised = np.zeros((B,Q), dtype=np.uint8)
    if region_map is None and len(arrays) > 8:
        region_map = arrays[8]
        if int(region_map.max()) + 1 != sensory_partitions:
            raise ValueError('sensory_partitions disagrees with bound layout')
    if region_map is None:
        from .senses import receptor_regions
        region_map = _region_lookup(tuple(tuple(range(a, b)) for a, b in receptor_regions(R, sensory_partitions)), R)
    region_map = np.ascontiguousarray(region_map, dtype=np.int32)
    if region_map.shape != (R,) or (region_map < 0).any() or (region_map >= sensory_partitions).any():
        raise ValueError('invalid receptor region map')
    inputs = [*arrays[:8], region_map, np.asarray(starts,dtype=np.int32)-spec.center, np.asarray(goals,dtype=np.int32)-spec.center,
              _random_states(seeds,phase,'embedding'), _random_states(seeds,phase,'output')]
    scalars = [np.int32(v) for v in (B,Q,R,E)]
    rates = [np.int32(learning),np.int32(boundary_mode=='clamp'),np.float64(learning_config.prediction_rate),
             np.float64(learning_config.embedding_rate),np.float64(learning_config.compatibility_threshold)]
    kernel = _kernel(Q,R,backend,environment)
    kernel(*scalars, *[ctypes.c_void_p(a.ctypes.data) for a in inputs], *rates, int(embedding_selection=="argmax"), sensory_partitions, int(action_policy=="vote"),
           ctypes.c_void_p(metrics.ctypes.data), ctypes.c_void_p(revised.ctypes.data), workers)
    return metrics


def _result(metrics, starts, goals):
    result = _unpack(metrics[:,:11],starts,goals)
    for episode, values in zip(result['episodes'],metrics):
        episode['empirical_loss_sum'] = float(values[11])
    return result


def learned_mappings(individual, arrays, index):
    from .sensory_genome import SensoryMappings
    if arrays and hasattr(arrays[0], 'inherited'):
        return arrays[index].inherited
    ids = individual.inherited.pair_ids; n = len(ids)
    return SensoryMappings(ids, individual.inherited.senses, dict(
        bias=arrays[0][index,:n].tolist(),weights=arrays[1][index,:,:n].tolist(),
        gates={q:arrays[2][index,j].tolist() for j,q in enumerate(ids)},
        prediction={q:[arrays[3][index,j].tolist()] for j,q in enumerate(ids)}))


def evaluate_batch(population, training_starts, assessment_starts, *, seeds, encoding_seed=0,
                   training_goals=None, assessment_goals=None, learning=True, learning_config=None,
                   backend='native', workers=8, return_weights=False, boundary_mode='clamp', goal_conditioned=True, environment="original", embedding_selection="argmax", sensory_partitions=1, action_policy="agreement"):
    from ..learning import LearningConfig
    spec = environment_spec(environment)
    if any(getattr(p.inherited, 'router', None) is not None for p in population):
        from .agent_y_accelerated import evaluate_batch as y_batch, adapter_config
        config=adapter_config(population,backend=backend,workers=workers,environment=environment,
            boundary_mode=boundary_mode,goal_conditioned=goal_conditioned,
            embedding_selection=embedding_selection,sensory_partitions=sensory_partitions,
            action_policy=action_policy,encoding_seed=encoding_seed,learning=learning,
            learning_config=learning_config,return_weights=return_weights)
        rows,parents=y_batch(population,training_starts,assessment_starts,training_goals,
                            assessment_goals,seeds=seeds,config=config)
        return (rows,parents) if return_weights else rows
    if not population or len(population)!=len(seeds) or not training_starts or not assessment_starts:
        raise ValueError('require nonempty populations, seeds and tasks')
    if type(workers) is not int or workers<1 or boundary_mode not in ('clamp','terminate'):
        raise ValueError('invalid workers or boundary mode')
    training_goals = [spec.center]*len(training_starts) if training_goals is None else training_goals
    assessment_goals = [spec.center]*len(assessment_starts) if assessment_goals is None else assessment_goals
    for starts, goals in ((training_starts,training_goals),(assessment_starts,assessment_goals)):
        if len(starts)!=len(goals) or any(type(s) is not int or type(g) is not int or not spec.lower<=s<=spec.upper or not spec.lower<=g<=spec.upper or s==g for s,g in zip(starts,goals)):
            raise ValueError('invalid sensory evaluation tasks')
        if spec.kind=='lights-out' and any(g!=0 for g in goals):raise ValueError('Lights Out goal must be 0')
    _bound_partitions(population[0].inherited.senses, sensory_partitions)
    _validate_parallel(embedding_selection, sensory_partitions, action_policy, population[0].inherited.senses.receptors)
    arrays = _pack(population,encoding_seed,goal_conditioned,environment)
    settings = dict(learning_config=learning_config or LearningConfig(),backend=backend,workers=workers,boundary_mode=boundary_mode,environment=environment,embedding_selection=embedding_selection,sensory_partitions=sensory_partitions,action_policy=action_policy)
    training = _phase(arrays,seeds,training_starts,training_goals,phase='training',learning=learning,**settings)
    learned = [a.copy() for a in arrays[:4]] if return_weights else None
    assessment = _phase(arrays,seeds,assessment_starts,assessment_goals,phase='assessment',learning=False,**settings)
    results = [dict(genotype_id=p.genotype_id,seed=seeds[i],model='senses',embedding_selection=embedding_selection,
                    sensory_partitions=sensory_partitions,action_policy=action_policy,
                    training=_result(training[i],training_starts,training_goals),
                    assessment=_result(assessment[i],assessment_starts,assessment_goals),
                    repertoire_size=len(p.repertoire.pairs),successor_count=len(p.repertoire.successors)) for i,p in enumerate(population)]
    return (results,learned) if return_weights else results


def evaluate_frozen(population, starts, goals, *, seeds, encoding_seed=0, backend='native', workers=8,
                    boundary_mode='clamp', goal_conditioned=True, goal_permutation=None, environment="original", embedding_selection="argmax", sensory_partitions=1, action_policy="agreement"):
    """Frozen control tasks; goal interventions affect sensory packets, not targets."""
    from ..learning import LearningConfig
    spec = environment_spec(environment)
    if not population or len(population)!=len(seeds) or not starts or len(starts)!=len(goals):
        raise ValueError('require nonempty matching populations, seeds and tasks')
    if any(type(s) is not int or type(g) is not int or not spec.lower<=s<=spec.upper or not spec.lower<=g<=spec.upper or s==g for s,g in zip(starts,goals)):
        raise ValueError('invalid frozen sensory tasks')
    if spec.kind=='lights-out' and any(g!=0 for g in goals):raise ValueError('Lights Out goal must be 0')
    if boundary_mode not in ('clamp','terminate') or type(workers) is not int or workers<1:
        raise ValueError('invalid boundary mode or worker count')
    _bound_partitions(population[0].inherited.senses, sensory_partitions)
    _validate_parallel(embedding_selection, sensory_partitions, action_policy, population[0].inherited.senses.receptors)
    arrays = _pack(population,encoding_seed,goal_conditioned,environment)
    if goal_permutation is not None and spec.kind=='lights-out':raise ValueError('Lights Out has a fixed goal')
    if goal_permutation is not None:
        np = _numpy()
        frames = arrays[6].reshape(41,41,-1)
        arrays[6] = np.ascontiguousarray(frames[:,[goal_permutation.get(g,g)-spec.lower for g in spec.positions],:])
    if any(getattr(p.inherited, 'router', None) is not None for p in population):
        from .agent_y_accelerated import adapter_config, router_arrays, _phase as y_phase
        config=adapter_config(population,backend=backend,workers=workers,environment=environment,
            boundary_mode=boundary_mode,goal_conditioned=goal_conditioned,
            embedding_selection=embedding_selection,sensory_partitions=sensory_partitions,
            action_policy=action_policy,encoding_seed=encoding_seed,learning=False)
        metrics,_=y_phase(arrays,router_arrays(population),seeds,starts,goals,
                          phase='assessment',learning=False,config=config)
        return [_result(row,starts,goals) for row in metrics]
    metrics = _phase(arrays,seeds,starts,goals,phase='assessment',learning=False,
        learning_config=LearningConfig(),backend=backend,workers=workers,boundary_mode=boundary_mode,environment=environment,embedding_selection=embedding_selection,sensory_partitions=sensory_partitions,action_policy=action_policy)
    return [_result(row,starts,goals) for row in metrics]
