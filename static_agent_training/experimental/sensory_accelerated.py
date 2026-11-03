"""CPU/CUDA evaluation of fixed sensory mappings, with Python replay RNGs."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile

from ..accelerated import SOURCE as SHARED_SOURCE, _numpy, _random_states, _check_allocation, _unpack
from .number_line import StimulusNumberLine

SOURCE = SHARED_SOURCE.with_name('evaluate_senses.cpp')


@lru_cache(maxsize=16)
def _kernel(capacity, receptors, backend):
    source = SOURCE.read_text().replace('#include "evaluate.cpp"', SHARED_SOURCE.read_text())
    options = ['-std=c++17', f'-DMAX_Q={capacity}', f'-DMAX_R={receptors}']
    if backend == 'cuda':
        try:
            import cupy as cp
        except ImportError as error:
            raise ValueError('CUDA evaluation requires CuPy') from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise ValueError('No CUDA device available')
        return cp.RawKernel(source, 'evaluate_senses_cuda', options=tuple(options + ['-DGPU_BUILD', '--fmad=false']))
    if backend != 'native':
        raise ValueError('backend must be native or cuda')
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
    function.argtypes = ([ctypes.c_int]*4 + [ctypes.c_void_p]*12 + [ctypes.c_int]*2 +
                         [ctypes.c_double]*3 + [ctypes.c_void_p]*2 + [ctypes.c_int])
    function.restype = None
    return function


@lru_cache(maxsize=16)
def _frames(senses, goal_conditioned):
    np = _numpy()
    world = StimulusNumberLine(goal_conditioned=goal_conditioned)
    frames = []
    for position in range(-20, 21):
        for goal in range(-20, 21):
            world.position, world.goal = position, goal
            frames.append(senses.spec.encode(senses.frame(world.stimulus())))
    return np.asarray(frames, dtype=np.uint8)


def _pack(population, encoding_seed, goal_conditioned):
    np = _numpy()
    senses = population[0].inherited.senses
    R, B = senses.receptors, len(population)
    Q = 1 << max(0, max(len(p.repertoire.pairs) for p in population)-1).bit_length()
    if Q > 4096 or R > 16384:
        raise ValueError('Accelerated sensory evaluation supports at most 4096 pairs and 16384 receptors')
    _check_allocation(B*Q*(2*R+12)*8 + 1681*(R+1))
    bias = np.zeros((B,Q)); weights = np.zeros((B,R+1,Q)); gates = np.zeros((B,Q,10)); prediction = np.zeros((B,Q,R))
    edges = np.full((B,Q), -1, dtype=np.int32); sizes = np.zeros(B, dtype=np.int32)
    for i, individual in enumerate(population):
        inherited = individual.inherited
        if inherited.senses != senses:
            raise ValueError('A sensory batch must share its fixed sensory capacity')
        ids = inherited.pair_ids; lookup = {q:j for j,q in enumerate(ids)}; n = len(ids); sizes[i] = n
        parameters = inherited.to_parameters()
        bias[i,:n] = parameters['bias']; weights[i,:,:n] = parameters['weights']
        for j,q in enumerate(ids):
            gates[i,j] = parameters['gates'][q]; prediction[i,j] = parameters['prediction'][q][0]
            edges[i,j] = lookup.get(individual.repertoire.successors.get(q), -1)
    moves = np.asarray(StimulusNumberLine(encoding_seed=encoding_seed).moves, dtype=np.int32)
    return [bias, weights, gates, prediction, edges, sizes, _frames(senses, goal_conditioned), moves]


def _phase(arrays, seeds, starts, goals, *, phase, learning, learning_config, backend, workers, boundary_mode):
    np = _numpy()
    B,Q,R = arrays[3].shape; E = len(starts)
    _check_allocation(sum(a.nbytes for a in arrays)*3+B*E*12*8)
    metrics = np.zeros((B,E,12)); revised = np.zeros((B,Q), dtype=np.uint8)
    inputs = [*arrays, np.asarray(starts,dtype=np.int32), np.asarray(goals,dtype=np.int32),
              _random_states(seeds,phase,'embedding'), _random_states(seeds,phase,'output')]
    scalars = [np.int32(v) for v in (B,Q,R,E)]
    rates = [np.int32(learning),np.int32(boundary_mode=='clamp'),np.float64(learning_config.prediction_rate),
             np.float64(learning_config.embedding_rate),np.float64(learning_config.compatibility_threshold)]
    kernel = _kernel(Q,R,backend)
    if backend == 'native':
        kernel(*scalars, *[ctypes.c_void_p(a.ctypes.data) for a in inputs], *rates,
               ctypes.c_void_p(metrics.ctypes.data), ctypes.c_void_p(revised.ctypes.data), workers)
    else:
        import cupy as cp
        device = [cp.asarray(a if i<len(inputs)-2 else a.T.copy()) for i,a in enumerate(inputs)]
        dm = cp.zeros(metrics.shape); dr = cp.zeros(revised.shape,dtype=cp.uint8)
        kernel(((B+63)//64,),(64,),(*scalars,*device,*rates,dm,dr))
        metrics = cp.asnumpy(dm)
        for i in range(4):
            arrays[i][:] = cp.asnumpy(device[i])
    return metrics


def _result(metrics, starts, goals):
    result = _unpack(metrics[:,:11],starts,goals)
    for episode, values in zip(result['episodes'],metrics):
        episode['empirical_loss_sum'] = float(values[11])
    return result


def learned_mappings(individual, arrays, index):
    from .sensory_genome import SensoryMappings
    ids = individual.inherited.pair_ids; n = len(ids)
    return SensoryMappings(ids, individual.inherited.senses, dict(
        bias=arrays[0][index,:n].tolist(),weights=arrays[1][index,:,:n].tolist(),
        gates={q:arrays[2][index,j].tolist() for j,q in enumerate(ids)},
        prediction={q:[arrays[3][index,j].tolist()] for j,q in enumerate(ids)}))


def evaluate_batch(population, training_starts, assessment_starts, *, seeds, encoding_seed=0,
                   training_goals=None, assessment_goals=None, learning=True, learning_config=None,
                   backend='native', workers=8, return_weights=False, boundary_mode='clamp', goal_conditioned=True):
    from ..learning import LearningConfig
    if not population or len(population)!=len(seeds) or not training_starts or not assessment_starts:
        raise ValueError('require nonempty populations, seeds and tasks')
    if type(workers) is not int or workers<1 or boundary_mode not in ('clamp','terminate'):
        raise ValueError('invalid workers or boundary mode')
    training_goals = [0]*len(training_starts) if training_goals is None else training_goals
    assessment_goals = [0]*len(assessment_starts) if assessment_goals is None else assessment_goals
    for starts, goals in ((training_starts,training_goals),(assessment_starts,assessment_goals)):
        if len(starts)!=len(goals) or any(type(s) is not int or type(g) is not int or not -20<=s<=20 or not -20<=g<=20 or s==g for s,g in zip(starts,goals)):
            raise ValueError('invalid sensory evaluation tasks')
    arrays = _pack(population,encoding_seed,goal_conditioned)
    settings = dict(learning_config=learning_config or LearningConfig(),backend=backend,workers=workers,boundary_mode=boundary_mode)
    training = _phase(arrays,seeds,training_starts,training_goals,phase='training',learning=learning,**settings)
    learned = [a.copy() for a in arrays[:4]] if return_weights else None
    assessment = _phase(arrays,seeds,assessment_starts,assessment_goals,phase='assessment',learning=False,**settings)
    results = [dict(genotype_id=p.genotype_id,seed=seeds[i],model='senses',
                    training=_result(training[i],training_starts,training_goals),
                    assessment=_result(assessment[i],assessment_starts,assessment_goals),
                    repertoire_size=len(p.repertoire.pairs),successor_count=len(p.repertoire.successors)) for i,p in enumerate(population)]
    return (results,learned) if return_weights else results


def evaluate_frozen(population, starts, goals, *, seeds, encoding_seed=0, backend='native', workers=8,
                    boundary_mode='clamp', goal_conditioned=True, goal_permutation=None):
    """Frozen control tasks; goal interventions affect sensory packets, not targets."""
    from ..learning import LearningConfig
    if not population or len(population)!=len(seeds) or not starts or len(starts)!=len(goals):
        raise ValueError('require nonempty matching populations, seeds and tasks')
    if any(type(s) is not int or type(g) is not int or not -20<=s<=20 or not -20<=g<=20 or s==g for s,g in zip(starts,goals)):
        raise ValueError('invalid frozen sensory tasks')
    if boundary_mode not in ('clamp','terminate') or type(workers) is not int or workers<1:
        raise ValueError('invalid boundary mode or worker count')
    arrays = _pack(population,encoding_seed,goal_conditioned)
    if goal_permutation is not None:
        np = _numpy()
        frames = arrays[6].reshape(41,41,-1)
        arrays[6] = np.ascontiguousarray(frames[:,[goal_permutation.get(g,g)+20 for g in range(-20,21)],:])
    metrics = _phase(arrays,seeds,starts,goals,phase='assessment',learning=False,
        learning_config=LearningConfig(),backend=backend,workers=workers,boundary_mode=boundary_mode)
    return [_result(row,starts,goals) for row in metrics]
