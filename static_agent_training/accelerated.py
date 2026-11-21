"""Optional native CPU lifetime evaluator, with reference RNG streams.

Each lane owns its mutable weights and runs lifetime episodes sequentially.
No reduction or numeric movement policy is performed by the agent.
"""
import ctypes
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import random
import subprocess
import tempfile

from .environments import environment_spec, categorical_world
from .environment import NumberLine
from .seeds import derive


SOURCE = Path(__file__).with_name('native') / 'evaluate.cpp'
REASONS = ('target', 'chain_terminated', 'out_of_bounds', 'action_limit', 'internal_limit')


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise ValueError('Accelerated evaluation requires NumPy; use .venv/bin/python or install requirements-accelerated.txt') from error
    return np


@lru_cache(maxsize=16)
def _cpu_kernel(capacity, environment="original"):
    source = SOURCE.read_bytes()
    puzzle = environment_spec(environment).kind=='lights-out'
    key = hashlib.sha256(source + str(capacity).encode()+str((puzzle,len(environment_spec(environment).positions))).encode()).hexdigest()[:24]
    directory = Path.home() / '.cache' / 'static-agent-training'
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f'evaluate-{key}.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            target = Path(tmp) / 'evaluate.so'
            try:
                subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp',
                                '-ffp-contract=off', f'-DMAX_Q={capacity}', *(['-DLIGHTS_OUT',f'-DPUZZLE_STATES={len(environment_spec(environment).positions)}'] if puzzle else []), str(SOURCE), '-o', str(target)],
                               check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f'Native evaluator compilation failed: {getattr(error, "stderr", str(error))}') from error
            target.replace(library)
    module = ctypes.CDLL(str(library))
    function = module.evaluate_cpu
    function.argtypes = ([ctypes.c_int]*4 + [ctypes.c_void_p]*13 + [ctypes.c_int]*2 +
                         [ctypes.c_double]*3 + [ctypes.c_void_p]*2 + [ctypes.c_int])
    function.restype = None
    return function


def _check_allocation(size_bytes):
    # Keep a single evaluator allocation well below the CLI process ceiling.
    if size_bytes > 512 * 1024 * 1024:
        raise MemoryError('Evaluator arrays exceed the 512 MiB safety budget; reduce --batch-size or episode counts')


def _pack(population, encoding_seed, environment="original"):
    spec = environment_spec(environment)
    np = _numpy()
    conditioned = any(s.startswith('goal_') for s in population[0].inherited.observations)
    world = categorical_world(environment,encoding_seed=encoding_seed,goal_conditioned=conditioned)
    observations = tuple(s.token for s in world.observation_tokens)
    outputs = tuple(s.token for s in world.output_tokens)
    size = len(population)
    capacity = 1 << max(0, max(len(p.repertoire.pairs) for p in population)-1).bit_length()
    if capacity > 4096:
        raise ValueError('Accelerated evaluator supports at most 4096 pairs per animal')
    _check_allocation(size * capacity * ((len(observations) + len(outputs)) * 8 + 4))
    embedding = np.zeros((size, len(observations), capacity), dtype=np.float64)
    output = np.zeros((size, capacity, len(outputs)), dtype=np.float64)
    edges = np.full((size, capacity), -1, dtype=np.int32)
    sizes = np.zeros(size, dtype=np.int32)
    for index, individual in enumerate(population):
        inherited = individual.inherited
        if inherited.observations != observations or inherited.outputs != outputs:
            raise ValueError('Accelerated batch must use the same NumberLine vocabulary and encoding')
        ids = inherited.pair_ids
        sizes[index] = len(ids)
        lookup = {q: j for j, q in enumerate(ids)}
        for s, signal in enumerate(observations):
            embedding[index, s, :len(ids)] = [inherited.embedding[signal][q] for q in ids]
        for j, q in enumerate(ids):
            output[index, j] = [inherited.output[q][o] for o in outputs]
            edges[index, j] = lookup.get(individual.repertoire.successors.get(q), -1)
    state = world.diagnostic_state()
    obs_lookup = {s: j for j, s in enumerate(observations)}
    observation_rows = np.array([obs_lookup[state['observation_encoding'][x]] for x in spec.positions], dtype=np.int32)
    goal_rows = np.array([obs_lookup.get(f'goal_{x:+d}', -1) for x in spec.positions], dtype=np.int32)
    expectation_rows = np.array([obs_lookup['obs_' + o[len('expect_'):]] for o in outputs[9:]], dtype=np.int32)
    values = np.array([state['move_decoding'].get(o, state['expectation_decoding'].get(o)) - (spec.center if o.startswith('expect_') else 0) for o in outputs], dtype=np.int32)
    target_columns = np.array([list(values[9:]).index(x-spec.center) for x in spec.positions], dtype=np.int32)
    return [embedding, output, edges, sizes, observation_rows, goal_rows, expectation_rows, values, target_columns]


def _random_states(seeds, phase, kind):
    np = _numpy()
    return np.array([random.Random(derive(seed, phase, kind)).getstate()[1] for seed in seeds], dtype=np.uint32)


def _phase(arrays, seeds, starts, goals, *, phase, learning, learning_config, backend, workers, boundary_mode, environment="original"):
    spec = environment_spec(environment)
    np = _numpy()
    embedding, output, edges, sizes, *encoding = arrays
    batch, signals, capacity = embedding.shape
    count = len(starts)
    _check_allocation(sum(a.nbytes for a in arrays) * 3 + batch * count * 11 * 8)
    metrics = np.zeros((batch, count, 11), dtype=np.float64)
    revised = np.zeros((batch, capacity), dtype=np.uint8)
    inputs = [embedding, output, edges, sizes, *encoding,
              np.asarray(starts, dtype=np.int32)-spec.center, np.asarray(goals, dtype=np.int32)-spec.center,
              _random_states(seeds, phase, 'embedding'),
              _random_states(seeds, phase, 'output')]
    scalar_args = [np.int32(batch), np.int32(capacity), np.int32(signals), np.int32(count)]
    rates = [np.int32(learning), np.int32(boundary_mode == "clamp"), np.float64(learning_config.prediction_rate),
             np.float64(learning_config.embedding_rate), np.float64(learning_config.compatibility_threshold)]
    if backend == 'native':
        pointers = [ctypes.c_void_p(a.ctypes.data) for a in inputs]
        _cpu_kernel(capacity,environment)(*[int(x) for x in scalar_args], *pointers, *rates,
                              ctypes.c_void_p(metrics.ctypes.data), ctypes.c_void_p(revised.ctypes.data), workers)
    else:
        raise ValueError('backend must be native')
    return metrics


def _unpack(metrics, starts, goals):
    episodes = []
    for values, start, goal in zip(metrics, starts, goals):
        success, reason, outputs, actions, expectations, tested, correct, loss, expired, updates, conflicts = values
        episodes.append(dict(start=start, goal=goal, success=bool(success), reason=REASONS[int(reason)],
                             outputs=int(outputs), actions=int(actions), expectations=int(expectations),
                             tested=int(tested), correct=int(correct), loss_sum=float(loss), expired=int(expired)))
    total = metrics.sum(axis=0)
    tested, correct = int(total[5]), int(total[6])
    return dict(fitness=float(total[0]/len(episodes)), episodes=episodes,
                outputs=int(total[2]), actions=int(total[3]), expectations=int(total[4]),
                tested=tested, correct=correct, prediction_accuracy=correct/tested if tested else None,
                prediction_nll=float(total[7]/tested) if tested else None, expired=int(total[8]),
                empirical_updates=int(total[9]), conflicts=int(total[10]))


def evaluate_batch(population, training_starts, assessment_starts, *, seeds, encoding_seed=0,
                   training_goals=None, assessment_goals=None, learning=True, learning_config=None,
                   backend='native', workers=8, return_weights=False, boundary_mode='clamp', environment='original'):
    from .experimental.sensory_genome import SensoryMappings
    if population and isinstance(population[0].inherited, SensoryMappings):
        from .experimental.sensory_accelerated import evaluate_batch as sensory_batch
        return sensory_batch(population, training_starts, assessment_starts, seeds=seeds,
            encoding_seed=encoding_seed, training_goals=training_goals, assessment_goals=assessment_goals,
            learning=learning, learning_config=learning_config, backend=backend, workers=workers,
            return_weights=return_weights, boundary_mode=boundary_mode, environment=environment)
    from .learning import LearningConfig
    spec = environment_spec(environment)
    if not population or len(seeds) != len(population):
        raise ValueError('require a seed for each member of a nonempty batch')
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not training_starts or not assessment_starts or set(training_starts) & set(assessment_starts):
        raise ValueError('training and assessment starts must be nonempty and disjoint')
    training_goals = [spec.center]*len(training_starts) if training_goals is None else training_goals
    assessment_goals = [spec.center]*len(assessment_starts) if assessment_goals is None else assessment_goals
    for starts, goals in ((training_starts, training_goals), (assessment_starts, assessment_goals)):
        if len(starts) != len(goals) or any(type(s) is not int or type(g) is not int or
                not spec.lower <= s <= spec.upper or not spec.lower <= g <= spec.upper or s == g for s, g in zip(starts, goals)):
            raise ValueError('require matching valid starts and goals that differ')
        if spec.kind=='lights-out' and any(g!=0 for g in goals):raise ValueError('Lights Out goal must be 0')
    arrays = _pack(population, encoding_seed, environment)
    if all(x == -1 for x in arrays[5]) and any(g != spec.center for g in [*training_goals, *assessment_goals]):
        raise ValueError('nonzero goals require a goal-conditioned animal')
    if boundary_mode not in ('clamp', 'terminate'):
        raise ValueError('boundary mode must be clamp or terminate')
    settings = dict(boundary_mode=boundary_mode, learning_config=learning_config or LearningConfig(), backend=backend, workers=workers, environment=environment)
    training = _phase(arrays, seeds, training_starts, training_goals, phase='training', learning=learning, **settings)
    learned = [arrays[0].copy(), arrays[1].copy()] if return_weights else None
    assessment = _phase(arrays, seeds, assessment_starts, assessment_goals, phase='assessment', learning=False, **settings)
    results = []
    for index, individual in enumerate(population):
        results.append(dict(genotype_id=individual.genotype_id, seed=seeds[index],
                            training=_unpack(training[index], training_starts, training_goals),
                            assessment=_unpack(assessment[index], assessment_starts, assessment_goals),
                            repertoire_size=len(individual.repertoire.pairs),
                            successor_count=len(individual.repertoire.successors)))
    return (results, learned) if return_weights else results


def assert_equivalent(actual, expected, path='result'):
    """Fail on behavioral differences; permit only small arithmetic roundoff."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'Accelerated replay keys differ at {path}')
        for key in expected:
            assert_equivalent(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f'Accelerated replay lengths differ at {path}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_equivalent(a, b, f'{path}[{i}]')
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10):
            raise ValueError(f'Accelerated replay differs at {path}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'Accelerated replay differs at {path}: {actual} != {expected}')


def evaluate_frozen(population, starts, goals, *, seeds, encoding_seed=0, backend='native', workers=8, boundary_mode='clamp', environment='original'):
    """Assess supplied immutable weights on arbitrary tasks, with no training block."""
    from .experimental.sensory_genome import SensoryMappings
    if population and isinstance(population[0].inherited, SensoryMappings):
        from .experimental.sensory_accelerated import evaluate_frozen as sensory_frozen
        return sensory_frozen(population, starts, goals, seeds=seeds, encoding_seed=encoding_seed,
                              backend=backend, workers=workers, boundary_mode=boundary_mode, environment=environment)
    from .learning import LearningConfig
    spec = environment_spec(environment)
    if not population or len(population)!=len(seeds) or not starts or len(starts)!=len(goals):
        raise ValueError('require nonempty matching populations, seeds, and tasks')
    if any(type(s) is not int or type(g) is not int or not spec.lower<=s<=spec.upper or not spec.lower<=g<=spec.upper or s==g for s,g in zip(starts,goals)):
        raise ValueError('invalid frozen assessment tasks')
    if spec.kind=='lights-out' and any(g!=0 for g in goals):raise ValueError('Lights Out goal must be 0')
    if boundary_mode not in ('clamp','terminate') or type(workers) is not int or workers<1:
        raise ValueError('invalid boundary mode or worker count')
    arrays=_pack(population,encoding_seed,environment)
    if all(x==-1 for x in arrays[5]) and any(g!=spec.center for g in goals):
        raise ValueError('nonzero goals require a goal-conditioned animal')
    metrics=_phase(arrays,seeds,starts,goals,phase='assessment',learning=False,
                   learning_config=LearningConfig(),backend=backend,workers=workers,boundary_mode=boundary_mode,environment=environment)
    return [_unpack(row,starts,goals) for row in metrics]
