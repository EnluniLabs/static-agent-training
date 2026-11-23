"""Compiled CPU Agent Y evaluation, including variable grouping and its updates."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile
from dataclasses import replace

from ..accelerated import _numpy, _random_states, _check_allocation
from ..environments import environment_spec
from . import sensory_accelerated as sensory

SOURCE = sensory.SOURCE.with_name('evaluate_agent_y.cpp')


@lru_cache(maxsize=16)
def _kernel(capacity, receptors, environment):
    source = SOURCE.read_text().replace('#include "evaluate_senses.cpp"',
        sensory.SOURCE.read_text().replace('#include "evaluate.cpp"', sensory.SHARED_SOURCE.read_text()))
    options = ['-std=c++17', f'-DMAX_Q={capacity}', f'-DMAX_R={receptors}']
    if environment_spec(environment).kind == 'lights-out':
        options += ['-DLIGHTS_OUT', f'-DPUZZLE_STATES={len(environment_spec(environment).positions)}']
    key = hashlib.sha256((source + repr(options)).encode()).hexdigest()[:24]
    directory = Path.home()/'.cache/static-agent-training'
    directory.mkdir(parents=True, exist_ok=True)
    library = directory/f'agent-y-{key}.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            target = Path(tmp)/'agent-y.so'
            try:
                subprocess.run(['g++','-O3','-shared','-fPIC','-fopenmp','-ffp-contract=off',
                    *options,'-x','c++','-','-o',str(target)], input=source, text=True,
                    capture_output=True, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f'Native Agent Y compilation failed: {getattr(error,"stderr",str(error))}') from error
            target.replace(library)
    function = ctypes.CDLL(str(library)).evaluate_agent_y_cpu
    function.argtypes = ([ctypes.c_int]*4 + [ctypes.c_void_p]*13 + [ctypes.c_int]*2 +
        [ctypes.c_double]*3 + [ctypes.c_int]*3 + [ctypes.c_void_p]*7 + [ctypes.c_int])
    function.restype = None
    return function


def _phase(arrays, router, seeds, starts, goals, *, phase, learning, config):
    np = _numpy()
    B,Q,R = arrays[3].shape
    E = len(starts)
    metrics=np.zeros((B,E,12));revised=np.zeros((B,Q),dtype=np.uint8)
    groups=np.zeros((B,E,12),dtype=np.int32)
    # Native scratch is allocated per simultaneous lane, not per population.
    _check_allocation(sum(a.nbytes for a in arrays)+sum(a.nbytes for a in router)+
                      min(B,config.workers)*(R*R*12+16*R*(R+16))+metrics.nbytes+groups.nbytes)
    spec=environment_spec(config.environment)
    inputs=[*arrays[:8],np.zeros(R,dtype=np.int32),
            np.asarray(starts,dtype=np.int32)-spec.center,np.asarray(goals,dtype=np.int32)-spec.center,
            _random_states(seeds,phase,'embedding'),_random_states(seeds,phase,'output')]
    lr=config.learning_config
    _kernel(Q,R,config.environment)(B,Q,R,E,*[ctypes.c_void_p(a.ctypes.data) for a in inputs],
        int(learning),int(config.boundary_mode=='clamp'),lr.prediction_rate,lr.embedding_rate,
        lr.compatibility_threshold,1,1,int(config.action_policy=='vote'),
        ctypes.c_void_p(metrics.ctypes.data),ctypes.c_void_p(revised.ctypes.data),
        *[ctypes.c_void_p(a.ctypes.data) for a in router],ctypes.c_void_p(groups.ctypes.data),config.workers)
    return metrics,groups


def router_arrays(population):
    np=_numpy()
    arrays=[np.asarray([p.inherited.router['weights'][key] for p in population],dtype=np.float64)
            for key in ['affinity','split']]
    return arrays + [np.asarray([float('nan') if p.inherited.router['baseline'] is None else p.inherited.router['baseline'] for p in population]),
                     np.asarray([p.inherited.router['rate'] for p in population])]


def adapter_config(population, *, backend, workers, environment, boundary_mode,
                   goal_conditioned, embedding_selection, sensory_partitions,
                   action_policy, encoding_seed, learning=True, learning_config=None,
                   return_weights=False):
    from ..evolution import TrainingConfig
    from ..learning import LearningConfig
    senses=population[0].inherited.senses
    return TrainingConfig(model='senses',agent_y=True,backend=backend,workers=workers,
        environment=environment,boundary_mode=boundary_mode,varying_goals=goal_conditioned,
        embedding_selection=embedding_selection,sensory_partitions=sensory_partitions,
        action_policy=action_policy,encoding_seed=encoding_seed,learning=learning,
        learning_config=learning_config or LearningConfig(),inherit_lifetime=return_weights,
        receptors=senses.receptors,sensory_interface=senses.interface,
        retina_width=getattr(senses,'width',16),retina_height=getattr(senses,'height',16))


def evaluate_batch(population, starts, tests, goals, test_goals, *, seeds, config):
    np=_numpy()
    if not population or not starts or not tests or len(seeds)!=len(population):
        raise ValueError('require nonempty agents and tasks with one seed per agent')
    if any(p.inherited.router is None for p in population):
        raise ValueError('Agent Y native evaluation requires learned grouping mappings')
    center=environment_spec(config.environment).center
    goals=[center]*len(starts) if goals is None else goals
    test_goals=[center]*len(tests) if test_goals is None else test_goals
    if len(goals)!=len(starts) or len(test_goals)!=len(tests):
        raise ValueError('task goal counts must match starts')
    arrays=sensory._pack(population,config.encoding_seed,config.varying_goals,config.environment)
    router=router_arrays(population)
    training,train_groups=_phase(arrays,router,seeds,starts,goals,phase='training',learning=config.learning,config=config)
    reproductive=None
    if config.inherit_lifetime:
        from .sensory_genome import SensoryMappings
        reproductive=[]
        for i,p in enumerate(population):
            mappings=sensory.learned_mappings(p,arrays,i)
            baseline=float(router[2][i])
            state=dict(weights=dict(affinity=router[0][i].tolist(),split=router[1][i].tolist()),
                       baseline=None if np.isnan(baseline) else baseline,rate=float(router[3][i]))
            reproductive.append(replace(p,inherited=SensoryMappings(mappings.pair_ids,mappings.senses,mappings.parameters,state)))
    assessment,test_groups=_phase(arrays,router,seeds,tests,test_goals,phase='assessment',learning=False,config=config)
    results=[]
    for i,p in enumerate(population):
        train=sensory._result(training[i],starts,goals);test=sensory._result(assessment[i],tests,test_goals)
        train['partition_groups']=train_groups[i][train_groups[i]>0].tolist()
        test['partition_groups']=test_groups[i][test_groups[i]>0].tolist()
        results.append(dict(genotype_id=p.genotype_id,seed=seeds[i],model='senses',agent_y=True,
            embedding_selection='argmax',sensory_partitions=config.sensory_partitions,action_policy=config.action_policy,
            training=train,assessment=test,repertoire_size=len(p.repertoire.pairs),successor_count=len(p.repertoire.successors)))
    return results,reproductive
