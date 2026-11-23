"""Frozen assessment of saved candidate populations; no training or evolution."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

from .accelerated import evaluate_frozen
from .baseline import random_task_mean
from .environments import environment_spec
from .evolution import TrainingConfig
from .experiment import write_json
from .experimental.sensory_accelerated import evaluate_frozen as sensory_frozen
from .genome import Species
from .populations import read_population, validate_compatibility, sensory_apparatus
from .seeds import derive


def validate_inputs(paths, environment, boundary_mode="clamp"):
    spec = environment_spec(environment)
    if not paths:
        raise ValueError('Select at least one saved population')
    sources = []
    for path in paths:
        path = Path(path).resolve()
        data = read_population(path)
        source = data['config']
        if data['schema_version'] in ('crt-agent-z-population-v1','crt-agent-q-population-v1'):
            if environment != 'original' or boundary_mode != 'clamp':
                raise ValueError('Agent Z and Q assessments require Original [−20, 20] with clamped walls')
            sources.append((path, data, TrainingConfig(varying_goals=True)))
            continue
        config = TrainingConfig(model=source.get('model','categorical'),**sensory_apparatus(source),
            environment=environment,embedding_selection=source.get('embedding_selection','sample'),
            action_policy=source.get('action_policy','agreement'),encoding_seed=source.get('encoding_seed',0),
            varying_goals=source.get('varying_goals',False))
        # A frozen assessment cannot initialize new categorical vocabulary entries.
        validate_compatibility(data,config)
        sources.append((path,data,config))
    if len({cfg.varying_goals for _,_,cfg in sources}) != 1:
        raise ValueError('Assess together populations with the same goal-input mode')
    return spec,sources


def run_assessment(paths, output, *, environment='original', episodes=200, seed=48, workers=4,
                   boundary_mode='clamp'):
    if type(episodes) is not int or not 1<=episodes<=1_000_000:
        raise ValueError('assessment episodes must be a positive integer, at most 1000000')
    if type(seed) is not int or not 0<=seed<=2**32-1 or type(workers) is not int or workers<1:
        raise ValueError('invalid assessment seed or worker count')
    if boundary_mode not in ('clamp','terminate'):
        raise ValueError('boundary mode must be clamp or terminate')
    spec,sources=validate_inputs(paths,environment,boundary_mode)
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new assessment output directory')
    output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(derive(seed,'frozen-population-tasks'))
    goals_vary=sources[0][2].varying_goals
    positions=spec.positions
    tasks=[]
    for _ in range(episodes):
        goal=0 if spec.kind=='lights-out' else rng.choice(positions) if goals_vary else spec.center
        start=rng.choice([x for x in positions if x!=goal])
        tasks.append((start,goal))
    starts,goals=[s for s,g in tasks],[g for s,g in tasks]
    report=dict(schema_version='crt-frozen-assessment-v1',status='running',pid=os.getpid(),updated=time.time(),
        config=dict(environment=environment,lower=spec.lower,upper=spec.upper,episodes=episodes,seed=seed,
                    workers=workers,boundary_mode=boundary_mode,backend='native',varying_goals=goals_vary),
        learning=False,evolution=False,weights='saved inherited genotype exactly as stored; no adaptation or warmup',
        task_source='shared uniform tasks over the selected world; may overlap source training tasks',tasks=tasks,
        task_id=hashlib.sha256(json.dumps(tasks,separators=(',',':')).encode()).hexdigest(),
        random_success=random_task_mean(starts,goals,boundary_mode=boundary_mode,**spec.baseline_kwargs),
        progress=dict(completed_animals=0,total_animals=sum(len(data['members']) for _,data,_ in sources)),populations=[])
    def save():
        report['updated']=time.time()
        write_json(output/'report.json',report)
    save()
    try:
        frozen=[]
        for index,(path,_,cfg) in enumerate(sources):
            snapshot=output/'inputs'/(f'{index:03d}.json.gz' if path.suffix=='.gz' else f'{index:03d}.json')
            snapshot.parent.mkdir(exist_ok=True)
            # Evaluate a retained copy, so the exact bytes used remain available.
            payload=path.read_bytes();snapshot.write_bytes(payload)
            data=read_population(snapshot)
            if data['schema_version'] not in ('crt-agent-z-population-v1','crt-agent-q-population-v1'):
                validate_compatibility(data,cfg)
            digest=hashlib.sha256(payload).hexdigest()
            frozen.append((path,snapshot,data,cfg,digest))
        report['progress']['total_animals']=sum(len(data['members']) for _,_,data,_,_ in frozen)
        for path,snapshot,data,cfg,digest in frozen:
            if data['schema_version'] in ('crt-agent-z-population-v1','crt-agent-q-population-v1'):
                from .experimental.agent_z import AgentZ
                from .experimental.agent_z_kernels import NativeKernels
                from .serialization import dumps
                from .experimental.agent_q import AgentQ
                from .experimental.agent_q_accelerated import NativeQ
                q = data['schema_version']=='crt-agent-q-population-v1'
                kernels, cls = (NativeQ(),AgentQ) if q else (NativeKernels(),AgentZ)
                population = dict(name=data['name'], source=str(path), snapshot=str(snapshot),
                                  sha256=digest, model='agent-q' if q else 'agent-z', count=len(data['members']),
                                  completed=0, individuals=[])
                report['populations'].append(population)
                for i, member in enumerate(data['members']):
                    agent = cls.from_data(member['individual'], kernels=kernels)
                    eval_seed = derive(seed, 'frozen-population-animal', i)
                    result = (kernels.evaluate(agent,tasks,seed=eval_seed,learning='off') if q else
                              kernels.evaluate(agent,tasks,random.Random(eval_seed),round_limit=64,
                                               empirical_rate=0,structural_rate=0,metrics=True))
                    identity = hashlib.sha256(dumps(member['individual'], sort_keys=True).encode()).hexdigest()
                    population['individuals'].append(dict(index=i, genotype_id=identity,
                        source_generation=member['generation'], assessment=dict(
                            fitness=result['successes'] / episodes, rounds=result['rounds'],
                            actions=result['actions'], tested=result['tested'],
                            prediction_accuracy=result['correct'] / result['tested'] if result['tested'] else None,
                            empirical_updates=0, conflicts=0)))
                    population['completed'] += 1
                    scores = [row['assessment']['fitness'] for row in population['individuals']]
                    population.update(mean_success=statistics.mean(scores),
                                      best_success=max(scores), worst_success=min(scores))
                    report['progress']['completed_animals'] += 1
                    save()
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                    raise AssertionError('Assessment input snapshot changed')
                print(f"{population['name']}: {population['completed']} frozen {population['model']} candidates; "
                      f"mean success {population['mean_success']:.2%}", flush=True)
                continue
            members=[Species.from_data(row['individual']) for row in data['members']]
            population=dict(name=data['name'],source=str(path),snapshot=str(snapshot),sha256=digest,
                model=cfg.model,embedding_selection=cfg.embedding_selection if cfg.model=="senses" else None,count=len(members),completed=0,individuals=[],
                **sensory_apparatus(data['config']),action_policy=cfg.action_policy)
            report['populations'].append(population)
            evaluator=sensory_frozen if cfg.model=='senses' else evaluate_frozen
            kwargs=dict(goal_conditioned=cfg.varying_goals,embedding_selection=cfg.embedding_selection,
                        sensory_partitions=cfg.sensory_partitions,action_policy=cfg.action_policy) if cfg.model=='senses' else {}
            for first in range(0,len(members),64):
                batch=members[first:first+64]
                identities=[p.genotype_id for p in batch]
                results=evaluator(batch,starts,goals,
                    seeds=[derive(seed,'frozen-population-animal',i) for i in range(first,first+len(batch))],
                    encoding_seed=cfg.encoding_seed,backend='native',workers=workers,
                    environment=environment,boundary_mode=boundary_mode,**kwargs)
                if identities != [p.genotype_id for p in batch]:
                    raise AssertionError('Frozen assessment changed candidate parameters or structure')
                for i,(identity,result) in enumerate(zip(identities,results),first):
                    if result['empirical_updates'] or result['conflicts']:
                        raise AssertionError('Frozen assessment performed a learning update')
                    population['individuals'].append(dict(index=i,genotype_id=identity,
                        source_generation=data['members'][i]['generation'],assessment=result))
                population['completed']+=len(batch)
                scores=[r['assessment']['fitness'] for r in population['individuals']]
                population.update(mean_success=statistics.mean(scores),best_success=max(scores),worst_success=min(scores))
                counts=Counter(e['reason'] for row in population['individuals'] for e in row['assessment']['episodes'])
                population['reasons']=dict(counts)
                report['progress']['completed_animals']+=len(batch)
                save()
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
                raise AssertionError('Assessment input snapshot changed')
            print(f"{population['name']}: {len(members)} frozen candidates; mean success {population['mean_success']:.2%}",flush=True)
        report['status']='completed';save()
    except KeyboardInterrupt:
        report['status']='stopped';save();raise
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--populations',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--environment',choices=('original','shifted','lights-out','switches-2x2','lights-out-2x2','switchboard'),default='original')
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--seed',type=int,default=48)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--boundary-mode',choices=('clamp','terminate'),default='clamp')
    args=vars(parser.parse_args())
    try:
        run_assessment(args.pop('populations'),**args)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__=='__main__':
    main()
