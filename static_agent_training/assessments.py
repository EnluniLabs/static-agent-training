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
from .populations import read_population, validate_compatibility
from .seeds import derive


def validate_inputs(paths, environment):
    spec = environment_spec(environment)
    if not paths:
        raise ValueError('Select at least one saved population')
    sources = []
    for path in paths:
        path = Path(path).resolve()
        data = read_population(path)
        source = data['config']
        config = TrainingConfig(model=source.get('model','categorical'),receptors=source.get('receptors',256),
            environment=environment,encoding_seed=source.get('encoding_seed',0),
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
    spec,sources=validate_inputs(paths,environment)
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new assessment output directory')
    output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(derive(seed,'frozen-population-tasks'))
    goals_vary=sources[0][2].varying_goals
    positions=spec.positions
    tasks=[]
    for _ in range(episodes):
        goal=rng.choice(positions) if goals_vary else spec.center
        start=rng.choice([x for x in positions if x!=goal])
        tasks.append((start,goal))
    starts,goals=[s for s,g in tasks],[g for s,g in tasks]
    report=dict(schema_version='crt-frozen-assessment-v1',status='running',pid=os.getpid(),updated=time.time(),
        config=dict(environment=environment,lower=spec.lower,upper=spec.upper,episodes=episodes,seed=seed,
                    workers=workers,boundary_mode=boundary_mode,backend='native',varying_goals=goals_vary),
        learning=False,evolution=False,weights='saved inherited genotype exactly as stored; no adaptation or warmup',
        task_source='shared uniform tasks over the selected world; may overlap source training tasks',tasks=tasks,
        task_id=hashlib.sha256(json.dumps(tasks,separators=(',',':')).encode()).hexdigest(),
        random_success=random_task_mean(starts,goals,boundary_mode=boundary_mode,**spec.world_kwargs),
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
            validate_compatibility(data,cfg)
            digest=hashlib.sha256(payload).hexdigest()
            frozen.append((path,snapshot,data,cfg,digest))
        report['progress']['total_animals']=sum(len(data['members']) for _,_,data,_,_ in frozen)
        for path,snapshot,data,cfg,digest in frozen:
            members=[Species.from_data(row['individual']) for row in data['members']]
            population=dict(name=data['name'],source=str(path),snapshot=str(snapshot),sha256=digest,
                model=cfg.model,count=len(members),completed=0,individuals=[])
            report['populations'].append(population)
            evaluator=sensory_frozen if cfg.model=='senses' else evaluate_frozen
            kwargs=dict(goal_conditioned=cfg.varying_goals) if cfg.model=='senses' else {}
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
    parser.add_argument('--environment',choices=('original','shifted'),default='original')
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
