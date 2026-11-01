"""Separate target-location effects from distance, with a random-walk control."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import statistics

from .accelerated import evaluate_frozen
from .baseline import random_success
from .candidates import load_saved_candidate
from .genome import Species
from .mappings import InheritedMappings
from .seeds import derive
from .tasks import distance_weights


def exposure(positions, mean=10, stddev=5, distribution="normal"):
    if distribution == "uniform":
        return {g:1/len(positions) for g in positions}
    buckets=defaultdict(list)
    for start in positions:
        for goal in positions:
            if start!=goal:
                buckets[abs(start-goal)].append((start,goal))
    distances=sorted(buckets)
    weights=distance_weights(distances,mean,stddev)
    total=sum(weights)
    probabilities={g:0.0 for g in positions}
    for d,w in zip(distances,weights):
        for s,g in buckets[d]:
            probabilities[g]+=w/total/len(buckets[d])
    return probabilities


def diagnose(directories, *, repeats=100, distances=(1,5,10)):
    population=[]
    metadata=[]
    encoding=None
    for directory in map(Path,directories):
        bundle=load_saved_candidate(directory)
        if not bundle.get('varying_goals'):
            raise ValueError('diagnostics require a goal-conditioned candidate')
        species=Species.from_data(bundle['individual'])
        mappings=InheritedMappings(species.inherited.observations,species.inherited.outputs,species.inherited.pair_ids,
                                    bundle['learned_embedding'],bundle['learned_output'])
        population.append(replace(species,inherited=mappings))
        saved=json.loads((directory/'config.json').read_text())
        metadata.append(dict(run=directory.name,generation=bundle['generation'],training_goals=saved['partitions']['training'],
                             expected_training_goal_exposure=exposure(saved['partitions']['training'],
                                 saved['config'].get('distance_mean',10),saved['config'].get('distance_stddev',5),
                                 distribution=saved['config'].get('distance_distribution','uniform'))))
        if encoding is not None and encoding!=bundle['encoding_seed']:
            raise ValueError('candidates must share an encoding')
        encoding=bundle['encoding_seed']
    tasks=[]
    for distance in distances:
        for goal in range(-20,21):
            for start in (goal-distance,goal+distance):
                if -20<=start<=20:
                    tasks.extend([(start,goal,distance)]*repeats)
    output=dict(candidates=metadata,repeats_per_pair=repeats,distances=list(distances),boundaries={})
    for mode in ('terminate','clamp'):
        results=evaluate_frozen(population,[s for s,g,d in tasks],[g for s,g,d in tasks],
                                 seeds=[derive(20261101,'goal-diagnostics',i) for i in range(len(population))],
                                 encoding_seed=encoding,workers=min(8,len(population)),boundary_mode=mode)
        by_goal=defaultdict(list)
        for result in results:
            cells=defaultdict(list)
            for (s,g,d),episode in zip(tasks,result['episodes']):
                cells[g,d].append(episode['success'])
            for cell,successes in cells.items():
                by_goal[cell].append(sum(successes)/len(successes))
        rows=[]
        for (goal,distance),rates in sorted(by_goal.items()):
            starts=[s for s in (goal-distance,goal+distance) if -20<=s<=20]
            rows.append(dict(goal=goal,distance=distance,animal=statistics.mean(rates),
                             random=statistics.mean(random_success(s,goal,boundary_mode=mode) for s in starts)))
        bands=[]
        for distance in distances:
            for lo,hi in ((0,5),(6,12),(13,20)):
                selected=[r for r in rows if r['distance']==distance and lo<=abs(r['goal'])<=hi]
                bands.append(dict(distance=distance,abs_goal_band=[lo,hi],
                                  animal=statistics.mean(r['animal'] for r in selected),
                                  random=statistics.mean(r['random'] for r in selected)))
        output['boundaries'][mode]=dict(per_goal=rows,bands=bands)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--repeats',type=int,default=100)
    parser.add_argument('--output',type=Path,default=Path('runs/goal-diagnostics.json'))
    args=parser.parse_args()
    result=diagnose(args.runs,repeats=args.repeats)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for mode,data in result['boundaries'].items():
        print(mode,flush=True)
        for r in data['bands']:
            print(f"distance {r['distance']:2d}, |goal| {r['abs_goal_band']}: animal {r['animal']:.1%}; random {r['random']:.1%}",flush=True)
    print(args.output,flush=True)


if __name__=='__main__':
    main()
