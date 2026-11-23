"""Bounded structural niches, persistent representatives, and fitness sharing.

This is a speciation treatment for lambda repertoires, not a NEAT implementation.
Global elite count and mutation operators remain identical to the control.
"""
from dataclasses import dataclass
import math
import random

from .initialize import pair_id
from .mutation import mutate
from .seeds import derive


@dataclass(frozen=True)
class Structure:
    seeds: frozenset
    pairs: frozenset
    edges: frozenset

    def to_data(self):
        return dict(seeds=sorted(self.seeds), pairs=sorted(self.pairs), edges=sorted(self.edges))

    @classmethod
    def from_data(cls, data):
        return cls(frozenset(data['seeds']), frozenset(data['pairs']),
                   frozenset(tuple(edge) for edge in data['edges']))


def describe(individual):
    return Structure(frozenset(pair_id(*p) for p in individual.initial_pairs),
                     frozenset(individual.repertoire.pairs),
                     frozenset(individual.repertoire.successors.items()))


def _jaccard(a, b):
    return 1-len(a & b)/len(a | b) if a or b else 0.0


def distance(a, b):
    return .25*_jaccard(a.seeds,b.seeds)+.5*_jaccard(a.pairs,b.pairs)+.25*_jaccard(a.edges,b.edges)


def partition(population, *, threshold=.6, limit=16, previous=None):
    """Assign each member once; preserve compatible species IDs across generations."""
    if not math.isfinite(threshold) or not 0<=threshold<=1 or type(limit) is not int or limit<1:
        raise ValueError('require threshold in [0,1] and a positive species limit')
    structures=[describe(p) for p in population]
    previous=previous or dict(next_id=0,representatives=[])
    next_id=previous['next_id']
    groups={r['id']:dict(id=r['id'],structure=Structure.from_data(r['structure']),members=[])
            for r in previous['representatives']}
    assignments=[None]*len(population)
    unassigned=[]
    for index,structure in enumerate(structures):
        choices=sorted((distance(structure,g['structure']),identity) for identity,g in groups.items())
        if choices and choices[0][0]<=threshold:
            identity=choices[0][1]
            groups[identity]['members'].append(index)
            assignments[index]=identity
        else:
            unassigned.append(index)
    extinct=sorted(identity for identity,g in groups.items() if not g['members'])
    groups={identity:g for identity,g in groups.items() if g['members']}
    created=[]
    forced=0
    for index in unassigned:
        structure=structures[index]
        choices=sorted((distance(structure,g['structure']),identity) for identity,g in groups.items())
        if choices and choices[0][0]<=threshold:
            identity=choices[0][1]
        elif len(groups)<limit:
            identity=next_id
            next_id+=1
            created.append(identity)
            groups[identity]=dict(id=identity,structure=structure,members=[])
        else:
            identity=choices[0][1]
            forced+=1
        groups[identity]['members'].append(index)
        assignments[index]=identity
    representatives=[]
    for identity,g in sorted(groups.items()):
        representative=min(g['members'],key=lambda i:(distance(structures[i],g['structure']),i))
        g['representative_index']=representative
        representatives.append(dict(id=identity,structure=structures[representative].to_data()))
    state=dict(schema_version='crt-speciation-v1',next_id=next_id,representatives=representatives)
    report=dict(assignments=assignments,created=created,extinct=extinct,forced_assignments=forced)
    return [g for _,g in sorted(groups.items())],state,report


def allocate(groups, scores, slots):
    """Reserve one mutated offspring per niche; distribute the rest by mean fitness."""
    if slots<len(groups):
        raise ValueError('not enough offspring slots for protected species')
    masses=[sum(scores[i] for i in g['members'])/len(g['members']) for g in groups]
    if not sum(masses):
        masses=[1.0]*len(groups)
    remainder=slots-len(groups)
    ideal=[remainder*m/sum(masses) for m in masses]
    quotas=[1+math.floor(value) for value in ideal]
    left=slots-sum(quotas)
    order=sorted(range(len(groups)),key=lambda i:(-(ideal[i]-math.floor(ideal[i])),groups[i]['id']))
    for index in order[:left]:
        quotas[index]+=1
    return quotas


def reproduce(population, results, config, generation, previous=None, progress=None):
    scores=[r['training']['fitness'] for r in results]
    limit=min(config.max_species,len(population)-config.elites)
    groups,state,report=partition(population,threshold=config.speciation_threshold,limit=limit,previous=previous)
    quotas=allocate(groups,scores,len(population)-config.elites)
    lookup={g['id']:g for g in groups}
    for g,quota in zip(groups,quotas):
        g['offspring']=quota
    order=sorted(range(len(population)),key=lambda i:(-scores[i],i))
    children=[]
    ancestry=[]
    for parent in order[:config.elites]:
        children.append(population[parent])
        ancestry.append(dict(kind='elite',parent_index=parent,genotype_id=population[parent].genotype_id,
                             species_id=report['assignments'][parent]))
    rng=random.Random(derive(config.seed,'selection',generation))
    schedule=[g['id'] for g in groups for _ in range(g['offspring'])]
    if len(groups)>1:
        rng.shuffle(schedule)
    jobs=[]
    for identity in schedule:
        members=lookup[identity]['members']
        competitors=rng.sample(members,min(config.tournament_size,len(members)))
        parent=max(competitors,key=lambda i:scores[i])
        jobs.append(dict(index=len(children)+len(jobs),parent_index=parent,competitors=competitors,species_id=identity))
    from .offspring import construct_offspring
    for job,(child,log) in zip(jobs,construct_offspring(population,jobs,config,generation)):
        children.append(child)
        ancestry.append(dict(kind='offspring',parent_index=job['parent_index'],competitors=job['competitors'],mutation=log,species_id=job['species_id']))
        if progress:
            progress(dict(phase='reproducing',generation=generation,completed=len(children),total=len(population)))
    report.update(enabled=True,count=len(groups),threshold=config.speciation_threshold,limit=limit,
                  elite_count=config.elites,
                  groups=[dict(id=g['id'],members=g['members'],size=len(g['members']),
                               representative_index=g['representative_index'],offspring=g['offspring'],
                               mean_fitness=sum(scores[i] for i in g['members'])/len(g['members']),
                               best_fitness=max(scores[i] for i in g['members'])) for g in groups])
    return children,ancestry,report,state
