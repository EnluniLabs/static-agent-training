"""Structural-shielding experiment with physically identical remapped observations."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from math import exp, log
from pathlib import Path
import random
from statistics import mean

from ..experiment import write_json
from ..serialization import dumps
from .agent_d import AgentD, RandomGraph
from .agent_w import AgentW
from .agent_w_accelerated import _kernel
from .agent_w_environment import NumberLine
from .agent_z_experiment import read_z_artifact
from .wd_learning_ablation import (MILESTONES, RATES, apply, changes, digest, embedded_logs,
                                   entropy, inputs, metrics, puzzle_frequency)

ORIGINAL = 'original'
PUZZLE = 'lights-out-2x2'
REMAPPED = 'remapped-number-line'
ENVIRONMENTS = (ORIGINAL, PUZZLE, REMAPPED)
MODELS = {'w': AgentW, 'd': AgentD, 'shuffled-w': AgentD}
CONTROL_ATTEMPTS = 256
CONTROL_EVALUATIONS = 128
MATCH_MARGIN = .03


def remap(byte):
    return ((byte << 2) | (byte >> 6)) & 255


def shape(nodes, edges):
    """Canonical unlabeled shape of a functional digraph, including attached trees."""
    indegree = Counter(edges.values())
    incoming = {node: [] for node in nodes}
    queue = deque(node for node in nodes if not indegree[node])
    sinks = []
    while queue:
        node = queue.popleft()
        tree = '(' + ''.join(sorted(incoming[node])) + ')'
        target = edges.get(node)
        if target is None:
            sinks.append(tree)
        else:
            incoming[target].append(tree)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    residual = {node for node in nodes if indegree[node]}
    cycles = []
    for node in nodes:
        if node not in residual:
            continue
        chain = []
        while node in residual:
            residual.remove(node)
            chain.append('(' + ''.join(sorted(incoming[node])) + ')')
            node = edges[node]
        cycles.append(min(tuple(chain[i:] + chain[:i]) for i in range(len(chain))))
    return tuple(sorted(sinks)), tuple(sorted(cycles))


def evaluate(agent, tasks, env, seed, mode='control', **kw):
    action, structural = RATES[mode]
    return agent.evaluate(tasks, seed=seed, environment=env, action_rate=action,
                          structural_rate=structural, **kw)


def calibration(candidate, tasks):
    return evaluate(candidate, tasks, ORIGINAL, 'shield-calibration:7')['successes']/len(tasks)


def control_for(model, identifier, tasks):
    original = model.structure.successors
    sources = sorted(original)
    targets = [original[x] for x in sources]
    signature = shape(model.ids, original)
    baseline = calibration(model, tasks)
    rng = random.Random(f'shield-rewire:{identifier}')
    choices = []
    seen = {tuple(targets)}
    for attempt in range(CONTROL_ATTEMPTS):
        rng.shuffle(targets)
        order = tuple(targets)
        if order in seen:
            continue
        seen.add(order)
        rewired = dict(zip(sources, order))
        candidate_shape = shape(model.ids, rewired)
        if candidate_shape == signature:
            continue
        candidate = AgentD(RandomGraph(model.ids, rewired), model.embedding, model.continuation,
                           actions=model.actions, regions=model.regions)
        score = calibration(candidate, tasks)
        changed = sum(rewired[node] != original[node] for node in sources)
        choices.append((abs(score-baseline),-changed,attempt,score,candidate))
        if len(choices)>=CONTROL_EVALUATIONS:
            break
    if not choices:
        raise ValueError(f'No non-isomorphic degree-matched graph available for {identifier}')
    _, negative, _, score, best = min(choices)
    assert Counter(original.values()) == Counter(best.structure.successors.values())
    assert set(original) == set(best.structure.successors)
    assert model.embedding == best.embedding and model.continuation == best.continuation
    return dict(individual=best.to_data(), calibration_original=baseline, calibration_control=score,
                nonisomorphic_variants_evaluated=len(choices), changed_edges=-negative,
                states=len(model.ids), edges=len(sources), original_shape=signature,
                shuffled_shape=shape(best.ids,best.structure.successors))


def line_frequency(counts, outcomes):
    import numpy as np
    total = sum(counts.values())
    if not total:
        return 0.
    grid = np.arange(41)
    values = np.eye(41)
    steps = [(np.clip(grid+int(move),0,40),count/total) for move,count in counts.items() if count]
    for _ in range(12):
        values = sum(weight*values[dest] for dest,weight in steps)
        np.fill_diagonal(values,1)
    return sum(values[s+20,g+20]*n for pair,(n,_) in outcomes.items()
               for s,g in [tuple(map(int,pair.split(',')))]) / sum(n for n,_ in outcomes.values())


def frozen(agent, tasks, env, seed):
    result = evaluate(agent,tasks,env,seed,task_diagnostics=True)
    assert result['embedding']==agent.embedding and result['continuation']==agent.continuation
    d=metrics(result)
    f=puzzle_frequency(d['action_counts']) if env==PUZZLE else line_frequency(d['action_counts'],d['task_outcomes'])
    assert sum(n for n,_ in d['task_outcomes'].values())==len(tasks)
    assert sum(n for _,n in d['task_outcomes'].values())==d['successes']
    d.update(frequency_matched=f,actual_minus_frequency=d['success']-f)
    return d


def information(model):
    """Average mutual information of sensory byte and selected state under uniform probes."""
    probes = sorted({s & 255 for s in range(-40,41)} | {remap(s&255) for s in range(-40,41)})
    by_region=[]
    for region in range(model.regions):
        logs=[embedded_logs(model,s,region) for s in probes]
        marginal=[mean(exp(p) for p in column) for column in zip(*logs)]
        h=-sum(p*log(p) for p in marginal if p)
        by_region.append(h-mean(entropy(row) for row in logs))
    return mean(by_region)


def weight_diagnostics(original, current):
    result=changes(original,current)
    for field,key in [('continuation','action'),('embedding','embedding')]:
        old,new=getattr(original,field),getattr(current,field)
        result[key]['fraction_parameters_changed']=mean(x!=y for a,b in zip(old,new) for x,y in zip(a,b))
    result['embedding']['sensory_information']=information(current)
    return result


def verify_remap(agent):
    # Verify physical identity by compensating the features only in this validation
    # copy. Actual experiment candidates always begin with their untouched weights.
    tasks=[(s,g) for s in range(-20,21) for g in range(-20,21)]
    assert len({remap(s&255) for s in range(-40,41)})==81
    assert all(((remap(s&255)>>2)|(remap(s&255)<<6))&255 == s&255 for s in range(-40,41))
    copy=agent.copy()
    for row in copy.embedding:
        prior=row[1:]
        row[1:]=prior[2:]+prior[:2]
    x=evaluate(agent,tasks,ORIGINAL,'verify-remap',trace=True)
    y=evaluate(copy,tasks,REMAPPED,'verify-remap',trace=True)
    assert x['successes']==y['successes'] and x['rounds']==y['rounds'] and x['actions']==y['actions']
    for (e,r,a),(other_e,other_r,b) in zip(x['trace'],y['trace']):
        assert (e,r,a.chain,a.action,a.successor)==(other_e,other_r,b.chain,b.action,b.successor)
        assert b.feedback==bytes([remap(a.feedback[0])])
        if a.compatibility is not None:
            assert abs(a.compatibility-b.compatibility)<1e-12


def execute(output,workers=10):
    _kernel()
    original_records,previous,prior_adapted,protected,probes,adaptation=inputs()
    output.mkdir(parents=True,exist_ok=True)
    probes[REMAPPED]=probes[ORIGINAL]
    adaptation[REMAPPED]=adaptation[ORIGINAL]
    probe_seeds={ORIGINAL:'large-transfer:shifted:7',PUZZLE:'large-transfer:puzzle:7',
                 REMAPPED:'large-transfer:shifted:7'}
    adaptation_seeds={ORIGINAL:'number-line-adaptation:7',PUZZLE:'puzzle-adaptation:7',
                      REMAPPED:'number-line-adaptation:7'}
    calibration_pairs=sorted(set(probes[ORIGINAL]))*5
    random.Random('shield-calibration:tasks:7').shuffle(calibration_pairs)
    calibration_tasks=calibration_pairs
    config=dict(schema_version='crt-wd-structural-shielding-v1',checkpoints=MILESTONES,rates=RATES,
                action_limit=12,round_limit=64,encoding_seed=0,remap='rotate byte left two bits',
                source_hashes=protected,probe_seeds=probe_seeds,adaptation_seeds=adaptation_seeds,
                controls=dict(max_attempts=CONTROL_ATTEMPTS,max_evaluations=CONTROL_EVALUATIONS,
                              tolerance=MATCH_MARGIN,calibration_tasks=calibration_tasks),
                tasks=dict(probes=probes,adaptation=adaptation))
    cfg=output/'config.json.gz'
    if cfg.exists():
        assert dumps(read_z_artifact(cfg),sort_keys=True)==dumps(config,sort_keys=True)
    else:
        write_json(cfg,config)
    verify_remap(AgentW.from_data(original_records['w'][0]['individual']))
    records=dict(original_records)
    control_path=output/'shuffled-controls.json.gz'
    if control_path.exists():
        records['shuffled-w']=read_z_artifact(control_path)['candidates']
    else:
        def select(item):
            ident=item['identity']
            model=AgentW.from_data(item['individual'])
            return dict(identity=ident,**control_for(model,ident,calibration_tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records['shuffled-w']=list(pool.map(select,records['w']))
        write_json(control_path,dict(candidates=records['shuffled-w']))
    starts={}
    for m,items in records.items():
        for item in items:
            ident=item['identity']
            agent=MODELS[m].from_data(item['individual'])
            path=output/'baseline'/m/f'{ident}.json.gz'
            result=read_z_artifact(path) if path.exists() else {
                'evaluation':{env:frozen(agent,probes[env],env,probe_seeds[env]) for env in ENVIRONMENTS},
                'weights':weight_diagnostics(agent,agent)}
            if m in previous:
                assert result['evaluation'][ORIGINAL]['successes']==previous[m][ident]['original_before']['successes']
                assert result['evaluation'][PUZZLE]['successes']==previous[m][ident]['puzzle_before']['successes']
            starts[m,ident]=result
            if not path.exists():
                write_json(path,result)
    match=[(ident, starts['shuffled-w',ident]['evaluation'][ORIGINAL]['success']-
                   starts['w',ident]['evaluation'][ORIGINAL]['success'])
           for ident,_ in [(x['identity'],x) for x in records['w']]]
    write_json(output/'baseline-matching.json',dict(tolerance=MATCH_MARGIN,
               pairs=[dict(identity=i,delta=v,passes=abs(v)<=MATCH_MARGIN) for i,v in match]))
    print('Baselines exact. Remap equivalent. Non-isomorphic controls matched within 3 points:',
          sum(abs(x)<=MATCH_MARGIN for _,x in match),'/19',flush=True)

    for m in MODELS:
        agent=MODELS[m].from_data(records[m][0]['individual'])
        for env,mode in product(ENVIRONMENTS,RATES):
            small=adaptation[env][:100]
            a=evaluate(agent,small,env,adaptation_seeds[env],mode,return_rng=True)
            b=evaluate(agent,small[:37],env,adaptation_seeds[env],mode,return_rng=True)
            middle=apply(agent,b)
            frozen(middle,probes[env][:30],env,probe_seeds[env])
            c=evaluate(middle,small[37:],env,adaptation_seeds[env],mode,rng_state=b['rng_state'],return_rng=True)
            assert all(a[k]==c[k] for k in ('embedding','continuation','rng_state'))
            assert a['successes']==b['successes']+c['successes']
    print('All split-RNG and frozen probe smoke checks passed.',flush=True)

    def trajectory(job):
        m,item,env,mode=job
        ident=item['identity']
        original=model=MODELS[m].from_data(item['individual'])
        anchor=starts[m,ident]
        state=None
        previous_step=0
        cumulative=0
        path=output/'checkpoints'/m/env/mode/ident
        wanted=(ORIGINAL,) if env==ORIGINAL else (ORIGINAL,env)
        pending=None
        for step in MILESTONES:
            destination=path/f'{step:05}.json.gz'
            if destination.exists():
                previous_step=step
                pending=destination
                continue
            if pending is not None:
                last=read_z_artifact(pending)
                state=last['rng_state']
                model=MODELS[m].from_data(last['individual'])
                cumulative=last['adaptation_successes']
                pending=None
            adaptation_stats=None
            if step:
                outcome=evaluate(model,adaptation[env][previous_step:step],env,
                                 adaptation_seeds[env],mode,diagnostics=True,rng_state=state,return_rng=True)
                state=outcome['rng_state']
                model=apply(model,outcome)
                cumulative+=outcome['successes']
                adaptation_stats=metrics(outcome)
                assert model.structure.successors==original.structure.successors
                if not RATES[mode][0]:
                    assert model.continuation==original.continuation
                if not RATES[mode][1]:
                    assert model.embedding==original.embedding
            measured={world:anchor['evaluation'][world] if not step or mode=='control'
                      else frozen(model,probes[world],world,probe_seeds[world]) for world in wanted}
            if m in previous and env==PUZZLE and mode=='both' and step==MILESTONES[-1]:
                assert model.to_data()==MODELS[m].from_data(prior_adapted[m][ident]).to_data()
                assert cumulative==previous[m][ident]['puzzle_during']['successes']
            write_json(destination,dict(model=m,identity=ident,environment=env,condition=mode,
                episodes=step,interval_start=previous_step,adaptation=adaptation_stats,
                adaptation_successes=cumulative,rng_state=state,individual=model.to_data(),
                weights=weight_diagnostics(original,model) if step and mode!='control' else anchor['weights'],
                evaluations=measured))
            previous_step=step
    jobs=[(m,item,env,mode) for m in MODELS for item in records[m] for env,mode in product(ENVIRONMENTS,RATES)]
    # Interleave architectures and environments in a deterministic work list.
    random.Random('shield-dispatch').shuffle(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(trajectory,jobs),1):
            if i%38==0 or i==len(jobs):
                print(f'{i}/{len(jobs)} trajectories complete.',flush=True)
    assert all(digest(path)==value for path,value in protected.items())
    write_json(output/'validation.json',dict(trajectories=len(jobs),checkpoints=len(jobs)*len(MILESTONES),
        baselines_exact=38,combined_puzzle_endpoints_exact=38,remap_and_rng_checks=True,
        original_sources_unchanged=True,shuffled_baseline_matches=sum(abs(x)<=MATCH_MARGIN for _,x in match)))
    print('ALL TRAJECTORIES COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args()
    execute(args.output,args.workers)
