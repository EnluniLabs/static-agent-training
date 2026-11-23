"""Run resumable W/D lifetime learning ablations on saved candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
from math import exp, fsum, log, sqrt
from pathlib import Path
from statistics import mean

from ..experiment import write_json
from .agent_d import AgentD
from .agent_w import AgentW
from .agent_z_experiment import read_z_artifact

MODELS = {'w': AgentW, 'd': AgentD}
ENVIRONMENTS = ('original', 'lights-out-2x2')
RATES = {'control': (0, 0), 'action': (.05, 0), 'structural': (0, .05), 'both': (.05, .05)}
MILESTONES = (0, 100, 500, 1000, 3000, 5000, 10000, 20000, 30000)
PROBES = tuple(range(-40, 41))
PREVIOUS = Path('runs/agent-w-d-puzzle-adaptation-retention-20261123-005527')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def log_probs(logits):
    peak = max(logits)
    centered = [x - peak for x in logits]
    normalizer = log(fsum(exp(x) for x in centered))
    return [x - normalizer for x in centered]


def divergence(original, latest):
    return max(0., fsum(exp(p) * (p - q) for p, q in zip(original, latest)))


def entropy(logs):
    return -fsum(exp(p) * p for p in logs)


def embedded_logs(agent, signal, region):
    return log_probs([row[0] + sum(row[i+1] for i in range(8)
                         if i % agent.regions == region and (signal & (128 >> i)))
                      for row in agent.embedding])


def changes(original, current):
    def amount(old, new):
        differences = [b - a for row_a, row_b in zip(old, new) for a, b in zip(row_a, row_b)]
        return dict(rms=sqrt(mean(x*x for x in differences)), max_absolute=max(map(abs, differences)),
                    fraction_rows_different=mean(a != b for a, b in zip(old, new)))
    def distribution_summary(before, after):
        return dict(kl_original_to_current=mean(divergence(a, b) for a, b in zip(before, after)),
                    entropy=mean(entropy(b) for b in after))
    action = amount(original.continuation, current.continuation)
    action.update(distribution_summary([log_probs(row) for row in original.continuation],
                                       [log_probs(row) for row in current.continuation]))
    embedding = amount(original.embedding, current.embedding)
    probes = [(s & 255, region) for s, region in product(PROBES, range(current.regions))]
    embedding.update(distribution_summary([embedded_logs(original, *p) for p in probes],
                                          [embedded_logs(current, *p) for p in probes]))
    return dict(action=action, embedding=embedding)


def puzzle_frequency(counts, budget=12):
    total = sum(counts.values())
    if not total:
        return 0.
    states = [1.] + [0.] * 15
    masks = [(int(mask), count / total) for mask, count in counts.items() if count]
    for _ in range(budget):
        states = [1.] + [fsum(weight * states[board ^ mask] for mask, weight in masks)
                         for board in range(1, 16)]
    return mean(states[1:])


def metrics(result):
    d = result['diagnostics']
    feedbacks = sum(d['compatibility_histogram'])
    actions, episodes = result['actions'], result['episodes']
    ratio = lambda x, n: x / n if n else None
    d.update(episodes=episodes, successes=result['successes'], actions=actions, rounds=result['rounds'],
             success=result['successes'] / episodes, actions_per_episode=actions / episodes,
             early_chain_or_round_termination=(d['termination']['chains_ended'] +
                                                d['termination']['round_limit']) / episodes,
             physical_action_fractions={a: ratio(n, actions) for a, n in d['action_counts'].items()},
             successor_feedbacks=feedbacks, mean_compatibility=ratio(d['compatibility_sum'], feedbacks),
             successor_unique_first_fraction=ratio(d['unique_first'], feedbacks),
             successor_tied_first_fraction=ratio(d['tied_first'], feedbacks),
             sampled_action_entropy=ratio(d['action_entropy_sum'], actions),
             action_update_sign_fraction={k:ratio(n,sum(d['action_update_sign'].values()))
                                          for k,n in d['action_update_sign'].items()})
    assert sum(d['action_counts'].values()) == actions
    assert sum(d['termination'].values()) == episodes
    assert sum(x[1] for x in d['start_outcomes'].values()) == result['successes']
    return d


def apply(model, result):
    return type(model).from_data(model.to_data() | {
        'embedding': result['embedding'], 'continuation': result['continuation']})


def evaluate(model, tasks, env, seed, rates, **kwargs):
    return model.evaluate(tasks, environment=env, seed=seed, action_rate=rates[0],
                          structural_rate=rates[1], diagnostics=True, **kwargs)


def frozen(model, probes, seeds):
    saved = model.to_data()
    result = {}
    for env in ENVIRONMENTS:
        r = evaluate(model, probes[env], env, seeds[env], RATES['control'])
        assert r['embedding'] == model.embedding and r['continuation'] == model.continuation
        result[env] = metrics(r)
    assert model.to_data() == saved
    p = result['lights-out-2x2']
    p['frequency_matched'] = puzzle_frequency(p['action_counts'])
    p['actual_minus_frequency'] = p['success'] - p['frequency_matched']
    return result


def inputs():
    previous = read_z_artifact(PREVIOUS/'report.json.gz')
    selection = read_z_artifact(previous['candidate_source'])
    enlarged = read_z_artifact(previous['task_source'])
    puzzle = enlarged['tasks']['puzzle']
    line = [(s - 50, g - 50) for s, g in enlarged['tasks']['shifted']]
    assert len(line)==20000 and len(puzzle)==30000
    # Equal empirical task distribution as original frozen probes, with 75
    # repeats of each held-out pair instead of 50, and an independent ordering.
    import random
    line_adapt = (sorted(set(line)) * 75)
    random.Random('ablation:line:tasks:7').shuffle(line_adapt)
    assert len(line_adapt)==30000
    candidates = {m:selection['models'][m]['candidates'] for m in MODELS}
    assert all(len(c)==19 for c in candidates.values())
    prior_by_id = {m:{x['identity']:x for x in previous['models'][m]['results']} for m in MODELS}
    saved_adaptations = {m:{x['source_identity']:x['individual'] for x in read_z_artifact(
        previous['models'][m]['candidate_list'])['members']} for m in MODELS}
    paths = [PREVIOUS/'report.json.gz', previous['candidate_source'], previous['task_source'],
             *(previous['models'][m]['candidate_list'] for m in MODELS)]
    return (candidates, prior_by_id, saved_adaptations, {str(p):digest(p) for p in paths},
            dict(zip(ENVIRONMENTS,(line,puzzle))), dict(zip(ENVIRONMENTS,(line_adapt,puzzle))))


def execute(out, workers):
    candidates, previous, saved, protected, probes, adaptation = inputs()
    probe_seeds = dict(zip(ENVIRONMENTS,('large-transfer:shifted:7','large-transfer:puzzle:7')))
    adaptation_seeds = dict(zip(ENVIRONMENTS,('number-line-adaptation:7','puzzle-adaptation:7')))
    out.mkdir(parents=True, exist_ok=True)
    config = dict(schema_version='crt-wd-learning-ablation-v1', source_hashes=protected,
                  probe_seeds=probe_seeds, adaptation_seeds=adaptation_seeds, rates=RATES,
                  checkpoints=MILESTONES, encoding_seed=0, action_limit=12, round_limit=64,
                  sensory_probe_signed_bytes=PROBES, tasks=dict(probes=probes,adaptation=adaptation),
                  candidate_ids={m:[x['identity'] for x in c] for m,c in candidates.items()})
    config_path = out/'config.json.gz'
    if config_path.exists():
        from ..serialization import dumps
        if dumps(read_z_artifact(config_path),sort_keys=True)!=dumps(config,sort_keys=True):
            raise ValueError('existing experiment has different configuration')
    else:
        write_json(config_path,config)

    baselines = {}
    for m, records in candidates.items():
        for record in records:
            identity = record['identity']
            original = MODELS[m].from_data(record['individual'])
            path = out/'baseline'/m/f'{identity}.json.gz'
            initial = read_z_artifact(path) if path.exists() else frozen(original,probes,probe_seeds)
            for env, key in zip(ENVIRONMENTS,('original_before','puzzle_before')):
                assert initial[env]['successes']==previous[m][identity][key]['successes']
            if not path.exists():
                write_json(path,initial)
            baselines[m,identity]=initial
    print('Reproduced frozen baselines for all 38 original candidates.',flush=True)

    # Validate split-call RNG and both independent switches before the full matrix.
    for m, records in candidates.items():
        rec=records[0]
        initial=MODELS[m].from_data(rec['individual'])
        for env,mode in product(ENVIRONMENTS,RATES):
            fresh=MODELS[m].from_data(rec['individual'])
            assert fresh.to_data()==initial.to_data()
            assert frozen(fresh,probes,probe_seeds)==baselines[m,rec['identity']]
            tasks=adaptation[env][:100]
            complete=evaluate(fresh,tasks,env,adaptation_seeds[env],RATES[mode],return_rng=True)
            first=evaluate(fresh,tasks[:37],env,adaptation_seeds[env],RATES[mode],return_rng=True)
            intermediate=apply(fresh,first)
            frozen(intermediate,{e:probes[e][:30] for e in ENVIRONMENTS},probe_seeds)
            split=evaluate(intermediate,tasks[37:],env,adaptation_seeds[env],RATES[mode],
                           rng_state=first['rng_state'],return_rng=True)
            assert all(complete[k]==split[k] for k in ('embedding','continuation','rng_state'))
            assert complete['successes']==first['successes']+split['successes']
            assert fresh.to_data()==initial.to_data()
            if not RATES[mode][0]:
                assert complete['continuation']==initial.continuation
            if not RATES[mode][1]:
                assert complete['embedding']==initial.embedding

    # Match the prior combined endpoint before starting the full matrix.
    for m, records in candidates.items():
        rec=records[0]; identity=rec['identity']
        model=MODELS[m].from_data(rec['individual'])
        prior=evaluate(model,adaptation['lights-out-2x2'],'lights-out-2x2',
                       adaptation_seeds['lights-out-2x2'],RATES['both'])
        assert prior['successes']==previous[m][identity]['puzzle_during']['successes']
        assert all(prior[k]==saved[m][identity][k] for k in ('embedding','continuation'))
    print('Smoke tests passed and both prior combined-learning endpoints reproduced.',flush=True)

    def lifetime(m,record,environment,mode):
        identity=record['identity']
        original=current=MODELS[m].from_data(record['individual'])
        path=out/'checkpoints'/m/environment/mode/identity
        starter=baselines[m,identity]
        row_count=len(original.ids)
        updated={'action':set(),'embedding':set()}
        cumulative=0
        prior_time=0
        state=None
        previous_file=None
        for checkpoint in MILESTONES:
            destination=path/f'{checkpoint:05}.json.gz'
            if destination.exists():
                previous_file=destination
                prior_time=checkpoint
                continue
            if previous_file is not None:
                saved_row=read_z_artifact(previous_file)
                current=type(original).from_data(saved_row['individual'])
                state=saved_row['rng_state']
                cumulative=saved_row['adaptation_successes']
                updated={k:set(v) for k,v in saved_row['ever_updated_indices'].items()}
                previous_file=None
            interval_metrics=None
            if checkpoint:
                evaluation=evaluate(current,adaptation[environment][prior_time:checkpoint],
                                    environment,adaptation_seeds[environment],RATES[mode],
                                    rng_state=state,return_rng=True)
                current=apply(current,evaluation)
                state=evaluation['rng_state']
                cumulative+=evaluation['successes']
                interval_metrics=metrics(evaluation)
                for key in updated:
                    updated[key].update(interval_metrics[f'{key}_updated_states'])
                if not RATES[mode][0]:
                    assert current.continuation==original.continuation
                if not RATES[mode][1]:
                    assert current.embedding==original.embedding
            evaluations=starter if not checkpoint or mode=='control' else frozen(current,probes,probe_seeds)
            if environment=='lights-out-2x2' and mode=='both' and checkpoint==MILESTONES[-1]:
                assert cumulative==previous[m][identity]['puzzle_during']['successes']
                assert current.to_data()==type(current).from_data(saved[m][identity]).to_data()
                assert evaluations['original']['successes']==previous[m][identity]['original_after']['successes']
                assert evaluations['lights-out-2x2']['successes']==previous[m][identity]['puzzle_after']['successes']
            weight=changes(original,current)
            for key in updated:
                weight[key]['fraction_ever_updated']=len(updated[key])/row_count
            write_json(destination,dict(model=m,identity=identity,environment=environment,condition=mode,
                episodes=checkpoint,interval_start=prior_time,adaptation=interval_metrics,
                adaptation_successes=cumulative,rng_state=state,individual=current.to_data(),
                ever_updated_indices={k:sorted(v) for k,v in updated.items()},weights=weight,evaluation=evaluations))
            prior_time=checkpoint
        return (m, environment, mode)

    jobs=[(m,record,env,mode) for m in MODELS for record in candidates[m]
          for env,mode in product(ENVIRONMENTS,RATES)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(lambda job:lifetime(*job),jobs),1):
            if i%19==0 or i==len(jobs):
                print(f'{i}/{len(jobs)} candidate-condition trajectories complete.',flush=True)
    assert all(digest(path)==value for path,value in protected.items())
    write_json(out/'validation.json',dict(baseline_candidates=38,trajectories=len(jobs),
                checkpoints_per_trajectory=len(MILESTONES),prior_puzzle_combined_endpoints=38,
                source_files_unchanged=True,rng_split_smoke_passed=True,zero_rate_and_isolation_passed=True))
    print('ALL TRAJECTORIES COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    execute(args.output,args.workers)
