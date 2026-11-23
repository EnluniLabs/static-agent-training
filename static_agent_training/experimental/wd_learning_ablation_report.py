"""Summarize the completed Agent W/D learning ablation and render its plots."""
import argparse
import csv
from itertools import product
from pathlib import Path
import random
from statistics import mean, median, stdev

from ..experiment import write_json
from .agent_z_experiment import read_z_artifact
from .wd_learning_ablation import ENVIRONMENTS, MILESTONES, MODELS, RATES

LABELS = dict(control='Frozen control', action='Action only',
              structural='Structural only', both='Both updates')
WORLDS = {'original':'Number line', 'lights-out-2x2':'2×2 Lights Out'}


def scalar(row):
    a, b = (row['evaluation'][e] for e in ENVIRONMENTS)
    weight = row['weights']
    return dict(
        number_line=a['success'], puzzle=b['success'], puzzle_frequency=b['frequency_matched'],
        puzzle_gap=b['actual_minus_frequency'],
        number_line_compatibility=a['mean_compatibility'], puzzle_compatibility=b['mean_compatibility'],
        number_line_first=a['successor_unique_first_fraction'], puzzle_first=b['successor_unique_first_fraction'],
        number_line_tied_first=a['successor_tied_first_fraction'], puzzle_tied_first=b['successor_tied_first_fraction'],
        number_line_actions=a['actions_per_episode'], puzzle_actions=b['actions_per_episode'],
        number_line_sampled_entropy=a['sampled_action_entropy'], puzzle_sampled_entropy=b['sampled_action_entropy'],
        number_line_early_termination=a['early_chain_or_round_termination'],
        puzzle_early_termination=b['early_chain_or_round_termination'],
        action_kl=weight['action']['kl_original_to_current'],
        embedding_kl=weight['embedding']['kl_original_to_current'],
        action_entropy=weight['action']['entropy'], embedding_entropy=weight['embedding']['entropy'],
        action_logit_rms=weight['action']['rms'], embedding_rms=weight['embedding']['rms'],
        action_fraction_ever_updated=weight['action']['fraction_ever_updated'],
        embedding_fraction_ever_updated=weight['embedding']['fraction_ever_updated'])


def stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=mean(values), median=median(values),
                sd=stdev(values) if len(values)>1 else 0.) if values else None


def paired(values, indices):
    observed=stats(values)
    if observed is None:
        return None
    import numpy as np
    if all(v == values[0] for v in values) and values[0] is not None:
        observed['ci95']=[values[0],values[0]]
        return observed
    sampled=np.asarray(values,dtype=float)[indices]
    valid=np.isfinite(sampled)
    counts=valid.sum(axis=1)
    estimates=np.nansum(sampled,axis=1)[counts>0] / counts[counts>0]
    estimates.sort()
    observed['ci95']=[float(estimates[int((len(estimates)-1)*p)]) for p in (.025,.975)]
    return observed


def difference(after,before):
    return after-before if after is not None and before is not None else None


def extract(out):
    config=read_z_artifact(out/'config.json.gz')
    if not (out/'validation.json').exists():
        raise ValueError('finish all trajectories before generating the report')
    rows={}
    all_rows=[]
    for m,env,mode,step in product(MODELS,ENVIRONMENTS,RATES,MILESTONES):
        candidates=[]
        for identifier in config['candidate_ids'][m]:
            path=out/'checkpoints'/m/env/mode/identifier/f'{step:05}.json.gz'
            data=read_z_artifact(path)
            candidates.append(scalar(data))
            adaptation=data['adaptation'] or {}
            initial=candidates[-1] if not step else rows[m,env,mode,0][len(candidates)-1]
            candidate_changes={f'delta_{key}':difference(value,initial[key])
                               for key,value in candidates[-1].items()}
            all_rows.append(dict(model=m,adaptation_environment=env,condition=mode,
                candidate=identifier,adaptation_episodes=step,**candidates[-1],**candidate_changes,
                interval_success=adaptation.get('success'),
                interval_compatibility=adaptation.get('mean_compatibility'),
                interval_positive_action_updates=adaptation.get('action_update_sign_fraction',{}).get('positive'),
                interval_negative_action_updates=adaptation.get('action_update_sign_fraction',{}).get('negative')))
        rows[m,env,mode,step]=candidates
    with (out/'candidates.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    randomizer=random.Random('candidate-paired-bootstrap:wd:7')
    import numpy as np
    draws=np.asarray([randomizer.choices(range(19),k=19) for _ in range(4000)])
    metric_keys=tuple(next(iter(rows.values()))[0])
    conditions, contrasts=[],[]
    for m,env,mode,step in product(MODELS,ENVIRONMENTS,RATES,MILESTONES):
        latest=rows[m,env,mode,step]
        initial=rows[m,env,mode,0]
        control=rows[m,env,'control',step]
        conditions.append(dict(model=m,environment=env,condition=mode,episodes=step,
            metrics={key:dict(level=stats([x[key] for x in latest]),
                        change=paired([difference(x[key],y[key]) for x,y in zip(latest,initial)],draws),
                        versus_control=paired([difference(x[key],y[key]) for x,y in zip(latest,control)],draws))
                     for key in metric_keys}))
    for m,env,step in product(MODELS,ENVIRONMENTS,MILESTONES):
        cohorts=[rows[m,env,mode,step] for mode in ('both','action','structural','control')]
        contrasts.append(dict(model=m,environment=env,episodes=step, metrics={
            key:paired([a[key]-b[key]-c[key]+d[key] for a,b,c,d in zip(*cohorts)],draws)
            for key in ('number_line','puzzle','puzzle_frequency','puzzle_gap')}))
    write_json(out/'summary.json',dict(bootstrap_replicates=len(draws),conditions=conditions,interactions=contrasts))
    return ({(c['model'],c['environment'],c['condition'],c['episodes']):c for c in conditions},
            {(c['model'],c['environment'],c['episodes']):c for c in contrasts})


def figures(out, summary, contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors=dict(control='#5a6574',action='#006aa4',structural='#b04d00',both='#6b49a2')
    destination=out/'plots'
    destination.mkdir(exist_ok=True)
    def plot(file,title,label,metric, *, scale=1, mode='level', extra=None):
        fig, axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
        for axis,(m,env) in zip(axes.flat,product(MODELS,ENVIRONMENTS)):
            for k,c in colors.items():
                series=[summary[m,env,k,i]['metrics'][metric][mode] for i in MILESTONES]
                axis.plot(MILESTONES,[v['mean']*scale if v else float('nan') for v in series],
                          label=LABELS[k],color=c,marker='o',markersize=3)
                if mode=='change':
                    axis.fill_between(MILESTONES,*[[v['ci95'][side]*scale for v in series]
                                     for side in (0,1)],color=c,alpha=.09)
                if extra:
                    series=[summary[m,env,k,i]['metrics'][extra]['level'] for i in MILESTONES]
                    axis.plot(MILESTONES,[v['mean']*scale for v in series],color=c,linestyle='--')
            if mode=='change':
                axis.axhline(0,color='#69717b',linewidth=.8)
            axis.set_title(f'Agent {m.upper()} · adaptation on {WORLDS[env]}')
            axis.set_xscale('symlog',linthresh=100)
            axis.set_xticks((0,100,500,1000,3000,10000,30000),('0','100','500','1k','3k','10k','30k'))
            axis.grid(axis='y',alpha=.25)
        fig.supxlabel('Adaptation episodes')
        fig.supylabel(label)
        fig.suptitle(title,fontweight='bold')
        axes.flat[0].legend(fontsize=8,ncol=2,loc='best',frameon=False)
        fig.savefig(destination/file,dpi=170)
        plt.close(fig)
    plot('01-number-line-retention.png','Original number-line retention','Change from baseline (pp)',
         'number_line',scale=100,mode='change')
    plot('02-lights-out-success.png','Frozen 2×2 Lights Out success','Success (%)','puzzle',scale=100)
    plot('03-action-kl.png','Action distribution drift','Mean KL (nats)','action_kl')
    plot('04-embedding-kl.png','Sensory embedding drift on fixed probes','Mean KL (nats)','embedding_kl')
    plot('05a-number-line-compatibility.png','Successor compatibility on number-line probes',
         'Mean compatibility (%)','number_line_compatibility',scale=100)
    plot('05b-puzzle-compatibility.png','Successor compatibility on Lights Out probes',
         'Mean compatibility (%)','puzzle_compatibility',scale=100)
    plot('06-action-entropy.png','Action distribution entropy across states','Mean entropy (nats)','action_entropy')
    plot('07-puzzle-frequency-control.png','Puzzle success (solid) and frequency-matched control (dashed)',
         'Success (%)','puzzle',scale=100,extra='puzzle_frequency')

    fig, axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
    for axis,(m,env) in zip(axes.flat,product(MODELS,ENVIRONMENTS)):
        for metric,color,label in [('number_line','#006aa4','Number-line evaluation'),
                                   ('puzzle','#b04d00','Lights Out evaluation')]:
            rows=[contrasts[m,env,n]['metrics'][metric] for n in MILESTONES]
            axis.plot(MILESTONES,[100*r['mean'] for r in rows],color=color,label=label,marker='o',markersize=3)
            axis.fill_between(MILESTONES,*[[100*r['ci95'][i] for r in rows]
                                         for i in (0,1)],color=color,alpha=.12)
        axis.axhline(0,color='#69717b',linewidth=.8)
        axis.set_title(f'Agent {m.upper()} · adaptation on {WORLDS[env]}')
        axis.set_xscale('symlog',linthresh=100)
        axis.set_xticks((0,100,500,1000,3000,10000,30000),('0','100','500','1k','3k','10k','30k'))
        axis.grid(axis='y',alpha=.25)
    axes.flat[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Action × structural learning interaction',fontweight='bold')
    fig.supxlabel('Adaptation episodes')
    fig.supylabel('Both − action − structural + control (pp)')
    fig.savefig(destination/'08-interaction.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    figures(args.output,*extract(args.output))
