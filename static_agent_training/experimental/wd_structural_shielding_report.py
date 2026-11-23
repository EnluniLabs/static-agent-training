"""Candidate-paired summaries and plots for the W/D structural-shielding study."""
import argparse
import csv
from itertools import product
from pathlib import Path
import random

from ..experiment import write_json
from .agent_z_experiment import read_z_artifact
from .wd_learning_ablation import MILESTONES, RATES
from .wd_learning_ablation_report import paired, stats
from .wd_structural_shielding import ENVIRONMENTS, MODELS, ORIGINAL, PUZZLE, REMAPPED

LABELS = {'w':'W', 'd':'D', 'shuffled-w':'Rewired W'}
WORLDS = {ORIGINAL:'Same number line', PUZZLE:'2×2 Lights Out', REMAPPED:'Remapped number line'}
COLORS = {'control':'#606970','action':'#0064a3','structural':'#b75500','both':'#7046a1'}
CONDITIONS = {'control':'Frozen','action':'Action only','structural':'Structural only','both':'Both updates'}


def scalar(row, base):
    line=row['evaluations'][ORIGINAL]
    novel=row['evaluations'][row['environment']]
    old_line=base['evaluation'][ORIGINAL]['success']
    old_novel=base['evaluation'][row['environment']]['success']
    weight=row['weights']
    return dict(
        original=line['success'], new=novel['success'],
        original_change=line['success']-old_line,new_change=novel['success']-old_novel,
        original_relative=line['success']/old_line if old_line else None,
        new_frequency=novel['frequency_matched'], new_gap=novel['actual_minus_frequency'],
        new_frequency_change=novel['frequency_matched']-base['evaluation'][row['environment']]['frequency_matched'],
        original_gap=line['actual_minus_frequency'],
        original_actions=line['actions_per_episode'],new_actions=novel['actions_per_episode'],
        original_compatibility=line['mean_compatibility'],new_compatibility=novel['mean_compatibility'],
        original_first=line['successor_unique_first_fraction'],new_first=novel['successor_unique_first_fraction'],
        embedding_entropy=weight['embedding']['entropy'],
        sensory_information=weight['embedding']['sensory_information'],
        sensory_information_change=weight['embedding']['sensory_information']-base['weights']['embedding']['sensory_information'],
        embedding_kl=weight['embedding']['kl_original_to_current'],
        embedding_param_fraction=weight['embedding']['fraction_parameters_changed'],
        action_entropy=weight['action']['entropy'], action_kl=weight['action']['kl_original_to_current'],
        action_param_fraction=weight['action']['fraction_parameters_changed'])


def summarize(output):
    import numpy as np
    assert (output/'validation.json').is_file()
    ids={m:[p.name.removesuffix('.json.gz') for p in sorted((output/'baseline'/m).glob('*.json.gz'))]
         for m in MODELS}
    baselines={(m,ident): read_z_artifact(output/'baseline'/m/f'{ident}.json.gz')
               for m in MODELS for ident in ids[m]}
    matching=read_z_artifact(output/'baseline-matching.json')['pairs']
    matched={r['identity'] for r in matching if r['passes']}
    rows={}
    candidate_rows=[]
    for m,env,mode,n in product(MODELS,ENVIRONMENTS,RATES,MILESTONES):
        scans={}
        for ident in ids[m]:
            path=output/'checkpoints'/m/env/mode/ident/f'{n:05}.json.gz'
            scans[ident]=scalar(read_z_artifact(path),baselines[m,ident])
            candidate_rows.append(dict(model=m,environment=env,condition=mode,episodes=n,
                                       candidate=ident,**scans[ident]))
        rows[m,env,mode,n]=scans
    with (output/'candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    rng=random.Random('shield:paired-bootstrap')
    def indices(n):
        return np.asarray([rng.choices(range(n),k=n) for _ in range(4000)])
    all_draws,matched_draws=indices(19),indices(len(matched))
    summary=[]
    for (m,e,c,n),group in rows.items():
        agg={key:stats([r[key] for r in group.values()]) for key in next(iter(group.values()))}
        for key in ('original_change','new_change','original_relative','new_gap','new_frequency_change',
                    'sensory_information_change'):
            agg[key]=paired([r[key] for r in group.values()],all_draws)
        summary.append(dict(model=m,environment=e,condition=c,episodes=n,metrics=agg))
    selectivity=[]
    cross=[]
    for c,n in product(RATES,MILESTONES):
        contrast={}
        for m in MODELS:
            contrast[m]={}
            for reference in (ORIGINAL,REMAPPED):
                a,b=rows[m,PUZZLE,c,n],rows[m,reference,c,n]
                gains=rows[m,REMAPPED,c,n]
                values={i:a[i]['original_change']-b[i]['original_change'] for i in ids[m]}
                contrast[m][reference]=values
                selectivity.append(dict(model=m,condition=c,episodes=n,reference=reference,
                    selective_retention=paired(list(values.values()),all_draws),
                    positive_and_compatible_gain=sum(v>0 and gains[i]['new_change']>0 for i,v in values.items())))
        for reference in (ORIGINAL,REMAPPED):
            left,right=contrast['w'][reference],contrast['shuffled-w'][reference]
            comparisons=[i for i in ids['w'] if i in matched]
            pa,pb=rows['w',PUZZLE,c,n],rows['shuffled-w',PUZZLE,c,n]
            ca,cb=rows['w',REMAPPED,c,n],rows['shuffled-w',REMAPPED,c,n]
            cross.append(dict(condition=c,episodes=n,reference=reference,matched_n=len(comparisons),
                difference_in_selectivity=paired([left[i]-right[i] for i in comparisons],matched_draws),
                difference_in_puzzle_retention=paired([pa[i]['original_change']-pb[i]['original_change']
                                                       for i in comparisons],matched_draws),
                difference_in_compatible_gain=paired([ca[i]['new_change']-cb[i]['new_change']
                                                       for i in comparisons],matched_draws)))
    base={m: {env:stats([baselines[m,i]['evaluation'][env]['success'] for i in ids[m]]) for env in ENVIRONMENTS}
          for m in MODELS}
    base_differences=[r['delta'] for r in matching]
    write_json(output/'summary.json',dict(conditions=summary,selectivity=selectivity,
        matched_rewiring=cross,baselines=base,rewiring_baseline_difference=paired(base_differences,all_draws)))
    return summary,selectivity,baselines


def plots(output,summary,selectivity,baselines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries={(x['model'],x['environment'],x['condition'],x['episodes']):x['metrics'] for x in summary}
    destination=output/'plots'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def draw(filename,title,metric,ylabel,scale=1,overlay=None,band=False):
        fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,sharey=True,layout='constrained')
        for axis,(m,e) in zip(axes.flat,product(MODELS,ENVIRONMENTS)):
            for c in RATES:
                values=[entries[m,e,c,n][metric] for n in MILESTONES]
                axis.plot(MILESTONES,[scale*x['mean'] for x in values],color=COLORS[c],
                          label=CONDITIONS[c],marker='o',markersize=2.7)
                if band:
                    axis.fill_between(MILESTONES,*[[scale*x['ci95'][side] for x in values]
                                                   for side in (0,1)],color=COLORS[c],alpha=.10)
                if overlay:
                    axis.plot(MILESTONES,[scale*entries[m,e,c,n][overlay]['mean'] for n in MILESTONES],
                              color=COLORS[c],linestyle='--')
            axis.set_title(f'{LABELS[m]} · {WORLDS[e]}')
            axis.set_xscale('symlog',linthresh=100)
            axis.set_xticks((0,100,500,1000,3000,10000,30000),('0','100','500','1k','3k','10k','30k'))
            axis.grid(axis='y',alpha=.2)
        axes.flat[0].legend(frameon=False,ncol=2,fontsize=7)
        fig.suptitle(title,fontweight='bold')
        fig.supxlabel('Adaptation episodes')
        fig.supylabel(ylabel)
        fig.savefig(destination/filename,dpi=160)
        plt.close(fig)
    draw('01-original-retention.png','Original number-line retention','original_change','Change (pp)',100,band=True)
    draw('02-adaptation-environment.png','Frozen success in adaptation environment','new','Success (%)',100)
    draw('03-embedding-kl.png','Sensory mapping drift','embedding_kl','KL (nats)')
    draw('04-sensory-information.png','Sensory information about selected state','sensory_information','Mutual information (nats)')
    draw('05-successor-compatibility.png','Successor compatibility on adaptation-environment probes','new_compatibility','Compatibility (%)',100)
    draw('06-frequency-control.png','Frozen success (solid) and frequency-matched control (dashed)','new','Success (%)',100,overlay='new_frequency')
    draw('07-action-kl.png','Action mapping drift','action_kl','KL (nats)')

    fig,axes=plt.subplots(1,3,figsize=(11,3.7),sharey=True,layout='constrained')
    for axis,m in zip(axes,MODELS):
        for c in RATES:
            series=[x for x in selectivity if x['model']==m and x['condition']==c and x['reference']==REMAPPED]
            axis.plot(MILESTONES,[100*x['selective_retention']['mean'] for x in series],
                      color=COLORS[c],marker='o',markersize=2.5,label=CONDITIONS[c])
            axis.fill_between(MILESTONES,*[[100*x['selective_retention']['ci95'][i] for x in series]
                                            for i in (0,1)],alpha=.10,color=COLORS[c])
        axis.set_title(LABELS[m])
        axis.axhline(0,color='#68727c',linewidth=.8)
        axis.set_xscale('symlog',linthresh=100)
        axis.set_xticks((0,100,1000,10000,30000),('0','100','1k','10k','30k'))
        axis.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Selective retention · incompatible minus compatible adaptation',fontweight='bold')
    fig.supxlabel('Adaptation episodes')
    fig.supylabel('Difference in original retention (pp)')
    fig.savefig(destination/'08-selective-retention.png',dpi=160)
    plt.close(fig)

    fig,axis=plt.subplots(figsize=(6,5),layout='constrained')
    ordered=sorted(ident for family,ident in baselines if family=='w')
    w=[baselines['w',i]['evaluation'][ORIGINAL]['success']*100 for i in ordered]
    s=[baselines['shuffled-w',i]['evaluation'][ORIGINAL]['success']*100 for i in ordered]
    limits=[min(w+s)-1,max(w+s)+1]
    axis.plot(limits,limits,color='#5b636a')
    axis.fill_between(limits,[x-3 for x in limits],[x+3 for x in limits],color='#d6e8ee')
    axis.scatter(w,s,color='#0064a3')
    axis.set(xlabel='Original W frozen success (%)',ylabel='Rewired W frozen success (%)',
             xlim=limits,ylim=limits,title='Rewiring baseline match · ±3 points')
    fig.savefig(destination/'09-baseline-matching.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    plots(args.output,*summarize(args.output))
