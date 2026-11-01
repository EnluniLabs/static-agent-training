"""Matched multi-seed speciation/control experiments with frozen final benchmarks."""
import argparse
import fcntl
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .benchmark import compare
from .evolution import TrainingConfig
from .experiment import write_json, load_checkpoint
from .run_records import read_summary
from .seeds import derive


ROOT=Path(__file__).resolve().parent.parent
# Two-sided 95% t critical values; uncertainty is over seed-paired differences.
T95={1:12.7062,2:4.3027,3:3.1824,4:2.7764,5:2.5706,6:2.4469,7:2.3646,8:2.3060,
     9:2.2622,10:2.2281,11:2.2010,12:2.1788,13:2.1604,14:2.1448,15:2.1314,
     16:2.1199,17:2.1098,18:2.1009,19:2.0930,20:2.0860,21:2.0796,22:2.0739,
     23:2.0687,24:2.0639,25:2.0595,26:2.0555,27:2.0518,28:2.0484,29:2.0452,30:2.0423}


def paired_summary(differences):
    if not differences:
        return None
    result=dict(seeds=len(differences),mean_percentage_points=100*statistics.mean(differences),interval95=None)
    if len(differences)>1:
        df=len(differences)-1
        critical=T95.get(df,2.0423)  # Conservative t(30) for larger seed counts.
        half=critical*statistics.stdev(differences)/math.sqrt(len(differences))
        mean=statistics.mean(differences)
        result['interval95']=[100*(mean-half),100*(mean+half)]
    return result


def _history(arm):
    return [read_summary(p) for p in sorted((Path(arm['directory'])/'generations').glob('*.json'))]


def report_for(manifest, status):
    rows=[]
    histories={}
    for arm in manifest['arms']:
        history=_history(arm)
        histories[(arm['seed'],arm['speciation'])]=history
        benchmark_path=Path(arm['directory'])/'demo'/'benchmark.json'
        benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
        rows.append(dict(seed=arm['seed'],speciation=arm['speciation'],run_name=arm['run_name'],
                         completed=len(history),target=manifest['config']['generations'],
                         latest=history[-1] if history else None,
                         final_benchmark=None if not benchmark else {k:benchmark[k] for k in ('episodes','animal','random','exact_random_success','generation')},
                         state=arm.get('state','queued'),returncode=arm.get('returncode')))
    pairs=[]
    deltas=[]
    for seed in manifest['seeds']:
        off=histories[(seed,False)]
        on=histories[(seed,True)]
        common=min(len(off),len(on))
        if common:
            if off[0]['population_id']!=on[0]['population_id']:
                raise ValueError(f'Unmatched initial populations for seed {seed}')
            for a,b in zip(off[:common],on[:common]):
                if a['task_id']!=b['task_id']:
                    raise ValueError(f'Unmatched tasks for seed {seed}, generation {a["generation"]+1}')
        paired=dict(seed=seed,common_generations=common,initialization_verified=bool(common),tasks_verified=bool(common))
        if common:
            window=min(5,common)
            paired['matched_window_population_heldout']={
                'control':statistics.mean(r['assessment_mean'] for r in off[common-window:common]),
                'speciation':statistics.mean(r['assessment_mean'] for r in on[common-window:common])}
        selected=[r for r in rows if r['seed']==seed]
        if all(r['final_benchmark'] for r in selected):
            rates={r['speciation']:r['final_benchmark']['animal']['success_rate'] for r in selected}
            delta=rates[True]-rates[False]
            paired.update(control=rates[False],speciation=rates[True],difference_percentage_points=100*delta)
            deltas.append(delta)
        pairs.append(paired)
    return dict(schema_version='crt-speciation-comparison-v1',name=manifest['name'],status=status,
                updated=time.time(),population=manifest['config']['population'],generations=manifest['config']['generations'],
                episodes=manifest['config']['episodes'],assessment_episodes=manifest['config']['assessment_episodes'],
                benchmark_episodes=manifest['benchmark_episodes'],seeds=manifest['seeds'],
                primary_endpoint='success of the highest-training saved champion on fresh shared held-out tasks, with learned weights frozen',
                boundary_mode=manifest['config'].get('boundary_mode','terminate'),
                distance_distribution=manifest['config']['distance_distribution'],
                distance_mean=manifest['config']['distance_mean'],distance_stddev=manifest['config']['distance_stddev'],
                speciation_threshold=manifest['config']['speciation_threshold'],max_species=manifest['config']['max_species'],
                arms=rows,pairs=pairs,paired_effect=paired_summary(deltas))


def _run_study(directory, config=None, *, seeds=(42,43,44), benchmark_episodes=2000, resume=False, runs_root=None):
    directory=Path(directory).resolve()
    manifest_path=directory/'manifest.json'
    runs_root=Path(runs_root or ROOT/'runs').resolve()
    if resume:
        manifest=json.loads(manifest_path.read_text())
        config=TrainingConfig.from_data(manifest['config'])
    else:
        if not config or len(set(seeds))!=len(seeds) or not seeds or benchmark_episodes<1:
            raise ValueError('require a config, unique seeds, and positive benchmark episode count')
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('comparison output already exists; use --resume')
        directory.mkdir(parents=True,exist_ok=True)
        manifest=dict(schema_version='crt-speciation-comparison-v1',name=directory.name,created=time.time(),
                      config=asdict(replace(config,speciation=False)),seeds=list(seeds),benchmark_episodes=benchmark_episodes,arms=[])
        for seed in seeds:
            for enabled in (False,True):
                name=f'{directory.name}-s{seed}-'+('speciation' if enabled else 'control')
                run=runs_root/name
                if run.exists():
                    raise ValueError(f'run directory already exists: {run}')
                manifest['arms'].append(dict(seed=seed,speciation=enabled,run_name=name,
                                             directory=str(run),log=str(directory/(name+'.log')),state='queued'))
        write_json(manifest_path,manifest)
    active=[]
    status='running'
    def save():
        write_json(manifest_path,manifest)
        write_json(directory/'report.json',report_for(manifest,status))
    try:
        save()
        for seed in manifest['seeds']:
            arms=[a for a in manifest['arms'] if a['seed']==seed]
            active=[]
            for arm in arms:
                run=Path(arm['directory'])
                history=_history(arm)
                if len(history)>=config.generations and (run/'checkpoint.json.gz').exists() and load_checkpoint(run)['next_generation']>=config.generations:
                    arm['state']='evaluating final benchmark'
                    continue
                if (run/'checkpoint.json.gz').exists():
                    args=['--resume',str(run)]
                else:
                    arm_config=replace(config,seed=seed,speciation=arm['speciation'])
                    path=directory/(arm['run_name']+'-config.json')
                    write_json(path,asdict(arm_config))
                    args=['--config',str(path),'--output',str(run)]
                with open(arm['log'],'ab') as log:
                    process=subprocess.Popen([sys.executable,'-u','-m','static_agent_training.experiment',*args,'--no-live'],
                                             cwd=ROOT,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
                arm.update(state='running',pid=process.pid)
                active.append((arm,process))
            while any(p.poll() is None for _,p in active):
                for arm,process in active:
                    code=process.poll()
                    if code is not None:
                        arm.update(returncode=code,state='trained' if code==0 else 'failed')
                save()
                time.sleep(2)
            for arm,process in active:
                code=process.wait()
                arm.update(returncode=code,state='trained' if code==0 else 'failed')
                if code:
                    raise RuntimeError(f"{arm['run_name']} failed (exit {code}); see {arm['log']}")
            active=[]
            for arm in arms:
                arm['state']='evaluating final benchmark'
                save()
                result=compare(arm['directory'],episodes=manifest['benchmark_episodes'],seed=derive(seed,'speciation-final-benchmark'))
                arm['state']='completed'
                print(f"seed {seed} | {'speciation' if arm['speciation'] else 'control'} | fresh held-out {result['animal']['success_rate']:.2%}",flush=True)
                save()
        status='completed'
        save()
        return json.loads((directory/'report.json').read_text())
    except KeyboardInterrupt:
        status='stopped'
        for arm,process in active:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for arm,process in active:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            arm['state']='stopped'
        save()
        raise
    except Exception as error:
        status='failed'
        for arm,process in active:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=30)
            arm['state']='failed'
        try:
            save()
        except Exception:
            # Never leave a stale 'running' result after input verification fails.
            report_path=directory/'report.json'
            if report_path.exists():
                failed=json.loads(report_path.read_text())
                failed.update(status='failed',error=str(error),updated=time.time(),paired_effect=None,pairs=[])
                write_json(report_path,failed)
        raise


def run_study(directory, config=None, **kwargs):
    directory=Path(directory).resolve()
    directory.parent.mkdir(parents=True,exist_ok=True)
    lock_path=directory.parent/('.'+directory.name+'.lock')
    with lock_path.open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('This comparison is already running') from error
        return _run_study(directory,config,**kwargs)


def list_reports(root=ROOT):
    reports=[]
    for path in (Path(root)/'runs'/'comparisons').glob('*/report.json'):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError,ValueError):
            continue
    return sorted(reports,key=lambda r:r['updated'],reverse=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--population',type=int,default=1024)
    parser.add_argument('--generations',type=int,default=30)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--assessment-episodes',type=int,default=200)
    parser.add_argument('--benchmark-episodes',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--backend',choices=['python','native','cuda'],default='native')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--speciation-threshold',type=float,default=.6)
    parser.add_argument('--max-species',type=int,default=16)
    parser.add_argument('--distance-mean',type=float,default=10)
    parser.add_argument('--distance-stddev',type=float,default=5)
    args=parser.parse_args()
    directory=args.resume or args.output or ROOT/'runs'/'comparisons'/datetime.now().strftime('speciation-%Y%m%d-%H%M%S')
    config=TrainingConfig(population=args.population,generations=args.generations,episodes=args.episodes,
                          assessment_episodes=args.assessment_episodes,elites=min(5,args.population-1),
                          backend=args.backend,workers=args.workers,batch_size=args.batch_size,
                          speciation_threshold=args.speciation_threshold,max_species=args.max_species,
                          distance_mean=args.distance_mean,distance_stddev=args.distance_stddev)
    print(f'Comparison: {directory}',flush=True)
    try:
        result=run_study(directory,config,seeds=args.seeds,benchmark_episodes=args.benchmark_episodes,resume=bool(args.resume))
    except KeyboardInterrupt:
        print(f'Stopped. Resume with: ./train compare --resume {directory}',flush=True)
        raise SystemExit(130)
    print(json.dumps(result['paired_effect']),flush=True)


if __name__=='__main__':
    main()
