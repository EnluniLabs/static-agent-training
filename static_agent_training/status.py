"""Inspect saved runs and live training processes without loading populations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .run_records import read_summary


ROOT = Path(__file__).resolve().parent.parent


def _option(args, flag):
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(flag + '='):
            return value[len(flag) + 1:]
    return None


def training_processes():
    processes = {}
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    ticks = os.sysconf('SC_CLK_TCK')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if 'static_agent_training.experiment' not in args:
                continue
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            directory = _option(args, '--resume') or _option(args, '--output')
            if directory is None:
                continue
            path = Path(directory)
            if not path.is_absolute():
                path = (entry / 'cwd').resolve() / path
            processes[str(path.resolve())] = dict(pid=int(entry.name),
                                                 elapsed_seconds=max(0, uptime - int(fields[19])/ticks))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue  # A process may exit while being inspected.
    return processes


def collect_runs(root=ROOT, *, processes=None, now=None):
    root = Path(root)
    processes = training_processes() if processes is None else processes
    now = time.time() if now is None else now
    rows = []
    for config_path in (root / 'runs').glob('*/config.json'):
        try:
            directory = config_path.parent
            config = json.loads(config_path.read_text())['config']
            records = sorted((directory / 'generations').glob('*.json'))
            latest = read_summary(records[-1]) if records else None
            completed = latest['generation'] + 1 if latest else 0
            process = processes.get(str(directory.resolve()))
            state = ('finishing' if completed >= config['generations'] else 'running') if process else (
                'completed' if completed >= config['generations'] else 'stopped')
            remaining = None
            if process and len(records) >= 2 and completed < config['generations']:
                recent = records[-min(11, len(records)):]
                timestamps = [p.stat().st_mtime for p in recent]
                seconds_per_generation = (timestamps[-1] - timestamps[0])/(len(timestamps)-1)
                since_last = max(0, now-timestamps[-1])
                remaining = max(0, seconds_per_generation*(config['generations']-completed)-min(since_last, seconds_per_generation))
            updated = records[-1].stat().st_mtime if records else config_path.stat().st_mtime
            rows.append(dict(name=directory.name, state=state, completed=completed,
                             target=config['generations'], population=config['population'],
                             backend=config.get('backend', 'python'), speciation=config.get('speciation', False),
                             progress=min(1, completed/config['generations']), summary=latest,
                             pid=process['pid'] if process else None,
                             elapsed_seconds=process['elapsed_seconds'] if process else None,
                             remaining_seconds=remaining, updated=updated,
                             checkpoint_age_seconds=max(0, now-updated)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda r: (r['state'] in ('running', 'finishing'), r['updated']), reverse=True)


def duration(seconds):
    if seconds is None:
        return '—'
    minutes = max(0, round(seconds/60))
    return f'{minutes//60}h {minutes%60}m' if minutes >= 60 else f'{minutes}m'


def render(rows):
    if not rows:
        return 'No saved training runs found.'
    lines = []
    for row in rows:
        summary = row['summary']
        line = f"{row['state'].upper():10} {row['completed']:3}/{row['target']:<3} ({row['progress']:.0%})  {row['name']}"
        lines.append(line)
        details = [f"population {row['population']}", row['backend'], 'speciation on' if row['speciation'] else 'speciation off']
        if summary:
            details += [f"mean {summary['mean']:.1%}", f"best {summary['best']:.1%}",
                        f"held-out {summary['assessment_mean']:.1%}"]
            if 'random_assessment' in summary:
                details.append(f"random {summary['random_assessment']:.1%}")
        if row['pid']:
            details += [f"elapsed {duration(row['elapsed_seconds'])}", f"ETA ~{duration(row['remaining_seconds'])}"]
        lines.append('  ' + ' | '.join(details))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='refresh every five seconds; Ctrl-C exits without stopping training')
    parser.add_argument('--json', action='store_true', help='print machine-readable status')
    args = parser.parse_args(argv)
    try:
        while True:
            rows = collect_runs()
            if args.watch and sys.stdout.isatty():
                print('\x1b[H\x1b[2J', end='')
            print(json.dumps(rows) if args.json else render(rows), flush=True)
            if not args.watch:
                break
            print('\nRefreshes every 5 seconds. Ctrl-C exits; training continues.', flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        pass


RUNS_PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Training runs</title>
<style>
:root{color-scheme:dark}body{font:17px system-ui;background:#101725;color:#eef4ff;max-width:1050px;margin:35px auto;padding:20px}h1{font-size:32px}p{color:#b8c5d8;line-height:1.5}a{color:#7fc4ff}button{font:inherit;background:#24344d;border:1px solid #62738e;color:white;padding:10px 15px;border-radius:8px;cursor:pointer}article{background:#182438;border:1px solid #344762;border-radius:14px;padding:22px;margin:18px 0}h2{font-size:20px;overflow-wrap:anywhere;margin:0 0 15px}.badge{font-weight:600;color:#b8c5d8}.running{color:#64db9f}.stopped{color:#ffc280}progress{width:100%;height:24px;accent-color:#65baff}.metrics{display:flex;gap:24px;flex-wrap:wrap;margin:20px 0}.metric strong{font-size:25px;display:block}.metric span{color:#b8c5d8;font-size:14px}.small{font-size:14px;color:#b8c5d8}#connection{min-height:24px}
</style>
<a href="/">← Animal playground</a> · <a href="/comparisons">Speciation benchmarks</a><h1>Training runs</h1>
<p>Live status of background training. Refreshes every 5 seconds. Closing this page leaves training running.</p>
<button id="refresh">Refresh now</button><p id="connection" role="status">Loading runs…</p><main id="runs"></main>
<script>
const $=id=>document.getElementById(id),pct=n=>(100*n).toFixed(1)+'%';
function duration(s){if(s==null)return 'Estimating…';const m=Math.max(0,Math.ceil(s/60));return m>=60?`${Math.floor(m/60)}h ${m%60}m`:`${m} min`}
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function metric(parent,label,value){const e=node('div',undefined,'metric');e.append(node('strong',value),node('span',label));parent.append(e)}
let pending=false;
async function refresh(){if(pending)return;pending=true;try{const response=await fetch('/runs.json');if(!response.ok)throw Error('Server unavailable');const rows=await response.json();const cards=[];for(const r of rows){const card=node('article');card.append(node('h2',r.name),node('div',r.state==='running'?'● Running':r.state==='finishing'?'● Finishing checkpoint':r.state==='completed'?'Completed':'Not running — checkpoint saved','badge '+r.state));card.append(node('p',`${r.completed} / ${r.target} generations complete · ${(100*r.progress).toFixed(0)}% · ${r.population} animals · ${r.backend} backend · speciation ${r.speciation?'on':'off'}`));const progress=node('progress');progress.max=r.target;progress.value=r.completed;progress.setAttribute('aria-label',`${r.completed} of ${r.target} generations complete`);card.append(progress);if(r.summary){const metrics=node('div',undefined,'metrics');metric(metrics,'Mean training',pct(r.summary.mean));metric(metrics,'Best training',pct(r.summary.best));metric(metrics,'Population held-out',pct(r.summary.assessment_mean));if(r.summary.random_assessment!==undefined)metric(metrics,'Random on same held-out tasks',pct(r.summary.random_assessment));card.append(metrics,node('div','Scores from the last completed generation.','small'))}if(r.pid){card.append(node('p',`Elapsed: ${duration(r.elapsed_seconds)} · Estimated remaining: ${duration(r.remaining_seconds)}`))}card.append(node('div',`Last generation saved ${new Date(r.updated*1000).toLocaleString()}`,'small'));cards.push(card)}$('runs').replaceChildren(...cards);$('connection').textContent=`Updated ${new Date().toLocaleTimeString()}`+(rows.length?'':' · No saved runs yet.')}catch(e){$('connection').textContent='Connection lost. Showing the last received status; retrying every 5 seconds.'}finally{pending=false}}
$('refresh').onclick=refresh;refresh();setInterval(refresh,5000);
</script></html>'''


if __name__ == '__main__':
    main()
