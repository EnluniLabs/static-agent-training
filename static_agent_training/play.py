"""Serve model/session management and frozen playback; retain legacy views."""
import argparse
import json
import random
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import webbrowser
from urllib.parse import parse_qs, quote, urlsplit

from .environments import environment_spec
from .agent import Agent, run_episode
from .candidates import CandidateStore, attach_benchmark, load_saved_candidate
from .environment import Output
from .baseline import random_success
from .demo import RecordingWorld
from .genome import Species
from .seeds import derive
from .status import ROOT, RUNS_PAGE, collect_runs
from .comparison import list_reports
from .comparison_view import PAGE as COMPARISON_PAGE
from .workspace import WEB, Workspace


def attempt(bundle, start, goal, seed, policy="animal"):
    if policy not in ('animal', 'random'):
        raise ValueError('Unknown policy')
    if any(type(v) is not int for v in (start, goal, seed)):
        raise ValueError('Start, goal, and seed must be integers')
    spec = environment_spec(bundle.get('environment','original'))
    if not (spec.lower <= start <= spec.upper and spec.lower <= goal <= spec.upper):
        raise ValueError(f'Start and goal must be between {spec.lower} and {spec.upper}')
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be between 0 and 4294967295')
    if bundle.get('model') == 'senses':
        from .experimental.sensory_playback import attempt as sensory_attempt
        return sensory_attempt(bundle, start, goal, seed, policy)
    conditioned = bundle.get('varying_goals', False) or policy == 'random'
    if not conditioned and goal != spec.center:
        raise ValueError("This older animal was trained only toward zero and has no goal input. Use a new varying-goal run for custom goals")
    if start == goal:
        frames = [dict(position=start, actions=0, label='Already at goal', done=True,
                       success=True, reason='target', correct=0, tested=0)]
    elif policy == 'random':
        world = RecordingWorld(encoding_seed=bundle['encoding_seed'], goal_conditioned=conditioned, boundary_mode="clamp", **spec.world_kwargs)
        rng = random.Random(derive(seed, 'interactive', 'random'))
        moves = tuple(world.diagnostic_state()['move_decoding'])
        transition = world.reset(start=start, goal=goal)
        while not transition.done:
            transition = world.emit(Output(rng.choice(moves)), source='random')
        frames = world.frames
    else:
        species = Species.from_data(bundle['individual'])
        mappings = species.inherited.lifetime()
        mappings.embedding = {k: dict(v) for k, v in bundle['learned_embedding'].items()}
        mappings.output = {k: dict(v) for k, v in bundle['learned_output'].items()}
        agent = Agent(species.repertoire, mappings, learning=False,
                      embedding_seed=derive(seed, 'interactive', 'embedding'),
                      output_seed=derive(seed, 'interactive', 'output'))
        world = RecordingWorld(encoding_seed=bundle['encoding_seed'], goal_conditioned=conditioned, boundary_mode="clamp", **spec.world_kwargs)
        run_episode(agent, world, start=start, goal=goal, learn=False)
        frames = world.frames
        for frame in frames:
            if frame['label'].startswith('Predict position '):
                position = int(frame['label'].split()[-1])
                frame['label'] = f'Predict next position {position}'
    return dict(start=start, goal=goal, seed=seed, policy=policy, frames=frames,
                lower=spec.lower, upper=spec.upper, boundary_mode="clamp",
                random_expected_success=random_success(start,goal,boundary_mode="clamp", **spec.world_kwargs))


PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Animal playground</title>
<style>
:root{color-scheme:dark}body{font:18px system-ui;background:#101725;color:#eef4ff;max-width:1050px;margin:35px auto;padding:20px}h1{font-size:32px}p{color:#b8c5d8;line-height:1.5}input,button,select{font:inherit;border:1px solid #62738e;background:#24344d;color:white;border-radius:8px;padding:10px;margin:6px}input[type=number]{width:100px}button{cursor:pointer}button:disabled{opacity:.4}#run{background:#205d9c}svg{background:#182438;width:100%;border-radius:16px;margin:25px 0}#status{font-size:25px;min-height:45px}#error{color:#ffb8b8}#scrub{width:95%}.small{font-size:15px}#detail{min-height:30px}
</style>
<a href="/" style="color:#7fc4ff">Model workspace</a> · <a href="/runs" style="color:#7fc4ff">Training runs →</a> · <a href="/comparisons" style="color:#7fc4ff">Benchmarks →</a>
<p id="background" class="small" aria-live="polite"></p>
<h1>Animal playground</h1><p>Choose a starting position and a goal. Watch the saved animal try to reach it within 12 moves.</p>
<label>Best candidate from run <select id="candidate" disabled style="max-width:100%"></select></label>
<button type="button" id="refresh-candidates">Refresh best</button>
<p class="small">Highest training success among saved generation champions; ties use the earliest generation. Refresh best to load a newer winner while training continues.</p>
<p id="scores" aria-live="polite" style="padding:14px;background:#24344d;border-radius:8px">Loading animal scores…</p>
<p id="comparison" class="small"></p>
<p id="mode" class="small"></p>
<form id="task"><label>Policy <select id="policy"><option value="animal">Selected candidate</option><option value="random">Random movement</option></select></label><label>Start <input id="start" type="number" min="-20" max="20" value="12" required></label>
<label>Goal <input id="goal" type="number" min="-20" max="20" value="0" required></label>
<label>Seed <input id="seed" type="number" min="0" max="4294967295" value="42" required></label>
<button id="run">Run attempt</button><button type="button" id="again">Try again</button></form>
<div id="error" role="alert"></div>
<svg viewBox="0 0 1000 185" role="img" aria-label="Animal moving toward your chosen goal">
<g id="scale"></g><circle id="target" cy="90" r="17" fill="none" stroke="#64db9f" stroke-width="4"/>
<circle id="animal" cy="90" r="10" fill="#65baff"/>
<text id="position" y="45" text-anchor="middle" fill="white" font-size="22"></text></svg>
<div id="status" aria-live="polite">Enter a task to begin.</div><div id="detail"></div>
<button id="play" disabled>Play</button><button id="step" disabled>Step</button><button id="restart" disabled>Restart</button>
<label>Speed <select id="speed"><option value="250">Normal</option><option value="70">Fast</option><option value="700">Slow</option></select></label>
<input type="range" id="scrub" min="0" max="0" value="0" aria-label="Output step">
<p class="small">Blue: animal · Green: your goal. Each attempt starts from the same frozen learned weights. A different seed changes its sampled choices; no training occurs here. Predictions describe what it expects next, and do not move it.</p>
<script>
const $=id=>document.getElementById(id);let result=null,index=0,timer=null,selectedCandidate=null,selectionVersion=0,loadingCandidate=true,attemptBusy=false;
function pause(){clearInterval(timer);timer=null;$('play').textContent='Play'}
function x(n){return 65+(n-result.lower)*870/(result.upper-result.lower)}
function render(){if(!result)return;const f=result.frames[index];$('target').style.visibility='visible';$('animal').style.visibility='visible';$('animal').setAttribute('cx',Math.max(15,Math.min(985,x(f.position))));$('position').setAttribute('x',Math.max(35,Math.min(965,x(f.position))));$('position').textContent=f.position;$('status').textContent=f.done?(f.success?'Solved — goal reached!':`Stopped — ${f.reason.replaceAll('_',' ')}`):f.label;$('detail').textContent=`${result.policy==='random'?'Random movement':`Candidate from generation ${result.generation}`} · Moves ${f.actions}/12 · Output ${index}/${result.frames.length-1} · ${f.label} · Random chance for this task: ${(result.random_expected_success*100).toFixed(1)}%`+(f.tested?` · Predictions correct ${f.correct}/${f.tested}`:'');$('scrub').max=result.frames.length-1;$('scrub').value=index;if(f.done)pause()}
function step(){if(result&&index<result.frames.length-1)index++;render()}
function play(){if(!result)return;if(timer){pause();return}if(index===result.frames.length-1)index=0;timer=setInterval(step,Number($('speed').value));$('play').textContent='Pause';render()}
async function run(event){event?.preventDefault();if(!selectedCandidate||loadingCandidate||attemptBusy)return;const version=selectionVersion,candidateId=selectedCandidate.candidate_id;pause();$('error').textContent='';attemptBusy=true;updateButtons();try{const response=await fetch('/attempt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start:Number($('start').value),goal:Number($('goal').value),seed:Number($('seed').value),policy:$('policy').value,candidate_id:candidateId})});const data=await response.json();if(version!==selectionVersion)return;if(!response.ok)throw Error(data.error);result=data;index=0;$('scale').replaceChildren();$('scale').insertAdjacentHTML('beforeend','<line x1="65" y1="90" x2="935" y2="90" stroke="#7c8da8" stroke-width="3"/>');for(let n=result.lower;n<=result.upper;n+=5){$('scale').insertAdjacentHTML('beforeend',`<line x1="${x(n)}" y1="82" x2="${x(n)}" y2="103" stroke="#7c8da8"/><text x="${x(n)}" y="135" text-anchor="middle" fill="#bcc9df" font-size="18">${n}</text>`)}$('target').setAttribute('cx',x(result.goal));$('scale').insertAdjacentHTML('beforeend',`<text x="${x(result.goal)}" y="166" text-anchor="middle" fill="#64db9f" font-size="17">Goal ${result.goal}</text>`);['play','step','restart'].forEach(id=>$(id).disabled=false);render();play()}catch(e){if(version===selectionVersion)$('error').textContent=e.message}finally{attemptBusy=false;updateButtons()}}
$('task').onsubmit=run;$('again').onclick=()=>{$('seed').value=(Number($('seed').value)+1)%4294967296;run()};$('play').onclick=play;$('step').onclick=()=>{pause();step()};$('restart').onclick=()=>{pause();index=0;render()};$('scrub').oninput=()=>{pause();index=Number($('scrub').value);render()};$('speed').onchange=()=>{if(timer){pause();play()}};
async function refreshBackground(){try{const r=await fetch('/runs.json');if(!r.ok)return;const rows=await r.json();const active=rows.filter(r=>r.state==='running'||r.state==='finishing');$('background').textContent=active.length?active.map(r=>`${r.name}: ${r.completed}/${r.target} generations complete (${(100*r.progress).toFixed(0)}%)`).join(' · '):'No background training currently running.'}catch(e){$('background').textContent='Background status unavailable — check Training runs.'}}
refreshBackground();setInterval(refreshBackground,5000);
function updateButtons(){$('run').disabled=loadingCandidate||attemptBusy||!selectedCandidate;$('again').disabled=$('run').disabled}
function clearPlayback(){pause();result=null;index=0;$('scale').replaceChildren();$('position').textContent='';$('target').style.visibility='hidden';$('animal').style.visibility='hidden';['play','step','restart'].forEach(id=>$(id).disabled=true);$('scrub').max=0;$('scrub').value=0;$('detail').textContent=''}
function showInfo(info){$('scores').textContent=`Generation ${info.generation} · Training success: ${(info.training_success*100).toFixed(0)}% · Held-out success: ${(info.assessment_success*100).toFixed(0)}% (${info.assessment_episodes} attempts). Saved evaluation used ${info.boundary_mode==='clamp'?'hard walls':'out-of-bounds failure'}.`;$('comparison').textContent=info.benchmark?`Independent comparison on ${info.benchmark.episodes.toLocaleString()} shared held-out tasks: animal ${(info.benchmark.animal.success_rate*100).toFixed(2)}% · random movement ${(info.benchmark.random.success_rate*100).toFixed(2)}%. Random chooses uniformly among all nine buttons, including zero.`:`Random movement expected success on the same assessment tasks: ${(info.random_assessment_success*100).toFixed(1)}%.`;$('mode').textContent=info.varying_goals?'The animal receives the current number and your target number separately, then chooses moves. No distance or suggested move is supplied. World boundaries: −20 to 20.':'This older saved animal has no goal input and can only be tested toward zero. Choose a varying-goal run to try custom goals.';$('mode').textContent+=' Live attempts use hard walls: moves stop at −20 or +20 and still count. Old saved scores may use different boundary rules.'}
async function loadCandidate(){const version=++selectionVersion;loadingCandidate=true;selectedCandidate=null;clearPlayback();updateButtons();$('scores').textContent='Loading best saved candidate…';$('comparison').textContent='';$('mode').textContent='';$('status').textContent='Loading candidate…';$('error').textContent='';try{const response=await fetch('/info?run='+encodeURIComponent($('candidate').value));const info=await response.json();if(version!==selectionVersion)return;if(!response.ok)throw Error(info.error);selectedCandidate=info;showInfo(info);$('status').textContent='Candidate ready. Enter a task to begin.'}catch(e){if(version===selectionVersion){$('error').textContent=e.message;$('scores').textContent='Candidate unavailable.';$('status').textContent='Choose another run or refresh best.'}}finally{if(version===selectionVersion){loadingCandidate=false;updateButtons()}}}
async function refreshCandidates(loadBest=false){$('refresh-candidates').disabled=true;try{const response=await fetch('/candidates');if(!response.ok)throw Error('Could not list training runs');const data=await response.json(),previous=$('candidate').value;const options=data.runs.map(r=>{const option=document.createElement('option');option.value=r.run_id;option.disabled=!r.available;option.textContent=`${r.name}${r.population?' · '+r.population+' animals':''} · ${r.state}${r.available?'':' · waiting for first saved candidate'}`;return option});$('candidate').replaceChildren(...options);const available=data.runs.filter(r=>r.available).map(r=>r.run_id);$('candidate').disabled=!available.length;const requested=new URLSearchParams(location.search).get('run');$('candidate').value=available.includes(previous)?previous:available.includes(requested)?requested:available.includes(data.default_run)?data.default_run:available[0]||'';if(!available.length)throw Error('No saved candidates yet');if(loadBest||!selectedCandidate)await loadCandidate()}catch(e){$('error').textContent=e.message}finally{$('refresh-candidates').disabled=false}}
$('candidate').onchange=loadCandidate;$('refresh-candidates').onclick=()=>refreshCandidates(true);updateButtons();refreshCandidates(true);

</script></html>'''


def handler_for(bundle, *, status_root=ROOT):
    candidates = CandidateStore(status_root, bundle)
    workspace = Workspace(status_root, bundle)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, content_type='application/json'):
            payload = content.encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path.startswith('/api/') or parsed.path.startswith('/assets/'):
                try:
                    if not self.local_host():
                        self.send(403, '{"error":"Local workspace access only"}')
                        return
                    query = parse_qs(parsed.query)
                    arg = lambda key, default='': query.get(key, [default])[0]
                    if parsed.path == '/api/bootstrap':
                        self.send(200, json.dumps(dict(token=workspace.token, runs=workspace.runs())))
                    elif parsed.path == '/api/runs':
                        self.send(200, json.dumps(workspace.runs()))
                    elif parsed.path == '/api/run':
                        self.send(200, json.dumps(workspace.detail(arg('run'))))
                    elif parsed.path == '/api/populations':
                        from .populations import list_populations
                        self.send(200,json.dumps(list_populations(workspace.root)))
                    elif parsed.path == '/api/models':
                        self.send(200, json.dumps(workspace.models.snapshots(arg('run'), arg('model', 'baseline'))))
                    elif parsed.path == '/api/export':
                        kind, snapshot, path = workspace.models.load(arg('snapshot'))
                        if path is not None:
                            self.download(path)
                        else:
                            self.send(200, json.dumps(snapshot))
                    elif parsed.path == '/api/log':
                        self.send(200, json.dumps(workspace.log(arg('run'), arg('source', 'output'))))
                    elif parsed.path == '/api/artifact':
                        self.download(workspace.artifact(arg('run'), arg('file')))
                    elif parsed.path == '/assets/workspace.css':
                        self.send(200, (WEB / 'workspace.css').read_text(), 'text/css')
                    elif parsed.path == '/assets/workspace.js':
                        self.send(200, (WEB / 'workspace.js').read_text(), 'application/javascript')
                    else:
                        self.send(404, '{"error":"Not found"}')
                except (ValueError, OSError, KeyError, TypeError) as error:
                    self.send(400, json.dumps(dict(error=str(error))))
                return
            if parsed.path == '/candidates':
                self.send(200, json.dumps(candidates.list()))
                return
            if parsed.path == '/info' and parsed.query:
                try:
                    run_id = parse_qs(parsed.query).get('run', [''])[0]
                    self.send(200, json.dumps(candidates.info(run_id)))
                except (ValueError, OSError) as error:
                    self.send(400, json.dumps(dict(error=str(error))))
                return
            if parsed.path == '/':
                self.send(200, (WEB / 'index.html').read_text(), 'text/html')
            elif parsed.path == '/playground':
                self.send(200, PAGE, 'text/html')
            elif self.path == '/comparisons':
                self.send(200, COMPARISON_PAGE, 'text/html')
            elif self.path == '/comparisons.json':
                self.send(200, json.dumps(list_reports(status_root)))
            elif self.path == '/runs':
                self.send(200, RUNS_PAGE, 'text/html')
            elif self.path == '/runs.json':
                self.send(200, json.dumps(collect_runs(status_root)))
            elif self.path == '/info':
                if not bundle:
                    self.send(400, '{"error":"Choose a saved candidate first"}')
                    return
                self.send(200, json.dumps(dict(varying_goals=bundle.get('varying_goals', False),
                                               training_success=bundle['training_success'],
                                               assessment_success=bundle['assessment_success'],
                                               assessment_episodes=len(bundle['episodes']),
                                               random_assessment_success=bundle.get('random_assessment_success'),
                                               boundary_mode=bundle.get('boundary_mode','terminate'),
                                               benchmark=bundle.get('benchmark'))))
            else:
                self.send(404, '{"error":"Not found"}')

        def do_POST(self):
            if self.path.startswith('/api/'):
                if not self.local_host() or not secrets.compare_digest(self.headers.get('X-Workspace-Token', ''), workspace.token):
                    self.send(403, '{"error":"Refresh this local workspace before starting a session"}')
                    return
                origin = self.headers.get('Origin')
                if origin and origin != 'http://' + self.headers.get('Host', ''):
                    self.send(403, '{"error":"Cross-origin session control is not allowed"}')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 16384:
                        raise ValueError('Invalid request size')
                    data = json.loads(self.rfile.read(length))
                    if not isinstance(data, dict):
                        raise ValueError('Expected a request object')
                    self.send(200, json.dumps(workspace.post(self.path, data)))
                except (ValueError, KeyError, TypeError, OSError) as error:
                    self.send(400, json.dumps(dict(error=str(error))))
                return
            if self.path != '/attempt':
                self.send(404, '{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected a task object')
                selected = candidates.load(data['candidate_id']) if 'candidate_id' in data else bundle
                result = attempt(selected, data['start'], data['goal'], data['seed'], data.get('policy', 'animal'))
                if 'candidate_id' in data:
                    result['generation'] = selected['generation']
                    result['candidate_id'] = data['candidate_id']
                self.send(200, json.dumps(result))
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, json.dumps(dict(error=str(error))))

        def log_message(self, *args):
            pass

        def local_host(self):
            host = self.headers.get('Host', '')
            return host in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def download(self, path):
            with path.open('rb') as handle:
                import os
                self.send_response(200)
                self.send_header('Content-Type', 'application/gzip' if path.suffix == '.gz' else 'application/json')
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(path.name))
                self.send_header('Content-Length', str(os.fstat(handle.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(handle, self.wfile)
    Handler.workspace = workspace
    return Handler


def open_browser(url):
    powershell = shutil.which('powershell.exe')
    if powershell and 'microsoft' in Path('/proc/version').read_text().lower():
        subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-Command',
                        f"Start-Process '{url}'"], check=True, timeout=20)
    else:
        webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, help='optional baseline run or exported champion for the legacy playground')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    source = args.source
    try:
        bundle = {}
        if source is not None:
            bundle = load_saved_candidate(source) if source.is_dir() else json.loads(source.read_text())
            bundle['run_name'] = source.name if source.is_dir() else source.parent.name
            if source.is_dir() and source.resolve().parent == (ROOT / 'runs').resolve():
                bundle['run_id'] = source.name
            benchmark_path = (source / 'demo' if source.is_dir() else source.parent) / 'benchmark.json'
            attach_benchmark(bundle, benchmark_path)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(bundle))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    url = f'http://localhost:{server.server_port}'
    print(f'Model workspace: {url}', flush=True)
    print('Playback is frozen. Sessions start only from an explicit launch. Ctrl-C stops the viewer, not training.', flush=True)
    if not args.no_open:
        try:
            open_browser(url)
        except (OSError, subprocess.SubprocessError) as error:
            print(f'Browser launch failed: {error}. Open {url} on Windows.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
