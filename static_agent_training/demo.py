"""Export the best saved training champion and a standalone visual assessment replay."""
import argparse
import gzip
import json
from pathlib import Path

from .agent import Agent, run_episode
from .baseline import random_task_mean
from .environment import NumberLine
from .genome import Species
from .seeds import derive


class RecordingWorld(NumberLine):
    """Viewer diagnostics stay outside the agent's symbolic interface."""
    def reset(self, **kwargs):
        transition = super().reset(**kwargs)
        self.frames = []
        self.record(transition, 'Start')
        return transition

    def record(self, transition, label):
        state = self.diagnostic_state()
        self.frames.append(dict(position=state['position'], actions=state['actions'],
                                label=label, done=transition.done,
                                success=transition.success, reason=transition.reason,
                                correct=sum(f.correct for f in transition.feedback),
                                tested=len(transition.feedback)))

    def emit(self, output, *, source=None):
        state = self.diagnostic_state()
        if output.token in state['move_decoding']:
            label = f"Move {state['move_decoding'][output.token]:+d}"
        else:
            label = f"Predict position {state['expectation_decoding'][output.token]}"
        transition = super().emit(output, source=source)
        if (output.token in state["move_decoding"] and self.boundary_mode == "clamp" and
                not -20 <= state["position"] + state["move_decoding"][output.token] <= 20):
            label += " (stopped at wall)"
        self.record(transition, label)
        return transition

    def stop(self):
        transition = super().stop()
        self.record(transition, 'Chain ended')
        return transition


def export_demo(directory):
    directory = Path(directory)
    records = [json.loads(p.read_text()) for p in sorted((directory / 'generations').glob('*.json'))]
    if not records:
        raise ValueError('run has no completed generations')
    # Stable tie break: first generation to achieve the highest training fitness.
    record = max(records, key=lambda r: r['summary']['best'])
    generation = record['summary']['generation']
    with gzip.open(directory / 'traces' / f'{generation:05d}-champion.json.gz', 'rt') as handle:
        trace = json.load(handle)
    config = json.loads((directory / 'config.json').read_text())['config']
    if config.get('model') == 'senses':
        from .candidates import load_saved_candidate
        from .experimental.bit_model import BitModel
        from .experimental.sensory_playback import record_episode
        bundle = load_saved_candidate(directory, generation)
        model = BitModel.from_data(bundle['learned_model'])
        seed = trace['result']['seed']
        model.embedding_rng.seed(derive(seed, 'assessment', 'embedding'))
        model.output_rng.seed(derive(seed, 'assessment', 'output'))
        episodes = []
        for expected in trace['result']['assessment']['episodes']:
            frames = record_episode(model, expected['start'], expected['goal'],
                encoding_seed=bundle['encoding_seed'], boundary_mode=bundle['boundary_mode'],
                goal_conditioned=bundle['varying_goals'])
            if (frames[-1]['success'], frames[-1]['reason']) != (expected['success'], expected['reason']):
                raise ValueError('sensory assessment replay does not match saved result')
            episodes.append(dict(start=expected['start'], goal=expected['goal'], frames=frames))
            model.events.clear()
        bundle['episodes'] = episodes
        destination = directory / 'demo'
        destination.mkdir(exist_ok=True)
        (destination / 'champion.json').write_text(json.dumps(bundle, separators=(',', ':')) + '\n')
        view = {k:bundle[k] for k in ('generation','training_success','assessment_success','episodes')}
        (destination / 'index.html').write_text(HTML.replace('__DATA__', json.dumps(view).replace('<', '\\u003c')))
        return destination, bundle
    species = Species.from_data(trace['individual'])
    mappings = species.inherited.lifetime()
    mappings.embedding = trace['learned_embedding']
    mappings.output = trace['learned_output']
    seed = trace['result']['seed']
    agent = Agent(species.repertoire, mappings, learning=False,
                  embedding_seed=derive(seed, 'assessment', 'embedding'),
                  output_seed=derive(seed, 'assessment', 'output'))
    world = RecordingWorld(encoding_seed=config['encoding_seed'],
                           goal_conditioned=config.get("varying_goals", False), boundary_mode=config.get("boundary_mode", "terminate"))
    episodes = []
    for start, expected in zip(record['assessment_starts'], trace['result']['assessment']['episodes']):
        transitions = run_episode(agent, world, start=start, learn=False, goal=expected.get("goal", 0))
        if (transitions[-1].success, transitions[-1].reason) != (expected['success'], expected['reason']):
            raise ValueError('assessment replay does not match saved result')
        episodes.append(dict(start=start, goal=expected.get("goal", 0), frames=world.frames))
    bundle = dict(schema_version='crt-demo-v1', generation=generation + 1,
                  training_success=trace['result']['training']['fitness'],
                  assessment_success=trace['result']['assessment']['fitness'],
                  individual=trace['individual'], learned_embedding=trace['learned_embedding'],
                  learned_output=trace['learned_output'], assessment_seed=seed,
                  encoding_seed=config['encoding_seed'], varying_goals=config.get('varying_goals', False), boundary_mode=config.get('boundary_mode', 'terminate'), episodes=episodes)
    bundle['random_assessment_success'] = random_task_mean(
        [e['start'] for e in episodes], [e.get('goal', 0) for e in episodes], boundary_mode=bundle['boundary_mode'])
    destination = directory / 'demo'
    destination.mkdir(exist_ok=True)
    (destination / 'champion.json').write_text(json.dumps(bundle, separators=(',', ':')) + '\n')
    view = {k: bundle[k] for k in ('generation', 'training_success', 'assessment_success', 'episodes')}
    payload = json.dumps(view).replace('<', '\\u003c')
    (destination / 'index.html').write_text(HTML.replace('__DATA__', payload))
    return destination, bundle


HTML = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Champion replay</title>
<style>
body{font:17px system-ui;background:#101725;color:#e7edf7;max-width:1050px;margin:40px auto;padding:20px}h1{font-size:30px}p{color:#b8c5d8;line-height:1.6}button,select{font:inherit;padding:10px;margin:5px;background:#26364e;color:white;border:1px solid #7283a0;border-radius:8px}svg{width:100%;background:#182438;border-radius:15px;margin:20px 0}#status{font-size:22px;min-height:65px}input{width:100%}#detail{min-height:30px}.legend{color:#b8c5d8}
</style>
<h1>Training champion · number-line replay</h1><p id="summary"></p>
<p>The animal must reach the goal within 12 moves. Blue is the animal; green is the target. Positions and decoded predictions are viewer diagnostics; the animal receives opaque symbols. Learned weights are frozen for these held-out attempts.</p>
<label>Attempt <select id="episode"></select></label>
<button id="play">Play</button><button id="step">Step</button><button id="restart">Restart</button>
<label>Speed <select id="speed"><option value="800">Normal</option><option value="200">Fast</option><option value="1500">Slow</option></select></label>
<svg viewBox="0 0 1000 170" role="img" aria-label="Animal position on number line">
<line x1="50" y1="90" x2="950" y2="90" stroke="#7283a0" stroke-width="3"/>
<g id="ticks"></g><circle id="target" cx="500" cy="90" r="16" fill="none" stroke="#64db9f" stroke-width="4"/>
<circle id="animal" cy="90" r="10" fill="#65baff"/><text id="position" y="45" text-anchor="middle" fill="white" font-size="22"></text></svg>
<div id="status" aria-live="polite"></div><div id="detail"></div>
<label>Output step <input id="scrub" type="range" min="0" value="0"></label>
<p>All assessment attempts are included in their original order. Selection uses the highest training success across saved generation champions; ties use the earliest generation. A successful replay does not imply reliable performance on every start.</p>
<script>
const data=__DATA__, $=id=>document.getElementById(id);let ep=0,frame=0,timer=null;
$('summary').textContent=`Generation ${data.generation} · Training success ${(data.training_success*100).toFixed(0)}% · Held-out success ${(data.assessment_success*100).toFixed(0)}% (${data.episodes.length} attempts)`;
data.episodes.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i+1}: start ${e.start} — ${e.frames.at(-1).success?'solved':e.frames.at(-1).reason.replaceAll('_',' ')}`;$('episode').append(o)});
for(let n=-20;n<=20;n+=5){$('ticks').insertAdjacentHTML('beforeend',`<line x1="${500+n*20}" y1="82" x2="${500+n*20}" y2="100" stroke="#7283a0"/><text x="${500+n*20}" y="133" text-anchor="middle" fill="#b8c5d8" font-size="18">${n}</text>`)}
function pause(){clearInterval(timer);timer=null;$('play').textContent='Play'}
function render(){const fs=data.episodes[ep].frames,f=fs[frame],goal=data.episodes[ep].goal||0,x=Math.max(25,Math.min(975,500+f.position*20));$('target').setAttribute('cx',500+goal*20);$('animal').setAttribute('cx',x);$('position').setAttribute('x',x);$('position').textContent=f.position;$('status').textContent=f.done?(f.success?'Solved — target reached!':`Stopped — ${f.reason.replaceAll('_',' ')}`):f.label;$('detail').textContent=`Step ${frame}/${fs.length-1} · Moves ${f.actions}/12 · ${f.label}`+(f.tested?` · Predictions correct: ${f.correct}/${f.tested}`:'');$('scrub').max=fs.length-1;$('scrub').value=frame;if(f.done)pause()}
function step(){if(frame<data.episodes[ep].frames.length-1)frame++;render()}
$('play').onclick=()=>{if(timer){pause();return}if(frame===data.episodes[ep].frames.length-1)frame=0;timer=setInterval(step,Number($('speed').value));$('play').textContent='Pause';render()};
$('step').onclick=()=>{pause();step()};$('restart').onclick=()=>{pause();frame=0;render()};$('episode').onchange=()=>{pause();ep=Number($('episode').value);frame=0;render()};$('scrub').oninput=()=>{pause();frame=Number($('scrub').value);render()};$('speed').onchange=()=>{if(timer){pause();$('play').click()}};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='saved training run directory')
    args = parser.parse_args()
    try:
        destination, bundle = export_demo(args.run)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Champion: generation {bundle['generation']}; training {bundle['training_success']:.0%}; held-out {bundle['assessment_success']:.0%}")
    print(f"Open in a browser: {(destination / 'index.html').resolve()}")
    print(f"Exported animal: {(destination / 'champion.json').resolve()}")


if __name__ == '__main__':
    main()
