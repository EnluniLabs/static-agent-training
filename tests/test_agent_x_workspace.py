"""Persisted parallel-senses wiring across training, workspace, and assessment."""
from dataclasses import replace
from http.server import ThreadingHTTPServer
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from static_agent_training.accelerated import assert_equivalent
from static_agent_training.assessments import run_assessment
from static_agent_training.comparison import run_study
from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import load_checkpoint, run_experiment
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.sensory_accelerated import evaluate_frozen
from static_agent_training.genome import Species
from static_agent_training.play import handler_for
from static_agent_training.populations import read_population, save_population
from static_agent_training.seeds import derive
from static_agent_training.workspace import Workspace


def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
        return json.load(handle)


class AgentXWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.runs = {}
        cls.populations = {}
        for backend in ('python', 'native'):
            for policy in ('agreement', 'vote'):
                config = TrainingConfig(model='senses', receptors=16, sensory_partitions=2,
                    action_policy=policy, backend=backend, workers=1, population=2, elites=1,
                    generations=1, episodes=2, assessment_episodes=2, initial_pairs=2,
                    chain_budget=4, seed=19)
                directory = cls.root / 'runs' / f'{backend}-{policy}'
                run_experiment(config, directory)
                cls.runs[backend, policy] = directory
                if backend == 'python':
                    output = cls.root / 'runs/populations' / f'{policy}.json.gz'
                    save_population(directory, output, count=2, name=policy)
                    cls.populations[policy] = output

    def assert_mode(self, data, policy):
        self.assertEqual(data['sensory_partitions'], 2)
        self.assertEqual(data['action_policy'], policy)

    def test_persisted_training_native_parity_and_saved_config_resume(self):
        for policy in ('agreement', 'vote'):
            python = read_json(self.runs['python', policy] / 'generations/00000.json')
            native = read_json(self.runs['native', policy] / 'generations/00000.json')
            assert_equivalent(native['individuals'], python['individuals'])
            for backend in ('python', 'native'):
                with self.subTest(backend=backend, policy=policy):
                    directory = self.runs[backend, policy]
                    saved = load_checkpoint(directory)
                    self.assert_mode(saved['config'], policy)
                    self.assert_mode(read_json(directory / 'config.json')['config'], policy)
                    trace = read_json(directory / 'traces/00000-champion.json.gz')
                    self.assert_mode(trace['learned_model'], policy)
                    model = BitModel.from_data(trace['learned_model'])
                    self.assert_mode(model.to_data(), policy)
                    config = TrainingConfig.from_data(saved['config'])
                    with self.assertRaisesRegex(ValueError, 'saved experiment configuration'):
                        run_experiment(replace(config, action_policy='vote' if policy == 'agreement'
                                               else 'agreement'), directory, resume=True)
                    resumed = run_experiment(replace(config, generations=2), directory, resume=True)
                    uninterrupted = run_experiment(replace(config, generations=2),
                                                   self.root / f'full-{backend}-{policy}')
                    self.assertEqual(resumed, uninterrupted)
                    self.assert_mode(load_checkpoint(directory)['config'], policy)
                    self.assertEqual(read_json(directory / 'generations/00001.json'),
                        read_json(self.root / f'full-{backend}-{policy}/generations/00001.json'))

    def test_saved_candidate_playback_and_brain_configuration(self):
        workspace = Workspace(self.root)
        for backend in ('python', 'native'):
            for policy in ('agreement', 'vote'):
                with self.subTest(backend=backend, policy=policy):
                    options = workspace.models.snapshots(f'{backend}-{policy}', 'senses')
                    self.assertTrue(options)
                    self.assert_mode(options[0], policy)
                    task = dict(snapshot=options[0]['id'], start=12, goal=-3, seed=42,
                                policy='animal', learning=False)
                    played = workspace.models.attempt(task)
                    self.assertEqual(played, workspace.models.attempt(task))
                    self.assertTrue(played['frames'])
                    self.assertTrue(any(f['active_pairs'] for f in played['frames']))
                    for frame in played['frames']:
                        self.assertIn('active_pairs', frame)
                        self.assertTrue(all(set(active) >= {'region', 'pair'}
                                            for active in frame['active_pairs']))
                        self.assertTrue(all(0 <= active['region'] < 2
                                            for active in frame['active_pairs']))
                        regions = [active['region'] for active in frame['active_pairs']]
                        self.assertEqual(len(regions), len(set(regions)))
                    resolutions = [f['action_resolution'] for f in played['frames']
                                   if 'action_resolution' in f]
                    self.assertTrue(resolutions)
                    for resolution in resolutions:
                        self.assertEqual(resolution['policy'], policy)
                        self.assertEqual(set(resolution), {'policy', 'action', 'conflict', 'counts'})
                    lifetime_task = dict(task, lifetime_id=f'agent-x-{backend}-{policy}')
                    graph = workspace.models.brain_scan(lifetime_task)
                    self.assert_mode(graph, policy)
                    self.assertEqual(graph['source'], 'Saved snapshot')
                    workspace.models.attempt(dict(lifetime_task, learning=True))
                    current = workspace.models.brain_scan(lifetime_task)
                    self.assert_mode(current, policy)
                    self.assertEqual(current['source'], 'Current playground lifetime')
                    self.assertEqual(current['attempts'], 1)

    def test_frozen_assessment_uses_each_persisted_source_policy(self):
        paths = [self.populations[p] for p in ('agreement', 'vote')]
        before = {p: p.read_bytes() for p in paths}
        with patch('static_agent_training.assessments.sensory_frozen',
                   wraps=evaluate_frozen) as evaluator:
            report = run_assessment(paths, self.root / 'assessment', episodes=2, workers=1)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(evaluator.call_count, 2)
        starts, goals = zip(*report['tasks'])
        for policy, call, population in zip(('agreement', 'vote'), evaluator.call_args_list,
                                            report['populations']):
            self.assert_mode(call.kwargs, policy)
            self.assert_mode(population, policy)
            data = read_population(self.populations[policy])
            members = [Species.from_data(m['individual']) for m in data['members']]
            expected = evaluate_frozen(members, starts, goals,
                seeds=[derive(48, 'frozen-population-animal', i) for i in range(len(members))],
                sensory_partitions=2, action_policy=policy, workers=1)
            for actual, reference in zip(population['individuals'], expected):
                assert_equivalent(actual['assessment'], reference)
                self.assertEqual(actual['assessment']['empirical_updates'], 0)
            self.assertEqual(Path(population['snapshot']).read_bytes(), before[self.populations[policy]])
        self.assertEqual(before, {p: p.read_bytes() for p in paths})
        self.assertEqual(read_json(self.root / 'assessment/report.json')['populations'],
                         report['populations'])

    def test_compare_import_keeps_target_policy_and_source_provenance(self):
        source = self.populations['agreement']
        before = source.read_bytes()
        config = TrainingConfig.from_data(read_population(self.populations['vote'])['config'])
        directory = self.root / 'compare'
        report = run_study(directory, config, seeds=[19], benchmark_episodes=2,
            runs_root=self.root / 'runs', treatment='starting-population',
            population_a=source, population_b=self.populations['vote'], champion_count=1)
        self.assertEqual(report['status'], 'completed')
        manifest = read_json(directory / 'manifest.json')
        self.assert_mode(manifest['config'], 'vote')
        self.assert_mode(read_population(manifest['population_a'])['config'], 'agreement')
        for arm in manifest['arms']:
            arm_dir = Path(arm['directory'])
            self.assert_mode(read_json(arm_dir / 'config.json')['config'], 'vote')
            self.assert_mode(load_checkpoint(arm_dir)['config'], 'vote')
            trace = read_json(arm_dir / 'traces/00000-champion.json.gz')
            self.assert_mode(trace['learned_model'], 'vote')
        self.assertEqual(source.read_bytes(), before)
        repeated = run_study(directory, resume=True)
        self.assertEqual(report['pairs'], repeated['pairs'])

    def test_model_comparison_keeps_categorical_single_and_sensory_parallel(self):
        config = TrainingConfig.from_data(read_population(self.populations['vote'])['config'])
        directory = self.root / 'model-compare'
        report = run_study(directory, config, seeds=[19], benchmark_episodes=2,
            runs_root=self.root / 'runs', treatment='model', champion_count=1)
        self.assertEqual(report['status'], 'completed')
        manifest = read_json(directory / 'manifest.json')
        self.assert_mode(manifest['config'], 'vote')
        for arm in manifest['arms']:
            saved = read_json(Path(arm['directory']) / 'config.json')['config']
            self.assertEqual(saved['model'], 'senses' if arm['enabled'] else 'categorical')
            self.assertEqual(saved['sensory_partitions'], 2 if arm['enabled'] else 1)
            self.assertEqual(saved['action_policy'], 'vote')

    def test_http_workspace_launch_args_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            server = ThreadingHTTPServer(('127.0.0.1', 0),
                                         handler_for({}, status_root=Path(temporary)))
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01},
                                      daemon=True)
            thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                with urlopen(base + '/api/bootstrap', timeout=5) as response:
                    token = json.load(response)['token']
                manager = server.RequestHandlerClass.workspace.sessions
                def post(config, kind='sensory-training'):
                    request = Request(base + '/api/sessions',
                        data=json.dumps(dict(kind=kind, config=config)).encode(),
                        headers={'Content-Type': 'application/json', 'X-Workspace-Token': token})
                    return urlopen(request, timeout=5)
                with patch.object(manager, 'spawn', return_value={'id': 'test'}) as spawn:
                    for policy in ('agreement', 'vote'):
                        config = dict(receptors=16, sensory_partitions=2, action_policy=policy,
                            embedding_selection='argmax', population=2, generations=1,
                            episodes=1, assessment_episodes=1, backend='python')
                        for kind in ('sensory-training', 'starting-population-benchmark'):
                            values = dict(config, model='senses') if kind.endswith('benchmark') else config
                            with post(values, kind) as response:
                                self.assertEqual(response.status, 200)
                            args = spawn.call_args.args[2]
                            self.assertEqual(args[args.index('--sensory-partitions') + 1], '2')
                            self.assertEqual(args[args.index('--action-policy') + 1], policy)
                        for invalid in ({'sensory_partitions': 17}, {'sensory_partitions': True},
                                        {'embedding_selection': 'sample'}, {'action_policy': 'bad'}):
                            spawn.reset_mock()
                            with self.assertRaises(HTTPError) as caught:
                                post(dict(config, **invalid))
                            self.assertEqual(caught.exception.code, 400)
                            caught.exception.close()
                            spawn.assert_not_called()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the DOM runtime check')
    def test_actual_config_and_playback_javascript_in_dom_harness(self):
        # Execute production functions, with a minimal DOM rather than reimplementing
        # config serialization or rendering. This does not test browser layout.
        web = Path(__file__).resolve().parents[1] / 'static_agent_training/web'
        source = (web / 'workspace.js').read_text()
        self.assertIn('id="parallel-senses-help"', (web / 'index.html').read_text())
        form = source[source.index('const common='):source.index("$('launch-form').onsubmit=")]
        frames = source[source.index('function bits('):source.index('function step()')]
        harness = r"""
const assert=require('node:assert/strict');
const elements=new Map();
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.style={};this.value='';}
  set id(value){this._id=value;elements.set(value,this);}
  get id(){return this._id;}
  append(...children){this.children.push(...children);}
  prepend(child){this.children.unshift(child);}
  replaceChildren(...children){
    const remove=e=>{if(e.id&&elements.get(e.id)===e)elements.delete(e.id);e.children.forEach(remove);};
    this.children.forEach(remove);this.children=children;
    if(this.tag==='select')this.value=children[0]?.value??'';
  }
  setCustomValidity(message){this.validationMessage=message;}
  setAttribute(name,value){this[name]=value;}
}
const document={getElementById:id=>elements.get(id)??null,
                createElement:tag=>new Element(tag)};
const $=id=>document.getElementById(id);
function node(tag,text,cls){const e=document.createElement(tag);e.textContent=text;e.className=cls;return e;}
for(const id of ['model-family','session-kind','parallel-senses-help','model-description',
 'switchboard-setup','switchboard-preset','session-fields','session-notes','setup-summary','animal','position','frame-label',
 'frame-detail','scrub','sensory-panel','receptor-count','receptors','pair-detail']){
 const e=new Element(id==='session-kind'?'select':'div');e.id=id;
}
const state={populations:[],runs:[]};
let playback=null,frameIndex=0;
const worldX=x=>x,pct=x=>String(x),pause=()=>{};
// Canvas rendering is outside this form/frame DOM harness.
function drawBrain(){}
function drawRetinaView(){}
function puzzleSize(env){return 0;}
"""
        checks = r"""
$('model-family').value='agent-x';setupKind();
assert.equal($('session-kind').value,'sensory-training');
assert.equal(configValues().sensory_partitions,4);
assert.equal(configValues().embedding_selection,'argmax');
assert.equal(configValues().sensory_interface,'signals');
assert.equal(configValues().retina_width,16);assert.equal(configValues().retina_height,16);
assert.equal($('config-sensory_partitions').max,256);
const fixed=JSON.stringify([configValues().receptors,configValues().sensory_partitions,configValues().retina_width,configValues().retina_height]);
for(const env of ['lights-out','switchboard','shifted','original']){
 $('config-environment').value=env;setupSummary();
 assert.equal(JSON.stringify([configValues().receptors,configValues().sensory_partitions,configValues().retina_width,configValues().retina_height]),fixed);
}
$('config-sensory_partitions').value=256;setupSummary();assert.equal($('config-sensory_partitions').validationMessage,'');
$('config-sensory_partitions').value=257;setupSummary();assert.notEqual($('config-sensory_partitions').validationMessage,'');
$('session-kind').value='starting-population-benchmark';renderFields();
assert.equal(configValues().model,'senses');
assert.equal(configValues().sensory_partitions,4);
$('session-kind').value='sensory-training';
$('model-family').value='senses';setupKind();
assert.equal($('parallel-senses-help').hidden,false);
assert.equal(configValues().sensory_partitions,1);
assert.equal(configValues().action_policy,'agreement');
assert.equal(configValues().environment,'original');
$('config-environment').value='switchboard';setupSummary();
assert.equal(configValues().receptors,256);
assert.equal(configValues().sensory_partitions,1);
$('config-environment').value='original';
$('config-receptors').value=16;$('config-sensory_partitions').value=2;setupSummary();
assert.equal($('config-sensory_partitions').max,16);
assert.equal($('config-sensory_partitions').validationMessage,'');
$('config-embedding_selection').value='sample';setupSummary();
assert.match($('config-embedding_selection').validationMessage,/require/);
$('config-embedding_selection').value='argmax';setupSummary();
assert.equal($('config-embedding_selection').validationMessage,'');
for(const invalid of [0,1.5,17]){
 $('config-sensory_partitions').value=invalid;setupSummary();
 assert.notEqual($('config-sensory_partitions').validationMessage,'');
}
$('config-sensory_partitions').value=2;$('config-action_policy').value='vote';setupSummary();
assert.equal(configValues().action_policy,'vote');
assert.equal(configValues().sensory_partitions,2);
$('session-kind').value='starting-population-benchmark';renderFields();
$('config-action_policy').value='vote';$('config-sensory_partitions').value=2;setupSummary();
assert.equal(configValues().model,'senses');
assert.equal(configValues().action_policy,'vote');
$('model-family').value='baseline';setupKind();
assert.equal($('parallel-senses-help').hidden,true);
assert.equal($('config-sensory_partitions'),null);
assert.equal(Object.hasOwn(configValues(),'action_policy'),false);
$('model-family').value='senses';setupKind();
assert.equal(configValues().sensory_partitions,1);
assert.equal(configValues().action_policy,'agreement');
const frame={position:5,actions:1,label:'round',sensory:'0000',
 active_pairs:[{region:0,pair:'left'},{region:1,pair:'right'}],
 action_resolution:{policy:'agreement',action:null,conflict:true,counts:{0:1,1:1}}};
playback={environment:'original',frames:[frame]};renderFrame();
assert.match($('pair-detail').textContent,/region 0: left/);
assert.match($('pair-detail').textContent,/region 1: right/);
assert.match($('pair-detail').textContent,/agreement.*conflict.*no action/);
frame.action_resolution={policy:'vote',action:0,conflict:false,counts:{0:2}};renderFrame();
assert.match($('pair-detail').textContent,/vote.*action 0/);
delete frame.active_pairs;delete frame.action_resolution;frame.pair='single';renderFrame();
assert.equal($('pair-detail').textContent,'Internal state: single');
console.log('form switching, validation, serialization, and frame rendering passed');
"""
        result = subprocess.run([shutil.which('node'), '-e', harness + form + frames + checks],
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
