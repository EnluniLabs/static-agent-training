"""Structural controls and command wiring; no training processes are started."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.structure_profiles import STRUCTURE_PRESETS
from static_agent_training.workspace_sessions import (
    COMPARISONS, TRAINING, STRUCTURE_INTEGERS, STRUCTURE_PROBABILITIES, SessionManager,
)


class ExploratoryWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = SessionManager(self.temp.name)
        self.directory = Path(self.temp.name) / 'runs/example'

    def test_only_explicit_settings_forwarded_for_training_and_comparisons(self):
        for kind in (*TRAINING, *COMPARISONS):
            with self.subTest(kind=kind):
                _, args = self.manager.command(kind, {}, self.directory)
                self.assertNotIn('--structure-preset', args)
                for key in (*STRUCTURE_INTEGERS, *STRUCTURE_PROBABILITIES):
                    self.assertNotIn('--' + key.replace('_', '-'), args)
                config = dict(structure_preset='exploratory', development_max_nodes=2048,
                              initial_pairs=16, combinator_probability=.15)
                _, args = self.manager.command(kind, config, self.directory)
                for flag, value in [('--structure-preset', 'exploratory'),
                                    ('--development-max-nodes', '2048'),
                                    ('--initial-pairs', '16'), ('--combinator-probability', '0.15')]:
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertNotIn('--chain-budget', args)

    def test_strict_validation_and_generous_integer_bounds(self):
        for key in STRUCTURE_INTEGERS:
            for value in (-1, True, 1.5, '16', None, 2**31):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.manager.command('training', {key: value}, self.directory)
            if key in ('chain_budget', 'new_pair_budget'):
                self.manager.command('training', {key: 0}, self.directory)
            else:
                with self.assertRaises(ValueError):
                    self.manager.command('training', {key: 0}, self.directory)
            if key in ('seed_term_max_nodes', 'seed_term_max_depth'):
                with self.assertRaises(ValueError):
                    self.manager.command('training', {key: 1}, self.directory)
            self.manager.command('training', {key: 1000000}, self.directory)
        for key in STRUCTURE_PROBABILITIES:
            for value in (-.01, 1.01, float('nan'), float('inf'), True, '.2', None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.manager.command('training', {key: value}, self.directory)
            for value in (0, 1, .35):
                self.manager.command('training', {key: value}, self.directory)
        for value in ('unknown', None, [], True):
            with self.assertRaises(ValueError):
                self.manager.command('training', {'structure_preset': value}, self.directory)
        for kind in ('senses', 'interfaces', 'assessment', 'population-benchmark'):
            with self.assertRaises(ValueError):
                self.manager.command(kind, {'initial_pairs': 16}, self.directory)

    def test_launch_factory_and_resume_preserve_settings(self):
        config = dict(structure_preset='custom', initial_pairs=19, chain_budget=300)
        with patch.object(self.manager, 'spawn', return_value={'id': 'mock'}) as spawn:
            self.manager.launch(dict(kind='sensory-training', config=config))
            self.assertEqual(spawn.call_args.kwargs['config'], config)
            self.assertIn('--initial-pairs', spawn.call_args.args[2])
        self.directory.mkdir(parents=True)
        (self.directory / 'checkpoint.json.gz').touch()
        (self.directory / 'config.json').write_text(json.dumps({'config': {'generations': 50}}))
        record = dict(id='old', state='stopped', kind='sensory-training',
                      directory=str(self.directory), name='Saved', config=config)
        with patch.object(self.manager, 'get', return_value=record), \
             patch.object(self.manager, 'records', return_value=[]), \
             patch.object(self.manager, 'spawn') as spawn:
            self.manager.resume('old', generations=60)
            args = spawn.call_args.args[2]
            self.assertNotIn('--structure-preset', args)
            self.assertNotIn('--initial-pairs', args)
            self.assertNotIn('--chain-budget', args)
            self.assertEqual(spawn.call_args.kwargs['config'], config)

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the production JS DOM harness')
    def test_form_presets_custom_serialization_and_growth_chart(self):
        web = Path(__file__).resolve().parents[1] / 'static_agent_training/web'
        source = (web / 'workspace.js').read_text()
        form = source[source.index('const common='):source.index("$('launch-form').onsubmit=")]
        chart = source[source.index('function chart('):source.index('function table(')]
        harness = r"""
const assert=require('node:assert/strict'), elements=new Map();
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.style={};this.value='';}
 set id(v){this._id=v;elements.set(v,this);} get id(){return this._id;}
 append(...v){this.children.push(...v);} prepend(v){this.children.unshift(v);}
 replaceChildren(...v){const rm=e=>{if(e.id&&elements.get(e.id)===e)elements.delete(e.id);e.children.forEach(rm);};this.children.forEach(rm);this.children=v;if(this.tag==='select')this.value=v[0]?.value??'';}
 setCustomValidity(v){this.validationMessage=v;} setAttribute(k,v){this[k]=v;}
}
const $=id=>elements.get(id)||null;
function node(tag,text){const e=new Element(tag);e.textContent=text;return e;}
function svg(tag,attrs,text){const e=node(tag,text);Object.assign(e,attrs);return e;}
const document={createTextNode:text=>node('text',text)};
const state={runs:[],populations:[]},colors=['blue','green','red'],pct=n=>String(n);
for(const id of ['model-family','session-kind','parallel-senses-help','model-description','switchboard-setup','switchboard-preset','session-fields','session-notes','setup-summary','chart-metric','chart-metric-control','chart-title','chart-caption','chart-note','trend','chart-legend']){const e=new Element(id==='session-kind'?'select':'div');e.id=id;}
"""
        checks = r"""
$('model-family').value='agent-x';setupKind();
assert.equal(configValues().structure_preset,'exploratory');
assert.equal(configValues().sensory_partitions,4);assert.equal(configValues().receptors,256);
for(const key of structureKeys)assert.equal(configValues()[key],structurePresets.exploratory[key]);
assert.match($('setup-summary').textContent,/Starting pairs: 16/);
$('config-deletion_probability').value=.123;$('config-deletion_probability').oninput();
assert.equal(configValues().structure_preset,'custom');assert.equal(configValues().deletion_probability,.123);
$('config-structure_preset').value='baseline';$('config-structure_preset').onchange();
for(const key of structureKeys)assert.equal(configValues()[key],structurePresets.baseline[key]);
$('session-kind').value='starting-population-benchmark';renderFields();
assert.equal(configValues().model,'senses');assert.equal(configValues().initial_pairs,16);
for(const family of ['senses','baseline']){$('model-family').value=family;setupKind();assert.equal(configValues().structure_preset,'baseline');assert.equal(configValues().initial_pairs,4);}
$('chart-metric').value='complexity';chart({kind:'training',series:[{x:1,mean_repertoire:12,mean_convergence_states:3,mean_seed_term_nodes:8}]});
assert.equal($('chart-title').textContent,'Structural growth');
assert.equal($('chart-legend').children.length,3);
assert.match($('chart-caption').textContent,/counts/);
chart({kind:'training',series:[{x:1}]});assert.match($('chart-note').textContent,/unavailable/);
"""
        parity = 'assert.deepEqual(structurePresets,' + json.dumps({key: STRUCTURE_PRESETS[key] for key in ('baseline', 'exploratory')}) + ');'
        result = subprocess.run(['node', '-e', harness + form + chart + checks + parity], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
