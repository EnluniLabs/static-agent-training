"""Immutable retina apparatus at the workspace and import boundaries."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import unittest
from unittest.mock import patch

from static_agent_training.workspace_sessions import SessionManager

from static_agent_training.populations import initialize_saved, sensory_apparatus, validate_compatibility


class RetinaWorkspaceTests(unittest.TestCase):
    def test_strict_source_apparatus(self):
        source=dict(model='senses',receptors=256,sensory_partitions=4,
                    sensory_interface='retina',retina_width=16,retina_height=16)
        config=dict(source,encoding_seed=0,varying_goals=False)
        validate_compatibility(dict(config=source),SimpleNamespace(**config))
        for key,value in [('receptors',128),('sensory_partitions',2),
                          ('sensory_interface','legacy-bytes'),('retina_width',8),('retina_height',32)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,key):
                validate_compatibility(dict(config=source),SimpleNamespace(**dict(config,**{key:value})))
        self.assertEqual(sensory_apparatus({})['sensory_interface'],'legacy-bytes')
        self.assertEqual(sensory_apparatus({})['sensory_partitions'],1)

    def test_import_rejects_parent_apparatus_mismatch(self):
        source=dict(model='senses',receptors=256,sensory_partitions=4,
                    sensory_interface='retina',retina_width=16,retina_height=16)
        config=SimpleNamespace(**source,encoding_seed=0,varying_goals=False,initial_population='source')
        for key,value in [('receptors',128),('partitions',2),('width',8),('interface','legacy-bytes')]:
            senses=dict(receptors=256,partitions=4,width=16,height=16,interface='retina')
            senses[key]=value
            parent=SimpleNamespace(inherited=SimpleNamespace(senses=SimpleNamespace(**senses)))
            with self.subTest(key=key), \
                 patch('static_agent_training.populations.read_population',return_value=dict(config=source,members=[dict(individual={})])), \
                 patch('static_agent_training.populations.Species.from_data',return_value=parent), \
                 self.assertRaisesRegex(ValueError,'Parent'):
                initialize_saved(config)

    def test_server_adopts_saved_apparatus_and_rejects_manual_conflicts(self):
        manager=SessionManager('/tmp/retina-workspace-test')
        source=dict(model='senses',receptors=128,sensory_partitions=4,
                    sensory_interface='retina',retina_width=8,retina_height=16)
        with patch.object(manager,'population_path',return_value=Path('/tmp/source.json.gz')), \
             patch('static_agent_training.populations.read_population',return_value=dict(config=source)):
            _,args=manager.command('sensory-training',dict(initial_population='source',population=2),Path('/tmp/output'))
            for flag,value in [('--receptors','128'),('--sensory-partitions','4'),
                               ('--sensory-interface','retina'),('--retina-width','8'),('--retina-height','16')]:
                self.assertEqual(args[args.index(flag)+1],value)
            for key,value in [('receptors',256),('sensory_partitions',2),('sensory_interface','legacy-bytes'),('retina_width',16)]:
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'conflicts'):
                    manager.command('sensory-training',dict(initial_population='source',**{key:value}),Path('/tmp/output'))

    @unittest.skipUnless(shutil.which('node'),'Node required for production DOM checks')
    def test_agent_x_environment_preserves_geometry_and_source_adoption(self):
        source=(Path(__file__).resolve().parents[1]/'static_agent_training/web/workspace.js').read_text()
        form=source[source.index('const common='):source.index("$('launch-form').onsubmit=")]
        layout=source[source.index('function brainHash('):source.index('const brain =')]
        frames=source[source.index('function bits('):source.index('function step()')]
        harness=HARNESS
        checks=r"""
$('model-family').value='agent-x';setupKind();
assert.equal(configValues().sensory_interface,'signals');
assert.equal(configValues().sensory_partitions,4);
assert.equal(configValues().retina_width,16);
assert.equal(configValues().retina_height,16);
for(const environment of ['original','switchboard','shifted','lights-out','switches-2x2']){
 $('config-environment').value=environment;setupSummary();
 assert.equal(configValues().receptors,256);assert.equal(configValues().sensory_partitions,4);
}
$('config-retina_width').value=8;setupSummary();assert.equal(configValues().receptors,128);
state.populations=[{id:'legacy',model:'senses',receptors:144,sensory_partitions:3}];
$('config-initial_population').value='legacy';adoptSourceApparatus();setupSummary();
assert.equal(configValues().sensory_interface,'legacy-bytes');
assert.equal(configValues().receptors,144);assert.equal(configValues().sensory_partitions,3);
assert.equal($('model-family').value,'agent-x');
for(const key of ['receptors','sensory_partitions','sensory_interface'])assert.equal($('config-'+key).disabled,true);
playback={environment:'original',sensory_interface:'retina',retina_width:8,retina_height:2,
 frames:[{sensory:'a000',label:'Step',position:1,actions:0}]};renderFrame();
assert.equal($('receptors').style.gridTemplateColumns,'repeat(8, minmax(0, 1fr))');
assert.match($('receptor-count').textContent,/RETINAL VIEW/);
const graph={sensory_geometry:{interface:'retina',width:8,height:4},nodes:[
 {id:'presence',kind:'input',feature:0}, {id:'a',kind:'input',pixel:[0,0]},
 {id:'b',kind:'input',pixel:[7,3]}, {id:'c',kind:'input',pixel:[7,0]}]};
const positions=brainLayout(graph);
assert.equal(positions.get('a').y,positions.get('c').y);
assert.equal(positions.get('b').x,positions.get('c').x);
assert.ok(positions.get('presence').x<positions.get('a').x);
"""
        result=subprocess.run([shutil.which('node'),'-e',harness+form+layout+frames+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


HARNESS = r"""
const assert=require('node:assert/strict');
const elements=new Map();
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.style={};this.value='';}
  set id(value){this._id=value;elements.set(value,this);}
  get id(){return this._id;}
  append(...children){this.children.push(...children);for(const c of children)c.parentElement=this;}
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
 'session-fields','session-notes','setup-summary','animal','position','frame-label',
 'frame-detail','scrub','sensory-panel','receptor-count','receptors','pair-detail']){
 const e=new Element(id==='session-kind'?'select':'div');e.id=id;
}
const state={populations:[],runs:[]};
let playback=null,frameIndex=0;
const worldX=x=>x,pct=x=>String(x),pause=()=>{};
// Canvas rendering is outside this form/frame DOM harness.
function drawBrain(){}
function puzzleSize(env){return 0;}
"""
