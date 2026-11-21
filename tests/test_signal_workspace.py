"""Universal signal apparatus controls and activity in the production workspace."""
from pathlib import Path
import shutil
import subprocess
import unittest

from tests.test_retina_workspace import HARNESS


@unittest.skipUnless(shutil.which('node'), 'Node required for production DOM checks')
class SignalWorkspaceTests(unittest.TestCase):
    def test_fixed_signal_layout_sources_and_activity(self):
        source = (Path(__file__).resolve().parents[1] / 'static_agent_training/web/workspace.js').read_text()
        form = source[source.index('const common='):source.index("$('launch-form').onsubmit=")]
        layout = source[source.index('function brainHash('):source.index('const brain =')]
        frames = source[source.index('function bits('):source.index('function step()')]
        checks = r"""
$('model-family').value='agent-x';setupKind();
assert.equal(configValues().sensory_interface,'signals');
assert.equal(configValues().receptors,256);assert.equal(configValues().sensory_partitions,4);
assert.match($('setup-summary').textContent,/indexing only/);
$('config-retina_width').value=8;$('config-retina_height').value=4;setupSummary();
assert.equal(configValues().receptors,32);assert.equal($('config-receptors').disabled,true);
assert.equal($('config-sensory_partitions').max,32);
$('config-sensory_partitions').value=33;setupSummary();
assert.notEqual($('config-sensory_partitions').validationMessage,'');
state.populations=[{id:'signal-source',model:'senses',sensory_interface:'signals',
 receptors:128,sensory_partitions:8,retina_width:8,retina_height:16}];
$('config-initial_population').value='signal-source';adoptSourceApparatus();setupSummary();
for(const environment of ['original','switchboard','shifted','lights-out','switches-2x2']){
 $('config-environment').value=environment;setupSummary();
 assert.equal(configValues().sensory_interface,'signals');
 assert.equal(configValues().receptors,128);assert.equal(configValues().sensory_partitions,8);
 assert.equal(configValues().retina_width,8);assert.equal(configValues().retina_height,16);
}
for(const key of ['receptors','sensory_partitions','sensory_interface','retina_width','retina_height'])
 assert.equal($('config-'+key).disabled,true);
const camera=new Element('section');camera.id='retina-camera';
playback={environment:'original',sensory_interface:'signals',retina_width:8,retina_height:16,
 frames:[{sensory:'a000',label:'Step',position:1,actions:0}]};renderFrame();
assert.equal(camera.hidden,true);
assert.equal($('receptors').style.gridTemplateColumns,'repeat(8, minmax(0, 1fr))');
assert.match($('receptor-count').textContent,/RECEPTOR SIGNALS.*fixed layout/);
assert.doesNotMatch($('receptor-count').textContent,/eye|pixel|view/i);
const graph={sensory_geometry:{interface:'signals',width:8,height:4},nodes:[
 {id:'presence',kind:'input',feature:0,label:'pixel presence'},
 {id:'a',kind:'input',feature:1,label:'pixel 1'},
 {id:'b',kind:'input',feature:32,label:'pixel 32'},
 {id:'c',kind:'input',feature:8,label:'pixel 8'}]};
const positions=brainLayout(graph);
assert.equal(positions.get('a').y,positions.get('c').y);
assert.equal(positions.get('b').x,positions.get('c').x);
for(const n of graph.nodes)assert.match(n.label,/Receptor channel/);
"""
        result = subprocess.run([shutil.which('node'), '-e', HARNESS + form + layout + frames + checks], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
