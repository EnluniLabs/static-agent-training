"""Connection-driven placement should expose neighborhoods without moving sensors."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node required for graph layout checks')
class BrainLayoutTests(unittest.TestCase):
    def test_connections_localize_and_camera_redraws_reuse_positions(self):
        source = (Path(__file__).resolve().parents[1]/'static_agent_training/web/workspace.js').read_text()
        layout = source[source.index('function brainHash('):source.index('const brain =')]
        checks = r"""
const assert=require('node:assert/strict');
const nodes=Array.from({length:24},(_,i)=>({id:'state:'+i,kind:'state'}));
const edges=[];
for(let group=0;group<4;group++)for(let i=0;i<6;i++){
 edges.push({source:'state:'+(group*6+i),target:'state:'+(group*6+(i+1)%6),kind:'successor',value:1});
}
const graph={nodes,edges};
const initial=brainLayout({nodes,edges:[]}),positions=brainLayout(graph);
const distance=(map,a,b)=>{const p=map.get(a),q=map.get(b);return Math.hypot(p.x-q.x,p.y-q.y,p.z-q.z);};
const average=(map)=>edges.reduce((sum,e)=>sum+distance(map,e.source,e.target),0)/edges.length;
assert.ok(average(positions)<average(initial)*.65,`${average(initial)} -> ${average(positions)}`);
let within=0,between=0,nw=0,nb=0;
for(let i=0;i<24;i++)for(let j=i+1;j<24;j++){
 const d=distance(positions,nodes[i].id,nodes[j].id);
 if(Math.floor(i/6)===Math.floor(j/6)){within+=d;nw++;}else{between+=d;nb++;}
}
assert.ok(within/nw<between/nb,'connected groups should form spatial neighborhoods');
assert.equal(brainLayout(graph),positions,'rotation and playback must reuse solved positions');
const reversed=brainLayout({nodes:[...nodes].reverse(),edges:[...edges].reverse()});
for(const n of nodes){
 const p=positions.get(n.id),q=reversed.get(n.id);
 for(const axis of ['x','y','z'])assert.ok(Number.isFinite(p[axis])&&Math.abs(p[axis]-q[axis])<1e-12);
 assert.ok(Math.abs(p.x)<1&&Math.abs(p.y)<1&&Math.abs(p.z)<1.3);
}
const input={id:'input:1',kind:'input',pixel:[0,0]},output={id:'out',kind:'output'};
const pinned={nodes:[...nodes,input,output],sensory_geometry:{interface:'signals',width:16,height:16},edges:[...edges,
 {source:input.id,target:nodes[0].id,kind:'embedding',value:-5},
 {source:nodes[1].id,target:output.id,kind:'output',value:.9}]};
const pinnedBefore=brainLayout({...pinned,edges:[]}),pinnedAfter=brainLayout(pinned);
for(const n of [input,output])for(const axis of ['x','y','z'])assert.equal(pinnedBefore.get(n.id)[axis],pinnedAfter.get(n.id)[axis]);
// The exact original site distribution is retained for each internal node kind.
const coordinates=(map,kind)=>[...map.values()].filter(p=>p.n.kind===kind).map(p=>[p.x,p.y,p.z].join(',')).sort();
assert.deepEqual(coordinates(positions,'state'),coordinates(initial,'state'));
const denseNodes=Array.from({length:96},(_,i)=>({id:'dense:'+i,kind:'state'}));
const dense={nodes:denseNodes,edges:denseNodes.map((n,i)=>({source:n.id,target:denseNodes[(i+1)%96].id,kind:'successor',value:1}))};
const denseBefore=brainLayout({...dense,edges:[]}),denseAfter=brainLayout(dense);
assert.deepEqual(coordinates(denseAfter,'state'),coordinates(denseBefore,'state'),'even a fully connected chain fills the original outline');
const mixed={nodes:[...nodes,...Array.from({length:16},(_,i)=>({id:'prediction:'+i,kind:'prediction'}))],
 edges:[...edges,...Array.from({length:16},(_,i)=>({source:nodes[i].id,target:'prediction:'+i,kind:'prediction',value:.9}))]};
const mixedBefore=brainLayout({...mixed,edges:[]}),mixedAfter=brainLayout(mixed);
for(const kind of ['state','prediction'])assert.deepEqual(coordinates(mixedAfter,kind),coordinates(mixedBefore,kind));
console.log('connected length',average(initial),'->',average(positions),'within/between',within/nw,between/nb);
"""
        result = subprocess.run([shutil.which('node'), '-e', layout+checks], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
