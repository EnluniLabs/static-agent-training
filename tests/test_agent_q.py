import math
import random
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from static_agent_training.experimental.agent_q import AgentQ, evaluate
from static_agent_training.experimental.agent_q_accelerated import NativeQ
from static_agent_training.experimental.agent_w_environment import NumberLine
from static_agent_training.experimental.agent_z import Chain, Structure
from static_agent_training.initialize import I, K, pair_id
from static_agent_training.workspace import Workspace


class Scripted:
    def __init__(self, action=0, draws=(.75,)*8):
        self.action, self.draws, self.sequence = action, iter(draws), []

    def choices(self, options, *, weights):
        self.sequence.append('action')
        return [options[self.action]]

    def random(self):
        self.sequence.append('expectation')
        return next(self.draws)


class AgentQTests(unittest.TestCase):
    def setUp(self):
        graph, _ = Structure.develop([(I,K)],max_transitions=1)
        self.source,self.target = pair_id(I,K),pair_id(K,I)
        self.model=AgentQ(graph,[[0.]*9 for _ in graph.pairs],[[[0.]*8]*9 for _ in graph.pairs])
        self.src,self.dst=[self.model.index[x] for x in (self.source,self.target)]
        self.chain=Chain(0,self.source)

    def test_exact_compatibilities_and_uniform_exploration(self):
        q=self.model
        q.embedding[self.dst][1]=math.log(3)
        q.expectations[self.src][0][0]=math.log(3)
        k=q.action_compatibilities(self.chain)
        self.assertAlmostEqual(sum(q.embed(128,0)),1)
        self.assertAlmostEqual(q.embed(128,0)[self.dst],.75)
        self.assertAlmostEqual(q.embed(128,1)[self.dst],.5)
        self.assertAlmostEqual(k[0],.75*.75+.25*.5)
        self.assertAlmostEqual(k[1],.5*.75+.5*.5)
        q.expectations[self.src][0][1]=50
        self.assertAlmostEqual(q.action_compatibilities(self.chain)[0],k[0])
        policy=q.policy(self.chain)
        self.assertAlmostEqual(sum(policy),1)
        self.assertAlmostEqual(policy[0],.95*k[0]/sum(k)+.05/9)
        self.assertNotIn('continuation',q.to_data())
        self.assertEqual(q.structure.successors[self.source],self.target)
        q.embedding[self.dst][0]=-1000
        self.assertEqual(q.policy(self.chain),[1/9]*9)

    def test_sample_timing_and_separate_targets(self):
        for mode in ('off','empirical','structural','both'):
            model=self.model.copy()
            world=NumberLine()
            world.reset((0,15))
            rng=Scripted(action=2)
            execute=world.execute
            def observed(signal):
                self.assertEqual(rng.sequence,['action']+['expectation']*8)
                self.assertEqual(model.to_data(),self.model.to_data())
                return execute(signal)
            with patch.object(world,'execute',side_effect=observed):
                chain,(event,)=model.step([self.chain],world,rng,learning=mode)
            self.assertEqual(chain,(Chain(0,self.target),))
            self.assertEqual(event.expectation,0)
            self.assertNotEqual(event.observation,0)
            self.assertAlmostEqual(event.empirical_error,8*math.log(2))
            self.assertAlmostEqual(event.structural_error,math.log(2))
            if mode in ('structural','both'):
                self.assertEqual(model.embedding[self.dst],[.025]+[0.]*8)
                self.assertEqual(model.embedding[self.src],[-.025]+[0.]*8)
            else:
                self.assertEqual(model.embedding,self.model.embedding)
            for i,pair in enumerate(model.expectations):
                for a,row in enumerate(pair):
                    expected=[0.]*8
                    if i==self.src and a==2 and mode in ('empirical','both'):
                        expected=[.025 if event.observation&(128>>bit) else -.025 for bit in range(8)]
                    self.assertEqual(row,expected)
            self.assertEqual(model.structure.successors,self.model.structure.successors)

    def test_both_updates_can_change_derived_policy(self):
        self.model.embedding[self.dst][1]=1
        self.model.expectations[self.src][0][0]=-.7
        self.model.expectations[self.src][1][0]=.7
        before=self.model.policy(self.chain)
        for mode in ('empirical','structural'):
            model=self.model.copy()
            world=NumberLine()
            world.reset((0,15))
            model.step([self.chain],world,Scripted(draws=(.01,)*8),learning=mode)
            self.assertNotEqual(model.policy(self.chain),before)

    def test_multiple_chains_use_pre_update_values_and_own_feedback(self):
        outcomes=[]
        for chains in ([Chain(0,self.source),Chain(1,self.source)],
                       [Chain(1,self.source),Chain(0,self.source)]):
            model=self.model.copy()
            world=NumberLine()
            world.reset((0,15))
            # Find a move producing different feedback for the next chain.
            a=next(i for i,m in enumerate(world._native_moves()) if m==1)
            _,events=model.step(chains,world,Scripted(a,(.75,)*16),round_index=1)
            self.assertEqual([e.chain.region for e in events],[1,0])
            self.assertEqual([e.observation for e in events],[14,13])
            self.assertEqual([e.empirical_error for e in events],[8*math.log(2)]*2)
            self.assertEqual([e.structural_error for e in events],[math.log(2)]*2)
            self.assertEqual(model.embedding[self.dst][0],.05)
            self.assertEqual(model.expectations[self.src][a][-1],0.)
            outcomes.append(model.to_data())
        self.assertEqual(*outcomes)

    def test_terminal_chain_has_no_feedback_or_action(self):
        world=NumberLine()
        world.reset((0,10))
        self.assertEqual(self.model.step([Chain(0,self.target)],world,Scripted()),((),()))

    def test_native_parity_all_modes_and_partition_sizes(self):
        graph,_=Structure.develop([(K,I),(I,I)],max_transitions=4)
        rng=random.Random(13)
        for regions in (1,2,3):
            q=AgentQ(graph,[[rng.gauss(0,.3) for _ in range(9)] for _ in graph.pairs],
                     [[[rng.gauss(0,.5) for _ in range(8)] for _ in range(9)] for _ in graph.pairs],
                     regions=regions)
            saved=q.to_data()
            for mode in ('off','empirical','structural','both'):
                opts=dict(seed=19,learning=mode,trace=True,return_weights=True)
                tasks=[(-4,3),(8,-5),(0,0)]
                reference=evaluate(q,tasks,**opts)
                with patch.object(AgentQ,'step',side_effect=AssertionError('Python fallback')):
                    native=NativeQ().evaluate(q,tasks,**opts)
                self.assertGreater(native['actions'],0)
                self.assertEqual(native.pop('world_frames'),reference.pop('world_frames'))
                for n,r in zip(native.pop('trace'),reference.pop('trace')):
                    self.assertEqual(len(n),len(r))
                    for a,b in zip(n,r):
                        for k in ('compatibility','empirical_error','structural_error'):
                            self.assertAlmostEqual(a.pop(k),b.pop(k),places=12)
                        self.assertEqual(a,b)
                def compare(left,right):
                    if isinstance(left,list):
                        self.assertEqual(len(left),len(right))
                        for a,b in zip(left,right):
                            compare(a,b)
                    else:
                        self.assertAlmostEqual(left,right,places=12)
                for k in ('embedding','expectations'):
                    compare(native.pop(k),reference.pop(k))
                self.assertEqual(native,reference)
                self.assertEqual(saved,q.to_data())


class AgentQWorkspaceTests(unittest.TestCase):
    def wait(self,workspace,job):
        for _ in range(500):
            row=workspace.sessions.get(job['id'])
            if row['state'] not in ('running','stopping'):
                return row['state']
            time.sleep(.02)
        self.fail(Path(job['log']).read_text())

    def test_launch_extend_seed_assess_and_play(self):
        with tempfile.TemporaryDirectory() as temp:
            w=Workspace(Path(temp))
            cfg=dict(generations=1,episodes=3,population=2,learning='empirical',exploration=.2)
            job=w.post('/api/sessions',dict(kind='agent-q-training',name='Q test',config=cfg))
            self.assertEqual(self.wait(w,job),'completed',Path(job['log']).read_text())
            run=job['run']
            self.assertEqual(w.detail(run)['kind'],'agent-q-training')
            self.assertEqual(w.detail(run)['config']['exploration'],.2)
            extended=w.post('/api/resume-run',dict(run=run,generations=2))
            self.assertEqual(self.wait(w,extended),'completed',Path(extended['log']).read_text())
            self.assertEqual(len(w.detail(run)['series']),2)
            options=w.models.snapshots(run,'agent-q')
            self.assertEqual(len(options),2)
            snapshot=options[0]['id']
            saved=w.models.load(snapshot)[1]
            base=dict(snapshot=snapshot,lifetime_id='agent-q-workspace-test')
            played=w.post('/api/attempt',dict(base,start=-3,goal=4,seed=13,
                                              learning=True,learning_mode='structural'))
            self.assertEqual(played['model'],'agent-q')
            self.assertEqual(played['learning']['empirical_updates'],0)
            self.assertTrue(played['frames'][-1]['done'])
            self.assertEqual(w.models.load(snapshot)[1],saved)
            self.assertEqual(w.post('/api/brain-scan',base)['source'],'Current playground lifetime')
            exported=w.post('/api/save-population',dict(run=run,count=2))
            assessment=w.post('/api/sessions',dict(kind='assessment',config=dict(populations=[exported['id']],episodes=4)))
            self.assertEqual(self.wait(w,assessment),'completed',Path(assessment['log']).read_text())
            self.assertEqual(w.detail(assessment['run'])['report']['populations'][0]['model'],'agent-q')
            seeded=w.post('/api/sessions',dict(kind='agent-q-training',
                          config=cfg | {'initial_population':exported['id']}))
            self.assertEqual(self.wait(w,seeded),'completed',Path(seeded['log']).read_text())


    def test_stop_and_resume_active_q_session(self):
        from static_agent_training.experimental.agent_z_experiment import read_z_artifact

        with tempfile.TemporaryDirectory() as temp:
            w=Workspace(Path(temp))
            cfg=dict(generations=1000,episodes=12,population=3,learning='off')
            job=w.post('/api/sessions',dict(kind='agent-q-training',config=cfg))
            active=job
            path=Path(job['directory'])/'checkpoint.json.gz'
            try:
                deadline=time.monotonic()+10
                while not path.is_file() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue(path.is_file(),Path(job['log']).read_text())
                w.post('/api/stop',dict(id=job['id']))
                self.assertEqual(self.wait(w,job),'stopped')
                count=len(read_z_artifact(path)['history'])
                active=w.post('/api/resume-run',dict(run=job['run']))
                deadline=time.monotonic()+10
                while len(read_z_artifact(path)['history'])<=count and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertGreater(len(read_z_artifact(path)['history']),count)
            finally:
                if w.sessions.get(active['id'])['state']=='running':
                    w.post('/api/stop',dict(id=active['id']))
                    self.assertEqual(self.wait(w,active),'stopped')
