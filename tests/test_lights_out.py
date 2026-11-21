from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.lights_out import MASKS, board_bits, solutions, stimulus, LightsOutWorld, StimulusLightsOut, random_success
from static_agent_training.environment import Output
from static_agent_training.experimental.signals import Action
from static_agent_training.evolution import TrainingConfig, initialize_population, partitions, generation_tasks
from static_agent_training.experiment import run_experiment
from static_agent_training.genome import develop, Species
from static_agent_training.initialize import I, K, pair_id
from static_agent_training.accelerated import evaluate_batch, assert_equivalent
from static_agent_training.evaluation import evaluate
from static_agent_training.experimental.sensory_evolution import evaluate as sensory_evaluate
from static_agent_training.experimental.sensory_accelerated import evaluate_batch as sensory_batch, learned_mappings
from static_agent_training.experimental.sensory_genome import SensoryMappings
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.mappings import InheritedMappings
from static_agent_training.populations import save_population, read_population
from static_agent_training.assessments import run_assessment
from static_agent_training.benchmark import compare
from static_agent_training.demo import export_demo
from static_agent_training.play import attempt
from static_agent_training.workspace import Workspace
from static_agent_training.seeds import derive


class LightsOutTests(unittest.TestCase):
    def config(self,**kwargs):
        return replace(TrainingConfig(environment='lights-out',model='senses',backend='native',population=2,
            elites=1,initial_pairs=2,chain_budget=8,generations=2,episodes=3,assessment_episodes=3),**kwargs)

    def test_masks_solutions_reversibility_and_terminal_feedback(self):
        self.assertEqual(MASKS[0],11);self.assertEqual(MASKS[4],186)
        self.assertEqual(len(solutions()),512)
        self.assertLessEqual(max(map(len,solutions().values())),9)
        for board,path in solutions().items():
            current=board
            for cell in path:current^=MASKS[cell]
            self.assertEqual(current,0)
        for seed in (0,19):
            raw=StimulusLightsOut(encoding_seed=seed)
            world=LightsOutWorld(encoding_seed=seed)
            for board in range(1,512):
                for button in range(9):
                    raw.reset((board,0));world.reset(start=board,goal=0)
                    raw.step(Action(button));world.emit(Output(f'move_{button:02d}'))
                    self.assertEqual(raw.position,world.diagnostic_state()['position'])
            world.reset(start=MASKS[0],goal=0)
            expectation=next(t for t,b in world.diagnostic_state()['expectation_decoding'].items() if b==0)
            world.emit(Output(expectation),source='q')
            cell0=world.cells.index(0)
            final=world.emit(Output(f'move_{cell0:02d}'),source='r')
            self.assertTrue(final.success)
            self.assertEqual(len(final.feedback),1);self.assertTrue(final.feedback[0].correct)
        self.assertEqual(stimulus(1),bytes([1]+[0]*8))
        self.assertAlmostEqual(random_success(11,1),1/9)
        self.assertEqual(random_success(0,12),1)

    def parent(self,model):
        world=LightsOutWorld()
        parent=develop([(I,K)],[o.token for o in world.observation_tokens],[o.token for o in world.output_tokens],
                       budgets=20,senses=FixedSenses() if model=='senses' else None)
        entry=pair_id(I,K);q=parent.repertoire.successors[entry];button=world.cells.index(0)
        if model=='senses':
            data=parent.inherited.to_parameters();ids=parent.inherited.pair_ids
            data['bias']=[1000.0 if x==entry else -1000.0 for x in ids]
            data['weights']=[[0.0]*len(ids) for _ in range(257)]
            for x in ids:
                data['gates'][x]=[-1000.0]*10;data['gates'][x][9 if x==q else button]=1000.0
            inherited=SensoryMappings(ids,parent.inherited.senses,data)
        else:
            data=parent.inherited.to_data()
            wrong=next(t for t,b in world.diagnostic_state()['expectation_decoding'].items() if b==1)
            for row in data['embedding'].values():
                for x in row:row[x]=1000.0 if x==entry else -1000.0
            for x,row in data['output'].items():
                for token in row:row[token]=1000.0 if token==(wrong if x==q else f'move_{button:02d}') else -1000.0
            inherited=InheritedMappings.from_data(data)
        return replace(parent,inherited=inherited)

    def test_native_reference_learning_full_weights_and_frozen_assessment(self):
        training=[11,23,38];assessment=[89,186,308]
        for model in ('senses','categorical'):
            parent=self.parent(model);before=parent.to_data();cfg=self.config(model=model,inherit_lifetime=True)
            function=sensory_batch if model=='senses' else evaluate_batch
            results,weights=function([parent],training,assessment,training_goals=[0]*3,assessment_goals=[0]*3,
                seeds=[19],environment='lights-out',return_weights=True)
            if model=='senses':
                reference,detail,learned=sensory_evaluate(parent,list(zip(training,[0]*3)),list(zip(assessment,[0]*3)),seed=19,config=cfg)
                assert_equivalent(learned_mappings(parent,weights,0).to_data(),learned.inherited.to_data())
            else:
                reference,detail=evaluate(parent,training,assessment,seed=19,training_goals=[0]*3,assessment_goals=[0]*3,environment='lights-out')
                for i,signal in enumerate(parent.inherited.observations):
                    for j,q in enumerate(parent.inherited.pair_ids):self.assertAlmostEqual(weights[0][0,i,j],detail['learned_embedding'][signal][q])
                for i,q in enumerate(parent.inherited.pair_ids):
                    for j,token in enumerate(parent.inherited.outputs):self.assertAlmostEqual(weights[1][0,i,j],detail['learned_output'][q][token])
            assert_equivalent(results[0],reference)
            self.assertGreater(results[0]['training']['empirical_updates'],0)
            self.assertGreater(results[0]['training']['conflicts'],0)
            self.assertEqual(results[0]['assessment']['empirical_updates'],0)
            self.assertEqual(results[0]['assessment']['conflicts'],0)
            self.assertEqual(parent.to_data(),before)

    def test_partitioned_tasks_are_solvable_and_not_board_id_distances(self):
        split=partitions(19,include_zero=True,environment='lights-out')
        self.assertFalse(set(split['training'])&set(split['assessment']))
        self.assertEqual(set(split['training'])|set(split['assessment']),set(range(1,512)))
        for varying in (True,False):
            starts,assess,goals,ag=generation_tasks(self.config(varying_goals=varying,distance_distribution='normal'),2,split)
            self.assertTrue(all(s in split['training'] and s in solutions() for s in starts))
            self.assertTrue(all(s in split['assessment'] for s in assess))
            self.assertEqual(set(goals+ag),{0})

    def test_training_resume_transfer_benchmark_assessment_and_recorded_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in ('senses','categorical'):
                cfg=self.config(model=model)
                full=run_experiment(cfg,root/model/'full')
                run_experiment(cfg,root/model/'resume',until_generation=1)
                resumed=run_experiment(cfg,root/model/'resume',resume=True)
                self.assertEqual(full,resumed)
                saved=root/(model+'.json.gz');save_population(root/model/'full',saved,count=2)
                snapshot=saved.read_bytes()
                report=run_assessment([saved],root/('assessment-'+model),environment='lights-out',episodes=5)
                self.assertTrue(all(s!=0 and g==0 for s,g in report['tasks']))
                self.assertEqual(snapshot,saved.read_bytes())
                result=compare(root/model/'full',episodes=5)
                self.assertEqual(result['environment'],'lights-out')
                _,bundle=export_demo(root/model/'full')
                playback=attempt(bundle,11,0,19)
                self.assertEqual(playback['frames'][0]['board'],board_bits(11))
                self.assertEqual(playback['environment'],'lights-out')
                record=json.loads((root/model/'full/generations/00000.json').read_text())
                self.assertNotIn('training_distances',record['task_sampling'])
            # Existing number-line sensory candidates can see puzzle boards without a new apparatus.
            oldcfg=self.config(environment='original',generations=1)
            run_experiment(oldcfg,root/'runs/old')
            save_population(root/'runs/old',root/'runs/populations/old.json.gz',count=2)
            newcfg=replace(self.config(generations=1),initial_population=str(root/'runs/populations/old.json.gz'))
            old=read_population(root/'runs/populations/old.json.gz')
            inherited=initialize_population(newcfg)
            self.assertEqual(inherited[0].to_data(),Species.from_data(old['members'][0]['individual']).to_data())
            workspace=Workspace(root)
            options=workspace.models.snapshots('old','senses')
            play=workspace.models.attempt(dict(snapshot=options[0]['id'],start=11,goal=0,seed=19,world='lights-out'))
            self.assertEqual(play['environment'],'lights-out')
            self.assertEqual(play['frames'][0]['board'],board_bits(11))

class SimpleSwitchTests(LightsOutTests):
    environment = "switches-2x2"
    """Exercise the same training/save/resume/frozen workflows on the simpler variant."""
    def config(self,**kwargs):
        kwargs.setdefault('environment',self.environment)
        return super().config(**kwargs)

    def parent(self,model):
        # The shared fixture builds its observation/output vocabulary from this world.
        import unittest.mock
        from functools import partial
        with unittest.mock.patch(__name__+'.LightsOutWorld',partial(LightsOutWorld,environment=self.environment)):
            return super().parent(model)

    def test_masks_solutions_reversibility_and_terminal_feedback(self):
        self.assertEqual(len(solutions(self.environment)),16)
        self.assertEqual(max(map(len,solutions(self.environment).values())),4)
        self.assertAlmostEqual(random_success(1,1,self.environment),2/9)
        for seed in (0,19):
            raw=StimulusLightsOut(environment=self.environment,encoding_seed=seed)
            world=LightsOutWorld(environment=self.environment,encoding_seed=seed)
            self.assertEqual(sorted(raw.moves),[0,1,1,2,2,4,4,8,8])
            for board in range(1,16):
                for button in range(9):
                    raw.reset((board,0));world.reset(start=board,goal=0)
                    raw.step(Action(button));world.emit(Output(f'move_{button:02d}'))
                    self.assertEqual(raw.position,board^raw.moves[button])
                    self.assertEqual(raw.position,world.diagnostic_state()['position'])
                    self.assertEqual(len(raw.board),4)

    def test_partitioned_tasks_are_solvable_and_not_board_id_distances(self):
        split=partitions(19,include_zero=True,environment=self.environment)
        self.assertEqual(len(split['training']),12)
        self.assertEqual(len(split['assessment']),3)
        self.assertEqual(set(split['training'])|set(split['assessment']),set(range(1,16)))
        self.assertFalse(set(split['training'])&set(split['assessment']))

    def test_native_reference_learning_full_weights_and_frozen_assessment(self):
        training=[1,3,5];assessment=[7,11,15]
        for model in ('senses','categorical'):
            parent=self.parent(model);before=parent.to_data();cfg=self.config(model=model,inherit_lifetime=True)
            function=sensory_batch if model=='senses' else evaluate_batch
            results,weights=function([parent],training,assessment,training_goals=[0]*3,assessment_goals=[0]*3,
                seeds=[19],environment=self.environment,return_weights=True)
            if model=='senses':
                reference,detail,learned=sensory_evaluate(parent,list(zip(training,[0]*3)),list(zip(assessment,[0]*3)),seed=19,config=cfg)
                assert_equivalent(learned_mappings(parent,weights,0).to_data(),learned.inherited.to_data())
            else:
                reference,detail=evaluate(parent,training,assessment,seed=19,training_goals=[0]*3,assessment_goals=[0]*3,environment=self.environment)
                for i,signal in enumerate(parent.inherited.observations):
                    for j,q in enumerate(parent.inherited.pair_ids):self.assertAlmostEqual(weights[0][0,i,j],detail['learned_embedding'][signal][q])
                for i,q in enumerate(parent.inherited.pair_ids):
                    for j,token in enumerate(parent.inherited.outputs):self.assertAlmostEqual(weights[1][0,i,j],detail['learned_output'][q][token])
            assert_equivalent(results[0],reference)
            self.assertGreater(results[0]['training']['empirical_updates'],0)
            self.assertGreater(results[0]['training']['conflicts'],0)
            self.assertEqual(results[0]['assessment']['empirical_updates'],0)
            self.assertEqual(results[0]['assessment']['conflicts'],0)
            self.assertEqual(parent.to_data(),before)


    def test_training_resume_transfer_benchmark_assessment_and_recorded_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in ('senses','categorical'):
                cfg=self.config(model=model)
                full=run_experiment(cfg,root/model/'full')
                run_experiment(cfg,root/model/'resume',until_generation=1)
                resumed=run_experiment(cfg,root/model/'resume',resume=True)
                self.assertEqual(full,resumed)
                saved=root/(model+'.json.gz');save_population(root/model/'full',saved,count=2)
                snapshot=saved.read_bytes()
                report=run_assessment([saved],root/('assessment-'+model),environment=self.environment,episodes=5)
                self.assertTrue(all(s!=0 and g==0 for s,g in report['tasks']))
                self.assertEqual(snapshot,saved.read_bytes())
                result=compare(root/model/'full',episodes=5)
                self.assertEqual(result['environment'],self.environment)
                _,bundle=export_demo(root/model/'full')
                playback=attempt(bundle,11,0,19)
                self.assertEqual(playback['frames'][0]['board'],board_bits(11,self.environment))
                self.assertEqual(playback['environment'],self.environment)
                record=json.loads((root/model/'full/generations/00000.json').read_text())
                self.assertNotIn('training_distances',record['task_sampling'])
            # Existing number-line sensory candidates can see puzzle boards without a new apparatus.
            oldcfg=self.config(environment='original',generations=1)
            run_experiment(oldcfg,root/'runs/old')
            save_population(root/'runs/old',root/'runs/populations/old.json.gz',count=2)
            newcfg=replace(self.config(generations=1),initial_population=str(root/'runs/populations/old.json.gz'))
            old=read_population(root/'runs/populations/old.json.gz')
            inherited=initialize_population(newcfg)
            self.assertEqual(inherited[0].to_data(),Species.from_data(old['members'][0]['individual']).to_data())
            workspace=Workspace(root)
            options=workspace.models.snapshots('old','senses')
            play=workspace.models.attempt(dict(snapshot=options[0]['id'],start=11,goal=0,seed=19,world=self.environment))
            self.assertEqual(play['environment'],self.environment)
            self.assertEqual(play['frames'][0]['board'],board_bits(11,self.environment))


class LightsOut2x2Tests(SimpleSwitchTests):
    environment = "lights-out-2x2"

    def test_masks_solutions_reversibility_and_terminal_feedback(self):
        from static_agent_training.lights_out import press_mask
        self.assertEqual([press_mask(i,self.environment) for i in range(4)],[7,11,13,14])
        paths=solutions(self.environment)
        self.assertEqual(set(paths),set(range(16)))
        self.assertEqual(max(map(len,paths.values())),4)
        for board,path in paths.items():
            state=board
            for cell in path:state ^= press_mask(cell,self.environment)
            self.assertEqual(state,0)
        self.assertAlmostEqual(random_success(7,1,self.environment),2/9)
        self.assertEqual(random_success(1,1,self.environment),0)
        for seed in (0,19):
            raw=StimulusLightsOut(environment=self.environment,encoding_seed=seed)
            world=LightsOutWorld(environment=self.environment,encoding_seed=seed)
            simple=StimulusLightsOut(environment='switches-2x2',encoding_seed=seed)
            self.assertEqual(raw.cells,simple.cells)
            self.assertEqual(sorted(raw.moves),[0,7,7,11,11,13,13,14,14])
            for board in range(1,16):
                for button in range(9):
                    raw.reset((board,0));world.reset(start=board,goal=0)
                    transition=raw.step(Action(button));world.emit(Output(f'move_{button:02d}'))
                    self.assertEqual(raw.position,board^raw.moves[button])
                    self.assertEqual(raw.position,world.diagnostic_state()['position'])
                    self.assertEqual(transition.success,raw.position==0)
                    self.assertEqual(len(raw.board),4)
            # A correct prediction is checked on the winning board as well.
            world.reset(start=7,goal=0)
            expect=next(t for t,b in world.diagnostic_state()['expectation_decoding'].items() if b==0)
            world.emit(Output(expect),source='q')
            final=world.emit(Output(f'move_{world.cells.index(0):02d}'),source='r')
            self.assertTrue(final.success)
            self.assertTrue(final.feedback[0].correct)
            # Different mechanics have distinct categorical vocabularies.
            switches=LightsOutWorld(environment='switches-2x2')
            self.assertFalse(set(o.token for o in switches.observation_tokens if o.token.startswith('obs_')) &
                             set(o.token for o in world.observation_tokens if o.token.startswith('obs_')))
