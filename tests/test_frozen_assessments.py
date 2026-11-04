from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.assessments import run_assessment
from static_agent_training.accelerated import evaluate_frozen, assert_equivalent
from static_agent_training.experimental.sensory_accelerated import evaluate_frozen as sensory_frozen
from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.experiment import write_json
from static_agent_training.genome import Species
from static_agent_training.seeds import derive
from static_agent_training.workspace_sessions import SessionManager
from static_agent_training.workspace import Workspace


class FrozenAssessmentTests(unittest.TestCase):
    def population(self,root,model,environment='original',varying_goals=True,suffix='.json.gz'):
        config=TrainingConfig(model=model,environment=environment,varying_goals=varying_goals,
            population=2,elites=1,initial_pairs=2,chain_budget=8)
        members=initialize_population(config)
        path=root/'runs/populations'/(model+'-'+environment+suffix)
        write_json(path,dict(schema_version='crt-population-v1',name=model+' '+environment,config=asdict(config),
            weights='inherited-genotype',source_type='best-evaluated-individuals',
            members=[dict(individual=p.to_data(),generation=0,index=i,training_fitness=0,assessment_fitness=0) for i,p in enumerate(members)]))
        return path,members

    def test_frozen_native_calls_shared_tasks_immutable_weights_and_reproducibility(self):
        from static_agent_training import accelerated
        from static_agent_training.experimental import sensory_accelerated
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,pa=self.population(root,'categorical');b,pb=self.population(root,'senses')
            before={p:p.read_bytes() for p in (a,b)}
            with patch('static_agent_training.evolution.reproduce',side_effect=AssertionError('evolution called')), \
                 patch('static_agent_training.engine.run',side_effect=AssertionError('development called')), \
                 patch('static_agent_training.accelerated.evaluate_batch',side_effect=AssertionError('training called')), \
                 patch('static_agent_training.experimental.sensory_accelerated.evaluate_batch',side_effect=AssertionError('training called')), \
                 patch.object(accelerated,'_phase',wraps=accelerated._phase) as old_phase, \
                 patch.object(sensory_accelerated,'_phase',wraps=sensory_accelerated._phase) as senses_phase:
                report=run_assessment([a,b],root/'assessment',episodes=7,seed=19)
                for phase in (old_phase,senses_phase):
                    self.assertEqual(phase.call_count,1)
                    self.assertIs(phase.call_args.kwargs['learning'],False)
                    self.assertEqual(phase.call_args.kwargs['phase'],'assessment')
            self.assertEqual(report['status'],'completed')
            self.assertIs(report['learning'],False);self.assertIs(report['evolution'],False)
            self.assertEqual(report['progress'],dict(completed_animals=4,total_animals=4))
            tasks=[tuple(t) for t in report['tasks']]
            starts,goals=[s for s,g in tasks],[g for s,g in tasks]
            for pop,members,evaluator in zip(report['populations'],(pa,pb),(evaluate_frozen,sensory_frozen)):
                reference=evaluator(members,starts,goals,seeds=[derive(19,'frozen-population-animal',i) for i in range(2)])
                for actual,expected in zip(pop['individuals'],reference):
                    assert_equivalent(actual['assessment'],expected)
                    self.assertEqual([(e['start'],e['goal']) for e in actual['assessment']['episodes']],tasks)
                    self.assertEqual(actual['assessment']['empirical_updates'],0)
                    self.assertEqual(actual['assessment']['conflicts'],0)
                self.assertEqual(Path(pop['snapshot']).read_bytes(),before[Path(pop['source'])])
            self.assertEqual(before,{p:p.read_bytes() for p in (a,b)})
            self.assertFalse((root/'assessment/generations').exists())
            repeated=run_assessment([a,b],root/'repeat',episodes=7,seed=19)
            self.assertEqual(report['tasks'],repeated['tasks'])
            self.assertEqual([p['individuals'] for p in report['populations']],[p['individuals'] for p in repeated['populations']])

    def test_shifted_sensory_and_fixed_goal_assessments_without_rebinding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,_=self.population(root,'senses',varying_goals=False,suffix='.json')
            report=run_assessment([a],root/'shifted',environment='shifted',episodes=5,boundary_mode='terminate')
            self.assertTrue(all(30<=s<=70 and g==50 and s!=g for s,g in report['tasks']))
            self.assertEqual(report['config']['boundary_mode'],'terminate')
            b,_=self.population(root,'categorical',environment='shifted',varying_goals=False)
            with self.assertRaisesRegex(ValueError,'original environment'):
                run_assessment([b],root/'invalid',environment='original')
            self.assertFalse((root/'invalid').exists())
            c,_=self.population(root,'senses',environment='shifted',varying_goals=True)
            with self.assertRaisesRegex(ValueError,'same goal-input mode'):
                run_assessment([a,c],root/'mixed',environment='shifted')
            self.assertFalse((root/'mixed').exists())

    def test_live_workspace_catalog_report_and_command_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,_=self.population(root,'senses')
            manager=SessionManager(root)
            _,args=manager.command('assessment',dict(populations=['populations/'+a.name],environment='shifted',episodes=3),root/'run')
            self.assertNotIn('--training-episodes',args)
            self.assertNotIn('--generations',args)
            self.assertNotIn('--learning',args)
            with self.assertRaises(ValueError):manager.command('assessment',dict(populations=[],learning=True),root/'bad')
            run_assessment([a],root/'runs/frozen',environment='shifted',episodes=3)
            workspace=Workspace(root)
            row=next(r for r in workspace.runs() if r['run']=='frozen')
            self.assertEqual((row['kind'],row['state']),('assessment','completed'))
            detail=workspace.detail('frozen')
            self.assertEqual(detail['kind'],'assessment')
            self.assertEqual(detail['config']['environment'],'shifted')
            self.assertEqual(len(detail['series']),1)
            self.assertIn('report.json',detail['artifacts'])

    def test_reject_accidental_learning_and_preserve_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,members=self.population(root,'senses')
            fake=sensory_frozen(members,[1],[2],seeds=[1,2]);fake[0]['empirical_updates']=1
            with patch('static_agent_training.assessments.sensory_frozen',return_value=fake):
                with self.assertRaisesRegex(AssertionError,'learning update'):
                    run_assessment([a],root/'failed',episodes=1)
            report=json.loads((root/'failed/report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertIn('learning update',report['error'])
