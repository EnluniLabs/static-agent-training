from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.comparison import run_study, report_for
from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import run_experiment
from static_agent_training.populations import save_population
from static_agent_training.workspace_sessions import SessionManager


class StartingPopulationBenchmarkTests(unittest.TestCase):
    def config(self,**kwargs):
        return replace(TrainingConfig(model='senses',environment='shifted',backend='native',population=2,
            elites=1,generations=2,episodes=2,assessment_episodes=2,initial_pairs=2,chain_budget=8),**kwargs)

    def candidates(self,root,seed):
        run=root/f'source-{seed}'
        run_experiment(self.config(seed=seed,environment='original',generations=1),run)
        path=root/f'candidates-{seed}.json.gz'
        save_population(run,path,count=2,name=f'Candidates {seed}')
        return path

    def check_report(self,report):
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['treatment'],'starting-population')
        self.assertEqual(report['pairs'][0]['common_generations'],2)
        self.assertTrue(report['pairs'][0]['tasks_verified'])
        self.assertEqual(len(report['arms']),2)
        for arm in report['arms']:
            self.assertEqual((arm['completed'],arm['target']),(2,2))
            self.assertEqual(len(arm['curve']),2)
            self.assertEqual(arm['final_benchmark']['episodes'],10)
            self.assertTrue(arm['learning'])

    def test_fresh_vs_saved_and_resume_after_original_collection_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=self.candidates(root,11);original=saved.read_bytes()
            report=run_study(root/'study',self.config(),seeds=[48],benchmark_episodes=10,
                runs_root=root/'arms',treatment='starting-population',population_b=saved)
            self.check_report(report)
            self.assertEqual(saved.read_bytes(),original)
            manifest=json.loads((root/'study/manifest.json').read_text())
            self.assertEqual(manifest['source_a']['kind'],'fresh')
            self.assertEqual(manifest['source_b']['original_sha256'],hashlib.sha256(original).hexdigest())
            self.assertTrue(Path(manifest['population_b']).is_file())
            configs=[json.loads((Path(a['directory'])/'config.json').read_text())['config'] for a in manifest['arms']]
            # Same budgets, tasks and algorithm settings; starting-population coordinates differ.
            for key in configs[0]:
                if key not in ('initial_population','initialization_seed'):
                    self.assertEqual(configs[0][key],configs[1][key],key)
            histories=[[json.loads(p.read_text()) for p in sorted((Path(a['directory'])/'generations').glob('*.json'))] for a in manifest['arms']]
            for a,b in zip(*histories):
                for key in ('training_starts','assessment_starts','training_goals','assessment_goals'):
                    self.assertEqual(a[key],b[key])
                self.assertEqual(a['summary']['task_id'],b['summary']['task_id'])
                self.assertEqual(len(a['reproduction']),len(b['reproduction']))
            finals=[json.loads((Path(a['directory'])/'demo/benchmark.json').read_text()) for a in manifest['arms']]
            self.assertEqual([[(o['start'],o['goal']) for o in f['outcomes']] for f in finals][0],
                             [[(o['start'],o['goal']) for o in f['outcomes']] for f in finals][1])
            saved.unlink()
            resumed=run_study(root/'study',resume=True)
            self.assertEqual(resumed['pairs'],report['pairs'])
            frozen=Path(manifest['population_b']);frozen.write_bytes(frozen.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'snapshot has changed'):
                run_study(root/'study',resume=True)

    def test_two_saved_sets_keep_independent_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=self.candidates(root,11);b=self.candidates(root,12)
            report=run_study(root/'study',self.config(),seeds=[48],benchmark_episodes=10,
                runs_root=root/'arms',treatment='starting-population',population_a=a,population_b=b)
            self.check_report(report)
            self.assertEqual([r['starting_source']['name'] for r in report['arms']],['Candidates 11','Candidates 12'])
            self.assertTrue(report['pairs'][0]['starting_populations_differ'])

    def test_two_fresh_starts_are_independent_but_tasks_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=run_study(root/'study',self.config(model='categorical'),seeds=[48],benchmark_episodes=10,
                runs_root=root/'arms',treatment='starting-population')
            self.check_report(report)
            self.assertTrue(report['pairs'][0]['starting_populations_differ'])
            self.assertNotEqual(report['arms'][0]['initialization_seed'],report['arms'][1]['initialization_seed'])
            manifest=json.loads((root/'study/manifest.json').read_text())
            # Tampered task IDs fail even though different populations are allowed.
            path=sorted((Path(manifest['arms'][1]['directory'])/'generations').glob('*.json'))[0]
            data=json.loads(path.read_text());data['summary']['task_id']='incorrect';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Unmatched tasks'):
                report_for(manifest,'completed')

    def test_setup_command_validates_both_sources_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manager=SessionManager(root)
            source=self.candidates(root,11)
            path=root/'runs/populations/candidates.json.gz';path.parent.mkdir(parents=True);path.write_bytes(source.read_bytes())
            _,args=manager.command('starting-population-benchmark',dict(model='senses',environment='shifted',
                population_a='populations/candidates.json.gz',population_b='',seed=49,learning=False,
                inherit_lifetime=True,speciation=True),root/'new')
            self.assertIn('--population-a',args)
            self.assertNotIn('--population-b',args)
            self.assertIn('--no-learning',args)
            self.assertIn('--inherit-lifetime',args)
            self.assertIn('--speciation',args)
            self.assertEqual(args[args.index('--treatment')+1],'starting-population')
            self.assertEqual(args[args.index('--seeds')+1],'49')
            with self.assertRaisesRegex(ValueError,'same model'):
                manager.command('starting-population-benchmark',dict(model='categorical',population_b='populations/candidates.json.gz'),root/'bad')

    def test_interrupted_paired_evolution_resumes_from_each_arm_checkpoint(self):
        from static_agent_training.experiment import load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=self.candidates(root,11);cfg=self.config()
            with patch('static_agent_training.comparison.subprocess.Popen',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_study(root/'study',cfg,seeds=[48],benchmark_episodes=10,runs_root=root/'arms',
                              treatment='starting-population',population_a=saved)
            manifest=json.loads((root/'study/manifest.json').read_text())
            self.assertEqual(json.loads((root/'study/report.json').read_text())['status'],'stopped')
            retained=[]
            for arm in manifest['arms']:
                arm_config=replace(TrainingConfig.from_data(manifest['config']),seed=arm['seed'],
                    initial_population=arm['initial_population'],initialization_seed=arm['initialization_seed'])
                path=Path(arm['directory'])
                run_experiment(arm_config,path,until_generation=1)
                retained.append((path/'generations/00000.json').read_bytes())
                self.assertEqual(load_checkpoint(path)['next_generation'],1)
            saved.unlink()
            resumed=run_study(root/'study',resume=True)
            self.check_report(resumed)
            for arm,original in zip(manifest['arms'],retained):
                self.assertEqual((Path(arm['directory'])/'generations/00000.json').read_bytes(),original)
                self.assertEqual(load_checkpoint(Path(arm['directory']))['next_generation'],2)
