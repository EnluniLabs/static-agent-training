from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.comparison import paired_summary, run_study, report_for
from static_agent_training.evolution import TrainingConfig
from static_agent_training.mutation import MutationConfig


class ComparisonTests(unittest.TestCase):
    def test_seed_level_effect_interval(self):
        self.assertIsNone(paired_summary([]))
        self.assertIsNone(paired_summary([.01])['interval95'])
        result=paired_summary([.01,.02,.03])
        self.assertAlmostEqual(result['mean_percentage_points'],2)
        self.assertLess(result['interval95'][0],0)
        self.assertGreater(result['interval95'][1],4)

    def test_learning_treatment_is_matched_and_disables_all_lifetime_updates(self):
        none=MutationConfig(weight_probability=0,subtree_probability=0,term_probability=0,
                            insertion_probability=0,deletion_probability=0)
        config=TrainingConfig(population=4,generations=2,episodes=3,assessment_episodes=3,
                              elites=1,initial_pairs=2,chain_budget=8,mutation=none)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'learning-study'
            result=run_study(directory,config,seeds=[19],benchmark_episodes=20,
                             runs_root=root/'runs',treatment='learning')
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['treatment'],'learning')
            self.assertEqual(result['control_label'],'Learning off')
            self.assertEqual(result['treatment_label'],'Learning on')
            self.assertTrue(result['pairs'][0]['initialization_verified'])
            self.assertTrue(result['pairs'][0]['tasks_verified'])
            manifest=json.loads((directory/'manifest.json').read_text())
            configs=[];updates={}
            for arm in manifest['arms']:
                run=Path(arm['directory'])
                saved=json.loads((run/'config.json').read_text())['config']
                self.assertFalse(saved['speciation'])
                self.assertEqual(saved['learning'],arm['enabled'])
                saved.pop('learning');configs.append(saved)
                records=[json.loads(p.read_text()) for p in sorted((run/'generations').glob('*.json'))]
                empirical=sum(i['training']['empirical_updates'] for r in records for i in r['individuals'])
                conflicts=sum(i['training']['conflicts'] for r in records for i in r['individuals'])
                updates[arm['enabled']]=empirical
                if not arm['enabled']:
                    self.assertEqual(empirical,0);self.assertEqual(conflicts,0)
                self.assertEqual(sum(i['assessment']['empirical_updates'] for r in records for i in r['individuals']),0)
            self.assertEqual(configs[0],configs[1])
            self.assertGreater(updates[True],0)
            resumed=run_study(directory,resume=True)
            self.assertEqual(result['pairs'],resumed['pairs'])
            self.assertEqual(resumed['treatment'],'learning')

    def test_matched_study_final_benchmarks_and_resume(self):
        none=MutationConfig(weight_probability=0,subtree_probability=0,term_probability=0,
                            insertion_probability=0,deletion_probability=0)
        config=TrainingConfig(population=4,generations=2,episodes=3,assessment_episodes=3,
                              elites=1,initial_pairs=2,chain_budget=8,max_species=2,mutation=none)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directory=root/'study'
            report=run_study(directory,config,seeds=[11,12],benchmark_episodes=20,runs_root=root/'runs')
            self.assertEqual(report['status'],'completed')
            self.assertEqual(len(report['arms']),4)
            self.assertEqual(report['paired_effect']['seeds'],2)
            for pair in report['pairs']:
                self.assertTrue(pair['initialization_verified'])
                self.assertTrue(pair['tasks_verified'])
                self.assertEqual(pair['common_generations'],2)
            for arm in report['arms']:
                self.assertEqual(arm['state'],'completed')
                self.assertEqual(arm['final_benchmark']['episodes'],20)
            manifest=json.loads((directory/'manifest.json').read_text())
            for seed in [11,12]:
                arms=[a for a in manifest['arms'] if a['seed']==seed]
                configs=[json.loads((Path(a['directory'])/'config.json').read_text())['config'] for a in arms]
                self.assertFalse(configs[0].pop('speciation'))
                self.assertTrue(configs[1].pop('speciation'))
                self.assertEqual(configs[0],configs[1])
            resumed=run_study(directory,resume=True)
            self.assertEqual(report['pairs'],resumed['pairs'])
            # A changed task checksum must invalidate the study rather than report a result.
            arm=manifest['arms'][0]
            file=Path(arm['directory'])/'generations/00000.json'
            record=json.loads(file.read_text())
            record['summary']['task_id']='invalid'
            file.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Unmatched tasks'):
                report_for(manifest,'completed')
