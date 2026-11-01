import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from static_agent_training.baseline import random_task_mean
from static_agent_training.bias_audit import audit_checkpoint, movement_baseline
from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import run_experiment


class AuditBaselineTests(unittest.TestCase):
    def test_exact_baselines_and_directional_controls(self):
        tasks=[(-20,20),(19,20),(-1,0),(5,-5)]
        moves=(-5,-3,-2,-1,0,1,2,3,5)
        self.assertAlmostEqual(movement_baseline(tasks,moves,'clamp'),
                               random_task_mean([s for s,g in tasks],[g for s,g in tasks]))
        self.assertEqual(movement_baseline([(19,20)],(5,),'clamp'),1)
        self.assertEqual(movement_baseline([(19,20)],(-5,),'clamp'),0)
        self.assertEqual(movement_baseline([(19,20)],(5,),'terminate'),0)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy unavailable')
    def test_audit_is_repeatable_and_leaves_checkpoint_untouched(self):
        config=TrainingConfig(population=3,generations=1,episodes=2,assessment_episodes=2,
                              elites=1,initial_pairs=2,chain_budget=8,seed=19)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'
            run_experiment(config,run)
            checkpoint=run/'checkpoint.json.gz'
            before=checkpoint.read_bytes()
            a=audit_checkpoint(run,sample=2,repeats=1,workers=2)
            b=audit_checkpoint(checkpoint,sample=2,repeats=1,workers=2)
            self.assertEqual(a,b)
            self.assertEqual(checkpoint.read_bytes(),before)
            self.assertEqual(a['checkpoint_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(a['population_sample'],2)
            self.assertEqual(len(a['controls']['correct_goal']['per_animal_fixed']),2)
            self.assertTrue(all(g!=replacement for g,replacement in a['goal_permutation'].items()))
            self.assertEqual(set(a['controls']),{'correct_goal','permuted_goal_input','goal_input_removed'})
