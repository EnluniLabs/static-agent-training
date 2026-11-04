from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import load_checkpoint, run_experiment, write_json


class BackendCompatibilityTests(unittest.TestCase):
    def test_new_configs_reject_removed_backend_but_old_configs_load_without_mutation(self):
        with self.assertRaises(ValueError):
            TrainingConfig(backend='cuda')
        saved = asdict(TrainingConfig())
        saved['backend'] = 'cuda'
        with self.assertWarnsRegex(RuntimeWarning, 'native CPU'):
            restored = TrainingConfig.from_data(saved)
        self.assertEqual(restored.backend, 'native')
        self.assertEqual(saved['backend'], 'cuda')

    def test_legacy_checkpoint_resumes_on_cpu_and_preserves_weights_history_and_provenance(self):
        cfg = TrainingConfig(model='senses',backend='native',population=2,elites=1,generations=2,
                             episodes=3,assessment_episodes=3,initial_pairs=2,chain_budget=8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run_experiment(cfg,root/'full')
            run_experiment(cfg,root/'legacy',until_generation=1)
            state=load_checkpoint(root/'legacy')
            state['config']['backend']='cuda'
            write_json(root/'legacy/checkpoint.json.gz',state)
            history_path=root/'legacy/generations/00000.json'
            original=history_path.read_bytes()
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                resumed_cfg=TrainingConfig.from_data(state['config'])
                run_experiment(resumed_cfg,root/'legacy',resume=True)
            self.assertTrue(any('native CPU' in str(w.message) for w in notices))
            self.assertEqual(history_path.read_bytes(),original)
            resumed=load_checkpoint(root/'legacy');full=load_checkpoint(root/'full')
            self.assertEqual(resumed['population'],full['population'])
            self.assertEqual(resumed['history'],full['history'])
            self.assertEqual(resumed['config']['backend'],'native')
            self.assertEqual(resumed['backend_migration']['previous_backend'],'cuda')
            self.assertEqual(json.loads((root/'legacy/config.json').read_text())['backend_migration'],resumed['backend_migration'])
