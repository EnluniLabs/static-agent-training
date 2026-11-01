import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.demo import export_demo
from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import run_experiment


class DemoTests(unittest.TestCase):
    def test_export_replays_frozen_assessment_and_selects_training_champion(self):
        config = TrainingConfig(population=3, generations=3, episodes=3,
                                assessment_episodes=3, elites=1, initial_pairs=2,
                                chain_budget=8, seed=19)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            state = run_experiment(config, root)
            expected = max(state['history'], key=lambda r: r['best'])
            with patch('static_agent_training.engine.run', side_effect=AssertionError('runtime reduction')):
                destination, bundle = export_demo(root)
            self.assertEqual(bundle['generation'], expected['generation'] + 1)
            self.assertEqual(bundle['training_success'], expected['best'])
            with gzip.open(root / 'traces' / f"{expected['generation']:05d}-champion.json.gz", 'rt') as f:
                trace = json.load(f)
            self.assertEqual(bundle['learned_embedding'], trace['learned_embedding'])
            self.assertEqual(bundle['learned_output'], trace['learned_output'])
            for episode, saved in zip(bundle['episodes'], trace['result']['assessment']['episodes']):
                frames = episode['frames']
                self.assertEqual(episode['start'], saved['start'])
                self.assertEqual(frames[0]['position'], saved['start'])
                self.assertEqual(frames[-1]['success'], saved['success'])
                self.assertEqual(frames[-1]['reason'], saved['reason'])
                self.assertEqual(frames[-1]['actions'], saved['actions'])
                self.assertEqual(sum(f['tested'] for f in frames), saved['tested'])
                self.assertEqual(sum(f['correct'] for f in frames), saved['correct'])
            html = (destination / 'index.html').read_text()
            self.assertNotIn('__DATA__', html)
            self.assertNotIn('<script src=', html)
            self.assertEqual(json.loads((destination / 'champion.json').read_text()), bundle)

    def test_empty_run_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'no completed generations'):
                export_demo(tmp)
