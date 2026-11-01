from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.baseline import random_success, random_task_mean
from static_agent_training.benchmark import compare
from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import run_experiment
from static_agent_training.play import attempt
from static_agent_training.demo import export_demo
from static_agent_training.environment import MOVES


class BaselineTests(unittest.TestCase):
    def test_exact_probability_matches_exhaustive_move_sequences(self):
        for start, goal in ((1, 0), (20, 19), (-20, 20), (0, 3)):
            solved = 0
            for moves in product(MOVES, repeat=2):
                position = start
                for move in moves:
                    position += move
                    if not -20 <= position <= 20:
                        break
                    if position == goal:
                        solved += 1
                        break
            self.assertAlmostEqual(random_success(start, goal, 2, boundary_mode="terminate"), solved/81)
        self.assertEqual(random_success(0, 0), 1)
        self.assertEqual(random_success(21, 0), 0)
        self.assertEqual(random_success(1, 0, 0), 0)
        self.assertAlmostEqual(random_task_mean([1, 20], [0, 19]),
                               (random_success(1, 0)+random_success(20, 19))/2)

    def test_shared_task_benchmark_and_random_playground_are_reproducible(self):
        config = TrainingConfig(population=2, generations=1, episodes=3,
                                assessment_episodes=3, elites=1, initial_pairs=2,
                                chain_budget=8, seed=19)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            state = run_experiment(config, run)
            self.assertIn('random_assessment', state['history'][0])
            a = compare(run, episodes=30, seed=77)
            b = compare(run, episodes=30, seed=77)
            self.assertEqual(a, b)
            heldout = json.loads((run/'config.json').read_text())['partitions']['assessment']
            for row in a['outcomes']:
                self.assertIn(row['start'], heldout)
                self.assertIn(row['goal'], heldout)
                self.assertNotEqual(row['start'], row['goal'])
            for name in ('animal', 'random'):
                self.assertEqual(a[name]['successes'], sum(r[name+'_success'] for r in a['outcomes']))
                self.assertEqual(sum(a[name]['reasons'].values()), 30)
            _, bundle = export_demo(run)
            result = attempt(bundle, 8, -4, 7, 'random')
            self.assertEqual(result, attempt(bundle, 8, -4, 7, 'random'))
            self.assertEqual(result['policy'], 'random')
            self.assertLessEqual(result['frames'][-1]['actions'], 12)
            self.assertTrue(all(f['label'].startswith('Move ') for f in result['frames'][1:]))
            with self.assertRaisesRegex(ValueError, 'Unknown policy'):
                attempt(bundle, 8, -4, 7, 'unknown')
