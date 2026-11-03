import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.catalog import catalog, resolve_run
from static_agent_training.experimental.model_comparison import run_study
from static_agent_training.experimental.number_line import StimulusNumberLine
from static_agent_training.experimental.signals import Action
from static_agent_training.environment import NumberLine, Output


class ModelComparisonTests(unittest.TestCase):
    def test_both_worlds_have_identical_actions_walls_and_termination(self):
        for start, goal in [(20, 19), (-20, -19), (0, 19), (1, -19)]:
            old = NumberLine(goal_conditioned=True)
            new = StimulusNumberLine()
            old.reset(start=start, goal=goal)
            new.reset((start, goal))
            old_moves = old.diagnostic_state()['move_decoding']
            self.assertEqual(list(old_moves.values()), new.moves)
            for action in list(range(9)) * 2:
                left = old.emit(Output(f'move_{action:02d}'))
                right = new.step(Action(action))
                self.assertEqual(old.diagnostic_state()['position'], new.position)
                self.assertEqual((left.done, left.success, left.reason), (right.done, right.success, right.reason))
                if left.done:
                    break

    def test_matched_tasks_frozen_models_and_pair_resume(self):
        plan = dict(population=2, seeds=[48], episodes=2, assessment_episodes=8, receptors=256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'runs' / 'interfaces' / 'paired'
            report = run_study(directory, plan)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['completed'], 2)
            task_list = json.loads((directory / 'tasks/48.json').read_text())
            self.assertFalse(set(task_list['partitions']['training']) & set(task_list['partitions']['assessment']))
            self.assertEqual(report['pairs'][0]['task_hash'], report['pairs'][1]['task_hash'])
            self.assertEqual(report['paired_effect']['seeds'], 1)
            self.assertIsNone(report['paired_effect']['interval95'])
            result_path = directory / 'seed-48/animal-0000/result.json'
            stamp = result_path.stat().st_mtime_ns
            result = json.loads(result_path.read_text())
            for arm in ('old', 'senses'):
                self.assertEqual(result[arm]['training']['episodes'], 2)
                self.assertEqual(result[arm]['assessment']['episodes'], 8)
                self.assertEqual(result[arm]['assessment']['empirical_updates'], 0)
                self.assertEqual(result[arm]['assessment']['structural_updates'], 0)
            resumed = run_study(directory, resume=True)
            self.assertEqual(resumed['population_mean'], report['population_mean'])
            self.assertEqual(result_path.stat().st_mtime_ns, stamp)
            rows = catalog(root, processes={})
            self.assertEqual(rows[0]['kind'], 'interfaces')
            self.assertEqual(resolve_run(root, 'latest-interfaces', rows=rows), directory)
            with self.assertRaises(ValueError):
                resolve_run(root, 'latest-interfaces', kind='comparison', rows=rows)
            with self.assertRaises(ValueError):
                run_study(directory, plan)

    def test_invalid_plans_do_not_start_a_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'invalid'
            for plan in [dict(population=0, seeds=[1], episodes=1, assessment_episodes=1, receptors=256),
                         dict(population=1, seeds=[1, 1], episodes=1, assessment_episodes=1, receptors=256)]:
                with self.assertRaises(ValueError):
                    run_study(directory, plan)
                self.assertFalse(directory.exists())
