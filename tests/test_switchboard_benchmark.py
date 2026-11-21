from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training import switchboard_benchmark as harness
from static_agent_training.environments import ENVIRONMENTS, EnvironmentSpec


class SwitchboardBenchmarkTests(unittest.TestCase):
    def test_validation_and_collision_preserve_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'study'
            for kwargs in ({'population': 0}, {'generations': True}, {'episodes': -1},
                           {'benchmark_episodes': 0}, {'seeds': []}, {'seeds': [48, 48]},
                           {'seeds': [-1]}, {'backend': 'cuda'}, {'workers': 0}):
                with self.assertRaises(ValueError):
                    harness.run_benchmark(output, **kwargs)
                self.assertFalse(output.exists())
            collision = output.with_name('study-seed48-parallel3vote')
            collision.mkdir()
            marker = collision / 'user.txt'
            marker.write_text('keep')
            with patch.object(harness, 'run_experiment') as run:
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    harness.run_benchmark(output)
                run.assert_not_called()
            self.assertEqual(marker.read_text(), 'keep')
            self.assertFalse(output.exists())
            output.write_text('user report')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                harness.run_benchmark(output)

            self.assertEqual(output.read_text(), 'user report')

    def test_report_matching_shared_assessment_tasks_and_work(self):
        def fake_run(config, directory):
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps(asdict(config)))
            (directory / 'generations').mkdir()
            initial = harness.initialize_population(config)
            summary = dict(population_id=harness.fingerprint([p.genotype_id for p in initial]), task_id='same', generation=0)
            phase = dict(outputs=3, actions=1, expectations=2, empirical_updates=1)
            record = dict(summary=summary, individuals=[dict(training=phase, assessment=phase)] * 2)
            (directory / 'generations/00000.json').write_text(json.dumps(record))

        def fake_benchmark(directory, bundle, tasks, seed):
            return dict(outcomes=[dict(start=s, goal=g) for s, g in tasks], animal=dict(success_rate=0))

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                ENVIRONMENTS, switchboard=EnvironmentSpec(0, 511, 'lights-out', 'switchboard')), patch.object(
                harness, 'partitions', return_value=dict(training=[1, 2], assessment=[3, 4, 5])), patch.object(
                harness, 'generation_tasks', return_value=([1], [3], [0], [0])), patch.object(
                harness, 'run_experiment', side_effect=fake_run), patch.object(
                harness, 'load_saved_candidate', return_value={}), patch.object(
                harness, 'benchmark', side_effect=fake_benchmark), patch.object(
                harness, '_replay_metrics', return_value=dict(action_conflicts=0)):
            output = Path(tmp) / 'study'
            report = harness.run_benchmark(output, population=2, generations=1, episodes=1,
                assessment_episodes=1, benchmark_episodes=8, backend='python')
            self.assertTrue(report['smoke_only'])
            self.assertTrue(report['pairs'][0]['initialization_verified'])
            self.assertTrue(report['pairs'][0]['tasks_verified'])
            self.assertEqual(report['heldout'][48]['pool'], [3, 4, 5])
            self.assertEqual(report['heldout'][48]['prior_assessment_unique_starts'], 1)
            self.assertEqual(report['heldout'][48]['prior_assessment_overlap_episodes'],
                             sum(s == 3 for s, _ in report['heldout'][48]['tasks']))
            self.assertEqual(len({a['benchmark_task_id'] for a in report['arms']}), 1)
            common = []
            for arm in report['arms']:
                config = arm['config'].copy()
                self.assertEqual(config['environment'], 'switchboard')
                self.assertEqual(config['receptors'], 144)
                self.assertEqual(config['embedding_selection'], 'argmax')
                self.assertIsNone(config['initial_population'])
                self.assertEqual(config.pop('sensory_partitions'), harness.MODES[arm['mode']][0])
                self.assertEqual(config.pop('action_policy'), harness.MODES[arm['mode']][1])
                common.append(config)
                self.assertEqual(Path(arm['directory']).parent, output.parent)
                self.assertEqual(arm['work_totals']['training']['outputs'], 6)
                self.assertEqual(arm['work_totals']['training']['empirical_updates'], 2)
            self.assertTrue(all(c == common[0] for c in common))
            self.assertEqual(json.loads((output / 'report.json').read_text())['seeds'], [48])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                harness.run_benchmark(output)

            def mismatched_run(config, directory):
                fake_run(config, directory)
                path = directory / 'generations/00000.json'
                data = json.loads(path.read_text())
                data['summary']['population_id'] = str(config.sensory_partitions)
                path.write_text(json.dumps(data))

            with patch.object(harness, 'run_experiment', side_effect=mismatched_run), patch.object(
                    harness, 'benchmark') as evaluate:
                with self.assertRaisesRegex(ValueError, 'fingerprints'):
                    harness.run_benchmark(Path(tmp) / 'mismatch', population=2, generations=1,
                        episodes=1, assessment_episodes=1, benchmark_episodes=1, backend='python')
                evaluate.assert_not_called()

    @unittest.skipUnless('switchboard' in ENVIRONMENTS, 'teammate switchboard environment not installed yet')
    def test_tiny_real_backends(self):
        for backend in ('python', 'native'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                report = harness.run_benchmark(Path(tmp) / 'smoke', population=2, generations=1,
                    episodes=1, assessment_episodes=1, benchmark_episodes=1, backend=backend)
                self.assertTrue(report['pairs'][0]['tasks_verified'])
                self.assertTrue(report['pairs'][0]['initialization_verified'])
                self.assertEqual(len({a['benchmark_task_id'] for a in report['arms']}), 1)
                for arm in report['arms']:
                    self.assertEqual(arm['champion_metrics']['empirical_updates'], 0)


if __name__ == '__main__':
    unittest.main()
