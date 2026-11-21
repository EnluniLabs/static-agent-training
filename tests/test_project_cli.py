from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.catalog import catalog, resolve_run
from static_agent_training.cli import cleanup, main
from static_agent_training.experimental import bit_experiment
from tests.test_bit_model import repertoire


class ProjectCliTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary.cleanup()

    def fixture(self):
        training = self.root / 'runs' / 'baseline'
        (training / 'generations').mkdir(parents=True)
        (training / 'config.json').write_text(json.dumps(dict(config=dict(population=3, generations=2))))
        (training / 'generations' / '00000.json').write_text(json.dumps(dict(summary=dict(
            generation=0, mean=.1, best=.2, assessment_mean=.1))))
        sensory = self.root / 'runs' / 'sensory'
        sensory.mkdir()
        (sensory / 'report.json').write_text(json.dumps(dict(schema_version='crt-bit-experiment-v2', results={})))
        comparison = self.root / 'runs' / 'comparisons' / 'paired'
        comparison.mkdir(parents=True)
        (comparison / 'report.json').write_text(json.dumps(dict(
            schema_version='crt-treatment-comparison-v1', status='running')))
        return training, sensory, comparison

    def test_catalog_kinds_alias_safety_and_stale_process_metadata(self):
        training, sensory, comparison = self.fixture()
        rows = catalog(self.root, processes={})
        self.assertEqual({r['kind'] for r in rows}, {'training', 'senses', 'comparison'})
        for alias, path in [('latest-training', training), ('latest-senses', sensory),
                            ('latest-comparison', comparison)]:
            self.assertEqual(resolve_run(self.root, alias, rows=rows), path)
        with self.assertRaises(ValueError):
            resolve_run(self.root, 'latest-senses', kind='training', rows=rows)
        self.assertEqual(next(r for r in rows if r['kind'] == 'comparison')['state'], 'recorded-running')
        (sensory / 'run.json').write_text(json.dumps(dict(schema_version='crt-sensory-run-v1',
                                                        status='running', pid=999999999)))
        rows = catalog(self.root, processes={})
        sensory_rows = [r for r in rows if r['kind'] == 'senses']
        self.assertEqual(len(sensory_rows), 1)
        self.assertEqual(sensory_rows[0]['state'], 'stopped')

    def test_senses_routing_unique_output_and_read_only_help(self):
        training, _, _ = self.fixture()
        with patch('static_agent_training.cli.execute') as execute:
            main(self.root, ['senses', '--candidate', 'latest-training'])
            module, args = execute.call_args.args
            self.assertEqual(module, 'static_agent_training.experimental.bit_experiment')
            self.assertEqual(args[args.index('--candidate') + 1], str(training))
            output = Path(args[args.index('--output') + 1])
            self.assertEqual(output.parent, self.root / 'runs')
            self.assertFalse(output.exists())
            execute.reset_mock()
            main(self.root, ['senses', '--help'])
            self.assertEqual(execute.call_args.args[1], ['--help'])
        with self.assertRaises(SystemExit):
            main(self.root, ['benchmark', 'latest-senses'])

    def test_switchboard_comparison_routing_is_explicit(self):
        with patch('static_agent_training.cli.execute') as execute:
            main(self.root, ['switchboard-test', '--help'])
            self.assertEqual(execute.call_args.args,
                             ('static_agent_training.switchboard_benchmark', ['--help']))
        self.assertFalse((self.root / 'runs').exists())

    def test_stream_routing_is_explicit_and_help_does_not_read_data(self):
        with patch('static_agent_training.cli.execute') as execute:
            main(self.root, ['stream', '--help'])
            self.assertEqual(execute.call_args.args,
                             ('static_agent_training.stream_inspect', ['--help']))
        self.assertFalse((self.root / 'runs').exists())

    def test_cleanup_preserves_runs_and_environment(self):
        cache = self.root / 'static_agent_training' / '__pycache__'
        cache.mkdir(parents=True)
        (cache / 'module.pyc').write_bytes(b'cache')
        paths = [self.root / 'runs' / 'important.json', self.root / '.venv' / 'keep.pyc',
                 self.root / 'tests' / 'test_keep.py']
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('keep')
        with redirect_stdout(io.StringIO()):
            cleanup(self.root, dry_run=True)
            self.assertTrue(cache.exists())
            cleanup(self.root)
        self.assertFalse(cache.exists())
        self.assertTrue(all(path.read_text() == 'keep' for path in paths))

    def test_sensory_run_metadata_completed_and_failure_without_experiment(self):
        data = dict(initial_pairs=[])
        for failure in (False, True):
            output = self.root / ('failure' if failure else 'success')
            argv = ['bit_experiment', '--output', str(output), '--label', 'fixture', '--notes', 'metadata test']
            with patch('sys.argv', argv), patch.object(bit_experiment, 'initialize', return_value=data), \
                 patch.object(bit_experiment, 'run', return_value=dict(repertoire=repertoire().to_data())), \
                 patch.object(bit_experiment, 'experiment', side_effect=RuntimeError('fixture failure') if failure else None,
                              return_value=(dict(results={}), {})), redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(RuntimeError):
                        bit_experiment.main()
                else:
                    bit_experiment.main()
            metadata = json.loads((output / 'run.json').read_text())
            self.assertEqual(metadata['status'], 'failed' if failure else 'completed')
            self.assertEqual(metadata['label'], 'fixture')
            self.assertEqual(metadata['notes'], 'metadata test')

    def test_entry_point_from_different_directory_does_not_launch_work(self):
        project = Path(__file__).resolve().parents[1] / 'project'
        result = subprocess.run([str(project), 'help'], cwd=self.root, capture_output=True, text=True, check=True)
        self.assertIn('latest-senses', result.stdout)
        self.assertFalse((self.root / 'runs').exists())
