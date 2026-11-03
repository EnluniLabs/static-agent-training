import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.run_records import read_summary


class DiagnosticsTests(unittest.TestCase):
    def test_failed_sync_preserves_previous_checkpoint(self):
        from static_agent_training.experiment import write_json, load_checkpoint
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'checkpoint.json.gz'
            original = dict(schema_version='crt-training-v1', next_generation=2)
            write_json(path, original)
            with patch('static_agent_training.experiment.os.fsync', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    write_json(path, dict(original, next_generation=3))
            self.assertEqual(load_checkpoint(root), original)

    def test_oversized_evaluator_allocation_is_rejected(self):
        from static_agent_training.accelerated import _check_allocation
        with self.assertRaisesRegex(MemoryError, 'batch-size'):
            _check_allocation(513 * 1024 * 1024)

    def test_fatal_signal_is_logged(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / 'run'
            script = ('import os, signal, resource; '
                      'from static_agent_training.diagnostics import diagnostics; '
                      'resource.setrlimit(resource.RLIMIT_CORE, (0,0)); '
                      f'ctx=diagnostics({str(run)!r}); ctx.__enter__(); '
                      'os.kill(os.getpid(), signal.SIGBUS)')
            result = subprocess.run([sys.executable, '-c', script], capture_output=True)
            self.assertEqual(result.returncode, -signal.SIGBUS)
            self.assertIn('Bus error', (Path(str(run)+'-diagnostics')/'crash.log').read_text())

    def test_memory_exhaustion_is_logged(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / 'run'
            script = ('from static_agent_training.diagnostics import diagnostics\n'
                      f'with diagnostics({str(run)!r}, 256):\n'
                      '    data = bytearray(512 * 1024 * 1024)\n')
            result = subprocess.run([sys.executable, '-c', script], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            logs = Path(str(run)+'-diagnostics')
            self.assertIn('MemoryError', (logs/'crash.log').read_text())
            self.assertEqual(json.loads((logs/'resources.jsonl').read_text().splitlines()[-1])['event'], 'failed')

    def test_summary_across_chunk_boundary_and_truncation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'record.json'
            path.write_text('{"padding":"' + 'x'*65518 + '","summary":{"generation":2,"best":0.5}}')
            self.assertEqual(read_summary(path)['generation'], 2)
            path.write_text('')
            with self.assertRaises(ValueError):
                read_summary(path)
