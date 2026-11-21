import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training import stream_inspect
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.initialize import pair_id
from static_agent_training.repertoire import Repertoire


class StreamInspectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'input.bin'
        self.source.write_bytes(bytes(range(80)))

    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = stream_inspect.main(list(args))
        return result, stdout.getvalue()

    def snapshot(self, *, bundle=False, compressed=False):
        term = ('abs', ('var', 0))
        identity = pair_id(term, term)
        repertoire = Repertoire({identity: (term, term)}, {identity: identity})
        model = BitModel(repertoire, senses=FixedSenses(64), sensory_partitions=3,
                         action_policy='vote')
        # Every region emits an action so the runner reaches the input budget.
        for gate in model.gates.values():
            gate[:] = [1000.] + [-1000.] * (len(gate) - 1)
        data = model.to_data()
        if bundle:
            data = {'learned_model': data}
        target = self.root / ('model.json.gz' if compressed else 'model.json')
        if compressed:
            with gzip.open(target, 'wt', encoding='utf-8') as file:
                json.dump(data, file)
        else:
            target.write_text(json.dumps(data), encoding='utf-8')
        return target, model

    def test_help_is_read_only_and_does_not_import_stream_core(self):
        with patch.dict('sys.modules', {'static_agent_training.experimental.stream_input': None}), \
                patch('builtins.open', side_effect=AssertionError('help opened a file')), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            with self.assertRaises(SystemExit) as caught:
                stream_inspect.main(['--help'])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('--max-frames', stdout.getvalue())

    def test_default_geometry_and_raw_binary(self):
        report, stdout = self.invoke('--input', str(self.source))
        self.assertEqual(json.loads(stdout), report)
        self.assertEqual((report['receptors'], report['partitions'], report['window_bytes']), (256, 4, 16))
        self.assertEqual(report['input_window_sizes'], [16] * 5)
        self.assertEqual(report['stop_reason'], 'end_of_stream')
        self.assertEqual(report['region_bounds'], [[0, 64], [64, 128], [128, 192], [192, 256]])
        self.assertEqual(report['frames'][0]['receptor_hex'], FixedSenses().frame(bytes(range(16)))[0].hex())
        self.assertEqual(sum(report['frames'][0]['region_active_counts']), 128)

    def test_utf8_is_bytes_and_partial_windows_remain_visible(self):
        data = 'é🌍'.encode('utf-8')
        self.source.write_bytes(data)
        report, _ = self.invoke('--input', str(self.source), '--window-bytes', '2')
        self.assertEqual(report['input_window_sizes'], [2, 2, 2])
        self.assertEqual([f['receptor_hex'] for f in report['frames']],
                         [FixedSenses().frame(data[i:i + 2])[0].hex() for i in range(0, 6, 2)])
        self.source.write_bytes(b'abc')
        report, _ = self.invoke('--input', str(self.source), '--window-bytes', '2')
        self.assertEqual(report['input_window_sizes'], [2, 1])

    def test_stdin_limit_does_not_probe_or_close(self):
        class Bounded(io.BytesIO):
            def read(self, size=-1):
                if self.tell() >= 32:
                    raise AssertionError('read beyond frame limit')
                return super().read(size)
        binary = Bounded(bytes(range(64)))
        with patch.object(stream_inspect.sys, 'stdin', type('Stdin', (), {'buffer': binary})()):
            report, _ = self.invoke('--input', '-', '--max-frames', '2')
        self.assertFalse(binary.closed)
        self.assertEqual(binary.tell(), 32)
        self.assertEqual(report['stop_reason'], 'frame_limit')
        # Even exact-length input must not claim EOF without observing it.
        self.source.write_bytes(bytes(32))
        report, _ = self.invoke('--input', str(self.source), '--max-frames', '2')
        self.assertEqual(report['stop_reason'], 'frame_limit')

    def test_report_and_no_output_collisions(self):
        output = self.root / 'report.json'
        report, stdout = self.invoke('--input', str(self.source), '--output', str(output))
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertIn('5 frames;', stdout)
        self.assertNotIn('receptor_hex', stdout)
        original = self.source.read_bytes()
        for target in (output, self.source):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('--input', str(self.source), '--output', str(target))
        self.assertEqual(self.source.read_bytes(), original)
        link = self.root / 'link'
        link.symlink_to(self.root / 'absent')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke('--input', str(self.source), '--output', str(link))
        self.assertFalse((self.root / 'absent').exists())

    def test_saved_snapshot_frozen_parameters_and_policy(self):
        path, model = self.snapshot(bundle=True, compressed=True)
        original = path.read_bytes()
        before = model.parameters()
        with patch.object(stream_inspect, '_load_model', return_value=model):
            report, _ = self.invoke('--input', str(self.source), '--snapshot', str(path), '--max-frames', '2')
        self.assertEqual(model.parameters(), before)
        self.assertFalse(report['parameter_changed'])
        self.assertEqual(report['model_mode'], 'frozen')
        self.assertEqual((report['receptors'], report['partitions'], report['window_bytes']), (64, 3, 4))
        self.assertEqual(report['action_policy'], 'vote')
        self.assertEqual(report['stop_reason'], 'frame_limit')
        self.assertTrue(report['outputs'])
        self.assertTrue(report['actions'])
        self.assertTrue(report['resolved_actions'])
        # Exercise real local compressed bundle loading too.
        loaded, _ = self.invoke('--input', str(self.source), '--snapshot', str(path), '--learn')
        self.assertEqual(loaded['model_mode'], 'learn')
        self.assertEqual(path.read_bytes(), original)

    def test_snapshot_overrides_and_collision_rejected(self):
        path, _ = self.snapshot()
        original = path.read_bytes()
        for extra in (['--receptors', '256'], ['--partitions', '4'], ['--output', str(path)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('--input', str(self.source), '--snapshot', str(path), *extra)
        report, _ = self.invoke('--input', str(self.source), '--snapshot', str(path),
                                '--receptors', '64', '--partitions', '3')
        self.assertEqual(report['receptors'], 64)
        self.assertEqual(path.read_bytes(), original)

    def test_empty_stream_and_dead_chain_keep_input_clock(self):
        self.source.write_bytes(b'')
        report, _ = self.invoke('--input', str(self.source))
        self.assertEqual(report['frames'], [])
        self.assertEqual(report['stop_reason'], 'end_of_stream')
        path, model = self.snapshot()
        report, _ = self.invoke('--input', str(self.source), '--snapshot', str(path))
        self.assertEqual(report['frame_count'], 0)
        self.assertEqual(report['stop_reason'], 'end_of_stream')
        self.source.write_bytes(bytes(80))
        model.repertoire = Repertoire(dict(model.repertoire.pairs), {})
        with patch.object(stream_inspect, '_load_model', return_value=model):
            report, _ = self.invoke('--input', str(self.source), '--snapshot', str(path))
        self.assertEqual(report['frame_count'], 16)
        self.assertEqual(report['stop_reason'], 'frame_limit')

    def test_prediction_only_model_consumes_all_windows(self):
        path, model = self.snapshot()
        for gate in model.gates.values():
            gate[:] = [-1000.] * len(gate)
            gate[model.spec.actions] = 1000.
        self.source.write_bytes(bytes(range(19)))
        original = path.read_bytes()
        before = model.parameters()
        with patch.object(stream_inspect, '_load_model', return_value=model):
            report, _ = self.invoke('--input', str(self.source), '--snapshot', str(path),
                                    '--rounds-per-frame', '2')
        self.assertEqual(report['input_window_sizes'], [4, 4, 4, 4, 3])
        self.assertEqual(report['frame_count'], 5)
        self.assertEqual(report['bytes_consumed'], 19)
        self.assertEqual(report['stop_reason'], 'end_of_stream')
        self.assertEqual(report['rounds_per_frame'], 2)
        self.assertEqual(report['actions'], [])
        self.assertTrue(all(frame['predictions'] for frame in report['frames']))
        self.assertTrue(report['event_summary'].get('FEEDBACK'))
        self.assertEqual(model.parameters(), before)
        self.assertEqual(path.read_bytes(), original)
        with patch.object(stream_inspect, '_load_model', return_value=model):
            learned, _ = self.invoke('--input', str(self.source), '--snapshot', str(path), '--learn')
        self.assertEqual(learned['model_mode'], 'learn')
        self.assertTrue(learned['parameter_changed'])
        self.assertEqual(path.read_bytes(), original)

    def test_snapshot_stdin_exact_limit_without_read_ahead(self):
        path, _ = self.snapshot()

        class Bounded(io.BytesIO):
            def read(self, size=-1):
                if size < 0 or self.tell() + size > 64:
                    raise AssertionError('read beyond sixteen windows')
                return super().read(size)

        for length in (64, 80):
            with self.subTest(input_bytes=length):
                binary = Bounded(bytes(range(length)))
                with patch.object(stream_inspect.sys, 'stdin', type('Stdin', (), {'buffer': binary})()):
                    report, _ = self.invoke('--input', '-', '--snapshot', str(path))
                self.assertEqual(report['frame_count'], 16)
                self.assertEqual(report['bytes_consumed'], 64)
                self.assertEqual(report['stop_reason'], 'frame_limit')
                self.assertEqual(binary.tell(), 64)
                self.assertFalse(binary.closed)

    def test_invalid_geometry_and_limits(self):
        for extra in (['--rounds-per-frame', '0'], ['--max-frames', '0'], ['--window-bytes', '0'], ['--receptors', '17'],
                      ['--partitions', '257'], ['--window-bytes', '17'], ['--learn']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke('--input', str(self.source), *extra)


if __name__ == '__main__':
    unittest.main()
