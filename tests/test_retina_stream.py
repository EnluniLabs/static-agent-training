"""Stream transport and graph contracts for bound pixel geometry."""
import unittest
import tempfile
from pathlib import Path

from static_agent_training.brain_scan import mapping_graph
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.experimental.stream_input import ReceptorStream
from static_agent_training.repertoire import Repertoire


class PixelSurface:
    interface = 'retina'
    width, height, receptors, partitions = 4, 4, 16, 4
    raw_window_bytes = 2
    scheme = 'tiles'
    partition_rectangles = ((0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2))

    def partition_indices(self, partitions=None):
        if partitions not in (None, self.partitions):
            raise ValueError('bound partitions')
        return ((0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15))

    def frame(self, stimulus):
        if stimulus is None:
            return (None,)
        if len(stimulus) != self.raw_window_bytes:
            raise ValueError('eye requires an exact raster')
        return (bytes(stimulus),)


class RetinaStreamTests(unittest.TestCase):
    def test_capacity_is_raw_bytes_and_transport_does_not_resize_surface(self):
        senses = PixelSurface()
        self.assertEqual(ReceptorStream(senses).window_bytes, 2)
        frames = list(ReceptorStream(senses, 1).iter_frames(b'\x80\x01\xff'))
        self.assertEqual(frames, [(b'\x80\0',), (b'\x01\0',), (b'\xff\0',)])
        self.assertEqual(senses.receptors, 16)
        with self.assertRaises(ValueError):
            ReceptorStream(senses, 3)

    def test_short_eof_padding_is_transport_owned(self):
        senses = PixelSurface()
        with self.assertRaises(ValueError):
            senses.frame(b'\x80')
        stream = ReceptorStream(senses)
        self.assertEqual(list(stream.iter_frames(b'\xff\xff\x80')),
                         [(b'\xff\xff',), (b'\x80\0',)])
        self.assertEqual(stream.convert(None), (None,))
        with self.assertRaises(ValueError):
            stream.convert(b'abc')

    def test_legacy_capacity_remains_dual_rail(self):
        self.assertEqual(ReceptorStream(FixedSenses(256)).window_bytes, 16)

    def test_graph_uses_tile_membership_and_presence_has_no_pixel(self):
        senses = PixelSurface()
        graph = mapping_graph(Repertoire({}, {}), senses=senses, sensory_partitions=4,
                              parameters=dict(weights=[[] for _ in range(17)], bias=[], gates={}, prediction={}))
        inputs = {node['id']: node for node in graph['nodes']}
        self.assertEqual(inputs['input:5']['pixel'], [0, 1])
        self.assertEqual(inputs['input:5']['region'], 0)
        self.assertEqual(inputs['input:3']['region'], 1)
        self.assertNotIn('pixel', inputs['input:0'])
        self.assertIsNone(inputs['input:0']['region'])
        self.assertEqual(graph['sensory_geometry'], dict(interface='retina', width=4, height=4,
                                                         resolution=16, partitions=4, scheme='tiles'))

    def test_retina_cli_short_eof_preserves_original_byte_counts(self):
        from static_agent_training.stream_inspect import inspect, parser
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'pixels.bin'
            source.write_bytes(bytes(range(33)))
            options = parser().parse_args(['--input', str(source), '--sensory-interface', 'retina'])
            report = inspect(options)
        self.assertEqual(report['input_window_sizes'], [32, 1])
        self.assertEqual(report['bytes_consumed'], 33)
        self.assertEqual(bytes.fromhex(report['frames'][1]['receptor_hex']), b'\x20' + bytes(31))

    def test_retina_model_stream_short_eof_preserves_original_byte_counts(self):
        from static_agent_training.experimental.senses import RetinaSenses
        from static_agent_training.experimental.bit_model import BitModel
        from static_agent_training.experimental.stream_input import run_stream
        from static_agent_training.initialize import pair_id
        term = ('abs', ('var', 0))
        identity = pair_id(term, term)
        repertoire = Repertoire({identity: (term, term)}, {identity: identity})
        model = BitModel(repertoire, senses=RetinaSenses(width=16, height=16, partitions=4),
                         sensory_partitions=4)
        report = run_stream(model, bytes(range(33)))
        self.assertEqual([frame['input_window_size'] for frame in report['frames']], [32, 1])
        self.assertEqual(report['bytes_consumed'], 33)
        self.assertEqual(bytes.fromhex(report['frames'][1]['sensory']), b'\x20' + bytes(31))

    def test_real_retina_when_core_is_available(self):
        from static_agent_training.experimental import senses as core
        if not hasattr(core, 'RetinaSenses'):
            self.skipTest('RetinaSenses core is being added independently')
        senses = core.RetinaSenses(width=16, height=16, partitions=4)
        self.assertEqual(senses.interface, 'retina')
        self.assertEqual(ReceptorStream(senses).window_bytes, 32)
        self.assertEqual(senses.frame(bytes(range(32)))[0], bytes(range(32)))
        indices = senses.partition_indices()
        self.assertEqual(sorted(pixel for region in indices for pixel in region), list(range(256)))
        self.assertEqual(len(senses.partition_rectangles), 4)
        with self.assertRaises(ValueError):
            senses.bind_partitions(2)


if __name__ == '__main__':
    unittest.main()
