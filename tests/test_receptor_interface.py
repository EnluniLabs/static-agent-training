"""Content-independent conversion and partition coordinates at the agent boundary."""
import json
import unittest

from static_agent_training.brain_scan import mapping_graph
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.senses import FixedSenses, receptor_regions
from tests.test_bit_model import repertoire


class ReceptorInterfaceTests(unittest.TestCase):
    def test_bytes_like_payloads_share_exact_conversion_without_aliases(self):
        senses = FixedSenses()
        for payload in (b'', bytes(range(256)), 'hello 🌎'.encode(), b'\0\xff'*40):
            expected = senses.frame(payload)
            self.assertEqual(senses.frame(bytearray(payload)), expected)
            self.assertEqual(senses.frame(memoryview(payload)), expected)
        mutable = bytearray(b'abcd')
        saved = senses.frame(mutable)
        mutable[:] = b'efgh'
        self.assertEqual(saved, senses.frame(b'abcd'))
        for bad in ('text', 3, [0, 1], {'data': b'a'}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                senses.frame(bad)

    def test_regions_cover_every_receptor_once_without_content_labels(self):
        senses = FixedSenses(receptors=16)
        self.assertEqual(senses.regions(3), ((0, 6), (6, 11), (11, 16)))
        for count in range(1, 17):
            bounds = senses.regions(count)
            self.assertEqual([i for start,end in bounds for i in range(start,end)], list(range(16)))
            self.assertLessEqual(max(b-a for a,b in bounds)-min(b-a for a,b in bounds), 1)
            for payload in (None, b'', b'A', b'\x00\xff'):
                frame = senses.frame(payload)
                encoded = senses.spec.encode(frame)
                for region,(start,end) in enumerate(bounds):
                    expected = [i for i,value in enumerate(encoded) if value and (i==0 or start<i<=end)]
                    self.assertEqual(senses.active_features(frame,region,count), expected)
        for count in (0,17,True,2.5):
            with self.assertRaises(ValueError):
                senses.regions(count)
        for region in (-1,3,True,1.5):
            with self.assertRaises(ValueError):
                senses.active_features(senses.frame(b'a'),region,3)

    def test_model_and_visualizer_use_the_same_generic_geometry(self):
        senses = FixedSenses(receptors=16)
        model = BitModel(repertoire(), senses=senses, sensory_partitions=3)
        for payload in (None, b'', b'\x01', bytes(range(64))):
            frame = senses.frame(payload)
            for region in range(3):
                self.assertEqual(model.feature_indices(frame, region),
                                 senses.active_features(frame, region, 3))
        graph = mapping_graph(model.repertoire,parameters=model.parameters(),sensory_partitions=3)
        inputs = {int(n['id'].split(':')[1]):n for n in graph['nodes'] if n['kind']=='input'}
        self.assertIsNone(inputs[0]['region'])
        for region,(start,end) in enumerate(senses.regions(3)):
            self.assertTrue(all(inputs[i+1]['region']==region for i in range(start,end)))

    def test_existing_transducer_serialization_stays_unchanged(self):
        senses = FixedSenses(144)
        expected = dict(receptors=144, actions=9, transducer='dual-rail-position-fold-or-v1')
        self.assertEqual(senses.to_data(), expected)
        self.assertEqual(FixedSenses.from_data(json.loads(json.dumps(expected))), senses)
        self.assertEqual(receptor_regions(144,3), ((0,48),(48,96),(96,144)))
