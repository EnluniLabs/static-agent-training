"""Owned spatial geometry and packed pixel boundary regressions."""
import json
import unittest
from dataclasses import FrozenInstanceError

from static_agent_training.experimental.bit_model import BitModel, run_episode
from static_agent_training.experimental.senses import (FixedSenses, RetinaSenses,
    SensoryAdapter, StimulusStep, compatible_senses)
from static_agent_training.experimental.sensory_genome import SensoryMappings
from tests.test_bit_model import repertoire


class RetinaTests(unittest.TestCase):
    def test_geometry_and_every_partition_count(self):
        self.assertEqual(RetinaSenses().partition_rectangles,
                         ((0, 0, 8, 8), (0, 8, 8, 8), (8, 0, 8, 8), (8, 8, 8, 8)))
        for width, height in ((16,16), (1,16), (16,1), (6,8)):
            for count in range(1, width*height+1):
                senses = RetinaSenses(width, height, count)
                regions = senses.partition_indices()
                self.assertEqual(len(regions), count)
                self.assertTrue(all(regions))
                self.assertEqual(sorted(i for group in regions for i in group), list(range(width*height)))
                self.assertEqual(regions, senses.partition_indices(count))
                self.assertEqual(senses.regions(), senses.partition_rectangles)
        self.assertTrue(all(len(r)==1 for r in RetinaSenses(partitions=256).partition_indices()))

    def test_validation_and_frozen_ownership(self):
        for kwargs in (dict(width=0), dict(width=True), dict(height=3,width=3),
                       dict(partitions=0), dict(partitions=257), dict(partitions=True),
                       dict(actions=8), dict(scheme='stripes')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetinaSenses(**kwargs)
        senses = RetinaSenses()
        with self.assertRaises(FrozenInstanceError):
            senses.width = 8
        self.assertIs(senses.bind_partitions(4), senses)
        for operation in (senses.bind_partitions, senses.partition_indices, senses.regions):
            with self.assertRaises(ValueError):
                operation(3)
        model = BitModel(repertoire(), senses=senses)
        self.assertEqual(model.sensory_partitions, 4)
        for key,value in (('senses', FixedSenses()), ('spec', FixedSenses().spec), ('sensory_partitions', 1)):
            with self.assertRaises(AttributeError):
                setattr(model, key, value)
        with self.assertRaises(ValueError):
            BitModel(repertoire(), senses=senses, sensory_partitions=3)
        shape = (model.spec, model.senses, len(model.weights))
        with self.assertRaises(ValueError):
            model.observe((bytes(33),))
        self.assertEqual((model.spec, model.senses, len(model.weights)), shape)

    def test_pixels_remain_original_features(self):
        senses = RetinaSenses()
        payload = bytes(range(32))
        self.assertEqual(senses.frame(payload), (payload,))
        self.assertEqual(senses.frame(memoryview(payload)), (payload,))
        self.assertEqual(senses.frame(None), (None,))
        for bad in (b'', b'\x80', b'x'*31, b'x'*33, b'{"value":1}', '123', 123, [1,0]):
            with self.assertRaises(ValueError):
                senses.frame(bad)
        for region, indices in enumerate(senses.partition_indices()):
            features = senses.active_features(senses.frame(b'\xff'*32), region)
            self.assertEqual(features, [0]+[i+1 for i in indices])
            self.assertEqual(senses.active_features(senses.frame(None), region), [])
        self.assertEqual(senses.active_features(senses.frame(b'\x80'+bytes(31))), [0,1])

    def test_legacy_binding_and_genotype_identity(self):
        senses = FixedSenses()
        record = dict(receptors=256, actions=9, transducer='dual-rail-position-fold-or-v1')
        self.assertEqual(senses.to_data(), record)
        self.assertEqual(FixedSenses.from_data(record).to_data(), record)
        model = BitModel(repertoire(), senses=senses, sensory_partitions=3)
        self.assertIsNone(senses.partitions)
        self.assertEqual(model.senses.partitions, 3)
        self.assertTrue(compatible_senses(model.senses, senses))
        self.assertFalse(compatible_senses(model.senses, FixedSenses(128)))
        self.assertFalse(compatible_senses(model.senses, FixedSenses(partitions=2)))
        self.assertIs(model.senses.bind_partitions(3), model.senses)
        with self.assertRaises(ValueError):
            model.senses.bind_partitions(2)
        mappings = SensoryMappings.initialize(repertoire(), senses=senses)
        loaded = SensoryMappings.from_data(json.loads(json.dumps(mappings.to_data())))
        self.assertEqual(loaded.senses.to_data(), record)
        loaded.model(repertoire(), sensory_partitions=3)
        self.assertEqual(loaded.perturb(seed=1, scale=0)[0].senses.to_data(), record)
        with self.assertRaises(ValueError):
            SensoryMappings.initialize(repertoire(), senses=FixedSenses(partitions=1), previous=loaded)

    def test_save_load_and_missing_top_level_count(self):
        for senses in (RetinaSenses(), FixedSenses(partitions=3)):
            model = BitModel(repertoire(), senses=senses)
            data = json.loads(json.dumps(model.to_data()))
            self.assertEqual(BitModel.from_data(data).to_data(), model.to_data())
            del data['sensory_partitions']
            self.assertEqual(BitModel.from_data(data).senses, senses)
            data['sensory_partitions'] = 2
            with self.assertRaises(ValueError):
                BitModel.from_data(data)
        data = BitModel(repertoire(), senses=FixedSenses()).to_data()
        data['senses'].pop('partitions')
        data.pop('sensory_partitions')
        data.pop('embedding_selection')
        restored = BitModel.from_data(data)
        self.assertEqual(restored.embedding_selection, 'sample')
        self.assertEqual(restored.sensory_partitions, 1)
        self.assertEqual(restored.senses.partitions, 1)
        retina = RetinaSenses()
        self.assertEqual(FixedSenses.from_data(retina.to_data()), retina)
        mappings = SensoryMappings.initialize(repertoire(), senses=retina)
        self.assertEqual(mappings.model(repertoire()).sensory_partitions, 4)
        with self.assertRaises(ValueError):
            mappings.model(repertoire(), sensory_partitions=1)

    def test_adapter_uses_only_rendered_pixels(self):
        class World:
            def reset(self, task):
                return StimulusStep(b'not pixels or numeric metadata', done=True)
            def step(self, action):
                return StimulusStep(None, done=True)
            def render_retina(self, senses):
                self.seen = senses
                return b'\x80'+bytes(senses.raw_window_bytes-1)
        senses = RetinaSenses()
        world = World()
        adapter = SensoryAdapter(world, senses)
        self.assertEqual(adapter.reset(None).frame, adapter.step(None).frame)
        self.assertIs(world.seen, senses)
        model = BitModel(repertoire(), senses=senses)
        run_episode(model, adapter, None, learn=False)
        for other in (FixedSenses(), RetinaSenses(8,32), RetinaSenses(partitions=3)):
            with self.assertRaises(ValueError):
                run_episode(model, SensoryAdapter(world, other), None)
        world.render_retina = lambda eye: b'{"value":1}'
        with self.assertRaises(ValueError):
            adapter.reset(None)
        world.render_retina = None
        with self.assertRaises(ValueError):
            adapter.reset(None)
        self.assertEqual((FixedSenses().interface, senses.interface), ('legacy-bytes','retina'))
        self.assertEqual((FixedSenses().raw_window_bytes, senses.raw_window_bytes), (16,32))


if __name__ == '__main__':
    unittest.main()
