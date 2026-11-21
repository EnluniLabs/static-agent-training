from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

from static_agent_training.experimental.senses import (
    FixedSenses, RetinaSenses, SignalSenses, SensoryAdapter, StimulusStep, stimulus_payload,
)
from static_agent_training.experimental.apparatus import make_senses
from static_agent_training.experimental.number_line import StimulusNumberLine


class SignalSensesTests(unittest.TestCase):
    def test_universal_bytes_and_output_capacity(self):
        for width, height in ((4, 4), (16, 16), (8, 32)):
            senses = SignalSenses(width=width, height=height)
            fixed = FixedSenses(receptors=senses.receptors)
            self.assertEqual(senses.raw_window_bytes, senses.receptors // 16)
            self.assertEqual(senses.spec.bytes_per_signal, senses.receptors // 8)
            for payload in (None, b'', b'\x00', b'\xff', bytes(range(256)),
                            bytearray(b'not pixels'), memoryview(b'\x80\x01')):
                self.assertEqual(senses.frame(payload), fixed.frame(payload))
                self.assertEqual(senses.spec.encode(senses.frame(payload)),
                                 fixed.spec.encode(fixed.frame(payload)))
            with self.assertRaises(ValueError):
                senses.frame('text')

    def test_serialization_immutable_geometry_and_legacy(self):
        senses = make_senses(sensory_interface='signals', receptors=256,
                             retina_width=8, retina_height=32, sensory_partitions=7)
        self.assertIsInstance(senses, RetinaSenses)
        self.assertEqual(senses.interface, 'signals')
        self.assertEqual(FixedSenses.from_data(senses.to_data()), senses)
        self.assertEqual(senses.partition_indices(), RetinaSenses(width=8, height=32, partitions=7).partition_indices())
        self.assertEqual(sorted(i for region in senses.partition_indices() for i in region), list(range(256)))
        with self.assertRaises(FrozenInstanceError):
            senses.width = 16
        with self.assertRaises(ValueError):
            senses.bind_partitions(4)
        with self.assertRaises(ValueError):
            make_senses(sensory_interface='signals', receptors=16)
        with self.assertRaises(ValueError):
            FixedSenses.from_data(dict(senses.to_data(), version=2))
        for old in (FixedSenses(), RetinaSenses()):
            self.assertEqual(FixedSenses.from_data(old.to_data()), old)

    def test_adapter_requires_no_renderer_or_stimulus_method(self):
        class World:
            def reset(self, task):
                return StimulusStep(b'ordinary arbitrary bytes')
        senses = SignalSenses()
        world = World()
        self.assertEqual(SensoryAdapter(world, senses).reset(None).frame,
                         senses.frame(b'ordinary arbitrary bytes'))
        world.receptor_stimulus = lambda: b'local readings'
        self.assertEqual(SensoryAdapter(world, senses).reset(None).frame,
                         senses.frame(b'local readings'))
        world.reset = lambda task: StimulusStep(None, done=True)
        self.assertIsNone(SensoryAdapter(world, senses).reset(None).frame)

    def test_local_readings_and_invariances(self):
        world = StimulusNumberLine(lower=-1000, upper=1000)
        world.reset((0, 3))
        expected = bytearray([64] * 16)
        expected[8], expected[11] = 128, 255
        self.assertEqual(world.receptor_stimulus(), bytes(expected))
        world.reset((100, 103))
        self.assertEqual(world.receptor_stimulus(), bytes(expected))
        for goal in (-900, 900):
            world.reset((0, goal))
            self.assertEqual(world.receptor_stimulus(), bytes([64]*8 + [128] + [64]*7))
        world.goal_conditioned = False
        world.reset((0, 3))
        self.assertNotIn(255, world.receptor_stimulus())
        world.reset((-1000, -999))
        self.assertEqual(world.receptor_stimulus()[:9], bytes([0]*8 + [128]))
        self.assertEqual(stimulus_payload(world, SignalSenses(width=4, height=4)),
                         stimulus_payload(world, SignalSenses(width=32, height=8)))

    def test_native_sources_match_python_without_renderer(self):
        from static_agent_training.experimental.sensory_accelerated import native_frames
        from static_agent_training.environments import environment_spec, stimulus_world
        senses = SignalSenses(width=4, height=4)
        native_frames.cache_clear()
        with patch.object(StimulusNumberLine, 'render_retina', side_effect=AssertionError('renderer called')):
            for environment in ('original', 'lights-out-2x2'):
                spec = environment_spec(environment)
                frames = native_frames(senses, True, environment)
                world = stimulus_world(environment)
                position = spec.positions[1] if spec.kind == 'lights-out' else spec.positions[0]
                goal = 0 if spec.kind == 'lights-out' else spec.positions[-1]
                transition = SensoryAdapter(world, senses).reset((position, goal))
                index = 1 if spec.kind == 'lights-out' else len(spec.positions)-1
                self.assertEqual(frames[index].tolist(), list(senses.spec.encode(transition.frame)))

    def test_native_learning_and_playback_parity(self):
        from static_agent_training.evolution import TrainingConfig, initialize_population
        from static_agent_training.accelerated import assert_equivalent
        from static_agent_training.experimental.sensory_genome import SensoryMappings
        from static_agent_training.experimental.sensory_evolution import evaluate
        from static_agent_training.experimental.sensory_accelerated import evaluate_batch, learned_mappings
        from static_agent_training.experimental.sensory_playback import record_episode
        from static_agent_training.experimental.bit_model import BitModel
        config = TrainingConfig(model='senses', population=2, elites=1, initial_pairs=2,
                                chain_budget=8, receptors=16, inherit_lifetime=True,
                                sensory_partitions=2)
        parent = initialize_population(config)[0]
        senses = SignalSenses(width=4, height=4, partitions=2)
        parent = replace(parent, inherited=SensoryMappings.initialize(parent.repertoire, senses=senses, seed=11))
        train, assess = [(-19, 0), (1, 3)], [(19, 0), (-1, -3)]
        with patch.object(StimulusNumberLine, 'render_retina', side_effect=AssertionError('renderer called')):
            actual, weights = evaluate_batch([parent], [s for s,g in train], [s for s,g in assess],
                training_goals=[g for s,g in train], assessment_goals=[g for s,g in assess],
                seeds=[43], return_weights=True, sensory_partitions=2)
            reference, detail, learned = evaluate(parent, train, assess, seed=43, config=config)
            assert_equivalent(actual[0], reference)
            assert_equivalent(learned_mappings(parent, weights, 0).to_parameters(), learned.inherited.to_parameters())
            model = BitModel(parent.repertoire, senses=senses, sensory_partitions=2)
            frames = record_episode(model, 1, 3)
            world = StimulusNumberLine()
            world.reset((1, 3))
            self.assertEqual(frames[0]['sensory'], senses.frame(world.receptor_stimulus())[0].hex())
