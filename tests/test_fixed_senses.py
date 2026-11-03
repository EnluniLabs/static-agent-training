import json
import unittest

from static_agent_training.experimental.bit_model import BitModel, run_episode
from static_agent_training.experimental.number_line import StimulusNumberLine
from static_agent_training.experimental.senses import FixedSenses, SensoryAdapter, StimulusStep
from static_agent_training.learning import LearningConfig
from tests.test_bit_model import repertoire


class FixedSensesTests(unittest.TestCase):
    def test_vocabulary_growth_and_variable_input_lengths_use_same_receptors(self):
        senses = FixedSenses()
        first = [senses.frame(str(v).encode()) for v in range(41)]
        expanded = [senses.frame(str(v).encode()) for v in range(61)]
        self.assertEqual(first, expanded[:41])
        self.assertEqual(len(set(expanded)), 61)
        for value in [b'', b'\x00', b'\xff', b'hello', 'new sensory input 🔔'.encode(), b'x' * 10000]:
            frame = senses.frame(value)
            self.assertEqual(len(frame), 1)
            self.assertEqual(len(frame[0]) * 8, senses.receptors)
        self.assertNotEqual(senses.frame(None), senses.frame(b''))
        self.assertNotEqual(senses.frame(b''), senses.frame(b'\x00'))
        self.assertEqual(senses, FixedSenses.from_data(json.loads(json.dumps(senses.to_data()))))

    def test_shared_receptors_include_matching_zero_bits_and_loss_is_explicit(self):
        senses = FixedSenses(receptors=16)
        left = int.from_bytes(senses.frame(b'\x06')[0], 'big')
        right = int.from_bytes(senses.frame(b'\x07')[0], 'big')
        self.assertEqual((left & right).bit_count(), 7)
        self.assertEqual((left ^ right).bit_count(), 2)
        # Folding is intentionally lossy outside the finite receptor capacity.
        self.assertEqual(senses.frame(b'\x00\xff'), senses.frame(b'\xff\x00'))

    def test_same_animal_different_environments_and_sensory_checkpoint(self):
        senses = FixedSenses()
        model = BitModel(repertoire(), senses=senses, seed=12)
        baseline = model.parameters()
        structure = model.structure_hash()
        weights = model.weights
        for bounds, task in [((-20, 20), (1, 2)), ((-30, 30), (25, -25)),
                             ((-100000, 100000), (99999, -99999))]:
            environment = SensoryAdapter(StimulusNumberLine(lower=bounds[0], upper=bounds[1]), senses)
            run_episode(model, environment, task, learn=False)
        class Symbols:
            def reset(self, task):
                return StimulusStep('unfamiliar symbol ☀'.encode())
            def step(self, action):
                return StimulusStep(b'different input', done=True)
        run_episode(model, SensoryAdapter(Symbols(), senses), None, learn=False)
        self.assertIs(model.weights, weights)
        self.assertEqual(model.parameters(), baseline)
        self.assertEqual(model.structure_hash(), structure)
        restored = BitModel.from_data(json.loads(json.dumps(model.to_data())))
        self.assertEqual(restored.senses, senses)
        self.assertEqual(restored.parameters(), baseline)
        with self.assertRaises(ValueError):
            run_episode(restored, SensoryAdapter(Symbols(), FixedSenses(128)), None)

    def test_structural_learning_changes_shared_sensory_map_without_new_rows(self):
        senses = FixedSenses()
        model = BitModel(repertoire(), senses=senses, seed=12,
                         learning_config=LearningConfig(compatibility_threshold=1))
        q = next(q for q in model.ids if q in model.repertoire.successors)
        frame = senses.frame(b'A')
        model.observe(frame)
        model.prediction[q][0] = [20. if (v >> bit) & 1 else -20.
                                  for v in frame[0] for bit in range(7, -1, -1)]
        model.revised[q] = 0
        before = model.probabilities(senses.frame(b'B'))
        shape = (len(model.weights), len(model.weights[0]))
        model.structural_update(q, enabled=True)
        self.assertEqual(model.events[-1]['kind'], 'CONFLICT')
        self.assertNotEqual(before, model.probabilities(senses.frame(b'B')))
        self.assertEqual(shape, (len(model.weights), len(model.weights[0])))
