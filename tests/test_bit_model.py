import json
import math
import unittest

from static_agent_training.engine import run
from static_agent_training.experimental.bit_model import BitModel, run_episode
from static_agent_training.experimental.bit_experiment import experiment
from static_agent_training.experimental.number_line import ByteNumberLine
from static_agent_training.experimental.signals import Action, Prediction, SignalSpec, Step
from static_agent_training.learning import LearningConfig
from static_agent_training.repertoire import Repertoire


def repertoire():
    identity = ('abs', ('var', 0))
    k = ('abs', ('abs', ('var', 1)))
    return Repertoire.from_data(run([(identity, identity), (k, k)], 32)['repertoire'])


class BitModelTests(unittest.TestCase):
    def model(self, **kwargs):
        return BitModel(repertoire(), seed=19, **kwargs)

    def test_fixed_shape_opaque_bytes_roles_and_overflow(self):
        model = self.model()
        initial = model.parameters()
        for byte in range(256):
            self.assertAlmostEqual(sum(model.probabilities((bytes([byte]), b'Z'))), 1)
        self.assertEqual(model.parameters(), initial)
        spec = model.spec
        self.assertNotEqual(spec.encode((b'A', b'B')), spec.encode((b'B', b'A')))
        self.assertNotEqual(spec.encode((None, b'A')), spec.encode((b'\x00', b'A')))
        for frame in ((b'long', None), (b'A',), ('A', None)):
            with self.assertRaises(ValueError):
                spec.encode(frame)
        with self.assertRaises(ValueError):
            ByteNumberLine(lower=-129)

    def test_shared_bits_structural_update_and_fixed_edges(self):
        model = self.model(learning_config=LearningConfig(compatibility_threshold=1))
        source = next(q for q in model.ids if q in model.repertoire.successors)
        target = model.repertoire.successors[source]
        before_hash = model.structure_hash()
        model.observe((b'\x06', b'\x10'))
        for c in range(model.spec.channels):
            model.prediction[source][c] = [-20.] * model.spec.bits
        model.prediction[source][0] = [-20., -20., -20., -20., -20., 20., 20., -20.]
        model.gates[source][model.spec.actions] = 100
        model.revised[source] = 0
        before = model.parameters()
        sibling = (b'\x07', b'\x10')
        sibling_before = model.probabilities(sibling)
        model.structural_update(source, enabled=True)
        event = model.events[-1]
        self.assertEqual(event['kind'], 'CONFLICT')
        self.assertEqual(event['signal'], '06')
        self.assertNotEqual(sibling_before, model.probabilities(sibling))
        target_index = model.ids.index(target)
        for j in range(len(model.ids)):
            if j != target_index:
                self.assertEqual(model.bias[j], before['bias'][j])
                self.assertEqual([r[j] for r in model.weights], [r[j] for r in before['weights']])
        self.assertEqual(model.gates, before['gates'])
        self.assertEqual(model.prediction, before['prediction'])
        self.assertEqual(model.structure_hash(), before_hash)

    def test_empirical_bit_loss_locality_exact_matching_and_freeze(self):
        model = self.model()
        q = model.ids[0]
        model.pending[0] = (q, Prediction(0, b'A'))
        before = model.parameters()
        model.receive((b'B', None))
        event = model.events[-1]
        self.assertFalse(event['correct'])
        self.assertEqual(event['kind'], 'EMPIRICAL_UPDATE')
        self.assertEqual(model.weights, before['weights'])
        self.assertEqual(model.gates, before['gates'])
        self.assertEqual(model.prediction[q][1], before['prediction'][q][1])
        bits = [(66 >> i) & 1 for i in range(7, -1, -1)]
        def loss(row):
            return sum(max(v, 0) - v * b + math.log1p(math.exp(-abs(v))) for v, b in zip(row, bits))
        self.assertLess(loss(model.prediction[q][0]), loss(before['prediction'][q][0]))
        frozen = model.parameters()
        model.pending[1] = (q, Prediction(0, b'B'))
        model.receive((b'B', None), learn=False)
        self.assertEqual(model.parameters(), frozen)
        self.assertTrue(model.events[-1]['correct'])
        model.pending[2] = (q, Prediction(1, b'B'))
        model.receive((b'B', None))
        self.assertEqual(model.events[-1]['kind'], 'EXPIRED')

    def test_non_numeric_environment_replay_and_shape_validation(self):
        class Letters:
            spec = SignalSpec()
            def reset(self, task):
                return Step((b'A', b'Z'))
            def step(self, action):
                return Step((b'B', b'Z'), done=True, success=True)
        model = self.model()
        snapshot = json.loads(json.dumps(model.to_data()))
        restored = BitModel.from_data(snapshot)
        first = run_episode(model, Letters(), object())
        second = run_episode(restored, Letters(), object())
        self.assertEqual(first, second)
        self.assertEqual(model.to_data(), restored.to_data())
        snapshot['parameters']['weights'][0].pop()
        with self.assertRaises(ValueError):
            BitModel.from_data(snapshot)

    def test_full_width_prediction_and_internal_limit(self):
        model = self.model()
        for q in model.ids:
            model.gates[q] = [-1000.] * (model.spec.actions + model.spec.channels)
            model.gates[q][model.spec.actions] = 1000
            model.prediction[q][0] = [1000.] * model.spec.bits
        q = next(q for q in model.ids if q in model.repertoire.successors)
        # Force embedding to a node with a continuation.
        model.bias = [1000. if p == q else -1000. for p in model.ids]
        for row in model.weights:
            row[:] = [0.] * len(model.ids)
        model.observe((b'A', None))
        self.assertEqual(model.emit(learn=False), Prediction(0, b'\xff'))
        model.receive(None, learn=False)
        model.reset_episode()
        final = run_episode(model, ByteNumberLine(), (1, 2), internal_limit=2)[-1]
        self.assertTrue(final.done)
        self.assertIn(final.reason, ('internal_limit', 'chain_terminated'))
        self.assertFalse(model.pending)

    def test_transfer_no_resizing_no_final_task_leak_and_reproducibility(self):
        report, snapshots = experiment(repertoire(), episodes=3, assessment_episodes=30)
        repeat, _ = experiment(repertoire(), episodes=3, assessment_episodes=30)
        self.assertEqual(report, repeat)
        seen = set(map(tuple, report['task_lists']['training'] + report['task_lists']['adaptation']))
        for rows in report['task_lists']['evaluation'].values():
            self.assertFalse(seen.intersection(map(tuple, rows)))
        specs = [snapshot['spec'] for snapshot in snapshots.values()]
        self.assertTrue(all(spec == specs[0] for spec in specs))
        for snapshot in snapshots.values():
            self.assertEqual(BitModel.from_data(snapshot).structure_hash(), report['structure_hash'])


if __name__ == '__main__':
    unittest.main()
