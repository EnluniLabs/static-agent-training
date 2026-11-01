import copy
import math
import unittest
from unittest.mock import patch

from static_agent_training.agent import Agent, run_episode
from static_agent_training.engine import run
from static_agent_training.environment import Feedback, NumberLine
from static_agent_training.initialize import I, K, pair_id
from static_agent_training.learning import LearningConfig
from static_agent_training.mappings import InheritedMappings, softmax
from static_agent_training.repertoire import Repertoire


def force(row, selected):
    for key in row:
        row[key] = 1000.0 if key == selected else -1000.0


def snapshot(mappings):
    return copy.deepcopy((mappings.embedding, mappings.output))


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.env = NumberLine(action_limit=1)
        self.rep = Repertoire.from_development(run([(I, K)], 20))
        self.inherited = InheritedMappings.initialize(
            self.rep, [s.token for s in self.env.observation_tokens],
            [o.token for o in self.env.output_tokens], seed=1, scale=0)
        self.maps = self.inherited.lifetime()
        self.agent = Agent(self.rep, self.maps)
        self.q = self.rep.successors[pair_id(I, K)]
        self.next = self.rep.successors[self.q]

    def register(self, token='expect_00', source=None):
        source = source or self.q
        self.agent.learner.register(token, source, 'obs_05', pair_id(I, K),
                                    [pair_id(I, K), source], 1, 2)
        return self.agent.learner.next_id - 1

    def test_empirical_gradient_locality_and_loss(self):
        before = snapshot(self.maps)
        identity = self.register()
        self.agent.learner.feedback(Feedback(identity, self.q, 'obs_00', 'obs_03', False))
        self.assertEqual(before[0], self.maps.embedding)
        for q, row in self.maps.output.items():
            for token, value in row.items():
                expected = before[1][q][token]
                if q == self.q and token.startswith('expect_'):
                    expected -= .05 * (1 / 41 - (token == 'expect_03'))
                self.assertAlmostEqual(value, expected)
        event = self.agent.events[-1]
        self.assertAlmostEqual(event['loss'], math.log(41))
        new = softmax(self.maps.output[self.q][o] for o in self.agent.learner.expectations)
        self.assertLess(-math.log(new[3]), event['loss'])
        self.assertEqual(dict(self.inherited.output[self.q]), before[1][self.q])
        self.assertFalse(self.agent.learner.pending)

    def test_correct_predictions_also_update_and_multiple_expectations_score(self):
        first = self.register('expect_03')
        second = self.register('expect_00')
        self.agent.learner.feedback(Feedback(first, self.q, 'obs_03', 'obs_03', True))
        self.agent.learner.feedback(Feedback(second, self.q, 'obs_00', 'obs_03', False))
        events = [e for e in self.agent.events if e['kind'] == 'EMPIRICAL_UPDATE']
        self.assertEqual([e['correct'] for e in events], [True, False])
        self.assertGreater(self.maps.output[self.q]['expect_03'], .05 * (1 - 1 / 41))

    def test_conflict_is_delayed_and_changes_one_embedding_cell(self):
        identity = self.register()
        before = snapshot(self.maps)
        # No correction before a mapping has been empirically revised.
        self.agent.learner.encounter(self.q)
        self.assertEqual(before, snapshot(self.maps))
        self.agent.learner.feedback(Feedback(identity, self.q, 'obs_00', 'obs_03', False))
        self.assertEqual(before[0], self.maps.embedding)
        # Revisiting another pair cannot scan or repair the revised pair.
        self.agent.learner.encounter(self.next)
        self.assertEqual(before[0], self.maps.embedding)
        force(self.maps.embedding['obs_03'], pair_id(I, K))
        before_embedding = copy.deepcopy(self.maps.embedding)
        self.agent.learner.encounter(self.q)
        for signal, row in self.maps.embedding.items():
            for q, value in row.items():
                delta = .05 if (signal, q) == ('obs_03', self.next) else 0
                self.assertEqual(value, before_embedding[signal][q] + delta)
        self.assertEqual(self.agent.events[-1]['kind'], 'CONFLICT')
        self.assertLess(self.agent.events[-1]['revision_event'], len(self.agent.events) - 1)
        force(self.maps.embedding['obs_03'], self.next)
        before = snapshot(self.maps)
        self.agent.learner.encounter(self.q)
        self.assertEqual(self.agent.events[-1]['kind'], 'COMPATIBLE')
        self.assertEqual(before, snapshot(self.maps))

    def configure_episode(self):
        entry = pair_id(I, K)
        for row in self.maps.embedding.values():
            force(row, entry)
        force(self.maps.output[self.q], 'expect_00')
        zero = next(k for k, v in self.env.diagnostic_state()['move_decoding'].items() if v == 0)
        force(self.maps.output[self.next], zero)

    def test_delayed_correction_across_episodes_and_frozen_assessment(self):
        self.configure_episode()
        before = snapshot(self.maps)
        with patch('static_agent_training.engine.contract', side_effect=AssertionError('reduction')):
            run_episode(self.agent, self.env, start=5)
            self.assertEqual(before[0], self.maps.embedding)
            self.assertEqual(len([e for e in self.agent.events if e['kind'] == 'EMPIRICAL_UPDATE']), 1)
            self.assertFalse(any(e['kind'] == 'CONFLICT' for e in self.agent.events))
            self.assertFalse(self.agent.learner.pending)
            run_episode(self.agent, self.env, start=5)
            self.assertTrue(any(e['kind'] == 'CONFLICT' for e in self.agent.events))
            before = snapshot(self.maps)
            count = len(self.agent.events)
            run_episode(self.agent, self.env, start=5, learn=False)
            self.assertEqual(before, snapshot(self.maps))
            self.assertFalse(any(e['kind'] in ('CONFLICT', 'EMPIRICAL_UPDATE')
                                 for e in self.agent.events[count:]))
            self.assertTrue(self.agent.learning_enabled)

    def test_expiration_and_action_only_do_not_learn(self):
        self.configure_episode()
        before = snapshot(self.maps)
        result = run_episode(self.agent, NumberLine(internal_limit=1), start=5)
        self.assertEqual(result[-1].reason, 'internal_limit')
        self.assertEqual(before, snapshot(self.maps))
        self.assertTrue(any(e['kind'] == 'EXPIRED' for e in self.agent.events))
        self.assertFalse(self.agent.learner.pending)
        zero = next(k for k, v in self.env.diagnostic_state()['move_decoding'].items() if v == 0)
        force(self.maps.output[self.q], zero)
        before = snapshot(self.maps)
        run_episode(self.agent, self.env, start=5)
        self.assertEqual(before, snapshot(self.maps))

    def test_terminal_feedback_and_out_of_bounds_expiration(self):
        self.configure_episode()
        diag = self.env.diagnostic_state()
        minus5 = next(k for k, v in diag['move_decoding'].items() if v == -5)
        force(self.maps.output[self.next], minus5)
        result = run_episode(self.agent, self.env, start=5)
        self.assertTrue(result[-1].success)
        self.assertTrue(any(e['kind'] == 'EMPIRICAL_UPDATE' for e in self.agent.events))
        plus5 = next(k for k, v in diag['move_decoding'].items() if v == 5)
        force(self.maps.output[self.next], plus5)
        before = snapshot(self.maps)
        # Freeze structural checks to isolate the legacy terminal expiration convention.
        self.env.boundary_mode = "terminate"
        run_episode(self.agent, self.env, start=20, learn=False)
        self.assertEqual(before, snapshot(self.maps))
        self.assertEqual(self.agent.events[-1]['kind'], 'EXPIRED')

    def test_seeded_learning_replays_across_episodes(self):
        agents = [Agent(self.rep, self.inherited.lifetime(), embedding_seed=7, output_seed=9)
                  for _ in range(2)]
        environments = [NumberLine(encoding_seed=4, episode_seed=6) for _ in range(2)]
        for agent, environment in zip(agents, environments):
            for _ in range(20):
                run_episode(agent, environment)
        self.assertEqual(agents[0].events, agents[1].events)
        self.assertEqual(snapshot(agents[0].mappings), snapshot(agents[1].mappings))

    def test_config_and_stable_underflow_loss(self):
        with self.assertRaises(ValueError):
            LearningConfig(compatibility_threshold=1.1)
        with self.assertRaises(ValueError):
            LearningConfig(prediction_rate=float('nan'))
        force(self.maps.output[self.q], 'expect_00')
        identity = self.register()
        self.agent.learner.feedback(Feedback(identity, self.q, 'obs_00', 'obs_03', False))
        self.assertTrue(math.isfinite(self.agent.events[-1]['loss']))
        self.assertAlmostEqual(self.agent.events[-1]['loss'], 2000)


if __name__ == '__main__':
    unittest.main()
