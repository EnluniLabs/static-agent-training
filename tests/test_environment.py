import unittest

from static_agent_training.environment import NumberLine, Output, sample_starts


class EnvironmentTests(unittest.TestCase):
    def token_for(self, env, kind, value):
        return Output(next(k for k, v in env.diagnostic_state()[kind].items() if v == value))

    def move(self, env, value):
        return env.emit(self.token_for(env, 'move_decoding', value))

    def expect(self, env, value, source='pair'):
        return env.emit(self.token_for(env, 'expectation_decoding', value), source=source)

    def test_expectations_and_reembedding(self):
        env = NumberLine()
        initial = env.reset(start=5)
        self.assertIsNone(self.expect(env, 4).observation)
        self.expect(env, 3, 'other')
        result = self.move(env, -1)
        self.assertNotEqual(initial.observation, result.observation)
        self.assertEqual([f.correct for f in result.feedback], [True, False])
        self.assertEqual([f.source for f in result.feedback], ['pair', 'other'])
        self.assertEqual(self.move(env, 0).observation, result.observation)
        self.assertEqual(self.move(env, 0).feedback, ())

    def test_target_and_action_limit(self):
        env = NumberLine(action_limit=1)
        env.reset(start=5)
        self.expect(env, 0)
        result = self.move(env, -5)
        self.assertTrue(result.success)
        self.assertTrue(result.feedback[0].correct)
        with self.assertRaises(RuntimeError):
            self.move(env, 0)
        env.reset(start=5)
        self.assertEqual(self.move(env, 0).reason, 'action_limit')

    def test_agent_chooses_expectation_timing(self):
        env = NumberLine()
        env.reset(start=10)
        # Acting without any expectation produces no empirical criticism.
        self.assertEqual(self.move(env, -1).feedback, ())
        self.assertEqual(self.move(env, -1).feedback, ())
        # The agent can later choose several expectations before one action.
        self.assertIsNone(self.expect(env, 7).observation)
        self.assertIsNone(self.expect(env, 6).observation)
        self.assertEqual(env.diagnostic_state()['position'], 8)
        result = self.move(env, -1)
        self.assertEqual([f.correct for f in result.feedback], [True, False])
        self.assertEqual(self.move(env, -1).feedback, ())

    def test_expiration_and_internal_limit(self):
        env = NumberLine(internal_limit=2, boundary_mode="terminate")
        env.reset(start=20)
        self.expect(env, 20)
        result = self.move(env, 1)
        self.assertEqual(result.reason, 'out_of_bounds')
        self.assertEqual(result.feedback, ())
        self.assertEqual(result.expired_expectations, (0,))
        env.reset(start=5)
        self.expect(env, 5)
        result = self.expect(env, 5)
        self.assertEqual(result.reason, 'internal_limit')
        self.assertEqual(result.expired_expectations, (0, 1))
        env.reset(start=5)
        self.expect(env, 5)
        self.assertEqual(env.stop().expired_expectations, (0,))

    def test_vocabulary_and_reproducibility(self):
        env = NumberLine(encoding_seed=9, episode_seed=12)
        other = NumberLine(encoding_seed=9, episode_seed=12)
        self.assertEqual(len(env.output_tokens), 50)
        self.assertEqual(len(env.observation_tokens), 41)
        self.assertEqual([env.reset() for _ in range(10)], [other.reset() for _ in range(10)])
        self.assertEqual(sample_starts(7), sample_starts(7))
        self.assertTrue(set(sample_starts(7, allowed=(1, 2))) <= {1, 2})


if __name__ == '__main__':
    unittest.main()
