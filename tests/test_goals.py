from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from static_agent_training.demo import export_demo
from static_agent_training.environment import NumberLine, Output
from static_agent_training.evolution import TrainingConfig, initialize_population, evaluate_generation, partitions
from static_agent_training.experiment import run_experiment
from static_agent_training.play import attempt, handler_for


class GoalTests(unittest.TestCase):
    def test_separate_current_and_goal_inputs_prediction_and_absolute_boundary(self):
        env = NumberLine(goal_conditioned=True)
        first = env.reset(start=8, goal=5)
        second = env.reset(start=0, goal=-3)
        self.assertEqual(first.observation.token, "obs_+8")
        self.assertEqual(first.observation.goal, "goal_+5")
        self.assertEqual(second.observation.token, "obs_+0")
        self.assertEqual(second.observation.goal, "goal_-3")
        self.assertNotEqual(first.observation, second.observation)
        state = env.diagnostic_state()
        expect_zero = next(k for k, v in state['expectation_decoding'].items() if v == -3)
        move_minus3 = next(k for k, v in state['move_decoding'].items() if v == -3)
        env.emit(Output(expect_zero))
        result = env.emit(Output(move_minus3))
        self.assertTrue(result.success)
        self.assertTrue(result.feedback[0].correct)
        self.assertEqual(env.diagnostic_state()['position'], -3)
        env.reset(start=20, goal=19)
        move_plus1 = next(k for k, v in state['move_decoding'].items() if v == 1)
        self.assertFalse(env.emit(Output(move_plus1)).done)
        self.assertEqual(env.diagnostic_state()['position'], 20)
        # Both endpoint numbers are valid inputs; physical boundaries remain ±20.
        env.reset(start=-20, goal=20)
        self.assertEqual(len(env.observation_tokens), 82)

    def config(self, **kwargs):
        return TrainingConfig(population=2, generations=2, episodes=4,
                              assessment_episodes=4, elites=1, initial_pairs=2,
                              chain_budget=8, seed=19, **kwargs)

    def test_generations_share_new_goals_and_keep_assessment_fixed(self):
        config = self.config()
        population = initialize_population(config)
        split = partitions(config.seed)
        a, _ = evaluate_generation(population, config, 0, split)
        b, _ = evaluate_generation(population, config, 1, split)
        self.assertNotEqual(a['training_goals'], b['training_goals'])
        self.assertEqual(a['assessment_goals'], b['assessment_goals'])
        self.assertFalse(set(a['training_goals']) & set(a['assessment_goals']))
        for record in (a, b):
            for individual in record['individuals']:
                self.assertEqual([e['goal'] for e in individual['training']['episodes']], record['training_goals'])
                self.assertEqual([e['goal'] for e in individual['assessment']['episodes']], record['assessment_goals'])
            self.assertTrue(all(s != g for s, g in zip(record['training_starts'], record['training_goals'])))

    def test_target_input_can_change_choices_at_the_same_current_number(self):
        population = initialize_population(self.config())
        maps = population[0].inherited.lifetime()
        ids = maps.pair_ids
        self.assertGreater(len(ids), 1)
        for signal in ('obs_+8', 'goal_+5', 'goal_-3'):
            for pair in ids:
                maps.embedding[signal][pair] = 0
        maps.embedding['goal_+5'][ids[0]] = 100
        maps.embedding['goal_-3'][ids[1]] = 100
        a = maps.embedding_probabilities('obs_+8', 'goal_+5')
        b = maps.embedding_probabilities('obs_+8', 'goal_-3')
        self.assertGreater(a[0], .99)
        self.assertGreater(b[1], .99)

    def test_old_configuration_keeps_original_task(self):
        data = asdict(self.config())
        del data['varying_goals']
        self.assertFalse(TrainingConfig.from_data(data).varying_goals)

    def test_interactive_endpoint_is_repeatable_frozen_and_validates_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            run_experiment(self.config(), root)
            _, bundle = export_demo(root)
            before = json.dumps(bundle, sort_keys=True)
            a = attempt(bundle, 12, -7, 42)
            self.assertEqual(a, attempt(bundle, 12, -7, 42))
            self.assertEqual(a['frames'][0]['position'], 12)
            self.assertEqual((a['lower'], a['upper']), (-20, 20))
            self.assertEqual(json.dumps(bundle, sort_keys=True), before)
            self.assertTrue(attempt(bundle, 3, 3, 42)['frames'][0]['success'])
            for values in [(21, 0, 42), (0, 0, -1), (True, 0, 42)]:
                with self.assertRaises(ValueError):
                    attempt(bundle, *values)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(bundle))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(url) as response:
                    self.assertIn('Animal playground', response.read().decode())
                request = Request(url + '/attempt', data=json.dumps(dict(start=12, goal=-7, seed=42)).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request) as response:
                    self.assertEqual(json.load(response), a)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url + '/attempt', data=b'{}'))
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_legacy_animal_rejects_unsupported_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            run_experiment(self.config(varying_goals=False), root)
            _, bundle = export_demo(root)
            result = attempt(bundle, 12, 0, 42)
            self.assertEqual(result['frames'][0]['position'], 12)
            self.assertEqual((result['lower'], result['upper']), (-20, 20))
            with self.assertRaisesRegex(ValueError, 'no goal input'):
                attempt(bundle, -20, 20, 42)
