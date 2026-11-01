from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.evaluation import evaluate
from static_agent_training.evolution import (TrainingConfig, evaluate_generation,
                                             initialize_population, partitions, reproduce)
from static_agent_training.experiment import load_checkpoint, run_experiment
from static_agent_training.mutation import MutationConfig
from static_agent_training.terminal import Dashboard, plot


NONE = MutationConfig(weight_probability=0, subtree_probability=0, term_probability=0,
                      insertion_probability=0, deletion_probability=0)


class TrainingTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(TrainingConfig(population=3, generations=3, episodes=3,
                                      assessment_episodes=3, elites=1, initial_pairs=2,
                                      chain_budget=8, seed=19), **kwargs)

    def test_common_episode_order_disjoint_assessment_and_no_runtime_reduction(self):
        config = self.config()
        population = initialize_population(config)
        before = [p.to_data() for p in population]
        split = partitions(config.seed)
        self.assertFalse(set(split['training']) & set(split['assessment']))
        self.assertEqual(set(split['training']) | set(split['assessment']), set(range(-20, 21)) - {0})
        with patch('static_agent_training.engine.run', side_effect=AssertionError('runtime development')), \
             patch('static_agent_training.engine.contract', side_effect=AssertionError('runtime reduction')):
            record, detail = evaluate_generation(population, config, 0, split)
        for result in record['individuals']:
            self.assertEqual([e['start'] for e in result['training']['episodes']], list(record['training_starts']))
            self.assertEqual([e['start'] for e in result['assessment']['episodes']], list(record['assessment_starts']))
            self.assertEqual(result['assessment']['empirical_updates'], 0)
            self.assertEqual(result['assessment']['conflicts'], 0)
        self.assertEqual(before, [p.to_data() for p in population])
        self.assertFalse(any(e['kind'] in ('EMPIRICAL_UPDATE', 'CONFLICT') for e in detail['assessment_events']))

    def test_selection_uses_training_fitness_and_elites_keep_inheritance(self):
        config = self.config(mutation=NONE, tournament_size=3)
        population = initialize_population(config)
        results = [dict(training=dict(fitness=s), assessment=dict(fitness=h))
                   for s, h in ((0, 1), (1, 0), (.5, .5))]
        with patch('static_agent_training.engine.run', side_effect=AssertionError('no structural edits')):
            children, ancestry = reproduce(population, results, config, 0)
        self.assertIs(children[0], population[1])
        self.assertTrue(all(e['parent_index'] == 1 for e in ancestry))
        self.assertTrue(all(c.genotype_id == population[1].genotype_id for c in children))

    def test_checkpoint_resume_matches_uninterrupted_run(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            full_path, resumed_path = Path(tmp) / 'full', Path(tmp) / 'resumed'
            full = run_experiment(config, full_path)
            partial = run_experiment(config, resumed_path, until_generation=1)
            self.assertEqual(partial['next_generation'], 1)
            resumed = run_experiment(config, resumed_path, resume=True)
            self.assertEqual(full, resumed)
            self.assertEqual(load_checkpoint(full_path), load_checkpoint(resumed_path))
            for generation in range(3):
                name = f'generations/{generation:05d}.json'
                self.assertEqual(json.loads((full_path / name).read_text()),
                                 json.loads((resumed_path / name).read_text()))
            with self.assertRaises(ValueError):
                run_experiment(config, full_path)
            with self.assertRaises(ValueError):
                run_experiment(replace(config, seed=20), resumed_path, resume=True)

    def test_resume_can_extend_generation_target(self):
        config = self.config(generations=1, mutation=NONE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run'
            run_experiment(config, path)
            state = run_experiment(replace(config, generations=2), path, resume=True)
            self.assertEqual(state['next_generation'], 2)
            self.assertEqual(len(state['history']), 2)
            run_experiment(replace(config, generations=4), path, resume=True, until_generation=2)
            self.assertEqual(load_checkpoint(path)['config']['generations'], 4)

    def test_fixed_control_and_validation(self):
        config = self.config(learning=False)
        individual = initialize_population(config)[0]
        result, detail = evaluate(individual, [1, 2], [3, 4], seed=5, learning=False)
        self.assertEqual(result['training']['empirical_updates'], 0)
        self.assertEqual(result['training']['conflicts'], 0)
        with self.assertRaises(ValueError):
            evaluate(individual, [1], [1], seed=5)
        with self.assertRaises(ValueError):
            self.config(population=1)
        self.assertEqual(TrainingConfig.from_data(asdict(config)), config)

    def test_terminal_graph_and_non_tty_fallback(self):
        history = [dict(generation=i, mean=.2, best=.8, assessment_mean=.4) for i in range(100)]
        rendered = plot(history, width=20, height=6)
        self.assertIn(' 80%', rendered)
        self.assertIn('  0%', rendered)
        self.assertIn('generations 1 to 100', rendered)
        self.assertIn('M', rendered)
        self.assertIn('B', rendered)
        self.assertIn('H', rendered)
        self.assertTrue(all(len(line) <= 26 for line in rendered.splitlines()[:7]))
        plain = io.StringIO()
        dashboard = Dashboard(self.config(), stream=plain)
        dashboard(dict(phase='generation-complete', history=history, summary=history[-1]))
        self.assertNotIn('\x1b', plain.getvalue())
        self.assertIn('held-out', plain.getvalue())
        class TTY(io.StringIO):
            def isatty(self):
                return True
        terminal = TTY()
        dashboard = Dashboard(self.config(), stream=terminal)
        dashboard(dict(phase='generation-complete', history=history, summary=history[-1]))
        self.assertIn('\x1b[H', terminal.getvalue())
        self.assertIn('mean held-out', terminal.getvalue())


if __name__ == '__main__':
    unittest.main()
