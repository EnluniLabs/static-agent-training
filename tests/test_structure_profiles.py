import argparse
from dataclasses import asdict, replace
import json
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.evolution import TrainingConfig
from static_agent_training import experiment, comparison
from static_agent_training.structure_profiles import (
    STRUCTURE_PRESETS, add_structure_arguments, apply_structure_arguments,
)


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_structure_arguments(parser)
    return parser.parse_args(flags)


class StructureProfileTests(unittest.TestCase):
    def test_baseline_defaults_unchanged(self):
        config = TrainingConfig()
        self.assertEqual((config.initial_pairs, config.chain_budget), (4, 64))
        self.assertEqual((config.mutation.max_nodes, config.mutation.max_depth), (31, 6))
        self.assertEqual((config.development.max_nodes, config.development.max_transitions), (256, 1000))
        self.assertEqual(apply_structure_arguments(config, arguments('--structure-preset', 'baseline')), config)
        self.assertIs(apply_structure_arguments(config, arguments()), config)
        self.assertIs(apply_structure_arguments(config, arguments('--structure-preset', 'custom')), config)

    def test_exploratory_resolves_all_flat_controls_and_roundtrips(self):
        config = apply_structure_arguments(TrainingConfig(), arguments('--structure-preset', 'exploratory'))
        expected = STRUCTURE_PRESETS['exploratory']
        actual = dict(initial_pairs=config.initial_pairs, chain_budget=config.chain_budget,
                      seed_term_max_nodes=config.mutation.max_nodes, seed_term_max_depth=config.mutation.max_depth,
                      development_max_nodes=config.development.max_nodes,
                      development_max_transitions=config.development.max_transitions)
        actual.update((key, getattr(config.mutation, key)) for key in expected if key not in actual)
        self.assertEqual(actual, expected)
        data = json.loads(json.dumps(asdict(config)))
        self.assertNotIn('structure_preset', data)
        self.assertEqual(TrainingConfig.from_data(data), config)

    def test_overrides_preserve_other_nested_fields_and_input(self):
        base = TrainingConfig()
        base = replace(base, mutation=replace(base.mutation, weight_probability=.23, weight_stddev=.7, attempts=9))
        before = asdict(base)
        resolved = apply_structure_arguments(base, arguments('--initial-pairs', '7', '--structure-preset', 'exploratory',
                                                             '--seed-term-max-depth', '8', '--subtree-probability', '.4'))
        self.assertEqual((resolved.initial_pairs, resolved.mutation.max_depth, resolved.mutation.subtree_probability), (7, 8, .4))
        self.assertEqual((resolved.mutation.weight_probability, resolved.mutation.weight_stddev, resolved.mutation.attempts), (.23, .7, 9))
        self.assertEqual(resolved.learning_config, base.learning_config)
        self.assertEqual(asdict(base), before)
        explicit = apply_structure_arguments(resolved, {'chain_budget': 0, 'new_pair_budget': 0})
        self.assertEqual(explicit.mutation, replace(resolved.mutation, new_pair_budget=0))
        self.assertEqual(explicit.chain_budget, 0)
        self.assertEqual(apply_structure_arguments(resolved, {'structure_preset': 'baseline'}).initial_pairs, 4)

    def test_invalid_numeric_controls(self):
        for key, value in [('initial_pairs', True), ('subtree_probability', False),
                           ('development_max_nodes', 0), ('seed_term_max_depth', -1),
                           ('initial_pairs', 0), ('new_pair_budget', -1),
                           ('term_probability', float('nan')), ('combinator_probability', float('inf')),
                           ('deletion_probability', 1.1), ('chain_budget', 1.2),
                           ('structure_preset', 'missing')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                apply_structure_arguments(TrainingConfig(), {key: value})
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            arguments('--initial-pairs', 'false')

    def test_cli_help_and_resume_rejection(self):
        for module in (experiment, comparison):
            with self.subTest(module=module.__name__), patch('sys.argv', ['cli', '--help']), patch('sys.stdout') as output:
                with self.assertRaises(SystemExit) as stopped:
                    module.main()
                self.assertEqual(stopped.exception.code, 0)
                self.assertIn('--structure-preset', ''.join(call.args[0] for call in output.write.call_args_list))
            for flags in (['--structure-preset', 'custom'], ['--initial-pairs', '4']):
                with patch('sys.argv', ['cli', '--resume', '/missing', *flags]), patch('sys.stderr'), self.assertRaises(SystemExit) as stopped:
                    module.main()
                self.assertEqual(stopped.exception.code, 2)

    def test_cli_saved_resume_preserves_resolved_config(self):
        config = apply_structure_arguments(TrainingConfig(), {'structure_preset': 'exploratory'})
        with tempfile.TemporaryDirectory() as directory:
            experiment.write_json(experiment.Path(directory) / experiment.CHECKPOINT,
                                  dict(schema_version='crt-training-v1', config=asdict(config)))
            with patch('sys.argv', ['cli', '--resume', directory, '--generations', '501', '--no-live']), \
                 patch.object(experiment, 'run_experiment') as run, patch('static_agent_training.diagnostics.diagnostics'), patch('builtins.print'):
                experiment.main()
            self.assertEqual(run.call_args.args[0], replace(config, generations=501))


if __name__ == '__main__':
    unittest.main()
