from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch

from static_agent_training.engine import run
from static_agent_training.environment import NumberLine
from static_agent_training.genome import DevelopmentConfig, Species, develop
from static_agent_training.initialize import Config, I, K, depth, initialize, is_closed, node_count
from static_agent_training.mutation import MutationConfig, mutate


NONE = MutationConfig(weight_probability=0, subtree_probability=0, term_probability=0,
                      insertion_probability=0, deletion_probability=0)


class MutationTests(unittest.TestCase):
    def setUp(self):
        env = NumberLine()
        self.observations = tuple(s.token for s in env.observation_tokens)
        self.outputs = tuple(o.token for o in env.output_tokens)
        self.parent = develop([(I, K), (I, I)], self.observations, self.outputs,
                              budgets=[10, 20], development_seed=7, mapping_seed=9)

    def assert_bounded(self, species, config):
        for pair in species.initial_pairs:
            for term in pair:
                self.assertTrue(is_closed(term))
                self.assertLessEqual(depth(term), config.max_depth)
                self.assertLessEqual(node_count(term), config.max_nodes)
        self.assertEqual(set(species.inherited.pair_ids), set(species.repertoire.pairs))

    def test_no_mutation_reuses_structure_without_reducing(self):
        before = self.parent.to_data()
        with patch('static_agent_training.engine.run', side_effect=AssertionError('unexpected reduction')):
            child, log = mutate(self.parent, seed=1, config=NONE)
        self.assertIs(child.repertoire, self.parent.repertoire)
        self.assertEqual(child.genotype_id, self.parent.genotype_id)
        self.assertFalse(log['redeveloped'])
        self.assertEqual(log['development_contractions'], 0)
        self.assertEqual(before, self.parent.to_data())

    def test_representation_noise_only_never_reduces_or_inherits_learning(self):
        lifetime = self.parent.inherited.lifetime()
        for row in lifetime.output.values():
            for token in row:
                row[token] = 10000
        before = self.parent.to_data()
        with patch('static_agent_training.engine.run', side_effect=AssertionError('unexpected reduction')):
            child, log = mutate(self.parent, seed=2, config=replace(NONE, weight_probability=1))
        self.assertIs(child.repertoire, self.parent.repertoire)
        self.assertFalse(log['structural_changed'])
        deltas = []
        for kind in ('embedding', 'output'):
            old, new = getattr(self.parent.inherited, kind), getattr(child.inherited, kind)
            deltas.extend(new[row][col] - old[row][col] for row in new for col in new[row])
        self.assertEqual(log['perturbed_logits'], len(deltas))
        self.assertTrue(all(delta != 0 for delta in deltas))
        self.assertLess(abs(sum(deltas) / len(deltas)), .03)
        self.assertTrue(.07 < math.sqrt(sum(d*d for d in deltas) / len(deltas)) < .13)
        self.assertEqual(before, self.parent.to_data())

    def test_structural_redevelopment_and_mapping_reconciliation(self):
        for field in ('subtree_probability', 'term_probability', 'insertion_probability', 'deletion_probability'):
            config = replace(NONE, **{field: 1})
            with patch('static_agent_training.engine.run', wraps=run) as reducer:
                child, log = mutate(self.parent, seed=3, config=config)
            self.assertTrue(log['structural_changed'], field)
            self.assertEqual(reducer.call_count, 1)
            self.assertEqual(tuple(reducer.call_args.args[0]), child.initial_pairs)
            self.assertEqual(tuple(reducer.call_args.args[1]), child.budgets)
            self.assertEqual(log['perturbed_logits'], 0)
            self.assert_bounded(child, config)
            for q in log['retained_pairs']:
                self.assertEqual(dict(self.parent.inherited.output[q]), dict(child.inherited.output[q]))
                for signal in self.observations:
                    self.assertEqual(self.parent.inherited.embedding[signal][q], child.inherited.embedding[signal][q])
            self.assertEqual(log['added_pairs'], sorted(set(child.repertoire.pairs) - set(self.parent.repertoire.pairs)))
            self.assertEqual(log['removed_pairs'], sorted(set(self.parent.repertoire.pairs) - set(child.repertoire.pairs)))
            if field == 'insertion_probability':
                self.assertEqual(len(child.initial_pairs), 3)
                self.assertIn(config.new_pair_budget, child.budgets)
            if field == 'deletion_probability':
                self.assertEqual(len(child.initial_pairs), 1)

    def test_structural_mutation_can_split_a_merger(self):
        parent = develop([(I, I), (I, I)], self.observations, self.outputs, budgets=10)
        delta = ('abs', ('app', ('var', 0), ('var', 0)))
        omega = ('app', delta, delta)
        with patch('static_agent_training.mutation.generate_term', return_value=omega):
            child, log = mutate(parent, seed=4, config=replace(NONE, term_probability=1))
        self.assertTrue(log['lost_mergers'])
        self.assertFalse(log['new_mergers'])
        development = json.loads(child.development_json)
        self.assertEqual(len(development['groups']), 2)
        self.assertFalse(any(e['kind'] == 'MERGE' for e in development['events']))
        self.assertEqual(development['accounting']['initial'], 20)

    def test_minimum_one_and_impossible_mutation_are_logged_noops(self):
        parent = develop([(I, I)], self.observations, self.outputs)
        with patch('static_agent_training.engine.run', side_effect=AssertionError('unexpected reduction')):
            child, log = mutate(parent, seed=5, config=replace(NONE, deletion_probability=1))
            self.assertEqual(child.initial_pairs, parent.initial_pairs)
            self.assertEqual(log['operations'][0]['reason'], 'minimum_one_pair')
            child, log = mutate(parent, seed=5, config=replace(NONE, term_probability=1, max_depth=2, max_nodes=2))
            self.assertFalse(log['structural_changed'])
            self.assertFalse(log['operations'][0]['changed'])
            self.assertEqual(log['operations'][0]['attempts'], 32)

    def test_reproducible_saved_species_and_offspring(self):
        saved = json.loads(json.dumps(self.parent.to_data(), sort_keys=True))
        with patch('static_agent_training.engine.run', side_effect=AssertionError('loading must not reduce')):
            restored = Species.from_data(saved)
        self.assertEqual(restored.genotype_id, self.parent.genotype_id)
        config = replace(NONE, weight_probability=1, term_probability=1)
        first, first_log = mutate(self.parent, seed=6, config=config)
        second, second_log = mutate(restored, seed=6, config=config)
        self.assertEqual(first.genotype_id, second.genotype_id)
        self.assertEqual(first_log, second_log)
        self.assertEqual(first.to_data(), second.to_data())
        json.dumps(first_log)

    def test_independent_mutations_and_context_safe_subtrees(self):
        config = MutationConfig(weight_probability=1, subtree_probability=1, term_probability=1,
                                insertion_probability=1, deletion_probability=1)
        for seed in range(50):
            child, log = mutate(self.parent, seed=seed, config=config)
            self.assertTrue(all(log['selected'].values()))
            self.assertEqual([o['operator'] for o in log['operations']], ['subtree', 'term', 'insert', 'delete'])
            self.assertEqual(len(child.initial_pairs), 2)
            self.assert_bounded(child, config)
        for seed in range(50):
            init = initialize(Config(seed=seed))
            pairs = [(p['left']['ast'], p['right']['ast']) for p in init['initial_pairs']]
            parent = develop(pairs, self.observations, self.outputs, budgets=3)
            child, log = mutate(parent, seed=seed, config=replace(NONE, subtree_probability=1))
            self.assert_bounded(child, config)

    def test_rejects_lifetime_state_and_invalid_config(self):
        with self.assertRaises(TypeError):
            mutate(self.parent.inherited.lifetime(), seed=1)
        for kwargs in ({'weight_probability': 2}, {'weight_stddev': -1}, {'max_depth': 1},
                       {'new_pair_budget': -1}, {'attempts': 0}):
            with self.assertRaises(ValueError):
                MutationConfig(**kwargs)
        with self.assertRaises(ValueError):
            DevelopmentConfig(max_nodes=0)


if __name__ == '__main__':
    unittest.main()
