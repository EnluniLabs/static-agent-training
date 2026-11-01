import unittest

from static_agent_training.initialize import Config, I, K, depth, initialize, is_closed, node_count, pair_id


class InitializationTests(unittest.TestCase):
    def test_reproducible_independent_streams(self):
        first = initialize(Config(seed=42))
        self.assertEqual(first, initialize(Config(seed=42)))
        self.assertEqual(first['initial_pairs'], initialize(Config(seed=42, count=8))['initial_pairs'][:4])

    def test_closed_bounded_terms(self):
        for max_depth in range(2, 8):
            for max_nodes in range(2, 24):
                config = Config(count=20, max_depth=max_depth, max_nodes=max_nodes, combinator_probability=0)
                for pair in initialize(config)['initial_pairs']:
                    for role in ('left', 'right'):
                        term = pair[role]['ast']
                        self.assertTrue(is_closed(term))
                        self.assertLessEqual(depth(term), max_depth)
                        self.assertLessEqual(node_count(term), max_nodes)

    def test_pair_order(self):
        self.assertNotEqual(pair_id(I, K), pair_id(K, I))

    def test_invalid_configuration(self):
        for config in (Config(count=0), Config(max_depth=1), Config(max_nodes=1), Config(combinator_probability=2)):
            with self.assertRaises(ValueError):
                initialize(config)


if __name__ == '__main__':
    unittest.main()
