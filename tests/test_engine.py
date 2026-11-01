import unittest

from static_agent_training.engine import run
from static_agent_training.initialize import I, K, Config, initialize
from static_agent_training.reduction import normalize, contract, redexes


class EngineTests(unittest.TestCase):
    def assert_conserved(self, result):
        a = result['accounting']
        self.assertEqual(a['initial'], a['spent'] + a['remaining'])
        self.assertEqual(a['spent'], sum(e['kind'] == 'CONTRACT' for e in result['events']))

    def test_single_chain_and_collapse(self):
        result = run([(I, K)], 30, max_transitions=4)
        a, b = I, K
        collapsed_a, collapsed_b = a, b
        edges = [e for e in result['events'] if e['kind'] == 'NORMAL']
        self.assertEqual(len(edges), 4)
        for event in edges:
            c = normalize(('app', b, a), 1000).term
            self.assertEqual(event['pair'], (b, c))
            collapsed_a, collapsed_b = collapsed_b, ('app', collapsed_b, collapsed_a)
            self.assertEqual(normalize(collapsed_b, 1000, max_nodes=10000).term, c)
            a, b = b, c
        self.assert_conserved(result)

    def test_duplicate_starts_pool_before_computation(self):
        result = run([(I, I), (I, I)], [2, 3])
        merges = [e for e in result['events'] if e['kind'] == 'MERGE']
        self.assertEqual(len(merges), 1)
        self.assertEqual(result['groups'][0]['sources'], [0, 1])
        self.assertEqual(result['accounting']['spent'], 5)
        self.assert_conserved(result)

    def test_one_shared_term_does_not_merge(self):
        result = run([(I, I), (I, ('abs', I))], 10, max_transitions=1)
        self.assertFalse(any(e['kind'] == 'MERGE' for e in result['events']))
        self.assertEqual(len(result['groups']), 2)

    def test_delayed_arrival_revives_exhausted_execution(self):
        result = run([(I, I), (('app', I, I), I)], [1, 8])
        merges = [e for e in result['events'] if e['kind'] == 'MERGE']
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]['target_status'], 'BUDGET')
        self.assertEqual(merges[0]['target_budget'], 0)
        self.assertGreater(merges[0]['incoming_budget'], 0)
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(result['groups'][0]['sources'], [0, 1])
        self.assertEqual(result['groups'][0]['status'], 'BUDGET')
        self.assert_conserved(result)

    def test_late_budget_follows_execution_that_has_moved_on(self):
        result = run([(I, K), (('app', I, I), K)], [30, 30], max_transitions=10)
        merge = next(e for e in result['events'] if e['kind'] == 'MERGE')
        self.assertNotEqual(merge['pair_id'], merge['target_pair_id'])
        self.assertEqual(merge['target_status'], 'ACTIVE')
        self.assertEqual(len(result['groups']), 1)
        self.assert_conserved(result)

    def test_distinct_starts_arrive_simultaneously(self):
        left1 = ('app', I, I)
        left2 = ('app', ('abs', I), K)
        result = run([(left1, I), (left2, I)], [5, 7])
        merges = [e for e in result['events'] if e['kind'] == 'MERGE']
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]['pooled_budget'], 8)
        self.assertEqual(merges[0]['round'], 2)
        self.assertEqual(result['groups'][0]['sources'], [0, 1])
        self.assert_conserved(result)

    def test_pooling_reaches_further(self):
        single = run([(I, I)], 2)
        pooled = run([(I, I), (I, I)], [2, 3])
        count = lambda r: sum(e['kind'] == 'NORMAL' for e in r['events'])
        self.assertGreater(count(pooled), count(single))

    def test_unknown_size_depth_and_empty_budget(self):
        delta = ('abs', ('app', ('var', 0), ('var', 0)))
        result = run([(delta, delta)], 9)
        self.assertEqual(result['groups'][0]['status'], 'BUDGET')
        self.assertFalse(any(e['kind'] == 'NORMAL' for e in result['events']))
        self.assertEqual(run([(I, I)], 2, max_nodes=2)['groups'][0]['status'], 'SIZE')
        self.assertEqual(run([(I, I)], 2, max_transitions=1)['groups'][0]['status'], 'DEPTH')
        self.assertEqual(run([(I, I)], 0)['accounting']['spent'], 0)

    def test_seeded_generated_runs_and_json_asts(self):
        for seed in range(20):
            data = initialize(Config(seed=seed))
            pairs = [(p['left']['ast'], p['right']['ast']) for p in data['initial_pairs']]
            result = run(pairs, [3, 7, 11, 13], seed=seed, max_nodes=100)
            self.assertEqual(result, run(pairs, [3, 7, 11, 13], seed=seed, max_nodes=100))
            self.assert_conserved(result)
            # Replay actual contraction witnesses, independently of engine state.
            expressions = {}
            for event in result['events']:
                execution = event['execution']
                if event['kind'] == 'PAIR':
                    left, right = event['pair']
                    expressions[execution] = ('app', right, left)
                elif event['kind'] == 'CONTRACT':
                    expression = expressions[execution]
                    self.assertIn(event['position'], tuple(redexes(expression)))
                    expressions[execution] = contract(expression, event['position'])
                elif event['kind'] == 'NORMAL':
                    expression = expressions[execution]
                    self.assertFalse(tuple(redexes(expression)))
                    self.assertEqual(expression, event['pair'][1])
        result = run([(['abs', ['var', 0]], ['abs', ['var', 0]])], 1)
        self.assertEqual(result['accounting']['spent'], 1)

    def test_invalid_inputs(self):
        for pairs, budgets in [([], 1), ([(I, I)], -1), ([(I, I)], []),
                               ([ (('var', 0), I)], 1), ([(('var', -1), I)], 1)]:
            with self.assertRaises(ValueError):
                run(pairs, budgets)


if __name__ == '__main__':
    unittest.main()
