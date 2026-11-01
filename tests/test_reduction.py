import unittest

from static_agent_training.initialize import I, K, is_closed
from static_agent_training.reduction import normalize, contract


class ReductionTests(unittest.TestCase):
    def test_identity_and_exact_budget(self):
        term = ('app', I, K)
        self.assertEqual(normalize(term, 0).status, 'BUDGET')
        result = normalize(term, 1)
        self.assertEqual((result.status, result.term, result.remaining), ('NORMAL', K, 0))

    def test_capture_avoidance_under_binder(self):
        # lambda y. ((lambda x. lambda z. x) y) -> lambda y. lambda z. y
        term = ('abs', ('app', ('abs', ('abs', ('var', 1))), ('var', 0)))
        result = normalize(term, 1)
        self.assertEqual(result.term, ('abs', ('abs', ('var', 1))))
        self.assertTrue(is_closed(result.term))

    def test_full_normalization(self):
        self.assertEqual(normalize(('abs', ('app', I, ('var', 0))), 1).term, I)

    def test_divergence_consumes_budget(self):
        delta = ('abs', ('app', ('var', 0), ('var', 0)))
        result = normalize(('app', delta, delta), 7)
        self.assertEqual((result.status, result.steps, result.remaining), ('BUDGET', 7, 0))

    def test_size_and_replay(self):
        term = ('app', ('app', I, I), ('app', I, I))
        self.assertEqual(normalize(term, 10, max_nodes=2).status, 'SIZE')
        self.assertEqual(normalize(term, 10, seed=42), normalize(term, 10, seed=42))
        results = [normalize(term, 10, seed=s) for s in range(20)]
        self.assertTrue(all(r.status == 'NORMAL' and r.term == I for r in results))


if __name__ == '__main__':
    unittest.main()
