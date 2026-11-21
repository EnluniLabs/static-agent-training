import unittest
from collections import Counter

from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.structural_diagnostics import population_structure


class StructuralDiagnosticsTests(unittest.TestCase):
    def test_read_only_metrics_are_equal_animal_weighted(self):
        animals = initialize_population(TrainingConfig(population=2,elites=1,initial_pairs=3))
        before = [a.to_data() for a in animals]
        summary = population_structure(animals, 1)
        expected = sum(sum(n>1 for n in Counter(a.repertoire.successors.values()).values()) for a in animals)/2
        self.assertEqual(summary['mean_convergence_states'], expected)
        self.assertGreater(summary['mean_seed_term_nodes'], 0)
        self.assertGreater(summary['mean_seed_term_depth'], 0)
        self.assertEqual(sum(summary['champion_development_stops'].values()), len(before[1]['development']['groups']))
        accounting=summary['champion_development_accounting']
        self.assertEqual(accounting['initial'],accounting['spent']+accounting['remaining'])
        self.assertEqual([a.to_data() for a in animals],before)
