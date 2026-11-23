"""Packing must preserve the thaw-based values and lifetime isolation."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.experimental.sensory_accelerated import _pack, _packed_sources
from static_agent_training.experimental.sensory_genome import SensoryMappings


class PackedGenomeCacheTests(unittest.TestCase):
    def setUp(self):
        self.population = initialize_population(TrainingConfig(
            model='senses', population=3, elites=1, initial_pairs=3,
            chain_budget=8, receptors=16))

    def test_exact_thaw_reference_values_padding_and_topology(self):
        parent = self.population[0]
        ids = parent.inherited.pair_ids
        # Exercise padding, reordered IDs, and a different repertoire size.
        repertoire = replace(parent.repertoire, pairs={q: parent.repertoire.pairs[q]
                                                      for q in reversed(ids[:2])},
                             successors={q: successor for q, successor in parent.repertoire.successors.items()
                                         if q in ids[:2] and successor in ids[:2]})
        mappings = SensoryMappings.initialize(repertoire, senses=parent.inherited.senses,
                                              previous=parent.inherited)
        population = [parent, replace(parent, repertoire=repertoire, inherited=mappings)]
        for _ in range(2):
            arrays = _pack(population, 0, True)
            expected = [np.zeros_like(a) for a in arrays[:6]]
            expected[4].fill(-1)
            for i, individual in enumerate(population):
                inherited = individual.inherited
                parameters = inherited.to_parameters()
                ids = inherited.pair_ids
                lookup = {q: j for j, q in enumerate(ids)}
                n = len(ids)
                expected[0][i, :n] = parameters['bias']
                expected[1][i, :, :n] = parameters['weights']
                expected[5][i] = n
                for j, q in enumerate(ids):
                    expected[2][i, j] = parameters['gates'][q]
                    expected[3][i, j] = parameters['prediction'][q][0]
                    expected[4][i, j] = lookup.get(individual.repertoire.successors.get(q), -1)
            for actual, reference in zip(arrays, expected):
                np.testing.assert_array_equal(actual, reference)

    def test_cache_readonly_reused_without_thaw(self):
        inherited = self.population[0].inherited
        with patch.object(SensoryMappings, 'to_parameters', side_effect=AssertionError('thaw')):
            _pack(self.population, 0, True)
            sources = _packed_sources(inherited)
            _pack(self.population, 0, True)
            self.assertIs(sources, _packed_sources(inherited))
        for source in sources:
            self.assertFalse(source.flags.writeable)
            with self.assertRaises(ValueError):
                source.flat[0] = 0
            with self.assertRaises(ValueError):
                source.setflags(write=True)

    def test_batch_and_public_parameters_are_independent(self):
        parent = self.population[0]
        population = [parent, parent]
        before = _pack(population, 0, True)
        expected = [a.copy() for a in before[:4]]
        for array in before[:4]:
            self.assertTrue(array.flags.writeable)
            array[0].fill(123)
            for source in _packed_sources(parent.inherited):
                self.assertFalse(np.shares_memory(array, source))
        for array, reference in zip(before[:4], expected):
            np.testing.assert_array_equal(array[1], reference[1])
        public = parent.inherited.to_parameters()
        public['bias'][0] = 999
        public['weights'][0][0] = 999
        public['gates'][parent.inherited.pair_ids[0]][0] = 999
        public['prediction'][parent.inherited.pair_ids[0]][0][0] = 999
        for actual, reference in zip(_pack(population, 0, True), expected):
            np.testing.assert_array_equal(actual, reference)
        self.assertNotEqual(parent.inherited.to_parameters()['bias'][0], 999)

    def test_new_instances_reconciliation_perturbation_and_inheritance(self):
        from static_agent_training.experimental.sensory_accelerated import learned_mappings
        parent = self.population[0]
        original = _packed_sources(parent.inherited)
        changed, _ = parent.inherited.perturb(seed=123, scale=.1)
        learned = _pack([parent], 0, True)
        learned[0][0, 0] += 1
        candidates = [
            SensoryMappings(parent.inherited.pair_ids, parent.inherited.senses,
                           parent.inherited.to_parameters()),
            SensoryMappings.initialize(parent.repertoire, senses=parent.inherited.senses,
                                       previous=parent.inherited),
            changed, learned_mappings(parent, learned, 0)]
        for inherited in candidates:
            self.assertFalse(hasattr(inherited, '_packed_numpy_sources'))
            sources = _packed_sources(inherited)
            self.assertIsNot(sources, original)
            for actual, old in zip(sources, original):
                self.assertFalse(np.shares_memory(actual, old))
            packed = _pack([replace(parent, inherited=inherited)], 0, True)
            np.testing.assert_array_equal(packed[0][0, :len(inherited.pair_ids)],
                                          inherited.to_parameters()['bias'])
        self.assertNotEqual(candidates[-1].parameters['bias'][0], parent.inherited.parameters['bias'][0])
