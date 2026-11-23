"""Cache regressions; golden digests captured before the cache implementation."""
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import unittest
from unittest.mock import patch

from static_agent_training.environment import NumberLine
from static_agent_training.genome import Species, develop, inherit_lifetime_mappings
from static_agent_training.initialize import I, K
from static_agent_training.mutation import MutationConfig, mutate

NONE = MutationConfig(weight_probability=0, subtree_probability=0, term_probability=0,
                      insertion_probability=0, deletion_probability=0)
GOLDEN_ID = '47d70964b9b6b64bb4db4b2af42decf47d577da9a26478d8af73bddebf014852'
# Canonical digests cover every log field, including streams, draws, operations,
# merger differences and child identities, without large serialized fixtures.
GOLDEN_LOGS = (
    '807712aad668c016eaa4bc97e8ecb92b562ea5fe7ab87ed7b32125851842dd43',
    '755abd319379ed4e59c5cc09339e311955b6b4b8ee90ad44f7851e2b100231aa',
    '641bb4fe22a3b3eae39b255ba88e0bf01a644ff2cd3b2fbf4f411dc4560e7c1c',
    '7997aa0b8566b0992844c5fbeb0cf8014da8d3bd9e8df9a975a42cf407039c4c',
    'a0cf6de0c855f676288d11304c8a47aaf20525cab756591a1dfc9d41dc65998a',
    '69dcd69d0beafeef566ef5733c4568a9062c10254bd4de856f8b36b4eeb43eda',
    'e15bc7c39ddb8db67d7d7c3c91b37bdc9143058846e27775329d920157301d21',
)


def uncached_id(species):
    payload = dict(initial_pairs=species.initial_pairs, budgets=species.budgets,
                   mappings=species.inherited.to_data(),
                   development_config=asdict(species.development_config))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class GenomeCacheTests(unittest.TestCase):
    def setUp(self):
        env = NumberLine()
        self.parent = develop([(I, K), (I, I)],
                              tuple(s.token for s in env.observation_tokens),
                              tuple(o.token for o in env.output_tokens),
                              budgets=[10, 20], development_seed=7, mapping_seed=9)

    def test_golden_hash_and_one_hash_per_instance(self):
        with patch('static_agent_training.genome.hashlib.sha256', wraps=hashlib.sha256) as digest:
            self.assertEqual([self.parent.genotype_id for _ in range(20)], [GOLDEN_ID] * 20)
            self.assertEqual(digest.call_count, 1)
        self.assertEqual(self.parent.genotype_id, uncached_id(self.parent))
        other = replace(self.parent)
        self.assertNotIn('genotype_id', other.__dict__)
        self.assertEqual(other.genotype_id, GOLDEN_ID)

    def test_replace_recomputes_identity_and_provenance(self):
        self.parent.genotype_id
        self.parent._merger_identities
        child = replace(self.parent, budgets=(11, 20))
        self.assertNotIn('genotype_id', child.__dict__)
        self.assertNotIn('_merger_identities', child.__dict__)
        self.assertEqual(child.genotype_id, uncached_id(child))
        self.assertNotEqual(child.genotype_id, self.parent.genotype_id)
        data = json.loads(self.parent.development_json)
        data['events'] = []
        child = replace(self.parent, development_json=json.dumps(data))
        self.assertEqual(child._merger_identities, frozenset())

    def test_mutation_and_inheritance_fresh_identities(self):
        self.parent.genotype_id
        child, _ = mutate(self.parent, seed=2, config=replace(NONE, weight_probability=1))
        self.assertEqual(child.genotype_id, uncached_id(child))
        self.assertNotEqual(child.genotype_id, self.parent.genotype_id)
        lifetime = self.parent.inherited.lifetime()
        row = next(iter(lifetime.embedding.values()))
        column = next(iter(row))
        row[column] += 1
        inherited = inherit_lifetime_mappings(self.parent, lifetime.embedding, lifetime.output)
        self.assertNotIn('genotype_id', inherited.__dict__)
        self.assertEqual(inherited.genotype_id, uncached_id(inherited))
        self.assertNotEqual(inherited.genotype_id, self.parent.genotype_id)
        row[column] += 1
        self.assertEqual(inherited.genotype_id, uncached_id(inherited))

    def test_full_logs_match_prechange_references(self):
        configs = [NONE, replace(NONE, weight_probability=1)]
        configs += [replace(NONE, **{field: 1}) for field in (
            'subtree_probability', 'term_probability', 'insertion_probability', 'deletion_probability')]
        configs += [MutationConfig(weight_probability=1, subtree_probability=1, term_probability=1,
                                   insertion_probability=1, deletion_probability=1)]
        for seed, (config, expected) in enumerate(zip(configs, GOLDEN_LOGS), 1):
            with self.subTest(seed=seed):
                child, log = mutate(self.parent, seed=seed, config=config)
                digest = hashlib.sha256(json.dumps(log, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(digest, expected)
                self.assertEqual(child.genotype_id, uncached_id(child))

    def test_no_structure_never_parses_development(self):
        with patch('static_agent_training.genome.json.loads', side_effect=AssertionError('parsed')):
            for config in (NONE, replace(NONE, weight_probability=1)):
                child, log = mutate(self.parent, seed=2, config=config)
                self.assertEqual(child.development_json, self.parent.development_json)
                self.assertEqual(log['lost_mergers'], [])
                self.assertEqual(log['new_mergers'], [])
        self.assertNotIn('_merger_identities', self.parent.__dict__)

    def test_parent_summary_parsed_once_and_only_immutable_cache(self):
        from static_agent_training import serialization
        with patch('static_agent_training.genome.json.loads', wraps=serialization.loads) as loads:
            for seed in (3, 4):
                mutate(self.parent, seed=seed, config=replace(NONE, term_probability=1))
            self.assertEqual(loads.call_count, 1)
        summary = self.parent._merger_identities
        self.assertIsInstance(summary, frozenset)
        merged = develop([(I, I), (I, I)], self.parent.inherited.observations,
                         self.parent.inherited.outputs, budgets=10)
        self.assertTrue(merged._merger_identities)
        for identity, sources in merged._merger_identities:
            self.assertIsInstance(identity, str)
            self.assertIsInstance(sources, tuple)
            self.assertTrue(all(isinstance(source, str) for source in sources))
        exported = self.parent.to_data()
        exported['development']['events'].clear()
        next(iter(exported['inherited']['embedding'].values())).clear()
        exported['repertoire']['pairs'].clear()
        self.assertEqual(self.parent._merger_identities, summary)
        self.assertEqual(self.parent.genotype_id, GOLDEN_ID)
        restored = Species.from_data(self.parent.to_data())
        self.assertEqual(restored.genotype_id, GOLDEN_ID)
        self.assertNotIn('genotype_id', self.parent.to_data())
        with self.assertRaises(FrozenInstanceError):
            self.parent.budgets = (0, 0)
        with self.assertRaises(TypeError):
            self.parent.repertoire.pairs['new'] = (I, I)
        with self.assertRaises(TypeError):
            next(iter(self.parent.inherited.embedding.values()))['new'] = 0


if __name__ == '__main__':
    unittest.main()
