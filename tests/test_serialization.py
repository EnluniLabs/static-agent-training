"""Regression coverage for deep development terms in checkpoint JSON."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from static_agent_training import serialization
from static_agent_training.experiment import CHECKPOINT, load_checkpoint, write_json
from static_agent_training.genome import Species, develop
from static_agent_training.initialize import I, canonical_bytes
from static_agent_training.repertoire import freeze


def deep_term():
    term = I
    for _ in range(2500):
        term = ('app', term, I)
    return term


class SerializationTests(unittest.TestCase):
    def test_shallow_compatibility(self):
        data = {'z': [None, True, False, 4, 1.5, 'λ\n"'], 'a': {'x': []}}
        for kwargs in ({}, {'sort_keys': True, 'separators': (',', ':')}, {'indent': 2}):
            self.assertEqual(serialization.dumps(data, **kwargs), json.dumps(data, **kwargs))
            self.assertEqual(serialization.loads(json.dumps(data)), data)
            stream = io.StringIO()
            serialization.dump(data, stream, **kwargs)
            self.assertEqual(stream.getvalue(), json.dumps(data, **kwargs))

    def test_deep_species_and_checkpoint_round_trip(self):
        limit = sys.getrecursionlimit()
        term = I
        for _ in range(1500):
            term = ('abs', term)
        species = develop([(I, I)], ['obs'], ['out'], budgets=0)
        development = serialization.loads(species.development_json)
        development['groups'][0]['partial_term'] = term
        from dataclasses import replace
        species = replace(species, development_json=serialization.dumps(development, sort_keys=True))
        from static_agent_training.mutation import MutationConfig, mutate
        child, log = mutate(species, seed=7, config=MutationConfig(
            weight_probability=0, subtree_probability=0, term_probability=0,
            insertion_probability=0, deletion_probability=0))
        self.assertEqual(child.development_json, species.development_json)
        self.assertFalse(log['redeveloped'])
        identity = species.genotype_id
        state = dict(schema_version='crt-training-v1', next_generation=14,
                     population=[species.to_data()])
        with tempfile.TemporaryDirectory() as directory:
            for name in (CHECKPOINT, 'checkpoint.json'):
                path = Path(directory) / name
                write_json(path, state)
                if name == CHECKPOINT:
                    loaded = load_checkpoint(directory)
                else:
                    loaded = serialization.loads(path.read_text())
                restored = Species.from_data(loaded['population'][0])
                self.assertEqual(restored.genotype_id, identity)
                partial = restored.to_data()['development']['groups'][0]['partial_term']
                self.assertEqual(canonical_bytes(freeze(partial)), canonical_bytes(term))
                self.assertFalse(path.with_name(name + '.tmp').exists())
        self.assertEqual(sys.getrecursionlimit(), limit)

    def test_deep_initial_ast_identity_and_shared_subtrees(self):
        term = deep_term()
        species = develop([(term, I)], ['obs'], ['out'], budgets=0)
        restored = Species.from_data(serialization.loads(serialization.dumps(species.to_data())))
        self.assertEqual(restored.genotype_id, species.genotype_id)
        self.assertEqual(canonical_bytes(restored.initial_pairs[0][0]), canonical_bytes(term))

    def test_fallback_matches_standard_json_and_rejects_invalid_data(self):
        # Exercise the iterative path on small values too, including JSON grammar errors.
        data = {'a:b': [[], {}, None, True, 'λ', -2.5], 'q': {'x': 2}}
        with patch.object(serialization.json, 'loads', side_effect=RecursionError):
            self.assertEqual(serialization.loads(json.dumps(data)), data)
            for document in ('[1,]', '{"x":1,}', '[1 2]', '{"x" 1}', '[]x', '[', '{', '[\f1]', '{true:1}', '[01]', '[NaN,]', '{\"x\":}'):
                with self.assertRaises(json.JSONDecodeError):
                    serialization.loads(document)
        # A deep payload forces the encoder fallback while remaining standard JSON.
        payload = {'term': deep_term(), **data}
        encoded = serialization.dumps(payload, sort_keys=True, separators=(',', ':'))
        loaded = serialization.loads(encoded)
        self.assertEqual(loaded['a:b'], data['a:b'])
        self.assertEqual(loaded['q'], data['q'])
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(ValueError):
            serialization.dumps(cycle)

    def test_boolean_and_other_standard_json_keys(self):
        for data in ({True: 'yes', False: 'no'}, {None: 1, 'null': 2},
                     {3: 'int', 1.5: 'float'}, {'quote"colon:': 1}):
            stream = io.StringIO()
            serialization.dump(data, stream)
            self.assertEqual(stream.getvalue(), json.dumps(data))
            self.assertEqual(serialization.loads(stream.getvalue()), json.loads(json.dumps(data)))
        for keys in ({True: 1, 'mixed': 2}, {(1, 2): 3}):
            with self.assertRaises(TypeError):
                serialization.dump(keys, io.StringIO(), sort_keys=True)

    def test_deep_bytes_and_bytearrays_with_standard_encodings(self):
        encoded = serialization.dumps({'term': deep_term(), 'text': 'λ'})
        for encoding in ('utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be',
                         'utf-32', 'utf-32-le', 'utf-32-be'):
            for value in (encoded.encode(encoding), bytearray(encoded.encode(encoding))):
                result = serialization.loads(value)
                self.assertEqual(result['text'], 'λ')
                self.assertEqual(canonical_bytes(freeze(result['term'])), canonical_bytes(deep_term()))

    def test_deep_saved_candidate_load_and_catalog_export(self):
        from dataclasses import replace
        from static_agent_training.candidates import load_saved_candidate
        from static_agent_training.catalog import read_json
        species = develop([(I, I)], ['obs'], ['out'], budgets=0)
        development = serialization.loads(species.development_json)
        development['groups'][0]['partial_term'] = deep_term()
        species = replace(species, development_json=serialization.dumps(development))
        trace = dict(individual=species.to_data(), learned_embedding={}, learned_output={},
                     result=dict(training=dict(fitness=0), assessment=dict(fitness=0,
                                 episodes=[dict(start=1, goal=0)])))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'config.json', dict(config=dict(encoding_seed=0)))
            write_json(root / 'traces/00000-champion.json.gz', trace)
            bundle = load_saved_candidate(root, generation=0)
            self.assertEqual(Species.from_data(bundle['individual']).genotype_id, species.genotype_id)
            export = root / 'champion.json'
            write_json(export, bundle)
            loaded = read_json(export)
            partial = loaded['individual']['development']['groups'][0]['partial_term']
            self.assertEqual(canonical_bytes(freeze(partial)), canonical_bytes(deep_term()))

    def test_failed_checkpoint_write_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CHECKPOINT
            state = dict(schema_version='crt-training-v1', population=[])
            write_json(path, state)
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                write_json(path, {'term': deep_term(), 'unsupported': object()})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(load_checkpoint(directory), state)
