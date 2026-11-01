import json
import random
import unittest
from unittest.mock import patch

from static_agent_training.agent import Agent, run_episode
from static_agent_training.engine import run
from static_agent_training.environment import NumberLine
from static_agent_training.initialize import I, K, pair_id
from static_agent_training.mappings import InheritedMappings, softmax
from static_agent_training.repertoire import Repertoire


def vocabulary(env):
    return (tuple(s.token for s in env.observation_tokens),
            tuple(o.token for o in env.output_tokens))


def force(row, chosen):
    for key in row:
        row[key] = 1000.0 if key == chosen else -1000.0


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.development = run([(I, K)], 12)
        self.repertoire = Repertoire.from_development(self.development)
        self.env = NumberLine()
        self.inherited = InheritedMappings.initialize(self.repertoire, *vocabulary(self.env), seed=42)

    def test_export_roundtrip_and_immutability(self):
        data = json.loads(json.dumps(self.development['repertoire']))
        restored = Repertoire.from_data(data)
        self.assertEqual(restored, self.repertoire)
        self.assertEqual(len(restored.pairs), len(set(restored.pairs)))
        with self.assertRaises(TypeError):
            restored.successors['new'] = 'invented'
        with self.assertRaises(TypeError):
            restored.pairs['new'] = (I, I)
        failed = Repertoire.from_development(run([(I, I)], 0))
        self.assertEqual(len(failed.pairs), 1)
        self.assertEqual(dict(failed.successors), {})

    def test_stable_softmax_and_sampling(self):
        self.assertEqual(softmax([1000, 1000]), (0.5, 0.5))
        self.assertEqual(softmax([-1000, -1000]), (0.5, 0.5))
        with self.assertRaises(ValueError):
            softmax([float('nan')])
        mappings = self.inherited.lifetime()
        signal = self.inherited.observations[0]
        self.assertAlmostEqual(sum(mappings.embedding_probabilities(signal)), 1)
        rng1, rng2 = random.Random(4), random.Random(4)
        self.assertEqual([mappings.embed(signal, rng1) for _ in range(100)],
                         [mappings.embed(signal, rng2) for _ in range(100)])
        # Sampling reflects probabilities, rather than always choosing an argmax.
        for q in mappings.pair_ids:
            mappings.embedding[signal][q] = 0
        counts = {q: 0 for q in mappings.pair_ids}
        for _ in range(10000):
            counts[mappings.embed(signal, rng1)] += 1
        expected = 10000 / len(counts)
        self.assertTrue(all(abs(n - expected) < 200 for n in counts.values()))

    def test_lifetime_weights_do_not_change_inheritance(self):
        first, second = self.inherited.lifetime(), self.inherited.lifetime()
        s, q = self.inherited.observations[0], self.inherited.pair_ids[0]
        original = self.inherited.embedding[s][q]
        first.embedding[s][q] += 1
        self.assertEqual(second.embedding[s][q], original)
        self.assertEqual(self.inherited.embedding[s][q], original)
        with self.assertRaises(TypeError):
            self.inherited.embedding[s][q] = 0

    def test_mutation_reconciliation_keeps_only_inherited_surviving_cells(self):
        # New development is external; reconciliation itself does no reduction.
        replacement = Repertoire.from_development(run([(I, I), (I, K)], 1))
        previous = self.inherited
        with patch('static_agent_training.engine.run', side_effect=AssertionError('runtime reduction')):
            updated = InheritedMappings.initialize(replacement, *vocabulary(self.env),
                                                   previous=previous, seed=7)
        self.assertEqual(set(updated.pair_ids), set(replacement.pairs))
        for q in set(updated.pair_ids) & set(previous.pair_ids):
            self.assertEqual(dict(updated.output[q]), dict(previous.output[q]))
            for s in updated.observations:
                self.assertEqual(updated.embedding[s][q], previous.embedding[s][q])
        again = InheritedMappings.initialize(replacement, *vocabulary(self.env), previous=previous, seed=7)
        self.assertEqual(updated, again)

    def test_agent_follows_edges_emits_after_successor_and_never_reduces(self):
        mappings = self.inherited.lifetime()
        entry = pair_id(I, K)
        successor = self.repertoire.successors[entry]
        diagnostics = self.env.diagnostic_state()
        zero_move = next(k for k, v in diagnostics['move_decoding'].items() if v == 0)
        for row in mappings.embedding.values():
            force(row, entry)
        force(mappings.output[successor], zero_move)
        agent = Agent(self.repertoire, mappings)
        before = self.repertoire.to_data()
        with patch('static_agent_training.engine.run', side_effect=AssertionError('runtime development')), \
             patch('static_agent_training.engine.contract', side_effect=AssertionError('runtime reduction')), \
             patch('static_agent_training.reduction.normalize', side_effect=AssertionError('runtime normalization')):
            transitions = run_episode(agent, self.env, start=5)
        self.assertEqual(transitions[-1].reason, 'action_limit')
        outputs = [e for e in agent.events if e['kind'] == 'OUTPUT']
        embeds = [e for e in agent.events if e['kind'] == 'EMBED']
        self.assertEqual(len(outputs), 12)
        # Even an unchanged observation re-embeds after each zero move.
        self.assertEqual(len(embeds), 12)
        self.assertTrue(all(e['pair'] == successor for e in outputs))
        self.assertEqual(before, self.repertoire.to_data())

    def test_expectation_continues_without_reembedding(self):
        mappings = self.inherited.lifetime()
        entry = pair_id(I, K)
        successor = self.repertoire.successors[entry]
        following = self.repertoire.successors[successor]
        for row in mappings.embedding.values():
            force(row, entry)
        force(mappings.output[successor], 'expect_00')
        move = next(k for k, v in self.env.diagnostic_state()['move_decoding'].items() if v == 0)
        force(mappings.output[following], move)
        env = NumberLine(action_limit=1)
        agent = Agent(self.repertoire, mappings)
        transitions = run_episode(agent, env, start=5)
        self.assertEqual([e['kind'] for e in agent.events if e['kind'] in ('EMBED', 'OUTPUT')],
                         ['EMBED', 'OUTPUT', 'OUTPUT'])
        self.assertIsNone(transitions[1].observation)
        self.assertEqual(len(transitions[-1].feedback), 1)
        self.assertEqual(transitions[-1].feedback[0].source, successor)

    def test_terminal_and_empty_repertoires_fail_without_invented_edges(self):
        for rep in (Repertoire({}, {}), Repertoire.from_development(run([(I, I)], 0))):
            mappings = InheritedMappings.initialize(rep, *vocabulary(self.env)).lifetime()
            agent = Agent(rep, mappings)
            result = run_episode(agent, self.env, start=5)
            self.assertEqual(result[-1].reason, 'chain_terminated')
            self.assertFalse(any(e['kind'] == 'OUTPUT' for e in agent.events))


if __name__ == '__main__':
    unittest.main()
