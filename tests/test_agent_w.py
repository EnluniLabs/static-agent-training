from math import log
import random
import unittest

from static_agent_training.experimental.agent_w import AgentW, probabilities
from static_agent_training.experimental.agent_w_environment import NumberLine
from static_agent_training.experimental.agent_z import Chain, Structure
from static_agent_training.initialize import I, K, pair_id


class FirstAction:
    def choices(self, population, *, weights):
        return [population[0]]


class AgentWTests(unittest.TestCase):
    def setUp(self):
        self.source = pair_id(I, K)
        self.target = pair_id(K, I)
        self.structure, _ = Structure.develop([(I, K)], max_transitions=1)
        self.world = NumberLine()
        self.world.reset((0, 15))
        self.agent = AgentW(self.structure, [[0.] * 9 for _ in self.structure.pairs],
                            [[0.] * len(self.world.action_signals) for _ in self.structure.pairs],
                            actions=self.world.action_signals)
        self.agent.embedding[self.agent.index[self.target]][0] = log(3)

    def test_action_updates_emitter_and_feedback_targets_successor(self):
        model = self.agent
        starting = model.copy()
        source = model.index[self.source]
        target = model.index[self.target]
        chains, events = model.step([Chain(0, self.source)], self.world, FirstAction())
        event, = events
        self.assertEqual(event.action, self.world.action_signals[0])
        self.assertEqual(event.successor, self.target)
        self.assertEqual(chains, (Chain(0, self.target),))
        self.assertEqual(model.structure.successors, {self.source: self.target})
        self.assertNotIn(event.feedback, self.world.action_signals)
        self.assertAlmostEqual(event.compatibility, .75)
        self.assertGreater(probabilities(model.continuation[source])[0], event.action_probability)
        self.assertEqual(model.continuation[target], starting.continuation[target])
        self.assertGreater(model.embed(event.feedback, 0)[target], event.compatibility)

    def test_learning_can_be_disabled_independently(self):
        for action, structural in ((0, 0), (0, .05), (.05, 0)):
            model = self.agent.copy()
            self.world.reset((0, 15))
            model.step([Chain(0, self.source)], self.world, FirstAction(),
                       action_rate=action, structural_rate=structural)
            self.assertEqual(model.embedding != self.agent.embedding, bool(structural))
            self.assertEqual(model.continuation != self.agent.continuation, bool(action))

    def test_two_chains_share_snapshot_and_get_individual_feedback(self):
        model = self.agent
        expected = model.copy()
        start = (Chain(0, self.source), Chain(1, self.source))
        first, second = model.step(start[::-1], self.world, FirstAction(), round_index=1)[1]
        self.assertEqual([first.chain.region, second.chain.region], [1, 0])
        self.assertNotEqual(first.feedback, second.feedback)
        self.assertAlmostEqual(first.compatibility, .75)
        self.assertAlmostEqual(second.compatibility, .75)
        self.assertEqual(first.action_probability, second.action_probability)
        self.world.reset((0, 15))
        expected.step(start, self.world, FirstAction(), round_index=1)
        self.assertEqual(model.to_data(), expected.to_data())

    def test_terminal_action_returns_zero_measurement_and_stops_next_emission(self):
        self.world.reset((19, 20))
        signals = self.world.action_signals
        for i, signal in enumerate(signals):
            world = NumberLine()
            world.reset((19, 20))
            if world.execute(signal) == bytes(1):
                break
        self.agent.continuation[self.agent.index[self.source]][i] = 1000
        events = self.agent.step([Chain(0, self.source), Chain(1, self.source)],
                                 self.world, random.Random(1))[1]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].feedback, bytes(1))
        self.assertTrue(self.world.success)

    def test_categorical_logit_update_weakens_low_compatibility_sample(self):
        model = self.agent
        model.embedding[model.index[self.target]][0] = -log(3)
        _, (event,) = model.step([Chain(0, self.source)], self.world, FirstAction(),
                                 structural_rate=0)
        result = probabilities(model.continuation[model.index[self.source]])
        self.assertAlmostEqual(sum(result), 1)
        self.assertTrue(all(0 <= p <= 1 for p in result))
        self.assertLess(result[0], event.action_probability)
        self.assertEqual(AgentW.from_data(model.to_data()).to_data(), model.to_data())


    def test_native_lifetime_matches_reference_for_all_learning_modes(self):
        from unittest.mock import patch

        structure, _ = Structure.develop([(I, K), (I, I)], max_transitions=5)
        world = NumberLine(encoding_seed=19, action_limit=9)
        generator = random.Random(9)
        size = len(structure.pairs)
        agent = AgentW(structure, [[generator.gauss(0, .5) for _ in range(9)] for _ in range(size)],
                       [[generator.gauss(0, .5) for _ in world.action_signals] for _ in range(size)],
                       actions=world.action_signals, regions=3)
        initial = agent.to_data()
        tasks = [(0, 0), (-14, 19), (20, 19), (0, -6), (13, -20)] * 25
        for action_rate, structural_rate in ((0, 0), (.05, 0), (0, .05), (.05, .05)):
            with self.subTest(action_rate=action_rate, structural_rate=structural_rate):
                reference = agent.copy()
                rng = random.Random('parity')
                expected = []
                successes = rounds = 0
                for episode, task in enumerate(tasks):
                    chains = reference.activate(world.reset(task), rng)
                    for turn in range(8):
                        if world.done or not chains:
                            break
                        chains, events = reference.step(
                            chains, world, rng, round_index=turn,
                            action_rate=action_rate, structural_rate=structural_rate)
                        expected.extend((episode, turn, event) for event in events)
                        rounds += 1
                    successes += world.success
                with patch.object(AgentW, 'step', side_effect=AssertionError('Python step')), \
                     patch.object(NumberLine, 'execute', side_effect=AssertionError('Python action')):
                    result = agent.evaluate(tasks, seed='parity', encoding_seed=19,
                                            action_limit=9, round_limit=8, trace=True,
                                            action_rate=action_rate, structural_rate=structural_rate)
                self.assertEqual((result['successes'], result['rounds'], result['actions']),
                                 (successes, rounds, len(expected)))
                self.assertGreater(len(expected), 624)
                for (ae, ar, actual), (ee, er, wanted) in zip(result['trace'], expected):
                    self.assertEqual((ae, ar, actual.chain, actual.action, actual.feedback, actual.successor),
                                     (ee, er, wanted.chain, wanted.action, wanted.feedback, wanted.successor))
                    if wanted.compatibility is None:
                        self.assertIsNone(actual.compatibility)
                    else:
                        self.assertAlmostEqual(actual.compatibility, wanted.compatibility, places=13)
                    self.assertAlmostEqual(actual.action_probability, wanted.action_probability, places=13)
                    self.assertAlmostEqual(actual.entropy, wanted.entropy, places=13)
                for family in ('embedding', 'continuation'):
                    for actual, wanted in zip(result[family], getattr(reference, family)):
                        for value, expected_value in zip(actual, wanted):
                            self.assertAlmostEqual(value, expected_value, places=13)
                self.assertEqual(agent.to_data(), initial)


    def test_native_diagnostics_do_not_change_adaptation_or_rng(self):
        tasks = [(0, 15), (1, -10), (19, 20)] * 20
        options = dict(seed=17, return_rng=True, trace=True)
        plain = self.agent.evaluate(tasks, **options)
        measured = self.agent.evaluate(tasks, diagnostics=True, **options)
        d = measured.pop('diagnostics')
        self.assertEqual(plain, measured)
        trace = [event for _, _, event in measured['trace']]
        self.assertEqual(sum(d['action_counts'].values()), len(trace))
        self.assertEqual(sum(d['termination'].values()), len(tasks))
        self.assertEqual(sum(d['compatibility_histogram']),
                         sum(event.compatibility is not None for event in trace))
        self.assertTrue(d['action_updated_states'])
        self.assertTrue(d['embedding_updated_states'])
        first = self.agent.evaluate(tasks[:19], **options)
        intermediate = AgentW.from_data(self.agent.to_data() | {
            'embedding': first['embedding'], 'continuation': first['continuation']})
        second = intermediate.evaluate(tasks[19:], rng_state=first['rng_state'], **options)
        for key in ('embedding','continuation','rng_state'):
            self.assertEqual(plain[key], second[key])


    def test_remapped_native_sensor_and_per_task_counts(self):
        from static_agent_training.experimental.wd_structural_shielding import remap, verify_remap

        verify_remap(self.agent)
        tasks = [(-2, 1), (-2, 3), (1, 1), (-2, 3)]
        result = self.agent.evaluate(tasks, environment='remapped-number-line',
                                     action_rate=0, structural_rate=0, task_diagnostics=True)
        outcomes = result['diagnostics']['task_outcomes']
        self.assertEqual({key: value[0] for key, value in outcomes.items()},
                         {'-2,1':1, '-2,3':2, '1,1':1})
        self.assertEqual(sum(x[1] for x in outcomes.values()), result['successes'])
        self.assertEqual(remap(1), 4)

    def test_shape_distinguishes_rewiring_from_relabeling(self):
        from static_agent_training.experimental.wd_structural_shielding import shape

        nodes = ('a', 'b', 'c')
        self.assertEqual(shape(nodes, {'a':'a', 'b':'c', 'c':'b'}),
                         shape(nodes, {'b':'b', 'a':'c', 'c':'a'}))
        self.assertNotEqual(shape(nodes, {'a':'a', 'b':'c', 'c':'b'}),
                            shape(nodes, {'a':'b', 'b':'c', 'c':'a'}))
