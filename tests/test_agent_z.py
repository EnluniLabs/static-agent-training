import unittest
from types import SimpleNamespace

from static_agent_training import engine, serialization
from static_agent_training.experimental.agent_z import AgentZ, Chain, Structure
from static_agent_training.experimental.agent_z_environment import NumberLine
from static_agent_training.experimental.agent_z_experiment import run
from static_agent_training.experimental.agent_z_kernels import NativeKernels
from static_agent_training.initialize import I, K, pair_id, canonical_bytes


class AgentZTests(unittest.TestCase):
    def setUp(self):
        self.structure, self.stops = Structure.develop([(I, K)], max_transitions=1)
        self.source = pair_id(I, K)
        self.successor = pair_id(K, I)

    def agent(self, kernels=None):
        return AgentZ(self.structure, [[0.] * 9 for _ in self.structure.pairs],
                      [[0.] * 8 for _ in self.structure.pairs], kernels=kernels)

    def test_asymmetric_development_and_legacy_orientation(self):
        self.assertEqual(self.structure.successors, {self.source: self.successor})
        legacy = engine.run([(I, K)], 10, max_transitions=1)
        edge = next(event for event in legacy['events'] if event['kind'] == 'NORMAL')
        self.assertEqual(edge['pair'][0], K)
        self.assertNotEqual(canonical_bytes(edge['pair'][1]), canonical_bytes(I))
        _, limited = Structure.develop([(I, K)], budget=0)
        self.assertEqual(limited, ('BUDGET',))
        with self.assertRaises(ValueError):
            Structure(dict(self.structure.pairs) | {pair_id(I, I): (I, I)},
                      {self.source: pair_id(I, I), self.successor: self.successor})
        with self.assertRaises(ValueError):
            AgentZ.from_data(legacy['repertoire'])

    def test_action_only_advances_without_feedback_or_reembedding(self):
        agent, world = self.agent(), NumberLine()
        world.reset((0, 20))
        original = agent.to_data()
        action = next(signal for signal, move in world.actions.items() if move == 1)
        draws = scripted(action)
        chains, feedback, emissions = agent.step([Chain(0, self.source)], world, draws)
        self.assertEqual(chains, (Chain(0, self.successor),))
        self.assertIsNone(feedback.observation)
        self.assertIsNone(emissions[0].compatibility)
        self.assertEqual(world.position, 1)
        self.assertEqual(agent.to_data(), original)

    def test_mixed_round_measures_once_after_actions_including_valid_zero(self):
        world = NumberLine()
        world.reset((0, 1))
        action = next(signal for signal, move in world.actions.items() if move == 1)
        agent = self.agent()
        _, feedback, emissions = agent.step([Chain(0, self.source), Chain(1, self.source)],
                                            world, scripted(action, b'\x80'), threshold=.9)
        self.assertEqual(feedback.observation, bytes(1))
        self.assertEqual(feedback.recipients, {1})
        self.assertTrue(feedback.success)
        self.assertIsNone(emissions[0].observation)
        self.assertEqual(emissions[1].observation, bytes(1))
        self.assertEqual(emissions[1].mismatched_bits, 1)
        self.assertEqual(agent.continuation[agent.index[self.source]], [-.025] * 8)

    def test_adapter_orders_actions_externally_and_keeps_duplicates(self):
        world = NumberLine(action_limit=5)
        world.reset((19, -20))
        actions = sorted(world.actions)
        low, high = next((left, right) for left in actions for right in actions
                         if left < right and world.actions[left] * world.actions[right] < 0)
        signals = (high, low, high, bytes(1), bytes([high[0] + 0x10]))
        feedback = world.interpret(signals)
        position = 19
        for action in (low, high, high):
            position = max(-20, min(20, position + world.actions[action]))
        self.assertEqual(world.position, position)
        self.assertEqual(world.remaining, 2)
        self.assertEqual(feedback.recipients, {3, 4})
        self.assertEqual(feedback.observation, (-20 - position).to_bytes(1, 'big', signed=True))

    def test_shared_feedback_snapshot_aggregation_and_native_parity(self):
        observation = b'\x01'
        expected = (b'\x80', b'\x40')
        chains = [Chain(0, self.source), Chain(1, self.source)]
        results = []
        for backend in (None, NativeKernels()):
            agent, world = self.agent(backend), NumberLine()
            world.reset((0, 1))
            order = chains if backend is None else chains[::-1]
            successors, feedback, emissions = agent.step(order, world, scripted(*expected),
                                                          threshold=.9)
            self.assertEqual(feedback.observation, observation)
            self.assertEqual([e.signal for e in emissions], list(expected))
            self.assertEqual([e.observation for e in emissions], [observation] * 2)
            self.assertEqual([e.compatibility for e in emissions], [.5, .5])
            self.assertEqual(successors, (Chain(0, self.successor), Chain(1, self.successor)))
            target = agent.embedding[agent.index[self.successor]]
            self.assertEqual(target, [.05, .025, .025, 0, 0, 0, 0, 0, 0])
            self.assertEqual(agent.continuation[agent.index[self.source]], [-.05] * 7 + [.05])
            self.assertEqual(agent.continuation[agent.index[self.successor]], [0.] * 8)
            restored = AgentZ.from_data(serialization.loads(serialization.dumps(agent.to_data())))
            self.assertEqual(restored.to_data(), agent.to_data())
            results.append((agent.to_data(), emissions))
        self.assertEqual(*results)

    def test_distinct_errors_and_ablations(self):
        for empirical, structural in ((0, .05), (.05, 0), (0, 0)):
            agent, world = self.agent(), NumberLine()
            original = agent.to_data()
            world.reset((0, 1))
            _, _, emissions = agent.step([Chain(0, self.source), Chain(1, self.source)],
                                         world, scripted(bytes(1), b'\xc0'), threshold=.9,
                                         empirical_rate=empirical, structural_rate=structural)
            self.assertEqual([e.mismatched_bits for e in emissions], [1, 3])
            self.assertEqual(agent.embedding != original['embedding'], bool(structural))
            self.assertEqual(agent.continuation != original['continuation'], bool(empirical))

    def test_seeded_evolution_backend_parity(self):
        options = dict(seed=11, generations=2, episodes=8, population=4)
        reference = run(**options)
        native = run(backend='native', **options)
        self.assertEqual(reference['history'], native['history'])
        self.assertEqual(reference['champion'], native['champion'])


def scripted(*signals):
    draws = iter(.25 if value & (1 << shift) else .75
                 for signal in signals for value in signal for shift in range(7, -1, -1))
    return SimpleNamespace(random=lambda: next(draws))
