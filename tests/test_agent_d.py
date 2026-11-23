from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.experimental.agent_d import AgentD, RandomGraph
from static_agent_training.experimental.agent_w_environment import NumberLine
from static_agent_training.experimental.agent_w_experiment import run
from static_agent_training.experimental.agent_z import Structure
from static_agent_training.experimental.agent_z_experiment import read_z_artifact


class AgentDTests(unittest.TestCase):
    def test_native_feedback_uses_the_random_graph_without_changing_it(self):
        nodes = ('0' * 64, '1' * 64)
        graph = RandomGraph(nodes, {nodes[0]: nodes[1], nodes[1]: nodes[1]})
        world = NumberLine()
        agent = AgentD(graph, [[0.] * 9, [1.] + [0.] * 8], [[0.] * 9] * 2,
                       actions=world.action_signals)
        replica = AgentD.from_data(agent.to_data())
        self.assertIsInstance(replica.copy(), AgentD)
        self.assertEqual(agent.to_data(), replica.to_data())
        with patch.object(AgentD, 'step', side_effect=AssertionError('Python step')):
            result = agent.evaluate([(-10, 20)] * 3, seed=7, trace=True)
        for _, _, event in result['trace']:
            self.assertEqual(event.successor, graph.successors[event.chain.pair])
            self.assertIn(event.action, agent.actions)
            self.assertIsNotNone(event.compatibility)
        self.assertGreater(result['actions'], 0)
        self.assertNotEqual(result['continuation'], agent.continuation)
        self.assertEqual(graph.successors, {nodes[0]: nodes[1], nodes[1]: nodes[1]})

    def test_seeded_evolution_uses_no_lambda_development(self):
        self.assertEqual(RandomGraph.generate(5, random.Random(3)).fingerprint,
                         RandomGraph.generate(5, random.Random(3)).fingerprint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(Structure, 'develop', side_effect=AssertionError('lambda development')):
                run(path, model='agent-d', initial_nodes=5, population=8,
                    generations=2, episodes=5, speciation=True)
            saved = read_z_artifact(path / 'checkpoint.json.gz')
            self.assertEqual(saved['schema_version'], 'crt-agent-d-checkpoint-v1')
            self.assertEqual(len(saved['niches']['assignments']), 8)
            self.assertEqual(len(saved['history']), 2)
            for member in saved['population']:
                agent = AgentD.from_data(member['individual'])
                self.assertTrue(set(agent.structure.successors.values()) <= set(agent.ids))
                self.assertNotIn('pairs', member['individual'])
