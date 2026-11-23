import copy
import unittest
from static_agent_training.experimental.agent_y import AgentY
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.genome import develop
from static_agent_training.initialize import I, K


class AgentYTests(unittest.TestCase):
    def make(self):
        repertoire = develop([(I, I), (K, I)], ['x'], ['y'], budgets=8).repertoire
        return AgentY(repertoire, senses=FixedSenses(receptors=16, partitions=2))

    def test_constructs_diverse_complete_partitions_without_candidates(self):
        model = self.make()
        frame = (b'\xff\xff',)
        partitions = set()
        before = model.parameters()
        for _ in range(20):
            model.observe(frame)
            partitions.add(model.partition)
            self.assertEqual(sorted(i for group in model.partition for i in group), list(range(16)))
            self.assertTrue(all(model.partition))
            groups = [set(model.feature_indices(frame, r)) for r in range(len(model.partition))]
            self.assertTrue(all(0 in g for g in groups))
            self.assertEqual(set().union(*groups), set(range(17)))
        self.assertGreater(len(partitions), 2)
        self.assertEqual(before, model.parameters())

    def test_snapshot_replays_assignments_pairs_and_learning(self):
        model = self.make()
        model.router_weights['split'][5][0] = 2
        restored = AgentY.from_data(model.to_data())
        frame = model.senses.frame(b'a')
        for _ in range(10):
            model.observe(frame)
            restored.observe(frame)
            self.assertEqual(model.partition, restored.partition)
            self.assertEqual(model.activations, restored.activations)
        model.reset_episode()
        restored.reset_episode()
        self.assertEqual(model.to_data(), restored.to_data())

    def test_feedback_learns_group_creation_and_joining(self):
        from static_agent_training.experimental.signals import Prediction
        model = self.make()
        frame = model.senses.frame(b'a')
        for learn in (False, True):
            model.observe(frame)
            q = model.ids[0]
            model.pending[0] = (q, Prediction(0, frame[0]))
            model.pending_logits[0] = [0.] * model.spec.bits
            model.router_baseline = 2.
            before = copy.deepcopy(model.router_weights)
            base_parameters = model.parameters()
            features, decisions = model._routing
            groups, probabilities, chosen = decisions[1]
            model.receive(frame, learn=learn)
            self.assertEqual(before == model.router_weights, not learn)
            if learn:
                # Reward reinforces the selected structural operation, joining
                # or making a new group; pair embedding weights are independent.
                if chosen == len(groups):
                    self.assertGreater(model.router_weights['split'][1][features[0]],
                                       before['split'][1][features[0]])
                else:
                    self.assertGreater(model.router_weights['affinity'][1][groups[chosen][0]],
                                       before['affinity'][1][groups[chosen][0]])
                self.assertEqual(base_parameters['weights'], model.weights)
            else:
                self.assertEqual(base_parameters, model.parameters())
                self.assertEqual(model.router_baseline, 2.)

    def test_full_signal_changes_number_of_groups_without_count_setting(self):
        model = self.make()
        for row in model.router_weights['split']:
            row[-1] = -100
            row[1] = 200
        model.observe((b'\x80\x00',))
        self.assertEqual(len(model.partition), 16)
        model.observe((b'\x00\x00',))
        self.assertEqual(len(model.partition), 1)
        # Returning input restores the learned split behavior without relearning.
        model.observe((b'\x80\x00',))
        self.assertEqual(len(model.partition), 16)

    def test_feedback_moves_probability_of_splitting(self):
        from static_agent_training.experimental.signals import Prediction
        model = self.make()
        frame = model.senses.frame(b'a')
        for row in model.router_weights['split']:
            row[-1] = -100
        model.router_weights['split'][1][-1] = 0
        model.observe(frame)
        features, decisions = model._routing
        groups, probabilities, chosen = decisions[1]
        before = model.router_weights['split'][1][-1]
        q = model.ids[0]
        model.pending[0] = (q, Prediction(0, frame[0]))
        model.pending_logits[0] = [0.] * model.spec.bits
        model.router_baseline = 2.
        model.receive(frame)
        after = model.router_weights['split'][1][-1]
        self.assertGreater((after-before) * (1 if chosen == len(groups) else -1), 0)

    def test_feedback_can_learn_more_or_fewer_groups(self):
        # Controlled feedback isolates structural credit assignment from whether
        # an untrained lambda repertoire solves a particular physical task.
        from static_agent_training.experimental.signals import Prediction
        frame = (b'\x00\x00',)
        for direction in (-1, 1):
            model = self.make()
            model.router_rate = .1
            initial = []
            for _ in range(100):
                model.observe(frame)
                initial.append(len(model.partition))
            for _ in range(500):
                model.observe(frame)
                count = len(model.partition)
                q = model.ids[0]
                model.pending[0] = (q, Prediction(0, frame[0]))
                # Zero observed bits: increasing logits increases prediction loss.
                penalty = (16-count if direction == 1 else count) / 4
                model.pending_logits[0] = [penalty] * model.spec.bits
                model.receive(frame)
            learned = []
            for _ in range(100):
                model.observe(frame)
                learned.append(len(model.partition))
            shift = sum(learned)/100 - sum(initial)/100
            self.assertGreater(shift * direction, .1)

    def test_invalid_snapshots_and_finite_capacity(self):
        model = self.make()
        data = model.to_data()
        data['router']['weights']['split'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            AgentY.from_data(data)
        # Arbitrarily high splitting is bounded only by receptor count.
        for row in model.router_weights['split']:
            row[-1] = 100
        model.observe((b'\xff\xff',))
        self.assertEqual(len(model.partition), model.senses.receptors)
        self.assertTrue(all(len(g)==1 for g in model.partition))

class AgentYTrainingTests(unittest.TestCase):
    def test_evolution_resume_inheritance_mutation_and_export(self):
        from dataclasses import replace
        import tempfile
        from pathlib import Path
        from static_agent_training.evolution import TrainingConfig, initialize_population
        from static_agent_training.experiment import run_experiment, load_checkpoint
        from static_agent_training.experimental.bit_model import BitModel
        from static_agent_training.experimental.sensory_genome import SensoryMappings
        from static_agent_training.mutation import mutate
        from static_agent_training.populations import save_population, initialize_saved
        from static_agent_training.catalog import read_json
        from static_agent_training.offspring import _pack_species, _unpack_species
        config = TrainingConfig(model='senses', agent_y=True, backend='python',
            receptors=16, sensory_partitions=2, population=2, elites=1,
            initial_pairs=2, chain_budget=8, generations=2, episodes=2,
            assessment_episodes=2, inherit_lifetime=True)
        parent = initialize_population(config)[0]
        self.assertIsNotNone(parent.inherited.router)
        self.assertEqual(_unpack_species(_pack_species(parent)).to_data(), parent.to_data())
        child, _ = mutate(parent, seed=5, config=config.mutation)
        self.assertIsNotNone(child.inherited.router)
        self.assertIsInstance(child.inherited.model(child.repertoire), AgentY)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = run_experiment(config, root/'full')
            run_experiment(config, root/'split', until_generation=1)
            actual = run_experiment(config, root/'split', resume=True)
            self.assertEqual(expected, actual)
            self.assertEqual(load_checkpoint(root/'full'), load_checkpoint(root/'split'))
            import gzip
            from static_agent_training import serialization
            with gzip.open(root/'full/traces/00001-champion.json.gz', 'rt') as handle:
                trace = serialization.load(handle)
            self.assertIsInstance(BitModel.from_data(trace['learned_model']), AgentY)
            output = root/'saved.json.gz'
            save_population(root/'full', output)
            imported = initialize_saved(replace(config, initial_population=str(output)))
            self.assertTrue(all(p.inherited.router is not None for p in imported))
        self.assertEqual(replace(config, backend='native').backend, 'native')

class AgentYParallelTests(unittest.TestCase):
    def test_parallel_results_and_learned_inheritance_match_serial(self):
        from dataclasses import replace
        from static_agent_training.evolution import TrainingConfig, initialize_population, partitions
        from static_agent_training.experimental.sensory_evolution import evaluate_generation
        config = TrainingConfig(model='senses', agent_y=True, backend='python', workers=1,
            receptors=16, sensory_partitions=1, population=3, elites=1,
            initial_pairs=2, chain_budget=8, generations=1, episodes=3,
            assessment_episodes=3, inherit_lifetime=True)
        population = initialize_population(config)
        split = partitions(config.seed, include_zero=config.varying_goals, environment=config.environment)
        serial = evaluate_generation(population, config, 0, split, return_lifetime=True)
        events=[]
        parallel = evaluate_generation(population, replace(config, workers=2), 0, split,
                                       progress=events.append, return_lifetime=True)
        self.assertEqual(serial[:2], parallel[:2])
        self.assertEqual([p.to_data() for p in serial[2]], [p.to_data() for p in parallel[2]])
        self.assertEqual([e['completed'] for e in events], [1,2,3])
