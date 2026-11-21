"""Exact behavioral contracts for experimental partitioned sensory activation."""
import copy
import json
import math
import random
import unittest

from static_agent_training.engine import run
from static_agent_training.experimental.bit_model import BitModel, run_episode, sigmoid
from static_agent_training.experimental.number_line import StimulusNumberLine
from static_agent_training.experimental.senses import FixedSenses, SensoryAdapter, StimulusStep
from static_agent_training.experimental.signals import Action, Prediction
from static_agent_training.repertoire import Repertoire
from static_agent_training.learning import LearningConfig


def repertoire():
    identity = ('abs', ('var', 0))
    constant = ('abs', ('abs', ('var', 1)))
    return Repertoire.from_data(run([(identity, identity), (constant, constant)], 32)['repertoire'])


class BytesWorld:
    """One action exposes a complete, known receptor target, then terminates."""
    def __init__(self):
        self.actions = []

    def reset(self, task):
        return StimulusStep(b'\x00\xff')

    def step(self, action):
        self.actions.append(action.index)
        return StimulusStep(b'\x06', done=True)


class AgentXTests(unittest.TestCase):
    def model(self, partitions=2, policy='agreement', seed=19):
        model = BitModel(repertoire(), senses=FixedSenses(16), seed=seed,
                         sensory_partitions=partitions, action_policy=policy)
        model.bias[:] = [0.] * len(model.ids)
        model.weights[:] = [[0.] * len(model.ids) for _ in model.weights]
        return model

    def force_regions(self, model, sources):
        """An all-one frame activates each contiguous region's original rows."""
        width, extra = divmod(model.senses.receptors, len(sources))
        start = 1  # Feature zero is the shared presence feature.
        for region, source in enumerate(sources):
            stop = start + width + (region < extra)
            for row in model.weights[start:stop]:
                row[model.ids.index(source)] = 10.
            start = stop

    def gate(self, model, pair, index):
        model.gates[pair] = [-1000.] * (model.spec.actions + model.spec.channels)
        model.gates[pair][index] = 1000.

    def embeds(self, model):
        return [event for event in model.events if event['kind'] == 'EMBED']

    def test_balanced_contiguous_masks_keep_original_indices_and_presence(self):
        model = self.model(3)
        sources = model.ids[:3]
        # Test every original receptor individually, including both boundaries.
        for receptor in range(16):
            with self.subTest(receptor=receptor):
                model.weights[:] = [[0.] * len(model.ids) for _ in model.weights]
                model.weights[0][model.ids.index(sources[0])] = 1.
                model.weights[1 + receptor][model.ids.index(sources[1])] = 10.
                bits = (1 << (15 - receptor)).to_bytes(2, 'big')
                model.events.clear()
                model.observe((bits,))
                expected = [sources[0]] * 3
                expected[0 if receptor < 6 else 1 if receptor < 11 else 2] = sources[1]
                self.assertEqual([e['pair'] for e in self.embeds(model)], expected)
                self.assertEqual([e['region'] for e in self.embeds(model)], [0, 1, 2])

    def test_successor_execution_preserves_duplicate_regions(self):
        model = self.model(3)
        source = model.ids[0]
        target = model.repertoire.successors[source]
        model.bias[0] = 100.
        self.gate(model, target, 2)
        model.observe(model.senses.frame(b'A'))
        self.assertEqual(model.emit_round(learn=False), (Action(2),) * 3)
        edges = [e for e in model.events if e['kind'] == 'EDGE']
        self.assertEqual([(e['source'], e['target'], e['region']) for e in edges],
                         [(source, target, i) for i in range(3)])

    def test_argmax_ties_use_seeded_choice_per_region_and_unique_max_uses_no_rng(self):
        model = self.model(3)
        model.embedding_rng.seed(51)
        expected = random.Random(51)
        for _ in range(12):
            model.events.clear()
            model.observe(model.senses.frame(b''))
            self.assertEqual([e['pair'] for e in self.embeds(model)],
                             [expected.choice(model.ids) for _ in range(3)])
        self.assertEqual(model.embedding_rng.getstate(), expected.getstate())
        model.bias[0] = 100.
        state = model.embedding_rng.getstate()
        model.observe(model.senses.frame(b'A'))
        self.assertEqual([e['pair'] for e in self.embeds(model)[-3:]], [model.ids[0]] * 3)
        self.assertEqual(model.embedding_rng.getstate(), state)

    def test_duplicate_predictions_keep_ids_full_targets_and_sequential_updates(self):
        model = self.model(3)
        model.bias[0] = 100.
        target = model.repertoire.successors[model.ids[0]]
        self.gate(model, target, model.spec.actions)
        model.prediction[target][0] = [0.] * model.spec.bits
        model.observe(model.senses.frame(b'A'))
        emissions = model.emit_round()
        self.assertEqual(len(emissions), 3)
        self.assertTrue(all(isinstance(e, Prediction) for e in emissions))
        self.assertEqual(len(model.pending), 3)
        self.assertEqual(len(set(model.pending)), 3)
        observed = model.senses.frame(b'\x06')
        expected = [0.] * model.spec.bits
        bits = model.spec.encode(observed)[1:]
        for _ in range(3):
            expected = [v - model.config.prediction_rate * (sigmoid(v) - b)
                        for v, b in zip(expected, bits)]
        before = model.parameters()
        model.receive(observed)
        self.assertEqual(model.prediction[target][0], expected)
        updates = [e for e in model.events if e['kind'] == 'EMPIRICAL_UPDATE']
        feedback = [e for e in model.events if e['kind'] == 'FEEDBACK']
        self.assertEqual(len(updates), 3)
        self.assertEqual(len(feedback), 3)
        self.assertEqual(len(set(e['expectation_id'] for e in feedback)), 3)
        self.assertTrue(all(e['observed'] == observed[0].hex() for e in feedback))
        self.assertTrue(all(math.isclose(e['prediction_nll'], model.spec.bits * math.log(2))
                            for e in feedback))
        self.assertEqual(updates[0]['after'], updates[1]['before'])
        self.assertEqual(updates[1]['after'], updates[2]['before'])
        self.assertEqual(model.weights, before['weights'])
        self.assertEqual(model.gates, before['gates'])
        self.assertFalse(model.pending)

    def test_same_round_predictions_are_registered_before_action_feedback(self):
        model = self.model(3)
        sources = model.ids[:3]
        self.force_regions(model, sources)
        for i, source in enumerate(sources):
            self.gate(model, model.repertoire.successors[source],
                      model.spec.actions if i != 1 else 2)
        world = BytesWorld()
        result = run_episode(model, SensoryAdapter(world, model.senses), None)
        self.assertTrue(result[-1].done)
        self.assertEqual(world.actions, [2])
        feedback = [e for e in model.events if e['kind'] == 'FEEDBACK']
        self.assertEqual(len(feedback), 2)
        observed = model.senses.frame(b'\x06')[0].hex()
        self.assertTrue(all(e['observed'] == observed for e in feedback))
        positions = [i for i, e in enumerate(model.events) if e['kind'] == 'EXPECTATION']
        resolution = next(i for i, e in enumerate(model.events) if e['kind'] == 'BATCH_ACTION')
        self.assertTrue(all(i < resolution for i in positions))
        self.assertFalse(model.pending)

    def test_environment_arbitration_executes_once_or_consumes_noop(self):
        cases = [('agreement', [1, 1, 1], 1), ('agreement', [1, 2, 1], None),
                 ('vote', [1, 2, 1], 1), ('vote', [1, 2], None),
                 ('vote', [1, 2, 3], None), ('agreement', [2], 2)]
        for policy, indices, winner in cases:
            with self.subTest(policy=policy, actions=indices):
                world = StimulusNumberLine(action_limit=1)
                adapter = SensoryAdapter(world, FixedSenses(16))
                adapter.reset((0, 20))
                step = adapter.step_batch(tuple(Action(i) for i in indices), policy)
                self.assertEqual(world.actions, 1)
                self.assertEqual(world.position, 0 if winner is None else world.moves[winner])
                self.assertTrue(step.done)
                self.assertEqual(step.frame, adapter.senses.frame(world.stimulus()))

    def test_frozen_replay_does_not_change_parameters_graph_or_observer_rng(self):
        model = self.model(3)
        model.bias[0] = 100.
        target = model.repertoire.successors[model.ids[0]]
        self.gate(model, target, 1)
        before = model.parameters(), model.structure_hash(), copy.deepcopy(model.revised)
        restored = BitModel.from_data(json.loads(json.dumps(model.to_data())))
        # Inspecting parameters and the immutable graph must not consume either RNG.
        states = model.embedding_rng.getstate(), model.output_rng.getstate()
        model.parameters(); model.structure_hash()
        self.assertEqual(states, (model.embedding_rng.getstate(), model.output_rng.getstate()))
        for candidate in (model, restored):
            run_episode(candidate, SensoryAdapter(BytesWorld(), candidate.senses), None, learn=False)
            self.assertEqual((candidate.parameters(), candidate.structure_hash(), candidate.revised), before)
        self.assertEqual(model.to_data(), restored.to_data())

    def test_snapshot_policies_and_historical_single_partition_defaults(self):
        model = self.model(3, 'vote')
        saved = json.loads(json.dumps(model.to_data()))
        restored = BitModel.from_data(saved)
        self.assertEqual(restored.sensory_partitions, 3)
        self.assertEqual(restored.action_policy, 'vote')
        self.assertEqual(json.loads(json.dumps(restored.to_data())), saved)
        old = self.model(1).to_data()
        for key in ('sensory_partitions', 'action_policy', 'embedding_selection'):
            old.pop(key, None)
        historical = BitModel.from_data(old)
        self.assertEqual(historical.sensory_partitions, 1)
        self.assertEqual(historical.action_policy, 'agreement')
        self.assertEqual(historical.embedding_selection, 'sample')

    def test_prediction_only_round_limit_expires_every_duplicate(self):
        model = self.model(3)
        model.bias[0] = 100.
        for pair in model.ids:
            self.gate(model, pair, model.spec.actions)
        before = model.parameters()
        world = BytesWorld()
        result = run_episode(model, SensoryAdapter(world, model.senses), None,
                             internal_limit=2)
        self.assertEqual(result[-1].reason, 'internal_limit')
        self.assertEqual(world.actions, [])
        self.assertEqual(len([e for e in model.events if e['kind'] == 'EXPECTATION']), 6)
        expired = [e for e in model.events if e['kind'] == 'EXPIRED']
        self.assertEqual(len(expired), 6)
        self.assertEqual([e['region'] for e in expired], [0, 1, 2] * 2)
        self.assertEqual(model.parameters(), before)
        self.assertFalse(model.pending)
        self.assertFalse(model.pending_regions)

    def test_structural_conflict_updates_only_active_original_region_rows(self):
        model = self.model(3)
        model.config = LearningConfig(compatibility_threshold=1.)
        source = model.ids[0]
        target = model.repertoire.successors[source]
        model.observe(model.senses.frame(b'A'))
        model.prediction[source][0] = [1000.] * model.spec.bits
        model.revised[source] = 1
        before = model.parameters()
        model.structural_update(source, enabled=False, region=1)
        self.assertEqual(model.parameters(), before)
        model.structural_update(source, enabled=True, region=1)
        event = model.events[-1]
        self.assertEqual(event['kind'], 'CONFLICT')
        self.assertEqual(event['active_features'], [0, 7, 8, 9, 10, 11])
        delta = model.config.embedding_rate / 7
        j = model.ids.index(target)
        for row in range(model.spec.features):
            for column in range(len(model.ids)):
                self.assertAlmostEqual(model.weights[row][column] - before['weights'][row][column],
                                       delta if column == j and row in event['active_features'] else 0.)
        self.assertAlmostEqual(model.bias[j] - before['bias'][j], delta)
        self.assertEqual(model.gates, before['gates'])
        self.assertEqual(model.prediction, before['prediction'])

    def test_playback_learning_matches_episode_and_freezes_after_learning(self):
        from static_agent_training.experimental.sensory_playback import record_episode
        for policy in ('agreement', 'vote'):
            with self.subTest(policy=policy):
                model = self.model(3, policy)
                source = next(q for q in model.ids
                              if model.repertoire.successors[q] != q)
                prediction_pair = model.repertoire.successors[source]
                action_pair = model.repertoire.successors[prediction_pair]
                model.bias[model.ids.index(source)] = 100.
                for pair in model.ids:
                    self.gate(model, pair, 0)
                self.gate(model, prediction_pair, model.spec.actions)
                self.gate(model, action_pair, 0)
                reference = BitModel.from_data(model.to_data())
                original = model.parameters()
                frames = record_episode(model, 1, 20, learn=True)
                run_episode(reference, SensoryAdapter(StimulusNumberLine(), model.senses),
                            (1, 20), learn=True)
                self.assertNotEqual(model.parameters(), original)
                self.assertEqual(model.parameters(), reference.parameters())
                self.assertEqual(model.events, reference.events)
                self.assertTrue(frames[-1]['done'])
                self.assertTrue(any(frame['tested'] == 3 for frame in frames))
                self.assertTrue(any(len(frame['predictions']) == 3 for frame in frames))
                self.assertTrue(any(frame.get('action_resolution', {}).get('policy') == policy
                                    for frame in frames))
                learned = model.parameters()
                revised = dict(model.revised)
                record_episode(model, 1, 20, learn=False)
                self.assertEqual(model.parameters(), learned)
                self.assertEqual(model.revised, revised)

    def test_partition_and_policy_validation(self):
        for count in (0, -1, True, 1.5, 17):
            with self.subTest(partitions=count), self.assertRaises(ValueError):
                self.model(count)
        with self.assertRaises(ValueError):
            self.model(2, 'unknown')
        with self.assertRaises(ValueError):
            BitModel(repertoire(), sensory_partitions=2)
        with self.assertRaises(ValueError):
            BitModel(repertoire(), senses=FixedSenses(16), sensory_partitions=2,
                     embedding_selection='sample')


class AgentXNativeParityTests(unittest.TestCase):
    def test_uneven_partitions_both_policies_metrics_and_learned_weights(self):
        from dataclasses import replace
        from static_agent_training.accelerated import assert_equivalent
        from static_agent_training.evolution import TrainingConfig, initialize_population
        from static_agent_training.experimental.sensory_accelerated import evaluate_batch, learned_mappings
        from static_agent_training.experimental.sensory_evolution import evaluate
        from static_agent_training.experimental.sensory_genome import SensoryMappings

        for policy in ('agreement', 'vote'):
            for environment in ('original', 'lights-out-2x2'):
                with self.subTest(policy=policy, environment=environment):
                    config = TrainingConfig(model='senses', population=2, elites=1,
                        initial_pairs=3, chain_budget=16, receptors=16,
                        inherit_lifetime=True, sensory_partitions=3, action_policy=policy,
                        environment=environment, learning_config=LearningConfig(compatibility_threshold=.9))
                    parent = initialize_population(config)[0]
                    params = parent.inherited.to_parameters()
                    # Exact ties exercise each region's seeded choice stream.
                    params['bias'] = [0.] * len(parent.inherited.pair_ids)
                    params['weights'] = [[0.] * len(parent.inherited.pair_ids) for _ in params['weights']]
                    parent = replace(parent, inherited=SensoryMappings(parent.inherited.pair_ids,
                                     parent.inherited.senses, params))
                    starts, goals = ([2, -7, 10], [0, 3, -4]) if environment == 'original' else ([1, 3, 7], [0, 0, 0])
                    rows, weights = evaluate_batch([parent], starts, starts,
                        training_goals=goals, assessment_goals=goals, seeds=[19],
                        sensory_partitions=3, action_policy=policy, environment=environment,
                        learning_config=config.learning_config, return_weights=True, workers=1)
                    expected, _, learned = evaluate(parent, list(zip(starts, goals)),
                        list(zip(starts, goals)), seed=19, config=config)
                    assert_equivalent(rows[0], expected)
                    assert_equivalent(learned_mappings(parent, weights, 0).to_parameters(),
                                      learned.inherited.to_parameters())

    def test_puzzle_conflicting_batch_is_noop_for_both_policies(self):
        from static_agent_training.environments import stimulus_world
        for policy in ('agreement', 'vote'):
            with self.subTest(policy=policy):
                raw = stimulus_world('lights-out-2x2')
                adapter = SensoryAdapter(raw, FixedSenses(16))
                adapter.reset((7, 0))
                before = raw.position
                result = adapter.step_batch((Action(0), Action(1)), policy)
                self.assertEqual(raw.position, before)
                self.assertEqual(raw.actions, 1)
                self.assertEqual(result.frame, adapter.senses.frame(raw.stimulus()))
                self.assertTrue(adapter.last_resolution.conflict)
                self.assertIsNone(getattr(adapter.last_resolution.action, 'index', None))


if __name__ == '__main__':
    unittest.main()
