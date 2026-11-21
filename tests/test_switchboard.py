"""Independent-panel mechanics and evaluator integration for switchboard."""
from dataclasses import replace
import unittest

from static_agent_training.accelerated import assert_equivalent, evaluate_batch as categorical_batch
from static_agent_training.environments import environment_spec, categorical_world, stimulus_world
from static_agent_training.environment import Output
from static_agent_training.evaluation import evaluate as categorical_evaluate
from static_agent_training.evolution import TrainingConfig, initialize_population, partitions
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.experimental.signals import Action, NoOp
from static_agent_training.experimental.sensory_accelerated import evaluate_batch, learned_mappings
from static_agent_training.experimental.sensory_evolution import evaluate
from static_agent_training.lights_out import action_cells, board_bits, grid_size, press_mask, random_success, solutions, stimulus


class SwitchboardTests(unittest.TestCase):
    def test_all_boards_have_exact_solutions_and_both_adapters_agree(self):
        self.assertEqual(environment_spec('switchboard').kind, 'lights-out')
        self.assertEqual(grid_size('switchboard'), 3)
        self.assertEqual(action_cells('switchboard'), list(range(9)))
        self.assertEqual([press_mask(c, 'switchboard') for c in range(9)], [1 << c for c in range(9)])
        paths = solutions('switchboard')
        self.assertEqual(set(paths), set(range(512)))
        for board, path in paths.items():
            self.assertEqual(len(path), board.bit_count())
            current = board
            for cell in path:
                current ^= press_mask(cell, 'switchboard')
            self.assertEqual(current, 0)
        for seed in (0, 19):
            raw = stimulus_world('switchboard', encoding_seed=seed)
            world = categorical_world('switchboard', encoding_seed=seed)
            self.assertEqual(sorted(raw.moves), [1 << c for c in range(9)])
            self.assertEqual(raw.action_limit, 12)
            self.assertEqual(world.action_limit, 12)
            for board in range(1, 512):
                raw.reset((board, 0)); world.reset(start=board, goal=0)
                for cell in paths[board]:
                    button = raw.cells.index(cell)
                    step = raw.step(Action(button))
                    other = world.emit(Output(f'move_{button:02d}'))
                    self.assertEqual(raw.position, world.diagnostic_state()['position'])
                    self.assertEqual(step.success, other.success)
                self.assertTrue(step.success)
                self.assertEqual(raw.position, 0)
            for board in range(1, 512):
                for button in range(9):
                    raw.reset((board, 0)); world.reset(start=board, goal=0)
                    raw.step(Action(button)); world.emit(Output(f'move_{button:02d}'))
                    self.assertEqual(raw.position, board ^ raw.moves[button])
                    self.assertEqual(raw.position, world.diagnostic_state()['position'])

    def test_categorical_tokens_and_random_control(self):
        world = categorical_world('switchboard')
        tokens = {o.token for o in world.observation_tokens if o.token.startswith('obs_')}
        self.assertEqual(len(tokens), 512)
        self.assertTrue(all(t.startswith('obs_panel_') for t in tokens))
        for variant in ('lights-out', 'lights-out-2x2', 'switches-2x2'):
            self.assertFalse(tokens & {o.token for o in categorical_world(variant).observation_tokens})
        for cell in range(9):
            self.assertAlmostEqual(random_success(1 << cell, 1, 'switchboard'), 1 / 9)
        self.assertEqual(random_success(3, 1, 'switchboard'), 0)
        self.assertEqual(random_success(0, 12, 'switchboard'), 1)

    def test_nine_bytes_and_isolated_receptor_regions(self):
        senses = FixedSenses(receptors=144)
        for board in range(512):
            raw = stimulus(board, 'switchboard')
            self.assertEqual(raw, bytes(board_bits(board, 'switchboard')))
            self.assertEqual(len(raw), 9)
            frame = senses.frame(raw)[0]
            self.assertEqual(len(frame), 18)
            # Exactly one rail per input bit, including seven zero bits per byte.
            self.assertEqual(sum(byte.bit_count() for byte in frame), 72)
            for panel in range(3):
                changed = senses.frame(stimulus(board ^ (7 << (panel * 3)), 'switchboard'))[0]
                start, end = panel * 6, (panel + 1) * 6
                self.assertEqual(frame[:start], changed[:start])
                self.assertEqual(frame[end:], changed[end:])
                self.assertNotEqual(frame[start:end], changed[start:end])

    def test_noops_consume_budget_and_starts_are_disjoint(self):
        world = stimulus_world('switchboard')
        world.reset((511, 0))
        for index in range(12):
            step = world.step(NoOp())
            self.assertEqual(world.position, 511)
            self.assertEqual(world.actions, index + 1)
            self.assertEqual(step.done, index == 11)
            self.assertFalse(step.success)
        self.assertEqual(step.reason, 'action_limit')
        split = partitions(19, environment='switchboard')
        self.assertEqual((len(split['training']), len(split['assessment'])), (409, 102))
        self.assertFalse(set(split['training']) & set(split['assessment']))
        self.assertEqual(set(split['training']) | set(split['assessment']), set(range(1, 512)))

    def test_seeded_task_pools_cover_all_local_panel_patterns(self):
        # Coverage belongs to these complete pools, not to sampled lifetimes.
        for seed in (0, 1, 19, 42, 2026):
            split = partitions(seed, environment='switchboard')
            for name, boards in split.items():
                for panel in range(3):
                    with self.subTest(seed=seed, pool=name, panel=panel):
                        self.assertEqual({(board >> (3 * panel)) & 7 for board in boards}, set(range(8)))

    def test_native_reference_both_policies_and_learned_parameters(self):
        training, assessment = [1, 7, 73, 511], [2, 56, 146, 510]
        for policy in ('agreement', 'vote'):
            config = TrainingConfig(environment='switchboard', model='senses', receptors=144,
                sensory_partitions=3, action_policy=policy, population=2, elites=1,
                initial_pairs=3, chain_budget=8, inherit_lifetime=True)
            population = initialize_population(config)
            snapshots = [p.to_data() for p in population]
            results, weights = evaluate_batch(population, training, assessment,
                training_goals=[0]*4, assessment_goals=[0]*4, seeds=[19, 31],
                environment='switchboard', sensory_partitions=3, action_policy=policy, return_weights=True)
            for i, parent in enumerate(population):
                reference, _, learned = evaluate(parent, [(b, 0) for b in training],
                    [(b, 0) for b in assessment], seed=[19, 31][i], config=config)
                assert_equivalent(results[i], reference)
                assert_equivalent(learned_mappings(parent, weights, i).to_parameters(), learned.inherited.to_parameters())
                self.assertEqual(results[i]['assessment']['empirical_updates'], 0)
                self.assertEqual(results[i]['assessment']['conflicts'], 0)
            self.assertEqual([p.to_data() for p in population], snapshots)
        config = replace(config, model='categorical', sensory_partitions=1)
        population = initialize_population(config)
        results, weights = categorical_batch(population, training, assessment, return_weights=True, training_goals=[0]*4,
            assessment_goals=[0]*4, seeds=[19, 31], environment='switchboard')
        for i, parent in enumerate(population):
            reference, detail = categorical_evaluate(parent, training, assessment, seed=[19, 31][i],
                training_goals=[0]*4, assessment_goals=[0]*4, environment='switchboard')
            assert_equivalent(results[i], reference)
            for row, signal in enumerate(parent.inherited.observations):
                for column, pair in enumerate(parent.inherited.pair_ids):
                    self.assertAlmostEqual(weights[0][i, row, column], detail['learned_embedding'][signal][pair])
            for row, pair in enumerate(parent.inherited.pair_ids):
                for column, token in enumerate(parent.inherited.outputs):
                    self.assertAlmostEqual(weights[1][i, row, column], detail['learned_output'][pair][token])
