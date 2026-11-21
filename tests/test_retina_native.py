"""Physical retinal payloads and spatial native/reference lifetime parity."""
from types import SimpleNamespace
import unittest

from static_agent_training.experimental.number_line import StimulusNumberLine
from static_agent_training.lights_out import StimulusLightsOut


class RetinaRenderingTests(unittest.TestCase):
    camera = SimpleNamespace(width=16, height=16)

    def image(self, position, goal, **kwargs):
        world = StimulusNumberLine(lower=-1000, upper=1000, **kwargs)
        world.reset((position, goal))
        return world.render_retina(self.camera)

    def test_translation_and_offscreen_goal_invariance(self):
        self.assertEqual(self.image(0, 3), self.image(400, 403))
        self.assertEqual(self.image(0, 30), self.image(1, 80))
        self.assertEqual(self.image(0, 30), self.image(0, -30))
        self.assertNotEqual(self.image(0, 3), self.image(0, 30))
        self.assertEqual(self.image(0, 3), self.image(0, 3, goal_conditioned=False))
        self.assertEqual(len(self.image(0, 3)), 32)

    def test_walls_and_center_agent(self):
        world = StimulusNumberLine(lower=-20, upper=20)
        world.reset((-20, 20))
        image = world.render_retina(self.camera)
        self.assertNotEqual(image, self.image(0, 40))
        bits = [bool(byte & (1 << shift)) for byte in image for shift in range(7, -1, -1)]
        self.assertTrue(bits[10 * 16 + 8])
        self.assertTrue(bits[11 * 16 + 8])
        self.assertTrue(all(bits[12 * 16:13 * 16]))

    def test_board_geometry_ignores_partition_and_puzzle_meaning(self):
        images = []
        for environment in ('lights-out', 'switchboard'):
            world = StimulusLightsOut(environment=environment)
            world.position = 1
            images.append(world.render_retina(self.camera))
            world.position = 0
            self.assertNotEqual(images[-1], world.render_retina(self.camera))
            self.assertFalse(hasattr(world, 'receptors'))
        self.assertEqual(*images)


class RetinaNativeTests(unittest.TestCase):
    def test_source_frames_are_actual_rendered_pixels(self):
        from static_agent_training.experimental.senses import RetinaSenses
        from static_agent_training.experimental.sensory_accelerated import native_frames
        from static_agent_training.environments import stimulus_world, environment_spec
        senses = RetinaSenses()
        for environment in ('original', 'lights-out', 'switchboard'):
            world = stimulus_world(environment)
            frames = native_frames(senses, False, environment)
            positions = environment_spec(environment).positions
            for index, position in enumerate(positions):
                world.position, world.goal = position, 0
                row = frames[index] if environment != 'original' else frames[index * len(positions) + 20]
                self.assertEqual(row.tolist(), list(senses.spec.encode(senses.frame(world.render_retina(senses)))))

    def test_native_reference_metrics_and_weights_spatial_regions(self):
        from static_agent_training.accelerated import assert_equivalent
        from static_agent_training.evolution import TrainingConfig, initialize_population
        from static_agent_training.experimental.sensory_accelerated import evaluate_batch, learned_mappings
        from static_agent_training.experimental.sensory_evolution import evaluate
        from static_agent_training.learning import LearningConfig
        for width, height, partitions in ((16, 16, 4), (4, 4, 16)):
            for policy in ('agreement', 'vote'):
                for environment in ('original', 'lights-out'):
                    with self.subTest(size=(width, height), policy=policy, environment=environment):
                        config = TrainingConfig(model='senses', sensory_interface='retina',
                            retina_width=width, retina_height=height, receptors=width*height,
                            sensory_partitions=partitions, action_policy=policy,
                            population=2, elites=1, initial_pairs=3, chain_budget=16,
                            inherit_lifetime=True, environment=environment,
                            learning_config=LearningConfig(compatibility_threshold=.99))
                        parent = initialize_population(config)[0]
                        starts, goals = ([2, -7, 19], [0, 3, -4]) if environment == 'original' else ([1, 3, 7], [0, 0, 0])
                        rows, weights = evaluate_batch([parent], starts, starts, seeds=[19],
                            training_goals=goals, assessment_goals=goals,
                            sensory_partitions=partitions, action_policy=policy,
                            environment=environment, return_weights=True, workers=1,
                            learning_config=config.learning_config)
                        expected, _, learned = evaluate(parent, list(zip(starts, goals)),
                            list(zip(starts, goals)), seed=19, config=config)
                        assert_equivalent(rows[0], expected)
                        assert_equivalent(learned_mappings(parent, weights, 0).to_parameters(),
                                          learned.inherited.to_parameters())
                        with self.assertRaises(ValueError):
                            evaluate_batch([parent], starts, starts, seeds=[19],
                                training_goals=goals, assessment_goals=goals,
                                sensory_partitions=1, environment=environment)
