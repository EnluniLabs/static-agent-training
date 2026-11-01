from collections import Counter
from dataclasses import asdict
import math
import random
import statistics
import unittest

from static_agent_training.environment import POSITIONS, sample_starts
from static_agent_training.evolution import TrainingConfig, generation_tasks, partitions
from static_agent_training.seeds import derive
from static_agent_training.tasks import distance_weights, sample_tasks


class TaskSamplingTests(unittest.TestCase):
    def test_normal_shape_boundaries_and_directions(self):
        tasks=sample_tasks(42,40000,starts=POSITIONS,goals=POSITIONS)
        distances=[abs(g-s) for s,g in tasks]
        self.assertTrue(all(1<=d<=40 for d in distances))
        self.assertAlmostEqual(statistics.mean(distances),10.34,delta=.12)
        self.assertAlmostEqual(statistics.pstdev(distances),4.65,delta=.12)
        self.assertAlmostEqual(sum(g>s for s,g in tasks)/len(tasks),.5,delta=.01)
        self.assertEqual(tasks[:10],sample_tasks(42,40000,starts=POSITIONS,goals=POSITIONS)[:10])

    def test_distance_mass_is_not_multiplied_by_number_of_pairs(self):
        # A very broad Gaussian gives nearly equal mass to distances 1 and 40,
        # despite there being forty times as many pairs at distance 1.
        tasks=sample_tasks(72,40000,starts=POSITIONS,goals=POSITIONS,mean=20,stddev=10000)
        counts=Counter(abs(g-s) for s,g in tasks)
        self.assertLess(abs(counts[1]-counts[40]),160)
        self.assertGreater(counts[40],800)

    def test_sparse_partitions_and_narrow_normal_remain_valid(self):
        starts=(-20,-3,2,20)
        goals=(-19,4,18)
        tasks=sample_tasks(3,200,starts=starts,goals=goals,mean=10.5,stddev=.001)
        self.assertTrue(all(s in starts and g in goals and s!=g for s,g in tasks))
        self.assertEqual(set(abs(s-g) for s,g in tasks),{7})
        self.assertEqual(distance_weights([1,40],20,1e-300),[1,0])
        with self.assertRaises(ValueError):
            sample_tasks(0,2,starts=[1],goals=[1])
        for mean,stddev in [(0,5),(41,5),(10,0),(10,-1),(math.inf,5),(10,math.nan)]:
            with self.assertRaises(ValueError):
                sample_tasks(0,2,starts=[1],goals=[2],mean=mean,stddev=stddev)

    def test_new_training_changes_tasks_but_assessment_is_fixed(self):
        config=TrainingConfig(distance_distribution="normal",population=2,elites=1,episodes=100,assessment_episodes=100,seed=42)
        split=partitions(config.seed)
        a=generation_tasks(config,0,split)
        b=generation_tasks(config,1,split)
        self.assertNotEqual(list(zip(a[0],a[2])),list(zip(b[0],b[2])))
        self.assertEqual(list(zip(a[1],a[3])),list(zip(b[1],b[3])))
        self.assertTrue(set(a[0]+a[2])<=set(split['training']))
        self.assertTrue(set(a[1]+a[3])<=set(split['assessment']))
        self.assertFalse(set(a[0]+a[2]) & set(a[1]+a[3]))

    def test_uniform_goals_include_zero_and_do_not_favor_center(self):
        config=TrainingConfig(population=2,elites=1,seed=42,episodes=31000,assessment_episodes=10000)
        self.assertEqual(config.distance_distribution,'uniform')
        split=partitions(config.seed,include_zero=config.include_zero)
        self.assertEqual(set(split['training'])|set(split['assessment']),set(POSITIONS))
        self.assertEqual(len(split['training']),31)
        starts,assessment,goals,assessment_goals=generation_tasks(config,0,split)
        for positions,draws in ((split['training'],goals),(split['assessment'],assessment_goals)):
            counts=Counter(draws)
            self.assertEqual(set(counts),set(positions))
            expected=len(draws)/len(positions)
            self.assertTrue(all(abs(count-expected)<170 for count in counts.values()))
        self.assertTrue(all(s!=g for s,g in zip(starts,goals)))
        self.assertTrue(all(s!=g for s,g in zip(assessment,assessment_goals)))
        old=asdict(config)
        del old['include_zero']
        self.assertFalse(TrainingConfig.from_data(old).include_zero)

    def test_old_saved_configuration_replays_uniform_tasks(self):
        config=TrainingConfig(population=2,elites=1,seed=42)
        data=asdict(config)
        for key in ('distance_distribution','distance_mean','distance_stddev'):
            del data[key]
        old=TrainingConfig.from_data(data)
        self.assertEqual(old.distance_distribution,'uniform')
        split=partitions(config.seed)
        starts,assessment,goals,assessment_goals=generation_tasks(old,3,split)
        expected=sample_starts(derive(old.seed,'training-starts',3),old.episodes,allowed=tuple(split['training']))
        self.assertEqual(starts,expected)
        rng=random.Random(derive(old.seed,'training-goals',3))
        self.assertEqual(goals,[rng.choice([g for g in split['training'] if g!=s]) for s in starts])
