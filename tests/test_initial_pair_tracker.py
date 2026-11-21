from dataclasses import replace
import unittest
from static_agent_training.evolution import TrainingConfig, initialize_population, partitions, evaluate_generation


class InitialPairTrackerTests(unittest.TestCase):
    def test_distinct_seeds_per_animal_before_population_average(self):
        for model in ('categorical','senses'):
            cfg=TrainingConfig(model=model,backend='native',population=2,initial_pairs=2,
                generations=1,episodes=1,assessment_episodes=1,elites=1,workers=1)
            population=initialize_population(cfg)
            animal=population[0]
            population[0]=replace(animal,initial_pairs=(animal.initial_pairs[0],)*2,
                budgets=(animal.budgets[0],)*2)
            self.assertEqual(len(set(population[1].initial_pairs)),2)
            before=[p.to_data() for p in population]
            record,_=evaluate_generation(population,cfg,0,partitions(cfg.seed,include_zero=True))
            self.assertEqual(record['summary']['mean_unique_initial_pairs'],1.5)
            self.assertEqual(before,[p.to_data() for p in population])
