from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from static_agent_training.accelerated import assert_equivalent, evaluate_batch
from static_agent_training.evaluation import evaluate
from static_agent_training.evolution import TrainingConfig, evaluate_generation, initialize_population, partitions
from static_agent_training.experiment import run_experiment
from static_agent_training.genome import Species
from static_agent_training.learning import LearningConfig
from static_agent_training.mappings import InheritedMappings
from static_agent_training.repertoire import Repertoire


NUMPY = importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(NUMPY, 'optional NumPy acceleration is not installed')
class AcceleratedTests(unittest.TestCase):
    def config(self, **kwargs):
        return TrainingConfig(population=8, generations=2, episodes=8, assessment_episodes=8,
                              elites=1, initial_pairs=3, chain_budget=16, seed=19, **kwargs)

    def compare(self, backend, *, conditioned=True, learning=True):
        import numpy as np
        settings=LearningConfig(compatibility_threshold=1.0)
        config=self.config(varying_goals=conditioned, learning=learning, learning_config=settings)
        population=initialize_population(config)
        before=[p.to_data() for p in population]
        starts=[-20,-10,-5,-1,0,2,8,20]
        assessment=[-19,-9,-4,1,3,7,10,19]
        train_goals=[20,2,5,1,-3,8,-10,-20] if conditioned else None
        if not conditioned:
            starts[4]=4
        assess_goals=[3,-20,20,2,-4,19,-19,10] if conditioned else None
        seeds=list(range(len(population)))
        kwargs=dict(encoding_seed=9,learning=learning,learning_config=settings,
                    training_goals=train_goals,assessment_goals=assess_goals)
        # Use matching encoding for inherited mappings.
        config=replace(config,encoding_seed=9)
        population=initialize_population(config)
        before=[p.to_data() for p in population]
        accelerated, weights=evaluate_batch(population,starts,assessment,seeds=seeds,
                                            backend=backend,return_weights=True,workers=2,**kwargs)
        for index,individual in enumerate(population):
            expected,detail=evaluate(individual,starts,assessment,seed=seeds[index],**kwargs)
            assert_equivalent(accelerated[index],expected)
            ids=individual.inherited.pair_ids
            for s,signal in enumerate(individual.inherited.observations):
                np.testing.assert_allclose(weights[0][index,s,:len(ids)],
                    [detail['learned_embedding'][signal][q] for q in ids],rtol=1e-12,atol=1e-12)
            for q,pair in enumerate(ids):
                np.testing.assert_allclose(weights[1][index,q],
                    [detail['learned_output'][pair][o] for o in individual.inherited.outputs],rtol=1e-12,atol=1e-12)
        self.assertEqual(before,[p.to_data() for p in population])

    def test_native_matches_actions_predictions_learning_and_learned_weights(self):
        self.compare('native')
        self.compare('native',conditioned=False)
        self.compare('native',learning=False)

    def test_terminal_empty_and_internal_limit(self):
        config=self.config()
        population=initialize_population(config)
        base=population[0]
        # A repertoire with no successors must terminate before producing an output.
        rep=Repertoire(dict(base.repertoire.pairs),{})
        terminal=replace(base,repertoire=rep)
        empty=Repertoire({}, {})
        inherited=InheritedMappings.initialize(empty,base.inherited.observations,base.inherited.outputs)
        no_pairs=replace(base,repertoire=empty,inherited=inherited)
        # Force expectations repeatedly on every available edge until a genuine stop.
        data=base.inherited.to_data()
        for row in data['output'].values():
            for token in row:
                row[token]=1000.0 if token=='expect_+0' else -1000.0
        expectations=replace(base,inherited=InheritedMappings.from_data(data))
        for backend in ['native']:
            members=[terminal,no_pairs,expectations]
            results=evaluate_batch(members,[1,2],[3,4],seeds=[1,2,3],backend=backend)
            for i,individual in enumerate(members):
                reference,_=evaluate(individual,[1,2],[3,4],seed=i+1)
                assert_equivalent(results[i],reference)
            self.assertEqual(results[0]['training']['outputs'],0)
            self.assertEqual(results[1]['training']['outputs'],0)

    def test_full_generation_and_checkpoint_resume(self):
        config=self.config(backend='native',batch_size=3,workers=2)
        population=initialize_population(config)
        native,_=evaluate_generation(population,config,0,partitions(config.seed))
        reference,_=evaluate_generation(population,replace(config,backend='python'),0,partitions(config.seed))
        assert_equivalent(native,reference)
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'full'
            second=Path(tmp)/'resumed'
            a=run_experiment(config,first)
            run_experiment(config,second,until_generation=1)
            b=run_experiment(config,second,resume=True)
            self.assertEqual(a,b)

    def test_rejects_invalid_batches(self):
        config=self.config()
        population=initialize_population(config)
        with self.assertRaisesRegex(ValueError,'disjoint'):
            evaluate_batch(population,[1],[1],seeds=list(range(8)))
        with self.assertRaisesRegex(ValueError,'matching valid'):
            evaluate_batch(population,[1],[2],seeds=list(range(8)),training_goals=[1])
