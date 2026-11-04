from dataclasses import replace
import unittest

from static_agent_training.accelerated import assert_equivalent
from static_agent_training.evolution import TrainingConfig, initialize_population, partitions, evaluate_generation
from static_agent_training.experimental.sensory_evolution import evaluate
from static_agent_training.experimental.sensory_accelerated import evaluate_batch, learned_mappings
from static_agent_training.learning import LearningConfig


class SensoryAcceleratedTests(unittest.TestCase):
    def check_backend(self, backend):
        for receptors, boundary, goals, learning in ((16,'clamp',True,True), (256,'terminate',True,True),
                                                    (64,'clamp',False,False), (256,'clamp',True,True)):
            with self.subTest(backend=backend,receptors=receptors,boundary=boundary,goals=goals,learning=learning):
                config = TrainingConfig(model='senses',population=3,elites=1,initial_pairs=3,chain_budget=16,
                    receptors=receptors,boundary_mode=boundary,varying_goals=goals,learning=learning,
                    inherit_lifetime=True,learning_config=LearningConfig(compatibility_threshold=.9))
                population = initialize_population(config)
                train = [(-19,18),(-9,3),(1,-4),(19,-20)]*5 if goals else [(-19,0),(-9,0),(1,0),(19,0)]*5
                assess = [(18,-19),(9,-3),(-1,4),(-18,20)]*3 if goals else [(18,0),(9,0),(-1,0),(-18,0)]*3
                seeds = [19,34,51]
                results, weights = evaluate_batch(population,[s for s,g in train],[s for s,g in assess],
                    training_goals=[g for s,g in train],assessment_goals=[g for s,g in assess],seeds=seeds,
                    backend=backend,workers=2,return_weights=True,learning=learning,
                    learning_config=config.learning_config,boundary_mode=boundary,goal_conditioned=goals)
                for i,parent in enumerate(population):
                    reference,detail,reproductive = evaluate(parent,train,assess,seed=seeds[i],config=config)
                    assert_equivalent(results[i],reference)
                    assert_equivalent(learned_mappings(parent,weights,i).to_parameters(),reproductive.inherited.to_parameters())

    def test_native_lifetime_metrics_and_all_learned_parameters(self):
        self.check_backend('native')

    def test_generation_batching_inheritance_and_reference_trace(self):
        config = TrainingConfig(model='senses',population=3,elites=1,episodes=6,assessment_episodes=6,
                                initial_pairs=2,chain_budget=8,inherit_lifetime=True)
        population=initialize_population(config); split=partitions(config.seed,include_zero=True)
        reference,detail,learned=evaluate_generation(population,config,0,split,return_lifetime=True)
        actual,trace,native=evaluate_generation(population,replace(config,backend='native',batch_size=2),0,split,return_lifetime=True)
        assert_equivalent(actual,reference)
        self.assertEqual(trace,detail)
        for a,b in zip(native,learned):
            assert_equivalent(a.inherited.to_parameters(),b.inherited.to_parameters())

    def test_prediction_limit_and_repeated_source_feedback(self):
        from static_agent_training.experimental.sensory_genome import SensoryMappings
        config=TrainingConfig(model='senses',population=2,elites=1,initial_pairs=2,chain_budget=8,inherit_lifetime=True)
        parent=initialize_population(config)[0]
        # A closed self-loop makes the gate control the whole prediction chain.
        from static_agent_training.repertoire import Repertoire
        from static_agent_training.initialize import pair_id
        identity=('abs',('var',0))
        q=pair_id(identity,identity)
        repertoire=Repertoire({q:(identity,identity)},{q:q})
        inherited=SensoryMappings.initialize(repertoire,senses=parent.inherited.senses,seed=11)
        for force_limit in (True,False):
            parameters=inherited.to_parameters()
            parameters['gates'][q]=[-100.0]*9+[100.0] if force_limit else [0.0]+[-100.0]*8+[2.0]
            animal=replace(parent,repertoire=repertoire,inherited=SensoryMappings((q,),inherited.senses,parameters))
            results,weights=evaluate_batch([animal],[-1,-2],[1,2],seeds=[43],return_weights=True)
            reference,detail,learned=evaluate(animal,[(-1,0),(-2,0)],[(1,0),(2,0)],seed=43,config=config)
            assert_equivalent(results[0],reference)
            assert_equivalent(learned_mappings(animal,weights,0).to_parameters(),learned.inherited.to_parameters())
            if force_limit:
                self.assertEqual(results[0]['training']['episodes'][0]['expectations'],16)
                self.assertEqual(results[0]['training']['episodes'][0]['expired'],16)

    def test_native_saved_run_export_playback_and_goal_controls(self):
        import tempfile
        from pathlib import Path
        from static_agent_training.experiment import run_experiment
        from static_agent_training.demo import export_demo
        from static_agent_training.play import attempt
        from static_agent_training.bias_audit import audit_checkpoint
        config=TrainingConfig(model='senses',backend='native',population=2,elites=1,generations=1,
                              episodes=3,assessment_episodes=3,initial_pairs=2,chain_budget=8,
                              encoding_seed=71,boundary_mode='terminate')
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'
            run_experiment(config,run)
            destination,bundle=export_demo(run)
            self.assertTrue((destination/'index.html').is_file())
            self.assertEqual(attempt(bundle,1,2,43)['model'],'senses')
            audit=audit_checkpoint(run,sample=1,repeats=1)
            self.assertEqual(set(audit['controls']),{'correct_goal','permuted_goal_input','goal_input_removed'})
