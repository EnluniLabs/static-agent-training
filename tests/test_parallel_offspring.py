"""Parallel mutation must be identical to sequential evolution, including logs."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from static_agent_training import serialization as json
from static_agent_training.evolution import TrainingConfig, initialize_population, reproduce
from static_agent_training.experiment import run_experiment, load_checkpoint
from static_agent_training.offspring import _pack_species, _unpack_species, _mutate_packet, _parent_cache, reproduction_workers
from static_agent_training.mutation import MutationConfig


class ParallelOffspringTests(unittest.TestCase):
    def config(self, **changes):
        return replace(TrainingConfig(population=4,elites=1,initial_pairs=2,chain_budget=8,
            generations=2,episodes=1,assessment_episodes=1,seed=19,workers=2), **changes)

    def test_processes_match_serial_for_model_speciation_and_lifetime_inheritance(self):
        for model,speciation,inherit in [('categorical',False,False),('senses',True,False),('senses',False,True)]:
            config=self.config(model=model,speciation=speciation,inherit_lifetime=inherit,
                sensory_interface='signals' if model=='senses' else 'legacy-bytes',sensory_partitions=2 if model=='senses' else 1)
            population=initialize_population(config)
            results=[dict(training=dict(fitness=i/4),assessment=dict(fitness=1-i/4)) for i in range(4)]
            lifetime=[replace(p) for p in population] if inherit else None
            kwargs=dict(return_speciation=True,lifetime_population=lifetime)
            original=[p.to_data() for p in population]
            sequential=reproduce(population,results,replace(config,reproduction_workers=1),2,**kwargs)
            parallel=reproduce(population,results,replace(config,reproduction_workers=2),2,**kwargs)
            self.assertEqual([p.to_data() for p in sequential[0]],[p.to_data() for p in parallel[0]])
            self.assertEqual(sequential[1:],parallel[1:])
            self.assertEqual(original,[p.to_data() for p in population])

    def test_parallel_checkpoint_resume_matches_uninterrupted(self):
        config=self.config(model='senses',sensory_interface='signals',sensory_partitions=2,reproduction_workers=2)
        with tempfile.TemporaryDirectory() as root:
            full=Path(root)/'full';resumed=Path(root)/'resumed'
            run_experiment(config,full)
            run_experiment(config,resumed,until_generation=1)
            run_experiment(config,resumed,resume=True)
            self.assertEqual(load_checkpoint(full),load_checkpoint(resumed))

    def test_transport_handles_deep_terms_and_does_not_reuse_different_provenance(self):
        parent=initialize_population(self.config())[0]
        development=json.loads(parent.development_json)
        term=('abs',('var',0))
        for _ in range(1500):term=('abs',term)
        development['groups'][0]['partial_term']=term
        changed=replace(parent,development_json=json.dumps(development))
        self.assertEqual(parent.genotype_id,changed.genotype_id)
        restored=_unpack_species(_pack_species(changed))
        self.assertEqual(restored.development_json,changed.development_json)
        self.assertEqual(restored.genotype_id,changed.genotype_id)
        no_mutation=MutationConfig(weight_probability=0,subtree_probability=0,term_probability=0,insertion_probability=0,deletion_probability=0)
        _parent_cache.clear()
        _mutate_packet(_pack_species(parent),3,no_mutation)
        packet,_=_mutate_packet(_pack_species(changed),3,no_mutation)
        self.assertEqual(_unpack_species(packet).development_json,changed.development_json)
        self.assertEqual(len(_parent_cache),2)
        _parent_cache.clear()

    def test_auto_avoids_processes_for_small_work_and_invalid_settings(self):
        config=self.config();population=initialize_population(config)
        self.assertEqual(reproduction_workers(population,[dict(parent_index=0,index=i) for i in range(4)],config),1)
        for value in [-1,True,1.5]:
            with self.assertRaises(ValueError):replace(config,reproduction_workers=value)
