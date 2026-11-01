from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from static_agent_training.evolution import TrainingConfig, initialize_population, reproduce
from static_agent_training.experiment import load_checkpoint, run_experiment
from static_agent_training.mutation import MutationConfig
from static_agent_training.speciation import describe, distance, partition, allocate


NONE=MutationConfig(weight_probability=0,subtree_probability=0,term_probability=0,
                    insertion_probability=0,deletion_probability=0)


class SpeciationTests(unittest.TestCase):
    def config(self,**kwargs):
        return replace(TrainingConfig(population=12,generations=3,episodes=3,assessment_episodes=3,
                                      elites=2,initial_pairs=2,chain_budget=8,seed=19),**kwargs)

    def test_structure_ignores_weight_changes_and_preserves_group_identity(self):
        config=self.config()
        p=initialize_population(config)
        changed=p[0].inherited.to_data()
        for row in changed['embedding'].values():
            for key in row:
                row[key]+=10
        inherited=type(p[0].inherited).from_data(changed)
        variant=replace(p[0],inherited=inherited)
        self.assertEqual(distance(describe(p[0]),describe(variant)),0)
        groups,state,report=partition([p[0],variant,p[1]],threshold=0,limit=3)
        self.assertEqual(report['assignments'][0],report['assignments'][1])
        regrouped,next_state,new=partition([variant,p[0],p[1]],threshold=0,limit=3,previous=state)
        self.assertEqual(report['assignments'],new['assignments'])
        self.assertEqual(state['next_id'],next_state['next_id'])

    def test_bound_membership_and_fitness_sharing_allocation(self):
        config=self.config()
        p=initialize_population(config)
        groups,state,report=partition(p,threshold=0,limit=4)
        self.assertLessEqual(len(groups),4)
        self.assertEqual(sorted(i for g in groups for i in g['members']),list(range(12)))
        self.assertGreater(report['forced_assignments'],0)
        # Duplicate members cannot inflate a niche's shared-fitness mass.
        groups=[dict(id=0,members=[0]),dict(id=1,members=list(range(1,10)))]
        self.assertEqual(allocate(groups,[.5]*10,10),[5,5])
        self.assertEqual(allocate(groups,[0]*10,10),[5,5])
        self.assertEqual(sum(allocate(groups,[.9]+[.1]*9,11)),11)
        self.assertTrue(all(q>=1 for q in allocate(groups,[1]+[0]*9,11)))

    def test_one_species_matches_control_and_assessment_does_not_select(self):
        config=self.config(mutation=NONE)
        p=initialize_population(config)
        results=[dict(training=dict(fitness=i/12),assessment=dict(fitness=1-i/12)) for i in range(12)]
        control,_=reproduce(p,results,config,2)
        grouped,ancestry,report,state=reproduce(p,results,replace(config,speciation=True,speciation_threshold=1),2,return_speciation=True)
        self.assertEqual([x.genotype_id for x in control],[x.genotype_id for x in grouped])
        self.assertEqual(sum(a['kind']=='elite' for a in ancestry),config.elites)
        config=replace(config,speciation=True,max_species=4)
        a,aa,report,state=reproduce(p,results,config,1,return_speciation=True)
        reversed_assessment=[dict(training=r['training'],assessment=dict(fitness=1-r['assessment']['fitness'])) for r in results]
        b,bb,other,_=reproduce(p,reversed_assessment,config,1,return_speciation=True)
        self.assertEqual([x.genotype_id for x in a],[x.genotype_id for x in b])
        self.assertEqual(report,other)
        lookup={g['id']:set(g['members']) for g in report['groups']}
        for event in aa:
            if event['kind']=='offspring':
                self.assertTrue(set(event['competitors'])<=lookup[event['species_id']])
        self.assertEqual(len(a),config.population)
        self.assertEqual(sum(g['offspring'] for g in report['groups']),config.population-config.elites)

    def test_checkpoint_resume_restores_representatives(self):
        config=self.config(speciation=True,max_species=4)
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/'full'
            resumed=Path(tmp)/'resumed'
            run_experiment(config,full)
            partial=run_experiment(config,resumed,until_generation=1)
            self.assertTrue(partial['speciation_state']['representatives'])
            run_experiment(config,resumed,resume=True)
            self.assertEqual(load_checkpoint(full),load_checkpoint(resumed))

    def test_option_validation(self):
        for values in [dict(speciation='yes'),dict(speciation_threshold=-1),dict(max_species=0)]:
            with self.assertRaises(ValueError):
                self.config(**values)
