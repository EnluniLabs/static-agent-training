from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.accelerated import assert_equivalent
from static_agent_training.comparison import run_study
from static_agent_training.evaluation import evaluate
from static_agent_training.evolution import TrainingConfig,initialize_population,evaluate_generation,generation_tasks,partitions,reproduce
from static_agent_training.experiment import load_checkpoint,run_experiment
from static_agent_training.genome import inherit_lifetime_mappings
from static_agent_training.mutation import MutationConfig
from static_agent_training.seeds import derive

NONE=MutationConfig(weight_probability=0,subtree_probability=0,term_probability=0,insertion_probability=0,deletion_probability=0)


class InheritanceTests(unittest.TestCase):
    def config(self,**kwargs):
        return replace(TrainingConfig(population=4,generations=2,episodes=4,assessment_episodes=4,
                                      elites=1,initial_pairs=2,chain_budget=16,seed=19,mutation=NONE),**kwargs)

    def test_commits_training_weights_for_every_parent_without_mutating_originals(self):
        c=self.config(inherit_lifetime=True)
        population=initialize_population(c);before=[p.to_data() for p in population]
        split=partitions(c.seed,include_zero=True)
        record,detail,parents=evaluate_generation(population,c,0,split,return_lifetime=True)
        starts,assessment,goals,assessment_goals=generation_tasks(c,0,split)
        self.assertGreater(sum(r['training']['empirical_updates'] for r in record['individuals']),0)
        for i,p in enumerate(population):
            result,expected=evaluate(p,starts,assessment,seed=derive(c.seed,'evaluation',0,i),
                                     training_goals=goals,assessment_goals=assessment_goals,boundary_mode=c.boundary_mode)
            self.assertEqual(parents[i].inherited.to_data()['embedding'],expected['learned_embedding'])
            self.assertEqual(parents[i].inherited.to_data()['output'],expected['learned_output'])
            self.assertIs(parents[i].repertoire,p.repertoire)
            self.assertEqual(parents[i].initial_pairs,p.initial_pairs)
        self.assertEqual(before,[p.to_data() for p in population])
        children,ancestry=reproduce(population,record['individuals'],c,0,lifetime_population=parents)
        for child,event in zip(children,ancestry):
            self.assertEqual(child.genotype_id,parents[event['parent_index']].genotype_id)
        with self.assertRaisesRegex(ValueError,'end-of-training'):
            reproduce(population,record['individuals'],c,0)
        # Committing snapshots makes immutable copies, rather than aliasing a live table.
        source=parents[0].inherited.to_data()
        committed=inherit_lifetime_mappings(population[0],source['embedding'],source['output'])
        signal=next(iter(source['embedding']));pair=next(iter(source['embedding'][signal]))
        old=committed.inherited.embedding[signal][pair]
        source['embedding'][signal][pair]+=100
        self.assertEqual(committed.inherited.embedding[signal][pair],old)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'optional acceleration unavailable')
    def test_accelerated_inheritance_matches_reference_weights(self):
        c=self.config(inherit_lifetime=True)
        p=initialize_population(c);split=partitions(c.seed,include_zero=True)
        reference,_,expected=evaluate_generation(p,c,0,split,return_lifetime=True)
        backends=['native']
        for backend in backends:
            actual,_,parents=evaluate_generation(p,replace(c,backend=backend,batch_size=3),0,split,return_lifetime=True)
            assert_equivalent(actual,reference)
            for a,b in zip(parents,expected):
                assert_equivalent(a.inherited.to_data(),b.inherited.to_data())

    def test_checkpoint_resume_and_provenance(self):
        c=self.config(inherit_lifetime=True)
        with tempfile.TemporaryDirectory() as tmp:
            full,resumed=Path(tmp)/'full',Path(tmp)/'resumed'
            run_experiment(c,full)
            run_experiment(c,resumed,until_generation=1)
            run_experiment(c,resumed,resume=True)
            self.assertEqual(load_checkpoint(full),load_checkpoint(resumed))
            record=json.loads((full/'generations/00000.json').read_text())
            self.assertTrue(record['inheritance']['enabled'])
            self.assertGreater(record['inheritance']['changed_parents'],0)
            for event in record['reproduction']:
                self.assertTrue(event['lifetime_inherited'])
                self.assertEqual(event['evaluated_parent_id'],record['individuals'][event['parent_index']]['genotype_id'])
                if event['kind']=='elite':self.assertEqual(event['genotype_id'],event['reproductive_parent_id'])
                else:self.assertEqual(event['mutation']['parent_id'],event['reproductive_parent_id'])

    def test_matched_inheritance_study_changes_only_inheritance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'study'
            report=run_study(directory,self.config(generations=1),seeds=[19],benchmark_episodes=20,
                             runs_root=root/'runs',treatment='inheritance')
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['treatment'],'inheritance')
            self.assertTrue(report['pairs'][0]['initialization_verified'])
            self.assertTrue(report['pairs'][0]['tasks_verified'])
            manifest=json.loads((directory/'manifest.json').read_text());configs=[]
            for arm in manifest['arms']:
                config=json.loads((Path(arm['directory'])/'config.json').read_text())['config']
                self.assertTrue(config['learning']);self.assertFalse(config['speciation'])
                self.assertEqual(config.pop('inherit_lifetime'),arm['enabled']);configs.append(config)
            self.assertEqual(configs[0],configs[1])
            restored=run_study(directory,resume=True)
            self.assertEqual(restored['pairs'],report['pairs'])
            self.assertEqual(restored['treatment'],'inheritance')
