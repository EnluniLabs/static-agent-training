from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.accelerated import assert_equivalent
from static_agent_training.baseline import random_success
from static_agent_training.environment import NumberLine, Output
from static_agent_training.environments import environment_spec
from static_agent_training.evolution import TrainingConfig, initialize_population, partitions, evaluate_generation, generation_tasks
from static_agent_training.experiment import run_experiment
from static_agent_training.genome import Species
from static_agent_training.populations import save_population, read_population, benchmark_populations
from static_agent_training.experimental.sensory_accelerated import _frames
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.workspace_sessions import SessionManager


class EnvironmentPopulationTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(TrainingConfig(model='senses',environment='shifted',backend='native',population=3,
            elites=1,generations=2,episodes=4,assessment_episodes=4,initial_pairs=2,chain_budget=8),**kwargs)

    def test_shifted_walls_tasks_and_exact_random_translation(self):
        for mode in ('clamp','terminate'):
            world=NumberLine(lower=30,upper=70,goal_conditioned=True,boundary_mode=mode)
            world.reset(start=70,goal=30)
            plus=next(t for t,v in world.diagnostic_state()['move_decoding'].items() if v==5)
            result=world.emit(Output(plus))
            self.assertEqual(world.diagnostic_state()['position'],70 if mode=='clamp' else 75)
            self.assertEqual(result.reason,None if mode=='clamp' else 'out_of_bounds')
            self.assertEqual(random_success(64,49,boundary_mode=mode,lower=30,upper=70),
                             random_success(14,-1,boundary_mode=mode))
        a=partitions(19,include_zero=True);b=partitions(19,include_zero=True,environment='shifted')
        self.assertEqual({k:[x+50 for x in v] for k,v in a.items()},b)
        for distribution in ('uniform','normal'):
            cfg=self.config(distance_distribution=distribution,varying_goals=False)
            tasks=generation_tasks(cfg,0,b)
            self.assertTrue(all(30<=v<=70 and v!=50 for v in tasks[0]))
        with self.assertRaises(ValueError):self.config(environment='unknown')

    def test_shifted_native_matches_reference_including_fixed_goals(self):
        for model in ('categorical','senses'):
            for varying,mode in ((True,'clamp'),(False,'terminate')):
                cfg=self.config(model=model,backend='python',varying_goals=varying,boundary_mode=mode,
                                episodes=20,assessment_episodes=12,inherit_lifetime=True)
                pop=initialize_population(cfg);split=partitions(cfg.seed,include_zero=varying,environment='shifted')
                reference,_,parents=evaluate_generation(pop,cfg,0,split,return_lifetime=True)
                for backend in ('native',):
                    actual,_,learned=evaluate_generation(pop,replace(cfg,backend=backend),0,split,return_lifetime=True)
                    assert_equivalent(actual,reference)
                    for a,b in zip(learned,parents):assert_equivalent(a.inherited.to_data(),b.inherited.to_data())

    def test_native_sensory_packets_are_physical_shifted_values(self):
        senses=FixedSenses()
        shifted=_frames(senses,True,'shifted')
        self.assertEqual(shifted[0].tolist(),list(senses.spec.encode(senses.frame(b'[30,30]'))))
        self.assertNotEqual(shifted[0].tolist(),_frames(senses,True)[0].tolist())
        self.assertEqual(shifted.shape,_frames(senses,True).shape)

    def test_population_export_import_resume_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.config()
            run_experiment(cfg,root/'full')
            run_experiment(cfg,root/'resumed',until_generation=1)
            run_experiment(cfg,root/'resumed',resume=True)
            self.assertEqual(read_population(root/'full/best-population.json.gz')['members'],
                             read_population(root/'resumed/best-population.json.gz')['members'])
            saved=root/'population.json.gz';result=save_population(root/'full',saved,count=3)
            self.assertEqual(result['count'],3)
            before=saved.read_bytes();data=read_population(saved)
            imported=replace(cfg,initial_population=str(saved),population=5)
            animals=initialize_population(imported)
            self.assertEqual([a.genotype_id for a in animals[:3]],
                             [Species.from_data(r['individual']).genotype_id for r in data['members']])
            self.assertEqual(saved.read_bytes(),before)
            self.assertEqual([a.to_data() for a in animals],[a.to_data() for a in initialize_population(imported)])
            run_experiment(replace(imported,generations=1),root/'imported')
            self.assertTrue((root/'imported/initial-population.json.gz').is_file())
            # Old sessions only contain saved champion genomes; never invent a full evaluated population.
            (root/'full/best-population.json.gz').unlink()
            fallback=save_population(root/'full',root/'legacy.json.gz',count=20)
            self.assertEqual(fallback['source_type'],'generation-champions-only')
            with self.assertRaises(ValueError):initialize_population(replace(imported,model='categorical'))
            with self.assertRaises(ValueError):initialize_population(replace(imported,receptors=512))

    def test_population_benchmark_shared_tasks_and_frozen_assessment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for i,env in enumerate(('original','shifted')):
                cfg=self.config(environment=env,generations=1)
                run_experiment(cfg,root/str(i));path=root/(str(i)+'.json.gz')
                save_population(root/str(i),path,count=2);paths.append(path)
            hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            report=benchmark_populations(paths,root/'benchmark',environment='shifted',episodes=4,assessment_episodes=5)
            self.assertEqual(report['status'],'completed')
            self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],hashes)
            tasks=[]
            for population in report['populations']:
                for row in population['individuals']:
                    self.assertEqual(row['assessment']['empirical_updates'],0)
                    self.assertEqual(row['assessment']['conflicts'],0)
                    tasks.append([(r['start'],r['goal']) for r in row['assessment']['episodes']])
            self.assertTrue(all(t==tasks[0] for t in tasks))
            self.assertTrue(all(30<=s<=70 and 30<=g<=70 for s,g in tasks[0]))

    def test_ui_command_includes_environment_and_validates_import_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manager=SessionManager(root)
            cfg=self.config(generations=1);directory=root/'runs/source';run_experiment(cfg,directory)
            path=root/'runs/populations/test.json.gz';save_population(directory,path,count=2)
            _,args=manager.command('sensory-training',dict(environment='shifted',initial_population='populations/test.json.gz'),root/'new')
            self.assertEqual(args[args.index('--environment')+1],'shifted')
            self.assertEqual(args[args.index('--initial-population')+1],str(path))
            with self.assertRaises(ValueError):manager.command('training',dict(initial_population='populations/test.json.gz'),root/'bad')

    def test_load_original_candidates_into_shifted_evolution_for_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in ('categorical','senses'):
                original=self.config(model=model,environment='original',generations=1)
                source=root/model/'source';run_experiment(original,source)
                path=root/model/'candidates.json.gz';save_population(source,path,count=2)
                saved=path.read_bytes();data=read_population(path)
                imported=replace(original,environment='shifted',initial_population=str(path),population=2,generations=2)
                parents=initialize_population(imported)
                for parent,row in zip(parents,data['members']):
                    ancestor=Species.from_data(row['individual'])
                    self.assertEqual(parent.repertoire.to_data(),ancestor.repertoire.to_data())
                    self.assertEqual(parent.initial_pairs,ancestor.initial_pairs)
                    if model=='senses':
                        self.assertEqual(parent.inherited.to_data(),ancestor.inherited.to_data())
                    else:
                        self.assertTrue(all(s.startswith('goal_') or s in {f'obs_{x:+d}' for x in range(30,71)} for s in parent.inherited.observations))
                        for q in parent.inherited.pair_ids:
                            for token in ancestor.inherited.outputs:
                                if token.startswith('move_'):
                                    self.assertEqual(parent.inherited.output[q][token],ancestor.inherited.output[q][token])
                        self.assertNotEqual(parent.inherited.observations,ancestor.inherited.observations)
                target=root/model/'target'
                run_experiment(imported,target,until_generation=1)
                self.assertEqual(path.read_bytes(),saved)
                provenance=json.loads((target/'candidate-import.json').read_text())
                self.assertEqual((provenance['source_environment'],provenance['target_environment']),('original','shifted'))
                # The imported source file is not needed to resume an existing session.
                path.unlink()
                run_experiment(imported,target,resume=True)
                record=json.loads((target/'generations/00001.json').read_text())
                self.assertTrue(all(30<=s<=70 for s in record['training_starts']))
