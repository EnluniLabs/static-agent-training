from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.benchmark import compare
from static_agent_training.catalog import catalog
from static_agent_training.comparison import run_study
from static_agent_training.evolution import TrainingConfig, evaluate_generation, initialize_population, partitions, reproduce
from static_agent_training.experiment import run_experiment
from static_agent_training.genome import Species, develop
from static_agent_training.mutation import MutationConfig, mutate
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.experimental.sensory_genome import SensoryMappings


NONE = MutationConfig(weight_probability=0, subtree_probability=0, term_probability=0,
                      insertion_probability=0, deletion_probability=0)


class SensoryEvolutionTests(unittest.TestCase):
    def config(self, **values):
        return replace(TrainingConfig(model='senses', population=3, generations=3, episodes=3,
            assessment_episodes=3, elites=1, initial_pairs=2, chain_budget=8, seed=19), **values)

    def parent(self):
        identity = ('abs', ('var', 0))
        k = ('abs', ('abs', ('var', 1)))
        return develop([(identity, identity), (k, k)], (), (), senses=FixedSenses(), budgets=16)

    def test_heritable_parameters_match_model_initialization_and_are_immutable(self):
        parent = self.parent()
        inherited = SensoryMappings.initialize(parent.repertoire, senses=FixedSenses(), seed=17)
        model = BitModel(parent.repertoire, senses=FixedSenses(), seed=17)
        self.assertEqual(inherited.to_parameters(), model.parameters())
        with self.assertRaises(TypeError):
            inherited.parameters['bias'][0] = 10
        restored = Species.from_data(json.loads(json.dumps(parent.to_data())))
        self.assertEqual(restored.genotype_id, parent.genotype_id)

    def test_weight_and_structural_mutation_reconcile_every_parameter_group(self):
        parent = self.parent()
        before = parent.to_data()
        child, log = mutate(parent, seed=20, config=replace(NONE, weight_probability=1))
        self.assertFalse(log['structural_changed'])
        self.assertGreater(log['perturbed_logits'], 0)
        for group in ('bias', 'weights', 'gates', 'prediction'):
            self.assertNotEqual(child.inherited.to_parameters()[group], parent.inherited.to_parameters()[group])
        structural, log = mutate(parent, seed=22, config=replace(NONE, insertion_probability=1))
        self.assertTrue(log['structural_changed'])
        self.assertEqual(structural.inherited.senses, parent.inherited.senses)
        for q in set(parent.inherited.pair_ids) & set(structural.inherited.pair_ids):
            old_index = parent.inherited.pair_ids.index(q)
            new_index = structural.inherited.pair_ids.index(q)
            self.assertEqual(parent.inherited.parameters['bias'][old_index], structural.inherited.parameters['bias'][new_index])
            self.assertEqual(parent.inherited.parameters['gates'][q], structural.inherited.parameters['gates'][q])
            self.assertEqual(parent.inherited.parameters['prediction'][q], structural.inherited.parameters['prediction'][q])
            self.assertEqual([row[old_index] for row in parent.inherited.parameters['weights']],
                             [row[new_index] for row in structural.inherited.parameters['weights']])
        self.assertEqual(before, parent.to_data())

    def test_learning_frozen_assessment_and_optional_inheritance(self):
        parent = self.parent()
        population = [parent] * 3
        before = parent.to_data()
        cfg = self.config(episodes=8)
        with patch('static_agent_training.engine.run', side_effect=AssertionError('runtime reduction')):
            record, detail = evaluate_generation(population, cfg, 0, partitions(cfg.seed, include_zero=True))
        self.assertGreater(sum(r['training']['empirical_updates'] for r in record['individuals']), 0)
        self.assertTrue(all(r['assessment']['empirical_updates'] == 0 and r['assessment']['conflicts'] == 0 for r in record['individuals']))
        self.assertEqual(before, parent.to_data())
        cfg = replace(cfg, inherit_lifetime=True)
        record, detail, learned = evaluate_generation(population, cfg, 0, partitions(cfg.seed, include_zero=True), return_lifetime=True)
        self.assertTrue(any(p.genotype_id != parent.genotype_id for p in learned))
        self.assertEqual(before, parent.to_data())
        children, ancestry = reproduce(population, record['individuals'], replace(cfg, mutation=NONE), 0,
                                       lifetime_population=learned)
        selected = record['champion_index']
        self.assertEqual(children[0].genotype_id, learned[selected].genotype_id)

    def test_checkpoint_resume_equals_uninterrupted_and_benchmark_is_frozen(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = run_experiment(cfg, root / 'runs/full')
            run_experiment(cfg, root / 'runs/resumed', until_generation=1)
            resumed = run_experiment(cfg, root / 'runs/resumed', resume=True)
            self.assertEqual(full, resumed)
            for generation in range(3):
                filename = f'generations/{generation:05d}.json'
                self.assertEqual(json.loads((root/'runs/full'/filename).read_text()), json.loads((root/'runs/resumed'/filename).read_text()))
            result = compare(root / 'runs/full', episodes=12)
            self.assertEqual(result['model'], 'senses')
            self.assertEqual(result['episodes'], 12)
            self.assertEqual({r['kind'] for r in catalog(root, processes={})}, {'sensory-training'})
            run_experiment(replace(cfg, generations=4), root/'runs/full', resume=True)
            self.assertEqual(len(list((root/'runs/full/generations').glob('*.json'))), 4)

    def test_speciation_and_both_initial_repertoire_populations_match(self):
        cfg = self.config(speciation=True, max_species=2)
        population = initialize_population(cfg)
        old = initialize_population(replace(cfg, model='categorical'))
        self.assertEqual([p.initial_pairs for p in population], [p.initial_pairs for p in old])
        self.assertEqual([p.repertoire.to_data() for p in population], [p.repertoire.to_data() for p in old])
        record, _ = evaluate_generation(population, cfg, 0, partitions(cfg.seed, include_zero=True))
        children, ancestry, report, state = reproduce(population, record['individuals'], cfg, 0, return_speciation=True)
        self.assertTrue(report['enabled'])
        self.assertEqual(len(children), cfg.population)
        self.assertTrue(all(isinstance(p.inherited, SensoryMappings) for p in children))

    def test_evolutionary_model_comparison_runs_both_models_and_resumes(self):
        cfg = self.config(population=2, elites=1, generations=2, episodes=2, assessment_episodes=2,
                          mutation=NONE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root/'study'
            result = run_study(study, replace(cfg, model='categorical'), seeds=[19], benchmark_episodes=10,
                               runs_root=root/'runs', treatment='model')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['control_label'], 'Old categorical')
            self.assertEqual(result['treatment_label'], 'Fixed senses')
            self.assertTrue(result['pairs'][0]['initialization_verified'])
            self.assertTrue(result['pairs'][0]['tasks_verified'])
            self.assertEqual(result['pairs'][0]['common_generations'], 2)
            self.assertEqual(result['arms'][0]['completed'], 2)
            self.assertEqual(result['arms'][1]['completed'], 2)
            resumed = run_study(study, resume=True)
            self.assertEqual(result['pairs'], resumed['pairs'])
            manifest = json.loads((study/'manifest.json').read_text())
            configs = [json.loads((Path(a['directory'])/'config.json').read_text())['config'] for a in manifest['arms']]
            self.assertEqual([c['model'] for c in configs], ['categorical', 'senses'])
