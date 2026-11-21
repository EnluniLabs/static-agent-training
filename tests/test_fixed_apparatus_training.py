"""A single eye survives new worlds, evolution and exact checkpoint resume."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.experiment import run_experiment, load_checkpoint
from static_agent_training.experimental.bit_model import BitModel, run_episode
from static_agent_training.experimental.senses import SensoryAdapter
from static_agent_training.environments import stimulus_world


class FixedApparatusTrainingTests(unittest.TestCase):
    def config(self):
        return TrainingConfig(model='senses',sensory_interface='retina',receptors=256,
            retina_width=16,retina_height=16,sensory_partitions=4,population=2,elites=1,
            initial_pairs=2,chain_budget=8,generations=2,episodes=1,assessment_episodes=1,
            workers=1,backend='native',seed=19)

    def test_identical_frozen_animal_can_see_multiple_worlds(self):
        config=self.config();species=initialize_population(config)[0]
        model=species.inherited.model(species.repertoire)
        apparatus=model.senses;parameters=model.parameters()
        for environment,task in [('original',(0,3)),('shifted',(50,53)),
                                  ('lights-out',(7,0)),('switchboard',(73,0))]:
            with self.subTest(environment=environment):
                world=SensoryAdapter(stimulus_world(environment),model.senses)
                before=len(model.events)
                run_episode(model,world,task,learn=False)
                self.assertIs(model.senses,apparatus)
                self.assertEqual(model.parameters(),parameters)
                for event in model.events[before:]:
                    if event['kind']=='EMBED':
                        self.assertEqual(len(bytes.fromhex(event['frame'][0])),32)
                self.assertEqual(model.senses.partition_indices(),apparatus.partition_indices())

    def test_native_resume_and_all_saved_retinas_stay_identical(self):
        config=self.config()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            full=run_experiment(config,root/'full')
            run_experiment(config,root/'resumed',until_generation=1)
            resumed=run_experiment(config,root/'resumed',resume=True)
            self.assertEqual(resumed,full)
            state=load_checkpoint(root/'resumed')
            for member in state['population']:
                senses=member['inherited']['senses']
                self.assertEqual((senses['width'],senses['height'],senses['partitions']),(16,16,4))
            source=initialize_population(config)[0].inherited.senses
            with self.assertRaisesRegex(ValueError,'saved experiment configuration'):
                run_experiment(replace(config,sensory_partitions=8),root/'resumed',resume=True)
