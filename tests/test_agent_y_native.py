"""Native partition-policy and whole-lifetime parity, not just task scores."""
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path

from static_agent_training.evolution import TrainingConfig, initialize_population, generation_tasks, partitions
from static_agent_training.experimental.sensory_evolution import evaluate
from static_agent_training.experimental.agent_y_accelerated import evaluate_batch
from static_agent_training.experimental.sensory_genome import SensoryMappings
from static_agent_training.accelerated import assert_equivalent
from static_agent_training.seeds import derive
from static_agent_training.experiment import run_experiment


class AgentYNativeTests(unittest.TestCase):
    def config(self, **values):
        return replace(TrainingConfig(model='senses',agent_y=True,backend='native',
            population=2,elites=1,receptors=16,sensory_partitions=1,initial_pairs=3,
            chain_budget=24,episodes=12,assessment_episodes=12,generations=2,workers=2,
            inherit_lifetime=True), **values)

    def parity(self, config, mode='mixed'):
        population=initialize_population(config)
        if mode!='mixed':
            adjusted=[]
            for p in population:
                model=p.inherited.model(p.repertoire)
                for row in model.router_weights['split']:
                    row[-1]=100 if mode=='singletons' else -100
                adjusted.append(replace(p,inherited=SensoryMappings(p.inherited.pair_ids,
                    p.inherited.senses,p.inherited.parameters,model.router_data())))
            population=adjusted
        split=partitions(config.seed,include_zero=config.varying_goals,environment=config.environment)
        s,t,g,h=generation_tasks(config,0,split)
        seeds=[derive(config.seed,'evaluation',0,i) for i in range(len(population))]
        rows,parents=evaluate_batch(population,s,t,g,h,seeds=seeds,config=config)
        for i,p in enumerate(population):
            reference,_,parent=evaluate(p,list(zip(s,g)),list(zip(t,h)),seed=seeds[i],config=config)
            assert_equivalent(rows[i],reference)
            assert_equivalent(parents[i].inherited.to_data(),parent.inherited.to_data())

    def test_variable_counts_policies_and_worlds(self):
        for environment in ['original','lights-out-2x2','switchboard']:
            for policy in ['agreement','vote']:
                with self.subTest(environment=environment,policy=policy):
                    self.parity(self.config(environment=environment,action_policy=policy))

    def test_singletons_single_group_frozen_and_signals(self):
        for mode in ['singletons','single']:
            for learning in [True,False]:
                with self.subTest(mode=mode,learning=learning):
                    self.parity(self.config(learning=learning,sensory_interface='signals',
                        retina_width=4,retina_height=4),mode)

    def test_native_checkpoint_resume(self):
        config=self.config(episodes=2,assessment_episodes=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            full=run_experiment(config,root/'full')
            run_experiment(config,root/'split',until_generation=1)
            resumed=run_experiment(config,root/'split',resume=True)
            self.assertEqual(full,resumed)

    def test_explicit_python_to_native_resume(self):
        python=self.config(backend='python',workers=1,episodes=2,assessment_episodes=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            run_experiment(python,root,until_generation=1)
            result=run_experiment(replace(python,backend='native'),root,resume=True)
            self.assertEqual(result['config']['backend'],'native')
            self.assertEqual(result['backend_migration'],dict(previous_backend='python',
                current_backend='native',resumed_at_generation=1))

    def test_shared_batch_and_frozen_entry_points_preserve_router(self):
        import random
        from static_agent_training.experimental.sensory_accelerated import evaluate_batch as shared_batch, evaluate_frozen
        from static_agent_training.experimental.sensory_evolution import block
        config=self.config(episodes=3,assessment_episodes=3)
        population=initialize_population(config)
        before=[p.inherited.to_data() for p in population]
        split=partitions(config.seed,include_zero=config.varying_goals,environment=config.environment)
        s,t,g,h=generation_tasks(config,0,split)
        seeds=[derive(config.seed,'evaluation',0,i) for i in range(2)]
        direct,_=evaluate_batch(population,s,t,g,h,seeds=seeds,config=config)
        shared=shared_batch(population,s,t,training_goals=g,assessment_goals=h,seeds=seeds,
            workers=2,sensory_partitions=1,goal_conditioned=True)
        assert_equivalent(direct,shared)
        frozen=evaluate_frozen(population,t,h,seeds=seeds,workers=2,sensory_partitions=1)
        for i,p in enumerate(population):
            model=p.inherited.model(p.repertoire,learning_config=config.learning_config)
            model.embedding_rng=random.Random(derive(seeds[i],'assessment','embedding'))
            model.output_rng=random.Random(derive(seeds[i],'assessment','output'))
            reference=block(model,list(zip(t,h)),learn=False,encoding_seed=0)
            reference.pop('partition_groups')
            assert_equivalent(frozen[i],reference)
        self.assertEqual(before,[p.inherited.to_data() for p in population])
