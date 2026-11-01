from dataclasses import asdict, replace
from itertools import product
import importlib.util
import unittest

from static_agent_training.accelerated import assert_equivalent, evaluate_batch, evaluate_frozen
from static_agent_training.baseline import random_success
from static_agent_training.environment import MOVES, NumberLine, Output
from static_agent_training.evaluation import evaluate
from static_agent_training.evolution import TrainingConfig,initialize_population,evaluate_generation,partitions
from static_agent_training.mappings import InheritedMappings
from static_agent_training.play import attempt


class BoundaryTests(unittest.TestCase):
    def token(self,env,kind,value):
        return Output(next(k for k,v in env.diagnostic_state()[kind].items() if v==value))

    def test_walls_count_moves_and_score_actual_observation(self):
        env=NumberLine(goal_conditioned=True,action_limit=2)
        env.reset(start=20,goal=-20)
        env.emit(self.token(env,'expectation_decoding',20),source='pair')
        transition=env.emit(self.token(env,'move_decoding',5))
        self.assertFalse(transition.done)
        self.assertTrue(transition.feedback[0].correct)
        self.assertEqual(env.diagnostic_state()['position'],20)
        transition=env.emit(self.token(env,'move_decoding',1))
        self.assertEqual(transition.reason,'action_limit')
        self.assertEqual(env.diagnostic_state()['actions'],2)
        for start,goal,move in [(19,20,5),(-19,-20,-5)]:
            env.reset(start=start,goal=goal)
            result=env.emit(self.token(env,'move_decoding',move))
            self.assertTrue(result.success)
            self.assertEqual(env.diagnostic_state()['position'],goal)

    def test_exact_random_probability_matches_clamped_exhaustive_sequences(self):
        for start,goal in [(20,19),(-20,-19),(18,20)]:
            successes=0
            for moves in product(MOVES,repeat=3):
                position=start
                for move in moves:
                    position=max(-20,min(20,position+move))
                    if position==goal:
                        successes+=1
                        break
            self.assertAlmostEqual(random_success(start,goal,3,boundary_mode='clamp'),successes/729)

    def test_old_config_preserves_legacy_boundary_rule(self):
        data=asdict(TrainingConfig())
        del data['boundary_mode']
        self.assertEqual(TrainingConfig.from_data(data).boundary_mode,'terminate')
        self.assertEqual(TrainingConfig().boundary_mode,'clamp')

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy is unavailable')
    def test_native_cuda_and_reference_agree_at_walls_and_on_legacy_mode(self):
        config=TrainingConfig(population=4,elites=1,episodes=4,assessment_episodes=4,
                              initial_pairs=2,chain_budget=16,seed=19)
        population=initialize_population(config)
        # Force +5; this distinguishes clamping at 20 from ending at 24.
        fixed=[]
        for species in population:
            if not species.repertoire.successors:
                continue
            data=species.inherited.to_data()
            entry=next(iter(species.repertoire.successors))
            world=NumberLine(goal_conditioned=True)
            plus5=self.token(world,'move_decoding',5).token
            for row in data['embedding'].values():
                for pair in row:
                    row[pair]=1000 if pair==entry else -1000
            for row in data['output'].values():
                for token in row:
                    row[token]=1000 if token==plus5 else -1000
            fixed.append(replace(species,inherited=InheritedMappings.from_data(data)))
        self.assertTrue(fixed)
        backends=['native']
        if importlib.util.find_spec('cupy'):
            import cupy
            if cupy.cuda.runtime.getDeviceCount():backends.append('cuda')
        for mode in ('clamp','terminate'):
            seeds=list(range(len(fixed)))
            expected=[evaluate(p,[19,-19],[18,-18],seed=i,training_goals=[20,-20],assessment_goals=[20,-20],
                               learning=False,boundary_mode=mode)[0] for i,p in enumerate(fixed)]
            for backend in backends:
                actual=evaluate_batch(fixed,[19,-19],[18,-18],seeds=seeds,training_goals=[20,-20],
                                      assessment_goals=[20,-20],learning=False,backend=backend,boundary_mode=mode)
                assert_equivalent(actual,expected)
                if mode=='clamp':
                    self.assertTrue(all(r['training']['episodes'][0]['success'] for r in actual))
                else:
                    self.assertTrue(all(r['training']['episodes'][0]['reason']=='out_of_bounds' for r in actual))
        # The extra frozen-only path uses the same assessment RNGs as reference.
        result=evaluate_frozen(fixed,[19,-19],[20,-20],seeds=list(range(len(fixed))),workers=2)
        self.assertTrue(all(r['episodes'][0]['success'] for r in result))
        # Legacy accelerated generation must replay its champion under saved rules.
        evaluate_generation(population,replace(config,backend='native',boundary_mode='terminate'),0,partitions(config.seed))

    def test_playground_random_attempt_never_leaves_line(self):
        bundle=dict(varying_goals=True,encoding_seed=0)
        for start,goal in [(20,-20),(-20,20),(19,20)]:
            for seed in range(10):
                result=attempt(bundle,start,goal,seed,policy='random')
                self.assertTrue(all(-20<=f['position']<=20 for f in result['frames']))
                self.assertNotEqual(result['frames'][-1]['reason'],'out_of_bounds')
