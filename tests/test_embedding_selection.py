"""Selection policy, historical defaults, exact MT19937 tie streams, and learned parity."""
import ctypes
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from static_agent_training.accelerated import assert_equivalent
from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.experiment import run_experiment
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.sensory_accelerated import evaluate_batch, learned_mappings, SOURCE
from static_agent_training.experimental.sensory_evolution import evaluate
from static_agent_training.experimental.sensory_genome import SensoryMappings
from static_agent_training.mappings import sample


class EmbeddingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=TrainingConfig(model='senses',population=2,elites=1,initial_pairs=3,
            generations=2,episodes=3,assessment_episodes=3,workers=1,backend='native')
        cls.parent=initialize_population(cls.cfg)[0]

    def model(self, mode='argmax'):
        m=self.parent.inherited.model(self.parent.repertoire,embedding_selection=mode)
        m.bias=[0.0]*len(m.ids)
        m.weights=[[0.0]*len(m.ids) for _ in m.weights]
        return m

    def test_unique_max_and_weight_change(self):
        m=self.model();frame=m.senses.frame(b'abc');m.bias[1]=2
        before=m.embedding_rng.getstate()
        for _ in range(100):m.observe(frame);self.assertEqual(m.pair,m.ids[1])
        self.assertEqual(before,m.embedding_rng.getstate())
        # Change an active sensory weight, leaving biases and sensory encoding alone.
        active=next(i for i,bit in enumerate(m.spec.encode(frame)) if bit)
        m.weights[active][0]=5
        m.observe(frame);self.assertEqual(m.pair,m.ids[0])

    def test_ties_are_uniform_seeded_and_exclude_lower_probabilities(self):
        m=self.model();m.bias=[-2.0]*len(m.ids)
        for i in (0,1,2):m.bias[i]=1
        m.embedding_rng.seed(19);expected=random.Random(19);counts={q:0 for q in m.ids[:3]}
        for _ in range(6000):
            m.observe(m.senses.frame(b''))
            self.assertEqual(m.pair,expected.choice(m.ids[:3]));counts[m.pair]+=1
        self.assertTrue(all(1800<n<2200 for n in counts.values()),counts)
        self.assertEqual(m.embedding_rng.getstate(),expected.getstate())

    def test_sample_uses_original_inverse_cdf_and_rng(self):
        m=self.model('sample');m.bias[0]=1;frame=m.senses.frame(b'abc');m.embedding_rng.seed(32)
        rng=random.Random(32);probabilities=m.probabilities(frame);seen=set()
        for _ in range(500):
            m.observe(frame);self.assertEqual(m.pair,sample(m.ids,probabilities,rng));seen.add(m.pair)
        self.assertGreater(len(seen),1)
        self.assertEqual(m.embedding_rng.getstate(),rng.getstate())

    def test_snapshot_and_config_defaults(self):
        m=self.model('sample');saved=m.to_data();saved.pop('embedding_selection')
        loaded=BitModel.from_data(saved);self.assertEqual(loaded.embedding_selection,'sample')
        frame=m.senses.frame(b'x')
        for _ in range(50):m.observe(frame);loaded.observe(frame);self.assertEqual(m.pair,loaded.pair)
        data=asdict(self.cfg);data.pop('embedding_selection')
        self.assertEqual(TrainingConfig.from_data(data).embedding_selection,'sample')
        self.assertEqual(TrainingConfig(model='senses').embedding_selection,'argmax')
        m.reset_episode();m.embedding_selection='argmax'
        self.assertEqual(BitModel.from_data(m.to_data()).embedding_selection,'argmax')
        with self.assertRaises(ValueError):self.model('invalid')

    def test_native_tie_rng_matches_python_choice_exactly(self):
        # Exercise the production native selector, including its RNG state, directly.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);source=path/'probe.cpp';library=path/'probe.so'
            source.write_text('#include "'+str(SOURCE)+'"\nextern "C" int choose(const double* v,int n,unsigned int* rng){return embedding_argmax(v,n,rng);}\n')
            subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-ffp-contract=off',str(source),'-o',str(library)],check=True,capture_output=True)
            fn=ctypes.CDLL(str(library)).choose
            fn.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_uint)];fn.restype=ctypes.c_int
            for n in (1,2,3,4,7,8,17):
                rng=random.Random(51);state=(ctypes.c_uint*625)(*rng.getstate()[1])
                logits=(ctypes.c_double*(n+1))(*([2.0]*n+[-1.0]))
                for _ in range(1000):
                    expected=0 if n==1 else rng.choice(list(range(n)))
                    self.assertEqual(fn(logits,n+1,state),expected)
                self.assertEqual(tuple(state),rng.getstate()[1])

    def test_native_python_metrics_and_all_learned_weights_both_modes(self):
        params=self.parent.inherited.to_parameters()
        params['bias']=[0.0]*len(self.parent.inherited.pair_ids)
        params['weights']=[[0.0]*len(self.parent.inherited.pair_ids) for _ in params['weights']]
        parent=replace(self.parent,inherited=SensoryMappings(self.parent.inherited.pair_ids,self.parent.inherited.senses,params))
        for mode in ('sample','argmax'):
            for env,starts,goals in [('original',[2,-7,10],[0,3,-4]),('lights-out-2x2',[1,3,7],[0,0,0])]:
                cfg=replace(self.cfg,embedding_selection=mode,environment=env,inherit_lifetime=True)
                rows,weights=evaluate_batch([parent],starts,starts,training_goals=goals,assessment_goals=goals,seeds=[19],environment=env,embedding_selection=mode,return_weights=True,workers=1)
                result,detail,learned=evaluate(parent,list(zip(starts,goals)),list(zip(starts,goals)),seed=19,config=cfg)
                assert_equivalent(rows[0],result)
                assert_equivalent(learned_mappings(parent,weights,0).to_data(),learned.inherited.to_data())
                self.assertEqual(detail['learned_model']['embedding_selection'],mode)

    def test_historical_checkpoint_resume_keeps_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=replace(self.cfg,embedding_selection='sample')
            expected=run_experiment(cfg,root/'full')
            run_experiment(cfg,root/'resume',until_generation=1)
            checkpoint=root/'resume/checkpoint.json.gz'
            with gzip.open(checkpoint,'rt') as f:state=json.load(f)
            del state['config']['embedding_selection']
            with gzip.open(checkpoint,'wt') as f:json.dump(state,f)
            oldcfg=TrainingConfig.from_data(state['config'])
            actual=run_experiment(oldcfg,root/'resume',resume=True)
            self.assertEqual(expected,actual)
            for f in (root/'resume/generations').glob('*.json'):
                self.assertEqual(json.loads(f.read_text())['summary']['embedding_selection'],'sample')
