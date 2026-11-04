import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from static_agent_training.environment import NumberLine
from static_agent_training.genome import develop
from static_agent_training.initialize import I, K, pair_id
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.workspace_models import WorkspaceModels


class PlaygroundLearningTests(unittest.TestCase):
    def fixture(self, kind):
        world=NumberLine(goal_conditioned=True)
        senses=FixedSenses()
        animal=develop([(I,K)], [o.token for o in world.observation_tokens],
                       [o.token for o in world.output_tokens], budgets=20,
                       senses=senses if kind=='senses' else None)
        ids=tuple(animal.repertoire.pairs); entry=pair_id(I,K); source=animal.repertoire.successors[entry]
        zero=next(t for t,v in world.diagnostic_state()['move_decoding'].items() if v==0)
        if kind=='senses':
            model=BitModel(animal.repertoire,senses=senses)
            model.bias=[1000.0 if q==entry else -1000.0 for q in ids]
            model.weights=[[0.0]*len(ids) for _ in range(model.spec.features)]
            for q in ids:
                model.gates[q]=[-1000.0]*10
                model.gates[q][9 if q==source else int(zero.split('_')[1])]=1000.0
            snapshot=model.to_data()
        else:
            maps=animal.inherited.lifetime()
            for row in maps.embedding.values():
                for q in ids: row[q]=1000.0 if q==entry else -1000.0
            for q,row in maps.output.items():
                for token in row: row[token]=1000.0 if token==('expect_+6' if q==source else zero) else -1000.0
            if kind=='candidate':
                snapshot=dict(individual=animal.to_data(),learned_embedding=maps.embedding,learned_output=maps.output,
                              encoding_seed=0,varying_goals=True,training_success=0,assessment_success=0,generation=1)
            else:
                snapshot=dict(schema_version='crt-tabular-reference-v1',repertoire=animal.repertoire.to_data(),mappings=dict(observations=maps.observations,
                              outputs=maps.outputs,pair_ids=ids,embedding=maps.embedding,output=maps.output))
        return snapshot

    @staticmethod
    def parameters(model):
        if isinstance(model,BitModel):return model.parameters()
        return copy.deepcopy(dict(embedding=model.mappings.embedding,output=model.mappings.output))

    def test_both_models_persist_freeze_isolate_and_reset(self):
        for kind in ('senses','baseline','candidate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                workspace=WorkspaceModels(Path(tmp),None,{})
                snapshot=self.fixture(kind); original=copy.deepcopy(snapshot)
                task=dict(snapshot='fixture',start=5,goal=0,seed=42,policy='animal',world='original',
                          lifetime_id='playground-test-lifetime',learning=True)
                with patch.object(workspace,'load',return_value=(kind,snapshot,None)):
                    first=workspace.attempt(task)
                    session=workspace.lifetimes[task['lifetime_id']]; model=session['model']
                    after_first=self.parameters(model)
                    self.assertGreater(first['learning']['empirical_updates'],0)
                    second=workspace.attempt(task)
                    self.assertIs(workspace.lifetimes[task['lifetime_id']]['model'],model)
                    self.assertNotEqual(after_first,self.parameters(model))
                    self.assertEqual(second['learning']['attempts'],2)
                    self.assertGreater(second['learning']['structural_corrections'],0)
                    self.assertEqual(second['learning']['total_empirical_updates'],2*first['learning']['empirical_updates'])
                    learned=self.parameters(model)
                    frozen=workspace.attempt(dict(task,learning=False))
                    self.assertEqual(frozen['learning']['empirical_updates'],0)
                    self.assertEqual(frozen['learning']['structural_corrections'],0)
                    self.assertEqual(learned,self.parameters(model))
                    workspace.attempt(dict(task,policy='random'))
                    self.assertEqual(learned,self.parameters(model))
                    other=workspace.attempt(dict(task,lifetime_id='another-playground-lifetime'))
                    self.assertEqual(other['frames'],first['frames'])
                    self.assertEqual(after_first,self.parameters(workspace.lifetimes['another-playground-lifetime']['model']))
                    workspace.reset_lifetime(task)
                    replay=workspace.attempt(task)
                    self.assertEqual(first,replay)
                    self.assertEqual(snapshot,original)

    def test_invalid_learning_and_snapshot_switch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace=WorkspaceModels(Path(tmp),None,{})
            task=dict(snapshot='fixture',start=5,goal=0,seed=42,learning=True,lifetime_id='playground-lifetime')
            with patch.object(workspace,'load',return_value=('senses',self.fixture('senses'),None)):
                with self.assertRaises(ValueError):workspace.attempt(dict(task,learning='yes'))
                with self.assertRaises(ValueError):workspace.attempt(dict(task,lifetime_id='bad'))
                workspace.attempt(task)
                with self.assertRaises(ValueError):workspace.attempt(dict(task,snapshot='other'))
