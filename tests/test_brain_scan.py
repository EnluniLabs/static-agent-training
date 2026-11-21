"""Graph inspection must preserve state and expose real mapping values."""
import copy
import tempfile
import unittest
from unittest.mock import Mock

from static_agent_training.brain_scan import mapping_graph
from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.experimental.bit_model import BitModel, sigmoid
from static_agent_training.mappings import softmax
from static_agent_training.workspace_models import WorkspaceModels


class BrainScanTests(unittest.TestCase):
    def test_graph_values_and_no_mutation(self):
        for kind in ('categorical', 'senses'):
            with self.subTest(model=kind):
                cfg = TrainingConfig(model=kind, population=2, elites=1, initial_pairs=2)
                species = initialize_population(cfg)[0]
                if kind == 'senses':
                    model = species.inherited.model(species.repertoire)
                    before = model.parameters()
                    rng = (model.embedding_rng.getstate(), model.output_rng.getstate())
                    graph = mapping_graph(model.repertoire, parameters=model.parameters())
                    for e in graph['edges']:
                        if e['kind'] == 'embedding':
                            i = int(e['source'].split(':')[1])
                            self.assertEqual(e['value'], model.weights[i][model.ids.index(e['target'])])
                        elif e['kind'] == 'prediction':
                            _, c, i = e['target'].split(':')
                            self.assertEqual(e['value'], sigmoid(model.prediction[e['source']][int(c)][int(i)]))
                    self.assertEqual(before, model.parameters())
                    self.assertEqual(rng, (model.embedding_rng.getstate(), model.output_rng.getstate()))
                else:
                    maps = species.inherited.lifetime()
                    before = copy.deepcopy(maps.embedding)
                    graph = mapping_graph(species.repertoire, mappings=maps)
                    for e in graph['edges']:
                        if e['kind'] == 'embedding':
                            row = maps.embedding[e['source'][6:]]
                            self.assertEqual(e['value'], dict(zip(row, softmax(row.values())))[e['target']])
                    self.assertEqual(before, maps.embedding)
                ids = {n['id'] for n in graph['nodes']}
                self.assertTrue(all(e['source'] in ids and e['target'] in ids for e in graph['edges']))
                self.assertEqual({(e['source'], e['target']) for e in graph['edges'] if e['kind']=='successor'}, set(species.repertoire.successors.items()))

    def test_current_lifetime_and_saved_fallback(self):
        species = initialize_population(TrainingConfig(model='senses', population=2, elites=1, initial_pairs=2))[0]
        model = species.inherited.model(species.repertoire)
        saved = model.to_data()
        with tempfile.TemporaryDirectory() as root:
            workspace = WorkspaceModels(root, Mock(), {})
            workspace.load = Mock(return_value=('senses', saved, None))
            request = dict(snapshot='test', lifetime_id='brain-test-lifetime')
            graph = workspace.brain_scan(request)
            self.assertEqual(graph['source'], 'Saved snapshot')
            model.weights[0][0] = -99
            workspace.lifetimes[request['lifetime_id']] = dict(snapshot='test', model=model, attempts=3)
            graph = workspace.brain_scan(request)
            self.assertEqual(graph['attempts'], 3)
            self.assertTrue(any(e['value']==-99 for e in graph['edges'] if e['kind']=='embedding'))
            workspace.reset_lifetime(request)
            self.assertEqual(workspace.brain_scan(request)['source'], 'Saved snapshot')
