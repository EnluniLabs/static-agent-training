"""Universal previews use isolated senses and the authenticated workspace route."""
import base64
import copy
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock

from static_agent_training.evolution import TrainingConfig, initialize_population
from static_agent_training.play import handler_for
from static_agent_training.workspace_models import WorkspaceModels


class StreamPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        species = initialize_population(TrainingConfig(model='senses', population=2, elites=1, initial_pairs=3))[0]
        self.model = species.inherited.model(species.repertoire)
        self.saved = self.model.to_data()
        self.workspace = WorkspaceModels(self.temp.name, Mock(), {})
        self.workspace.load = Mock(return_value=('senses', self.saved, None))
        self.request = dict(snapshot='selected', lifetime_id='stream-test-lifetime',
                            base64_data=base64.b64encode(bytes(range(64))).decode(), window_bytes=3)

    def test_frozen_current_lifetime_deterministic_and_saved_unchanged(self):
        self.model.weights[0][0] = 99
        self.workspace.lifetimes[self.request['lifetime_id']] = dict(snapshot='selected', model=self.model, attempts=7)
        before = copy.deepcopy(self.model.to_data())
        saved = copy.deepcopy(self.saved)
        first = self.workspace.stream_preview(self.request)
        self.assertEqual(first, self.workspace.stream_preview(self.request))
        self.assertEqual(self.model.to_data(), before)
        self.assertEqual(self.saved, saved)
        self.workspace.load.assert_not_called()
        self.assertEqual(first['origin'], 'Current playground lifetime')
        self.assertEqual(first['metadata']['windows'], 16)
        self.assertEqual(first['metadata']['bytes_consumed'], 48)
        self.assertEqual(first['metadata']['stop_reason'], 'frame_limit')
        self.assertTrue(any(e['value'] == 99 for e in first['graph']['edges'] if e['kind'] == 'embedding'))
        for frame in first['frames']:
            self.assertEqual(frame['input_window_size'], 3)
            self.assertIn('sensory', frame)
            self.assertNotIn('position', frame)
            self.assertNotIn('goal', frame)
            self.assertNotIn('expected', frame)

    def test_generic_bytes_empty_and_saved_fallback(self):
        for payload in (b'', b'\x00\xff', b'opaque encoded file bytes', bytes(range(256))):
            with self.subTest(length=len(payload)):
                request = dict(self.request, base64_data=base64.b64encode(payload).decode(), max_frames=1024)
                result = self.workspace.stream_preview(request)
                self.assertEqual(result['metadata']['bytes_consumed'], len(payload))
                self.assertEqual(result['metadata']['stop_reason'], 'end_of_stream')
                self.assertEqual(result['frame_count'], (len(payload)+2)//3)
                self.assertNotIn('base64_data', result)
        self.assertEqual(self.saved, self.model.to_data())

    def test_parallel_prediction_only_input_clock_and_feedback(self):
        from tests.test_bit_model import repertoire
        from static_agent_training.experimental.bit_model import BitModel
        from static_agent_training.experimental.senses import FixedSenses
        model = BitModel(repertoire(), senses=FixedSenses(), seed=71, sensory_partitions=3)
        loop = next(q for q, successor in model.repertoire.successors.items() if q == successor)
        model.bias[:] = [1000. if q == loop else -1000. for q in model.ids]
        for row in model.weights:
            row[:] = [0.] * len(row)
        for gates in model.gates.values():
            gates[:] = [-1000.] * len(gates)
            gates[model.spec.actions] = 1000.
        self.workspace.load.return_value = ('senses', model.to_data(), None)
        result = self.workspace.stream_preview(dict(self.request, max_frames=3))
        self.assertEqual(result['frame_count'], 3)
        self.assertEqual(result['bytes_consumed'], 9)
        for frame in result['frames']:
            self.assertEqual(len(frame['active_pairs']), 3)
            self.assertEqual(len(frame['predictions']), 3)
            self.assertEqual(frame['actions'], [])
            self.assertEqual({e['region'] for e in frame['traversals']}, {0, 1, 2})
        self.assertTrue(any(e['kind'] == 'FEEDBACK' for e in result['frames'][1]['events']))
        self.assertTrue(any(e['kind'] == 'EXPIRED' for e in result['frames'][-1]['events']))

    def test_reject_invalid_and_oversized(self):
        for extra in (dict(base64_data='%%%'), dict(base64_data='é'), dict(base64_data=None),
                      dict(base64_data=base64.b64encode(b'x'*(256*1024+1)).decode()),
                      dict(max_frames=0), dict(max_frames=True), dict(window_bytes=0),
                      dict(window_bytes=100000), dict(seed=True)):
            with self.subTest(extra=list(extra)):
                with self.assertRaises(ValueError):
                    self.workspace.stream_preview(dict(self.request, **extra))
        self.workspace.load.return_value = ('baseline', {}, None)
        with self.assertRaisesRegex(ValueError, 'fixed senses.*categorical'):
            self.workspace.stream_preview(self.request)

    def test_http_token_and_payload_above_standard_cap(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for({}, status_root=self.temp.name))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            server.RequestHandlerClass.workspace.models = self.workspace
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/api/bootstrap') as response:
                token = json.load(response)['token']
            payload = json.dumps(dict(self.request, base64_data=base64.b64encode(b'x'*20000).decode())).encode()
            for headers in ({}, {'X-Workspace-Token': token, 'Origin': 'https://unrelated.example'}):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base+'/api/stream-preview', data=payload, headers=headers))
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
            with urlopen(Request(base+'/api/stream-preview', data=payload,
                                 headers={'X-Workspace-Token': token, 'Content-Type': 'application/json'})) as response:
                result = json.load(response)
            self.assertEqual(result['metadata']['bytes'], 20000)
            self.assertEqual(result['source'], 'Frozen stream copy')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
