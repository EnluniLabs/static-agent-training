import base64
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode
from urllib.request import urlopen

from static_agent_training.play import handler_for
from static_agent_training.populations import list_populations
from static_agent_training.workspace import Workspace


class AgentZWorkspaceTests(unittest.TestCase):
    def wait(self, workspace, job):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            record = workspace.sessions.get(job['id'])
            if record['state'] not in ('running', 'stopping'):
                self.assertEqual(record['state'], 'completed', Path(job['log']).read_text())
                return
            time.sleep(.05)
        self.fail(Path(job['log']).read_text())

    def test_session_resume_snapshot_learning_stream_and_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))
            job = workspace.post('/api/sessions', dict(kind='agent-z-training', name='Z',
                config=dict(population=2, generations=1, episodes=3, speciation=True)))
            self.wait(workspace, job)
            run = job['run']
            self.assertEqual(workspace.runs()[0]['model_family'], 'agent-z')
            self.assertEqual(workspace.detail(run)['series'][0]['x'], 1)
            self.assertIn('--speciation', job['command'])
            self.assertEqual(workspace.detail(run)['series'][0]['species_sizes'], [2])
            checkpoint = workspace.artifact(run, 'checkpoint.json.gz')
            self.assertTrue(checkpoint.is_file())
            resumed = workspace.post('/api/resume-run', dict(run=run, generations=2))
            self.wait(workspace, resumed)
            self.assertEqual([point['x'] for point in workspace.detail(run)['series']], [1, 2])
            snapshot = workspace.models.snapshots(run, 'agent-z')[0]['id']
            storage = workspace.models.load(snapshot)[1]
            base = dict(snapshot=snapshot, lifetime_id='agent-z-workspace-test')
            original = workspace.post('/api/brain-scan', base)
            self.assertEqual(original['source'], 'Saved snapshot')
            self.assertTrue(any(e['kind'] == 'successor' for e in original['edges']))
            attempt = workspace.post('/api/attempt', dict(base, start=-2, goal=3,
                                                         seed=7, learning=True))
            self.assertEqual(attempt['learning']['attempts'], 1)
            self.assertGreater(attempt['learning']['empirical_updates'], 0)
            self.assertTrue(attempt['frames'][-1]['done'])
            active = workspace.post('/api/brain-scan', base)
            self.assertEqual(active['source'], 'Current playground lifetime')
            self.assertNotEqual(active['edges'], original['edges'])
            stream = workspace.post('/api/stream-preview', dict(
                base, base64_data=base64.b64encode(b'\x00\xff').decode(), max_frames=2))
            self.assertEqual([f['sensory'] for f in stream['frames']], ['00', 'ff'])
            self.assertTrue(all(e['observation'] is None for f in stream['frames'] for e in f['emissions']))
            self.assertEqual(workspace.models.load(snapshot)[1], storage)
            workspace.post('/api/reset-lifetime', base)
            self.assertEqual(workspace.post('/api/brain-scan', base), original)
            population = workspace.post('/api/save-population', dict(run=run, count=2))
            self.assertEqual(list_populations(workspace.root)[0]['model'], 'agent-z')
            assessment = workspace.post('/api/sessions', dict(kind='assessment', name='Frozen Z',
                config=dict(populations=[population['id']], episodes=4)))
            self.wait(workspace, assessment)
            report = workspace.detail(assessment['run'])['report']
            self.assertFalse(report['learning'])
            self.assertEqual(report['progress']['completed_animals'], 2)
            self.assertEqual(report['populations'][0]['model'], 'agent-z')
            self.assertEqual(report['populations'][0]['individuals'][0]['assessment']['empirical_updates'], 0)
            seeded = workspace.post('/api/sessions', dict(kind='agent-z-training', name='Seeded Z',
                config=dict(population=2, generations=1, episodes=3, initial_population=population['id'])))
            self.wait(workspace, seeded)
            self.assertIn('--initial-population', seeded['command'])

            with ThreadingHTTPServer(('127.0.0.1', 0), handler_for({}, status_root=workspace.root)) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base_url = f'http://127.0.0.1:{server.server_port}'
                    with urlopen(base_url + '/api/models?' + urlencode(dict(run=run, model='agent-z'))) as response:
                        self.assertEqual(json.load(response)[0]['id'], snapshot)
                    with urlopen(base_url + '/api/export?' + urlencode(dict(snapshot=snapshot))) as response:
                        self.assertEqual(json.load(response)['schema_version'], storage['schema_version'])
                    with urlopen(base_url + '/') as response:
                        self.assertIn(b'<option value="agent-z">Agent Z</option>', response.read())
                finally:
                    server.shutdown()
                    thread.join()



if __name__ == '__main__':
    unittest.main()
