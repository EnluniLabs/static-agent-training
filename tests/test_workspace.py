import gzip
import json
from pathlib import Path
import signal
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from static_agent_training.experimental.model_comparison import run_study
from static_agent_training.play import handler_for
from static_agent_training.workspace_sessions import SessionManager, process_identity
from static_agent_training.experimental.bit_model import BitModel, run_episode as sensory_episode
from static_agent_training.experimental.number_line import StimulusNumberLine
from static_agent_training.experimental.senses import SensoryAdapter
from static_agent_training.seeds import derive


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for({}, status_root=self.root))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.token = self.get('/api/bootstrap')['token']

    def tearDown(self):
        for path in (self.root / 'runs/ui-sessions').glob('*.json'):
            record = json.loads(path.read_text())
            if record.get('process_identity') and process_identity(record['pid']) == record['process_identity']:
                try:
                    os.killpg(record['pid'], signal.SIGINT)
                except ProcessLookupError:
                    pass
        for process in self.server.RequestHandlerClass.workspace.sessions.children.values():
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def get(self, path):
        with urlopen(self.base + path, timeout=20) as response:
            return json.load(response)

    def post(self, path, data, *, token=True, origin=None):
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['X-Workspace-Token'] = self.token
        if origin:
            headers['Origin'] = origin
        request = Request(self.base + path, data=json.dumps(data).encode(), headers=headers)
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    def wait_for(self, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(.1)
        self.fail('Timed out waiting for temporary workspace session')

    def test_workspace_can_open_empty_and_controls_require_same_origin_token(self):
        with urlopen(self.base) as response:
            self.assertIn('Animal playground', response.read().decode())
        self.assertEqual(self.get('/api/runs'), [])
        config = dict(kind='training', config=dict(population=2, generations=1, episodes=1, assessment_episodes=1, backend='python'))
        for options in (dict(token=False), dict(origin='https://unrelated.example')):
            with self.assertRaises(HTTPError) as error:
                self.post('/api/sessions', config, **options)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
        self.assertFalse((self.root / 'runs').exists())
        for path in ('/api/run?run=../', '/api/models?run=../../&model=senses', '/api/log?run=/tmp'):
            with self.assertRaises(HTTPError) as error:
                self.get(path)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_launch_track_extend_and_download_baseline_session(self):
        job = self.post('/api/sessions', dict(kind='training', name='temporary unit session',
                        config=dict(population=2, generations=1, episodes=1, assessment_episodes=1, backend='python')))
        path = Path(job['directory'])
        self.wait_for(lambda: (path / 'generations/00000.json').is_file())
        self.wait_for(lambda: next((r for r in self.get('/api/runs') if r['run'] == job['run'] and r['state'] == 'completed'), None))
        detail = self.get('/api/run?' + urlencode(dict(run=job['run'])))
        self.assertEqual(len(detail['series']), 1)
        self.assertEqual(detail['kind'], 'training')
        snapshots = self.get('/api/models?' + urlencode(dict(run=job['run'], model='baseline')))
        self.assertTrue(snapshots)
        task = dict(snapshot=snapshots[0]['id'], start=12, goal=-3, seed=42)
        self.assertEqual(self.post('/api/attempt', task), self.post('/api/attempt', task))
        with urlopen(self.base + '/api/artifact?' + urlencode(dict(run=job['run'], file='config.json'))) as response:
            self.assertIn('attachment', response.headers['Content-Disposition'])
        resumed = self.post('/api/resume-run', dict(run=job['run'], generations=2))
        self.assertEqual(resumed['previous'], job['id'])
        self.wait_for(lambda: (path / 'generations/00001.json').is_file())
        self.wait_for(lambda: next((r for r in self.get('/api/runs') if r['run'] == job['run'] and r['state'] == 'completed'), None))
        self.assertEqual(len(list((path / 'generations').glob('*.json'))), 2)

    def test_comparison_charts_and_both_frozen_snapshot_formats(self):
        directory = self.root / 'runs/interfaces/fixture'
        run_study(directory, dict(population=1, seeds=[48], episodes=2, assessment_episodes=3, receptors=256))
        detail = self.get('/api/run?' + urlencode(dict(run='interfaces/fixture')))
        self.assertEqual(detail['kind'], 'interfaces')
        self.assertEqual(len(detail['series']), 1)
        self.assertIsNotNone(detail['series'][0]['random'])
        for model in ('baseline', 'senses'):
            snapshots = self.get('/api/models?' + urlencode(dict(run='interfaces/fixture', model=model)))
            self.assertEqual(len(snapshots), 2)
            task = dict(snapshot=snapshots[0]['id'], start=12, goal=-3, seed=42)
            before = {path: path.read_bytes() for path in directory.rglob('*.json.gz')}
            result = self.post('/api/attempt', task)
            self.assertEqual(result, self.post('/api/attempt', task))
            self.assertTrue(result['frames'][-1]['done'])
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            if model == 'senses':
                self.assertEqual(len(result['frames'][0]['sensory']) * 4, 256)
                result = self.post('/api/attempt', dict(task, world='expanded', start=25, goal=-25))
                self.assertEqual((result['lower'], result['upper']), (-30, 30))
                with urlopen(self.base + '/api/export?' + urlencode(dict(snapshot=snapshots[0]['id']))) as response:
                    saved = json.loads(gzip.decompress(response.read()))
                    self.assertEqual(saved['schema_version'], 'crt-bit-model-v1')
                reference = BitModel.from_data(saved)
                reference.embedding_rng.seed(derive(42, 'interactive', 'embedding'))
                reference.output_rng.seed(derive(42, 'interactive', 'output'))
                raw = StimulusNumberLine(lower=-30, upper=30)
                final = sensory_episode(reference, SensoryAdapter(raw, reference.senses), (25, -25), learn=False)[-1]
                self.assertEqual((final.done, final.success, final.reason, raw.position, raw.actions),
                                 (result['frames'][-1]['done'], result['frames'][-1]['success'],
                                  result['frames'][-1]['reason'], result['frames'][-1]['position'],
                                  result['frames'][-1]['actions']))

    def test_adopt_external_baseline_and_reject_completed_target(self):
        from static_agent_training.evolution import TrainingConfig
        from static_agent_training.experiment import run_experiment
        directory = self.root / 'runs/external'
        run_experiment(TrainingConfig(population=2, generations=1, episodes=1, assessment_episodes=1,
                                      initial_pairs=1, chain_budget=4, elites=1), directory)
        with self.assertRaises(HTTPError) as error:
            self.post('/api/resume-run', dict(run='external', generations=1))
        self.assertEqual(error.exception.code, 400)
        error.exception.close()
        job = self.post('/api/resume-run', dict(run='external', generations=2))
        self.assertEqual(job['previous'], 'external')
        self.wait_for(lambda: (directory / 'generations/00001.json').is_file())

    def test_stop_preserves_checkpoint_and_session_validation_is_strict(self):
        manager = SessionManager(self.root)
        for config in [dict(population=True), dict(seed=-1), dict(backend='shell'), dict(unknown='value'), dict(notes=['invalid'])]:
            with self.assertRaises(ValueError):
                manager.launch(dict(kind='training', config=config))
        for kind, config in [('interfaces', dict(seeds=[[1]])), ('senses', dict(receptors=17)),
                             ('senses', dict(candidate='../outside'))]:
            with self.assertRaises(ValueError):
                manager.launch(dict(kind=kind, config=config))
        self.assertFalse((self.root / 'runs').exists())
        job = self.post('/api/sessions', dict(kind='training', config=dict(population=2, generations=100000,
                        episodes=1, assessment_episodes=1, backend='python')))
        path = Path(job['directory'])
        self.wait_for(lambda: (path / 'config.json').is_file())
        stopped = self.post('/api/stop', dict(id=job['id']))
        self.assertEqual(stopped['state'], 'stopping')
        self.wait_for(lambda: process_identity(job['pid']) is None)
        self.assertTrue((path / 'checkpoint.json.gz').is_file())
        with self.assertRaises(HTTPError) as error:
            self.post('/api/stop', dict(id='not-owned'))
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_failed_session_registration_does_not_leave_untracked_child(self):
        manager = SessionManager(self.root)
        with patch('static_agent_training.workspace_sessions.subprocess.Popen') as popen, \
             patch('static_agent_training.workspace_sessions.write_json', side_effect=OSError('fixture disk failure')), \
             patch('static_agent_training.workspace_sessions.os.killpg') as kill:
            process = popen.return_value
            process.pid = 999999999
            process.poll.return_value = None
            with self.assertRaises(OSError):
                manager.launch(dict(kind='training', config={}))
            kill.assert_called_once_with(process.pid, signal.SIGINT)
            process.wait.assert_called_once()
            self.assertFalse(manager.children)

    def test_current_model_family_metadata_keeps_legacy_runs_readable(self):
        from dataclasses import asdict
        from static_agent_training.evolution import TrainingConfig
        from static_agent_training.experiment import write_json
        for name, settings in [('categorical', {}),
                ('old-fixed', dict(model='senses')),
                ('agent-x', dict(model='senses',sensory_interface='signals',sensory_partitions=4)),
                ('agent-y', dict(model='senses',agent_y=True))]:
            config=TrainingConfig(**settings)
            write_json(self.root/'runs'/name/'config.json',dict(config=asdict(config)))
        self.server.RequestHandlerClass.workspace.invalidate()
        rows={r['run']:r for r in self.get('/api/runs')}
        self.assertNotIn('model_family',rows['categorical'])
        self.assertNotIn('model_family',rows['old-fixed'])
        self.assertEqual(rows['agent-x']['model_family'],'agent-x')
        self.assertEqual(rows['agent-y']['model_family'],'agent-y')

    def test_agent_y_session_resume_and_playback(self):
        job = self.post('/api/sessions', dict(kind='sensory-training', name='Agent Y fixture',
            config=dict(agent_y=True, backend='python', population=2, generations=1,
                episodes=1, assessment_episodes=1, receptors=16, sensory_partitions=2,
                initial_pairs=2, chain_budget=8, inherit_lifetime=True)))
        path = Path(job['directory'])
        self.wait_for(lambda: (path/'generations/00000.json').is_file())
        self.wait_for(lambda: next((r for r in self.get('/api/runs')
            if r['run']==job['run'] and r['state']=='completed'), None))
        detail = self.get('/api/run?' + urlencode(dict(run=job['run'])))
        self.assertTrue(detail['config']['agent_y'])
        snapshots = self.get('/api/models?' + urlencode(dict(run=job['run'], model='senses')))
        self.assertTrue(snapshots[0]['agent_y'])
        played = self.post('/api/attempt', dict(snapshot=snapshots[0]['id'], start=12, goal=-3, seed=42))
        self.assertEqual(played['model'], 'senses')
        self.post('/api/resume-run', dict(run=job['run'], generations=2, backend='native'))
        self.wait_for(lambda: (path/'generations/00001.json').is_file())
        self.assertEqual(self.get('/api/run?' + urlencode(dict(run=job['run'])))['config']['backend'], 'native')

    def test_sensory_generation_sessions_checkpoint_resume_and_playback(self):
        job = self.post('/api/sessions', dict(kind='sensory-training', name='sensory evolution fixture',
                         config=dict(population=2, generations=2, episodes=1, assessment_episodes=1, receptors=256)))
        path = Path(job['directory'])
        self.wait_for(lambda: (path/'generations/00001.json').is_file())
        self.wait_for(lambda: next((r for r in self.get('/api/runs') if r['run'] == job['run'] and r['state'] == 'completed'), None))
        detail = self.get('/api/run?' + urlencode(dict(run=job['run'])))
        self.assertEqual(detail['kind'], 'sensory-training')
        self.assertEqual(len(detail['series']), 2)
        self.assertEqual(detail['config']['model'], 'senses')
        snapshots = self.get('/api/models?' + urlencode(dict(run=job['run'], model='senses')))
        self.assertTrue(snapshots)
        played = self.post('/api/attempt', dict(snapshot=snapshots[0]['id'], start=12, goal=-3, seed=42))
        self.assertEqual(played['model'], 'senses')
        self.assertEqual(len(played['frames'][0]['sensory']) * 4, 256)
        resumed = self.post('/api/resume-run', dict(run=job['run'], generations=3))
        self.wait_for(lambda: (path/'generations/00002.json').is_file())
        self.assertEqual(resumed['kind'], 'sensory-training')

    def test_evolutionary_model_comparison_session_uses_generations(self):
        job = self.post('/api/sessions', dict(kind='model-comparison', name='evolutionary comparison fixture',
                         config=dict(population=2, generations=2, episodes=1, assessment_episodes=1,
                                     benchmark_episodes=2, seeds=[48], backend='python', receptors=256)))
        path = Path(job['directory'])
        def complete():
            report = path/'report.json'
            return report.is_file() and json.loads(report.read_text()).get('status') == 'completed'
        self.wait_for(complete)
        detail = self.get('/api/run?' + urlencode(dict(run=job['run'])))
        self.assertEqual(detail['kind'], 'model-comparison')
        self.assertTrue(detail['results'][0]['initialization_verified'])
        self.assertEqual(detail['results'][0]['common_generations'], 2)
        self.assertEqual([arm['completed'] for arm in detail['report']['arms']], [2, 2])
