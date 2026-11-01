import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from static_agent_training.play import handler_for
from static_agent_training.status import _option, collect_runs, render


class StatusTests(unittest.TestCase):
    def make_run(self, root):
        run = root/'runs'/'example'
        (run/'generations').mkdir(parents=True)
        (run/'config.json').write_text(json.dumps(dict(config=dict(population=64, generations=10))))
        for index, stamp in ((0, 1000), (1, 1020)):
            file = run/'generations'/f'{index:05d}.json'
            file.write_text(json.dumps(dict(summary=dict(generation=index, mean=.1, best=.3,
                                                         assessment_mean=.2, random_assessment=.15))))
            os.utime(file, (stamp, stamp))
        return run

    def test_running_stopped_completed_and_eta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            process = {str(run.resolve()): dict(pid=123, elapsed_seconds=45)}
            row = collect_runs(root, processes=process, now=1025)[0]
            self.assertEqual(row['state'], 'running')
            self.assertEqual(row['completed'], 2)
            self.assertEqual(row['remaining_seconds'], 155)
            self.assertEqual(row['progress'], .2)
            self.assertIn('random 15.0%', render([row]))
            self.assertEqual(collect_runs(root, processes={}, now=1025)[0]['state'], 'stopped')
            (run/'config.json').write_text(json.dumps(dict(config=dict(population=64,generations=2))))
            self.assertEqual(collect_runs(root, processes={}, now=1025)[0]['state'], 'completed')
            self.assertEqual(collect_runs(root, processes=process, now=1025)[0]['state'], 'finishing')

    def test_startup_and_incomplete_files_do_not_break_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=self.make_run(root)
            for record in (run/'generations').glob('*.json'):
                record.unlink()
            row=collect_runs(root, processes={str(run.resolve()):dict(pid=123,elapsed_seconds=1)})[0]
            self.assertEqual(row['completed'],0)
            self.assertIsNone(row['remaining_seconds'])
            self.assertIsNone(row['summary'])
            broken=root/'runs'/'broken'
            broken.mkdir()
            (broken/'config.json').write_text('{')
            self.assertEqual(len(collect_runs(root, processes={})),1)
            self.assertEqual(_option(['--output=somewhere'],'--output'),'somewhere')
            self.assertEqual(_option(['--resume','elsewhere'],'--resume'),'elsewhere')

    def test_browser_status_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.make_run(root)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_for({},status_root=root))
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/runs') as response:
                    page=response.read().decode()
                    self.assertIn('Training runs',page)
                    self.assertIn('setInterval(refresh,5000)',page)
                with urlopen(base+'/runs.json') as response:
                    rows=json.load(response)
                    self.assertEqual(rows[0]['name'],'example')
                    self.assertEqual(rows[0]['completed'],2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
