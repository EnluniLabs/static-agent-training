import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import run_experiment
from static_agent_training.populations import save_population
from static_agent_training.workspace import Workspace


class WorkspaceDeletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/'runs/fixture'
        run_experiment(TrainingConfig(model='senses',population=2,elites=1,generations=1,
            episodes=1,assessment_episodes=1,backend='native',workers=1),self.run)
        self.population=self.root/'runs/populations/saved.json.gz'
        save_population(self.run,self.population,count=1)
        self.workspace=Workspace(self.root)

    def test_delete_population_keeps_source_and_can_recover_exact_bytes(self):
        before=self.population.read_bytes()
        result=self.workspace.post('/api/delete-population',dict(population='populations/saved.json.gz'))
        self.assertFalse(self.population.exists());self.assertTrue(self.run.exists())
        trash=self.root/result['trash'];manifest=json.loads((trash/'manifest.json').read_text())
        self.assertEqual((trash/manifest['moves'][0]['stored']).read_bytes(),before)
        self.assertTrue(manifest['moves'][0]['source'].startswith('runs/populations/'))

    def test_delete_session_keeps_exported_candidates_and_other_sessions(self):
        log=self.run.with_name('fixture.log');log.write_text('log')
        diagnostics=self.run.with_name('fixture-diagnostics');diagnostics.mkdir();(diagnostics/'crash.log').write_text('crash')
        registry=self.root/'runs/ui-sessions';registry.mkdir()
        record=registry/'old.json';record.write_text(json.dumps(dict(directory=str(self.run))))
        result=self.workspace.post('/api/delete-session',dict(run='fixture'))
        self.assertFalse(self.run.exists());self.assertFalse(log.exists());self.assertFalse(diagnostics.exists());self.assertFalse(record.exists())
        self.assertTrue(self.population.exists())
        self.assertFalse(any(r['run']=='fixture' for r in self.workspace.runs()))
        manifest=json.loads((self.root/result['trash']/'manifest.json').read_text())
        self.assertEqual(len(manifest['moves']),4)

    def test_paths_containers_and_symlinks_are_rejected(self):
        outside=self.root/'outside';outside.write_text('preserve')
        link=self.root/'runs/populations/link.json.gz';link.symlink_to(outside)
        for route,key,value in [('/api/delete-session','run','..'),('/api/delete-session','run','populations'),
            ('/api/delete-session','run','.'),('/api/delete-population','population','../outside'),
            ('/api/delete-population','population','populations/link.json.gz')]:
            with self.subTest(value=value),self.assertRaises(ValueError):self.workspace.post(route,{key:value})
        self.assertEqual(outside.read_text(),'preserve')
        self.assertTrue(self.population.exists())

    def test_live_session_and_its_candidate_reference_are_protected(self):
        cfg=json.loads((self.run/'config.json').read_text());cfg['config']['initial_population']=str(self.population)
        (self.run/'config.json').write_text(json.dumps(cfg))
        # Harmless process carrying an experiment command identity; no training runs.
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
            'static_agent_training.experiment','--output',str(self.run)])
        try:
            with self.assertRaises(ValueError):self.workspace.post('/api/delete-session',dict(run='fixture'))
            with self.assertRaises(ValueError):self.workspace.post('/api/delete-population',dict(population='populations/saved.json.gz'))
            self.assertTrue(self.run.exists());self.assertTrue(self.population.exists())
        finally:process.terminate();process.wait()

    def test_failed_move_restores_previously_moved_session(self):
        from unittest.mock import patch
        log=self.run.with_name('fixture.log');log.write_text('preserve')
        original=Path.rename
        def rename(path,destination):
            if path==log:raise OSError('injected move failure')
            return original(path,destination)
        with patch.object(Path,'rename',rename):
            with self.assertRaises(OSError):self.workspace.post('/api/delete-session',dict(run='fixture'))
        self.assertTrue((self.run/'checkpoint.json.gz').exists())
        self.assertEqual(log.read_text(),'preserve')

    def test_linked_diagnostic_prevents_any_moves(self):
        outside=self.root/'outside';outside.mkdir()
        self.run.with_name('fixture-diagnostics').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):self.workspace.post('/api/delete-session',dict(run='fixture'))
        self.assertTrue(self.run.exists());self.assertTrue(outside.exists())
