from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from static_agent_training.candidates import CandidateStore, best_saved, load_saved_candidate
from static_agent_training.evolution import TrainingConfig
from static_agent_training.experiment import run_experiment, write_json
from static_agent_training.play import attempt, handler_for
from static_agent_training.run_records import read_summary


class CandidateTests(unittest.TestCase):
    def make_runs(self, root):
        config=TrainingConfig(population=3,generations=2,episodes=3,assessment_episodes=3,
                              elites=1,initial_pairs=2,chain_budget=8,seed=19)
        a,b=root/'runs'/'a',root/'runs'/'b'
        run_experiment(config,a)
        run_experiment(replace(config,seed=20),b)
        initial=load_saved_candidate(a)
        initial['run_id']='a'
        return a,b,initial

    def test_best_selection_snapshot_pinning_and_unpublished_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a,b,initial=self.make_runs(root)
            store=CandidateStore(root,initial)
            self.assertEqual({r['run_id'] for r in store.list()['runs']},{'a','b'})
            selected=store.info('a')
            self.assertEqual(selected['training_success'],best_saved(a)['best'])
            frozen=store.load(selected['candidate_id'])
            before=json.dumps(frozen,sort_keys=True)
            # A new record can precede its atomically published champion trace.
            source=best_saved(a)['generation']
            record=json.loads((a/'generations'/f'{source:05d}.json').read_text())
            record['summary']['generation']=2
            record['summary']['best']=.99
            write_json(a/'generations/00002.json',record)
            self.assertEqual(store.info('a')['candidate_id'],selected['candidate_id'])
            with gzip.open(a/'traces'/f'{source:05d}-champion.json.gz','rt') as f:
                trace=json.load(f)
            trace['generation']=2
            trace['result']['training']['fitness']=.99
            write_json(a/'traces/00002-champion.json.gz',trace)
            newer=store.info('a')
            self.assertEqual(newer['generation'],3)
            self.assertEqual(newer['training_success'],.99)
            self.assertNotEqual(newer['candidate_id'],selected['candidate_id'])
            self.assertEqual(json.dumps(store.load(selected['candidate_id']),sort_keys=True),before)
            for invalid in ('../b','/tmp','a/../b'):
                with self.assertRaises(ValueError):
                    store.info(invalid)
            with self.assertRaises(ValueError):
                store.load('a:0')
            # Selection / attempts don't rewrite the exported demo or checkpoint.
            self.assertFalse((a/'demo'/'champion.json').exists())

    def test_run_selection_api_uses_requested_candidate_without_global_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a,b,initial=self.make_runs(root)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(initial,status_root=root))
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/candidates') as response:
                    listing=json.load(response)
                self.assertEqual(listing['default_run'],'a')
                with urlopen(base+'/info?run=a') as response:
                    info_a=json.load(response)
                with urlopen(base+'/info?run=b') as response:
                    info_b=json.load(response)
                self.assertNotEqual(info_a['candidate_id'],info_b['candidate_id'])
                for info,path in ((info_a,a),(info_b,b),(info_a,a)):
                    task=dict(start=12,goal=-3,seed=42,candidate_id=info['candidate_id'])
                    request=Request(base+'/attempt',data=json.dumps(task).encode(),
                                    headers={'Content-Type':'application/json'})
                    with urlopen(request) as response:
                        result=json.load(response)
                    expected=attempt(load_saved_candidate(path),12,-3,42)
                    expected.update(candidate_id=info['candidate_id'],generation=info['generation'])
                    self.assertEqual(result,expected)
                with self.assertRaises(HTTPError) as error:
                    urlopen(base+'/info?run=unknown')
                self.assertEqual(error.exception.code,400)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_large_record_summary_is_read_without_decoding_episode_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'record.json'
            summary=dict(generation=0,best=.5)
            file.write_text(json.dumps(dict(individuals=['x'*1000000],summary=summary),separators=(',',':')))
            self.assertEqual(read_summary(file),summary)
            summary['best']=.7
            file.write_text(json.dumps(dict(summary=summary),indent=2))
            self.assertEqual(read_summary(file),summary)
