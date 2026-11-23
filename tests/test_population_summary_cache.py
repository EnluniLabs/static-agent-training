import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import threading

from static_agent_training.populations import list_populations, _population_summary
from static_agent_training.experiment import write_json
from static_agent_training.genome import develop
from static_agent_training.initialize import I


class PopulationSummaryCacheTests(unittest.TestCase):
    def test_cache_only_metadata_invalidates_on_replace_add_delete(self):
        from static_agent_training import populations
        species=develop([(I,I)],['x'],['y'],budgets=0)
        payload=dict(schema_version='crt-population-v1',name='first',config={},
                     source_type='fixture',members=[dict(individual=species.to_data())])
        _population_summary.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'runs/populations/one.json.gz'
            write_json(path,payload)
            with patch.object(populations,'read_population',wraps=populations.read_population) as read:
                first=list_populations(root)
                for _ in range(3):self.assertEqual(first,list_populations(root))
                self.assertEqual(read.call_count,1)
                first[0]['name']='client mutation'
                self.assertEqual(list_populations(root)[0]['name'],'first')
                write_json(path,dict(payload,name='other'))
                self.assertEqual(list_populations(root)[0]['name'],'other')
                self.assertEqual(read.call_count,2)
                write_json(path.with_name('two.json.gz'),payload)
                self.assertEqual(len(list_populations(root)),2)
                path.unlink()
                self.assertEqual(len(list_populations(root)),1)
                self.assertEqual(read.call_count,3)

    def test_concurrent_polls_share_cold_read(self):
        from static_agent_training import populations
        species=develop([(I,I)],['x'],['y'],budgets=0)
        payload=dict(schema_version='crt-population-v1',name='first',config={},
                     source_type='fixture',members=[dict(individual=species.to_data())])
        _population_summary.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'runs/populations/one.json.gz',payload)
            with patch.object(populations,'read_population',wraps=populations.read_population) as read:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    rows=list(pool.map(lambda _:list_populations(root),range(4)))
                self.assertTrue(all(row==rows[0] for row in rows))
                self.assertEqual(read.call_count,1)

    def test_snapshot_discovery_does_not_load_full_champions(self):
        from static_agent_training.workspace_models import WorkspaceModels
        from static_agent_training.workspace_sessions import SessionManager
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'runs/example'
            write_json(directory/'config.json',dict(config=dict(model='senses',agent_y=True,
                receptors=16,sensory_interface='signals',retina_width=4,retina_height=4,
                sensory_partitions=1)))
            write_json(directory/'champions.json',dict(champions=[dict(generation=i,
                training_success=.5,assessment_success=.4) for i in range(10)]))
            workspace=WorkspaceModels(root,SessionManager(root),{})
            with patch.object(workspace,'load',side_effect=AssertionError('full model loaded')):
                options=workspace.snapshots('example','senses')
            self.assertEqual(len(options),10)
            self.assertTrue(all(o['agent_y'] and o['sensory_geometry']['resolution']==16 for o in options))
            self.assertFalse(workspace.candidates.cache)
