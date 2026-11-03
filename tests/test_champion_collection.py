import json
from pathlib import Path
import tempfile
import unittest

from static_agent_training.experiment import update_champion_collection, write_json


class ChampionCollectionTests(unittest.TestCase):
    def test_training_selection_deduplication_ties_and_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for generation, (identity, training, heldout) in enumerate([
                    ('a', .6, .1), ('b', .5, 1), ('a', .7, 0), ('c', .7, .2)]):
                record = dict(summary=dict(generation=generation), champion_index=0,
                              individuals=[dict(genotype_id=identity,
                                                training=dict(fitness=training),
                                                assessment=dict(fitness=heldout))])
                write_json(root / 'generations' / f'{generation:05d}.json', record)
                write_json(root / 'traces' / f'{generation:05d}-champion.json.gz', {})
                result = update_champion_collection(root, 2, record)
            self.assertEqual([r['genotype_id'] for r in result['champions']], ['a', 'c'])
            self.assertEqual([r['generation'] for r in result['champions']], [2, 3])
            self.assertEqual(result, update_champion_collection(root, 2))
            self.assertEqual(result, json.loads((root / 'champions.json').read_text()))
            write_json(root / 'generations' / '00003.json', dict(summary=dict(generation=3)))
            write_json(root / 'traces' / '00003-champion.json.gz', dict(result=record['individuals'][0]))
            rebuilt = update_champion_collection(root, 2)
            self.assertEqual([r['genotype_id'] for r in rebuilt['champions']], ['a', 'c'])
            (root / 'generations' / '00004.json').write_text('')
            self.assertEqual(rebuilt, update_champion_collection(root, 2))
            for count in (0, -1, True):
                with self.assertRaises(ValueError):
                    update_champion_collection(root, count)
