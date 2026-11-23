import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from static_agent_training.agent_z_speciation import Niches
from static_agent_training.experimental.agent_z import Structure
from static_agent_training.experimental.agent_z_experiment import run, read_z_artifact
from static_agent_training.initialize import I, K


class AgentZSpeciationTests(unittest.TestCase):
    def test_exact_repeat_and_parent_skip_representative_comparisons(self):
        first, _ = Structure.develop([(I, I)])
        second, _ = Structure.develop([(K, K)], max_transitions=1)
        copied = Structure(dict(first.pairs), dict(first.successors))
        niches = Niches(32)
        niches.seed([first])
        self.assertEqual(niches.classify(second, parent=(first, 0)), 1)
        self.assertEqual(niches.classify(first), 0)
        with patch('static_agent_training.agent_z_speciation.distance',
                   side_effect=AssertionError('reclassified unchanged structure')):
            self.assertEqual(niches.classify(copied), 0)
            self.assertEqual(niches.classify(copied, parent=(first, 0)), 0)

    def test_checkpoint_resume_matches_uninterrupted_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(speciation=True, population=12, episodes=5, learning='off')
            full = run(**settings, generations=4, directory=root / 'full')
            run(**settings, generations=2, directory=root / 'split')
            resumed = run(**settings, generations=4, directory=root / 'split', resume=True)
            self.assertEqual(full['champion'], resumed['champion'])
            for a, b in zip(full['history'], resumed['history']):
                self.assertEqual(a | {'development_stops': list(a['development_stops'])},
                                 b | {'development_stops': list(b['development_stops'])})
                self.assertEqual(sum(a['species_sizes']), 12)
                self.assertGreaterEqual(a['species_count'], 1)
            full_checkpoint = read_z_artifact(root / 'full/checkpoint.json.gz')
            split_checkpoint = read_z_artifact(root / 'split/checkpoint.json.gz')
            self.assertEqual(full_checkpoint, split_checkpoint)
