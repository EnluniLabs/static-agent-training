"""Generic signals stay fixed through training, streams and saved imports."""
from dataclasses import replace
from pathlib import Path
import tempfile

from tests import test_fixed_apparatus_training as baseline
from static_agent_training.evolution import initialize_population
from static_agent_training.experiment import run_experiment, load_checkpoint
from static_agent_training.experimental.bit_model import BitModel
from static_agent_training.experimental.stream_input import ReceptorStream
from static_agent_training.populations import validate_compatibility, apparatus_arguments
from static_agent_training.brain_scan import mapping_graph


class SignalApparatusTrainingTests(baseline.FixedApparatusTrainingTests):
    def config(self):
        return replace(super().config(), sensory_interface='signals')

    def test_stream_snapshot_and_import_keep_same_transducer(self):
        config = self.config()
        species = initialize_population(config)[0]
        model = species.inherited.model(species.repertoire)
        restored = BitModel.from_data(model.to_data())
        self.assertEqual(restored.senses, model.senses)
        self.assertEqual(restored.senses.interface, 'signals')
        payload = bytes(range(33))
        stream = ReceptorStream(restored.senses)
        frames = list(stream.iter_frames([payload[:3], payload[3:]]))
        self.assertEqual(stream.window_bytes, 16)
        self.assertEqual(frames, [restored.senses.frame(payload[i:i+16]) for i in range(0,33,16)])
        geometry = restored.senses.partition_indices()
        self.assertEqual(geometry, model.senses.partition_indices())
        source = dict(config=apparatus_arguments(config))
        source['config'].update(model='senses', varying_goals=config.varying_goals, encoding_seed=config.encoding_seed)
        validate_compatibility(source, config)
        for different in (replace(config,sensory_partitions=2), replace(config,retina_width=8,retina_height=32), replace(config,sensory_interface='retina')):
            with self.assertRaises(ValueError):
                validate_compatibility(source, different)
        graph = mapping_graph(model.repertoire,parameters=model.parameters(),senses=model.senses)
        labels = [node['label'] for node in graph['nodes'] if node['kind']=='input']
        self.assertFalse(any('Pixel' in label for label in labels))
        self.assertEqual(graph['sensory_geometry']['interface'],'signals')
