import io
import itertools
import tracemalloc
import unittest

from static_agent_training.experimental.bit_model import BitModel, run_episode
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.experimental.signals import Action, NoOp, Prediction, Step
from static_agent_training.experimental.stream_input import (
    ReceptorStream, StreamEnvironment, iter_stimuli,
)
from tests.test_bit_model import repertoire


class StreamInputTests(unittest.TestCase):
    def test_fragmentation_and_binary_sources(self):
        data = bytes(range(33)) + b'\0\0'
        expected = [data[i:i + 4] for i in range(0, len(data), 4)]
        for source in (data, bytearray(data), memoryview(data), io.BytesIO(data)):
            self.assertEqual(list(iter_stimuli(source, 4)), expected)
        for size in range(1, len(data) + 1):
            chunks = [data[i:i + size] for i in range(0, len(data), size)]
            self.assertEqual(list(iter_stimuli(chunks, 4)), expected)
        self.assertEqual(list(iter_stimuli([b'', bytearray(b'a'), memoryview(b'bc'), b''], 2)),
                         [b'ab', b'c'])
        for source in (b'', [], [b'', b''], io.BytesIO()):
            self.assertEqual(list(iter_stimuli(source, 4)), [])

    def test_invalid_sources_chunks_and_dimensions(self):
        for source in ('text', 3, True, None, ['text'], [1], [None],
                       io.StringIO('text'), memoryview(b'abcd')[::2]):
            with self.subTest(source=source), self.assertRaises(TypeError):
                list(iter_stimuli(source, 2))
        for size in (0, -1, True, 1.5, '2', None):
            with self.assertRaises(ValueError):
                iter_stimuli(b'a', size)
        with self.assertRaises(ValueError):
            ReceptorStream(FixedSenses(), 17)
        for size in (0, True):
            with self.assertRaises(ValueError):
                ReceptorStream(FixedSenses(), size)
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                StreamEnvironment(lambda: b'a', FixedSenses(), frame_limit=limit)
        with self.assertRaises(TypeError):
            StreamEnvironment(b'a', FixedSenses())
        with self.assertRaises(TypeError):
            StreamEnvironment(lambda: b'a', FixedSenses(), action_sink=1)

    def test_read_sizes_short_reads_and_limit_no_read_ahead(self):
        class Reader(io.BytesIO):
            def __init__(self):
                super().__init__(b'abcdefghi')
                self.requests = []
            def read(self, n):
                self.requests.append(n)
                return super().read(min(n, 2))
        reader = Reader()
        world = StreamEnvironment(lambda: reader, FixedSenses(), 4, frame_limit=1)
        self.assertFalse(world.reset().done)
        self.assertEqual(reader.requests, [4, 2])
        self.assertEqual(reader.tell(), 4)
        self.assertEqual(world.step(NoOp()), Step(None, done=True, reason='frame_limit'))
        self.assertEqual(reader.tell(), 4)
        world.close()
        self.assertFalse(reader.closed)
        reader.seek(0)
        self.assertEqual(list(iter_stimuli(reader, 4)), [b'abcd', b'efgh', b'i'])
        class BadReader:
            def read(self, n):
                return b'x' * (n + 1)
        with self.assertRaises(ValueError):
            list(iter_stimuli(BadReader(), 2))

    def test_infinite_laziness_and_bounded_extra_memory(self):
        pulls = []
        closed = []
        def source():
            try:
                while True:
                    pulls.append(1)
                    yield b'abc'
            finally:
                closed.append(True)
        windows = iter_stimuli(source(), 2)
        self.assertEqual(pulls, [])
        self.assertEqual(next(windows), b'ab')
        self.assertEqual(len(pulls), 1)
        self.assertEqual(next(windows), b'ca')
        self.assertEqual(len(pulls), 2)
        windows.close()
        self.assertEqual(closed, [True])
        large = b'x' * 4_000_000
        tracemalloc.start()
        try:
            windows = iter_stimuli(iter((large,)), 16)
            self.assertEqual(list(itertools.islice(windows, 100)), [b'x' * 16] * 100)
            self.assertLess(tracemalloc.get_traced_memory()[1], 100_000)
            windows.close()
        finally:
            tracemalloc.stop()

    def test_frames_reset_counters_actions_and_partial_eof(self):
        senses = FixedSenses()
        self.assertEqual(ReceptorStream(senses).window_bytes, 16)
        self.assertEqual(list(ReceptorStream(senses, 2).iter_frames(b'a\0b')),
                         [senses.frame(b'a\0'), senses.frame(b'b')])
        sinks, factories = [], []
        def factory():
            factories.append(True)
            return [b'a', b'\0b']
        world = StreamEnvironment(factory, senses, 2, sinks.append)
        with self.assertRaises(RuntimeError):
            world.step(Action(0))
        self.assertEqual(world.reset('opaque').frame, senses.frame(b'a\0'))
        self.assertEqual((world.actions, world.observations), (0, 1))
        for action in (None, 0, Action(True), Action(-1), Action(senses.actions)):
            with self.assertRaises(ValueError):
                world.step(action)
        self.assertEqual(sinks, [])
        self.assertEqual(world.step_batch([Action(0), Action(1)]).frame, senses.frame(b'b'))
        self.assertIsInstance(sinks[-1], NoOp)
        self.assertTrue(world.last_resolution.conflict)
        self.assertEqual(world.step(Action(2)), Step(None, done=True, reason='end_of_stream'))
        self.assertEqual(sinks, [NoOp(), Action(2)])
        self.assertEqual((world.actions, world.observations), (2, 2))
        world.reset()
        self.assertEqual(len(factories), 2)
        self.assertEqual((world.actions, world.observations), (0, 1))
        self.assertIsNone(world.last_resolution)
        world.close()
        empty = StreamEnvironment(lambda: b'', senses)
        self.assertEqual(empty.reset(), Step(None, done=True, reason='end_of_stream'))

    def test_reset_and_close_release_iterator(self):
        closed = []
        def source():
            try:
                yield b'abcdef'
            finally:
                closed.append(True)
        world = StreamEnvironment(source, FixedSenses(), 1)
        world.reset()
        world.reset()
        self.assertEqual(len(closed), 1)
        world.close()
        world.close()
        self.assertEqual(len(closed), 2)

    def test_real_parallel_runner_frozen_parameters(self):
        senses = FixedSenses()
        for partitions in (1, 2, 4):
            model = BitModel(repertoire(), senses=senses, sensory_partitions=partitions, seed=12)
            # Ensure every activation emits an action, so the real runner consumes EOF.
            for gates in model.gates.values():
                gates[:] = [100.] + [-100.] * (len(gates) - 1)
            before, weights = model.parameters(), model.weights
            shape = (len(weights), len(weights[0]))
            def sink(action):
                if world.observations == 3:
                    model.pending[999] = (model.ids[0], Prediction(0, bytes(senses.spec.bytes_per_signal)))
            world = StreamEnvironment(lambda: [b'a\0', b'bcd'], senses, 2, sink)
            transitions = run_episode(model, world, None, learn=False)
            self.assertEqual(transitions[-1], Step(None, done=True, reason='end_of_stream'))
            self.assertEqual(world.observations, 3)
            for transition in transitions:
                if transition.frame is not None:
                    senses.spec.validate(transition.frame)
            self.assertEqual(model.parameters(), before)
            self.assertIs(model.weights, weights)
            self.assertEqual((len(model.weights), len(model.weights[0])), shape)
            self.assertFalse(model.pending)
            self.assertTrue(any(event['kind'] == 'EXPIRED' and event['expectation_id'] == 999
                                for event in model.events))


class ContinuousStreamTests(unittest.TestCase):
    def model(self, partitions=1, prediction=True):
        model = BitModel(repertoire(), senses=FixedSenses(), seed=71,
                         sensory_partitions=partitions)
        loop = next(q for q, successor in model.repertoire.successors.items() if q == successor)
        model.bias[:] = [1000. if q == loop else -1000. for q in model.ids]
        for row in model.weights:
            row[:] = [0.] * len(row)
        for gates in model.gates.values():
            gates[:] = [-1000.] * len(gates)
            gates[model.spec.actions if prediction else 0] = 1000.
        return model

    def test_prediction_only_input_clock_feedback_and_freeze(self):
        from static_agent_training.experimental.stream_input import run_stream
        for partitions in (1, 4):
            model = self.model(partitions)
            before, weights = model.parameters(), model.weights
            model.events.append(dict(kind='HISTORY'))
            calls = []
            result = run_stream(model, b'abcdefg', window_bytes=2, frame_limit=None,
                                rounds_per_frame=3, action_sink=calls.append)
            self.assertEqual((result['frame_count'], result['bytes_consumed'], result['stop_reason']),
                             (4, 7, 'end_of_stream'))
            self.assertEqual(calls, [])
            self.assertEqual(result['resolved_actions'], [])
            for index, record in enumerate(result['frames']):
                events = record['events']
                self.assertEqual(len(record['predictions']), partitions * 3)
                self.assertEqual(len(record['traversals']), partitions * 3)
                self.assertEqual(len(record['active_pairs']), partitions)
                self.assertEqual(len(record['current_pairs']), partitions)
                self.assertEqual(len({p['expectation_id'] for p in record['predictions']}),
                                 partitions * 3)
                self.assertTrue(all(p['channel'] == 0 for p in record['predictions']))
                expectations = [e for e in events if e['kind'] == 'EXPECTATION']
                self.assertEqual([p['expectation_id'] for p in record['predictions']],
                                 [e['expectation_id'] for e in expectations])
                self.assertEqual(record['input_window_size'], 1 if index == 3 else 2)
                self.assertEqual(record['actions'], [])
                feedback = [e for e in events if e['kind'] == 'FEEDBACK']
                self.assertEqual(len(feedback), partitions * 3 if index else 0)
                first_embed = next(i for i, e in enumerate(events) if e['kind'] == 'EMBED')
                self.assertTrue(all(e['kind'] == 'FEEDBACK' for e in events[:first_embed]))
                self.assertTrue(all(e['observed'] == record['sensory'] for e in feedback))
            self.assertEqual(sum(e['kind'] == 'EXPIRED' for e in model.events), partitions * 3)
            self.assertEqual(model.events[0], dict(kind='HISTORY'))
            self.assertEqual(model.parameters(), before)
            self.assertIs(model.weights, weights)
            self.assertIsNone(model.frame)
            self.assertFalse(model.pending)
            self.assertFalse(model.activations)

    def test_dead_chain_reembeds_on_next_input(self):
        from static_agent_training.experimental.stream_input import run_stream
        model = self.model()
        live = next(q for q, successor in model.repertoire.successors.items() if q == successor)
        dead = next(q for q in model.ids if q != live)
        successors = dict(model.repertoire.successors)
        successors.pop(dead, None)
        model.repertoire = type(model.repertoire)(model.repertoire.pairs, successors)
        model.bias[:] = [0.] * len(model.ids)
        model.weights[1][model.ids.index(dead)] = 1000.
        model.weights[2][model.ids.index(live)] = 1000.
        result = run_stream(model, b'\0\xff', window_bytes=1, frame_limit=None,
                            rounds_per_frame=3)
        self.assertEqual(result['frame_count'], 2)
        first, second = result['frames']
        self.assertEqual(first['active_pairs'][0]['pair'], dead)
        self.assertEqual(first['current_pairs'], [])
        self.assertEqual([e['kind'] for e in first['events']], ['EMBED', 'TERMINATED'])
        self.assertEqual(second['active_pairs'][0]['pair'], live)
        self.assertEqual(len(second['predictions']), 3)

    def test_limits_do_not_probe_and_default_is_bounded(self):
        from static_agent_training.experimental.stream_input import run_stream
        reader = io.BytesIO(b'x' * 100)
        model = self.model()
        result = run_stream(model, reader, window_bytes=1)
        self.assertEqual((result['frame_count'], reader.tell(), result['stop_reason']),
                         (16, 16, 'frame_limit'))
        self.assertFalse(reader.closed)
        self.assertFalse(model.pending)
        self.assertEqual(model.events[-1]['kind'], 'EXPIRED')
        pulls = []
        def source():
            while True:
                pulls.append(True)
                yield b'x'
        result = run_stream(model, source(), window_bytes=1, frame_limit=2)
        self.assertEqual(len(pulls), 2)
        self.assertEqual(result['bytes_consumed'], 2)
        self.assertEqual(run_stream(model, b'', frame_limit=None)['frames'], [])
        for argument in ('rounds_per_frame', 'frame_limit'):
            for invalid in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    run_stream(model, b'a', **{argument: invalid})

    def test_chunk_invariance_tied_embedding_rng_and_action_sink(self):
        from static_agent_training.experimental.stream_input import run_stream
        models = [self.model(4, prediction=False) for _ in range(2)]
        for model in models:
            # All embedding scores tie, exercising saved RNG state in every region.
            model.bias[:] = [0.] * len(model.ids)
        sinks = [[], []]
        results = [run_stream(models[0], b'abcde', window_bytes=2,
                              frame_limit=None, action_sink=sinks[0].append),
                   run_stream(models[1], [b'a', b'', memoryview(b'bc'), bytearray(b'de')],
                              window_bytes=2, frame_limit=None, action_sink=sinks[1].append)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(sinks[0], sinks[1])
        self.assertEqual(models[0].embedding_rng.getstate(), models[1].embedding_rng.getstate())
        self.assertEqual(models[0].output_rng.getstate(), models[1].output_rng.getstate())
        self.assertEqual(len(sinks[0]), 3)
        for record in results[0]['frames']:
            self.assertEqual(record['action_resolution']['action'], 0)
            self.assertEqual(record['events'][-1]['kind'], 'BATCH_ACTION')

    def test_mixed_round_arbitrates_after_predictions_and_preserves_learning(self):
        from static_agent_training.experimental.stream_input import run_stream
        model = self.model(4)
        # Keep the real synchronous emitter, forcing different gates per region.
        original = model.emit
        def emit(*, learn=True):
            pair = model.repertoire.successors[model.pair]
            model.gates[pair][:] = [-1000.] * len(model.gates[pair])
            model.gates[pair][model.spec.actions if model._region == 3 else model._region % 2] = 1000.
            return original(learn=learn)
        model.emit = emit
        sinks = []
        result = run_stream(model, b'ab', window_bytes=1, frame_limit=None,
                            action_sink=sinks.append, learn=True, rounds_per_frame=5)
        for record in result['frames']:
            self.assertEqual(record['actions'], [0, 1, 0])
            self.assertEqual(len(record['predictions']), 1)
            self.assertTrue(record['action_resolution']['conflict'])
            self.assertIsNone(record['action_resolution']['action'])
            kinds = [e['kind'] for e in record['events']]
            self.assertLess(kinds.index('EXPECTATION'), kinds.index('BATCH_ACTION'))
        self.assertEqual(sinks, [NoOp(), NoOp()])
        self.assertTrue(model.revised)
        self.assertFalse(model.pending)
        self.assertTrue(any(e['kind'] == 'EMPIRICAL_UPDATE' for e in model.events))
