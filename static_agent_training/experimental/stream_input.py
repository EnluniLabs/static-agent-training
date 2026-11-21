"""Lazy byte streams for the generic Python runner.

Sources and action sinks define transport only; no action meanings or rewards
are assigned here. Files remain caller-owned. Iterable iterators are closed on
exhaustion, reset, or explicit environment close when they support close().
"""
from .signals import Action, NoOp, Step, resolve_actions


_BYTES = (bytes, bytearray, memoryview)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _view(chunk):
    if not isinstance(chunk, _BYTES):
        raise TypeError('stream chunks must be bytes, bytearray, or memoryview')
    try:
        return memoryview(chunk).cast('B')
    except (TypeError, ValueError) as error:
        raise TypeError('stream chunks must expose contiguous bytes') from error


def iter_stimuli(source, window_bytes):
    """Yield fixed byte windows and a final partial window, without read-ahead.

    Extra storage is O(window_bytes), excluding the current caller-owned chunk.
    Memoryviews must be contiguous. Validation of sources/chunks is lazy.
    Binary readers must honor read(n)'s bound; empty reads mean EOF.
    """
    _positive(window_bytes, 'window_bytes')

    def windows():
        reader = getattr(source, 'read', None)
        iterator = None
        if isinstance(source, _BYTES):
            iterator = iter((source,))
        elif callable(reader):
            pass
        else:
            if isinstance(source, (str, int)):
                raise TypeError('source must be binary data, chunks, or a binary reader')
            try:
                iterator = iter(source)
            except TypeError as error:
                raise TypeError('source must be binary data, chunks, or a binary reader') from error
        pending = bytearray()
        try:
            while True:
                if iterator is None:
                    chunk = reader(window_bytes - len(pending))
                    view = _view(chunk)
                    if len(view) > window_bytes - len(pending):
                        raise ValueError('binary reader returned more bytes than requested')
                    if not view:
                        break
                else:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        break
                    view = _view(chunk)
                offset = 0
                while offset < len(view):
                    count = min(window_bytes - len(pending), len(view) - offset)
                    pending.extend(view[offset:offset + count])
                    offset += count
                    if len(pending) == window_bytes:
                        result = bytes(pending)
                        pending.clear()
                        yield result
            if pending:
                yield bytes(pending)
        finally:
            close = getattr(iterator, 'close', None)
            if callable(close):
                close()

    return windows()


class ReceptorStream:
    """Window stimuli within the non-folding capacity of saved senses."""

    def __init__(self, senses, window_bytes=None):
        capacity = getattr(senses, 'raw_window_bytes', senses.receptors // 16)
        self.window_bytes = _positive(capacity if window_bytes is None else window_bytes,
                                      'window_bytes')
        if self.window_bytes > capacity:
            raise ValueError('window_bytes exceeds the non-folding receptor capacity')
        self.senses = senses

    def convert(self, window):
        """Frame a transport window; only Retina transport adds zero pixels.

        The eye itself accepts exact rasters. Padding here never changes the
        original window used for input-size and byte-consumption accounting.
        """
        if getattr(self.senses, 'interface', None) == 'retina' and window is not None:
            window = bytes(_view(window))
            capacity = self.senses.raw_window_bytes
            if len(window) > capacity:
                raise ValueError('window exceeds the retina raw input capacity')
            window = window.ljust(capacity, b'\0')
        return self.senses.frame(window)

    def iter_frames(self, source):
        windows = iter_stimuli(source, self.window_bytes)
        try:
            for window in windows:
                yield self.convert(window)
        finally:
            windows.close()


class StreamEnvironment:
    """Action-paced stream adapter for Python run_episode, including partitions.

    source_factory() is called on each reset; task is intentionally opaque and
    ignored. The optional sink receives each validated Action or NoOp once.
    observations counts delivered windows; actions counts accepted step calls,
    including the advance that discovers EOF or the frame limit. No native
    runner integration or implicit ownership of binary readers is provided.
    """

    def __init__(self, source_factory, senses, window_bytes=None, action_sink=None,
                 frame_limit=None):
        if not callable(source_factory):
            raise TypeError('source_factory must be callable')
        if action_sink is not None and not callable(action_sink):
            raise TypeError('action_sink must be callable')
        if frame_limit is not None:
            _positive(frame_limit, 'frame_limit')
        self.stream = ReceptorStream(senses, window_bytes)
        self.senses, self.spec = senses, senses.spec
        self.source_factory, self.action_sink = source_factory, action_sink
        self.frame_limit = frame_limit
        self.actions = self.observations = 0
        self.last_resolution = None
        self.done = True
        self._frames = None

    def close(self):
        """Release stream iterators, never directly close a caller-owned file."""
        frames, self._frames = self._frames, None
        self.done = True
        if frames is not None:
            frames.close()

    def _advance(self):
        if self.frame_limit is not None and self.observations >= self.frame_limit:
            self.close()
            return Step(None, done=True, reason='frame_limit')
        try:
            frame = next(self._frames)
        except StopIteration:
            self.close()
            return Step(None, done=True, reason='end_of_stream')
        self.observations += 1
        return Step(frame)

    def reset(self, task=None):
        self.close()
        self.actions = self.observations = 0
        self.last_resolution = None
        self._frames = self.stream.iter_frames(self.source_factory())
        self.done = False
        return self._advance()

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        if not isinstance(action, NoOp) and (
                not isinstance(action, Action) or type(action.index) is not int or
                not 0 <= action.index < self.spec.actions):
            raise ValueError('unknown action')
        if self.action_sink is not None:
            self.action_sink(action)
        self.actions += 1
        return self._advance()

    def step_batch(self, actions, policy='agreement'):
        if self.done:
            raise RuntimeError('reset required')
        self.last_resolution = resolve_actions(actions, policy, action_count=self.spec.actions)
        return self.step(self.last_resolution.action)


def run_stream(model, source, *, window_bytes=None, frame_limit=16,
               action_sink=None, learn=False, rounds_per_frame=1):
    """Run the Python model on an input clock independent of emitted actions.

    Traces contain one record per actual input window. A finite limit never
    probes the following window (so exact-length input reports frame_limit).
    frame_limit=None explicitly enables unbounded tracing: reader buffering is
    bounded, but returned records and the model's retained event history grow.
    Caller-owned binary readers are never closed here. active_pairs records
    selected entry states (EMBED); traversals connect them to current_pairs,
    the surviving states after the final emission round. Prediction IDs are
    separate even when multiple regions select the same pair.
    """
    from itertools import islice

    _positive(rounds_per_frame, 'rounds_per_frame')
    if frame_limit is not None:
        _positive(frame_limit, 'frame_limit')
    if action_sink is not None and not callable(action_sink):
        raise TypeError('action_sink must be callable')
    if model.senses is None:
        raise ValueError('run_stream requires model-owned senses')
    stream = ReceptorStream(model.senses, window_bytes)
    if model.spec != model.senses.spec:
        raise ValueError('model and senses specifications differ')
    windows = iter_stimuli(source, stream.window_bytes)
    records, resolutions = [], []
    consumed = 0
    model.reset_episode()
    try:
        for window in islice(windows, frame_limit):
            offset = len(model.events)
            frame = stream.convert(window)
            # Pending predictions always see the next input, before reembedding.
            model.receive(frame, learn=learn)
            model.observe(frame)
            active = [dict(region=e.get('region', 0), pair=e['pair'])
                      for e in model.events[offset:] if e['kind'] == 'EMBED']
            record = dict(sensory=frame[0].hex(), input_window_size=len(window),
                          active_pairs=active, traversals=[], predictions=[], actions=[], events=[])
            records.append(record)
            consumed += len(window)
            for round_index in range(rounds_per_frame):
                emissions = model.emit_round(learn=learn)
                actions = [e for e in emissions if isinstance(e, Action)]
                if actions:
                    resolution = resolve_actions(actions, model.action_policy,
                                                 action_count=model.spec.actions)
                    summary = dict(policy=model.action_policy,
                                   action=getattr(resolution.action, 'index', None),
                                   conflict=resolution.conflict, counts=resolution.counts)
                    model.events.append(dict(kind='BATCH_ACTION', **summary, round=round_index))
                    record['action_resolution'] = summary
                    resolutions.append(summary)
                    if action_sink is not None:
                        action_sink(resolution.action)
                    break
                if not emissions:
                    break
            record['current_pairs'] = (
                [dict(activation) for activation in model.activations]
                if model.sensory_partitions > 1 else
                [dict(region=0, pair=model.pair)] if emissions else [])
            events = model.events[offset:]
            record['traversals'] = [dict(source=e['source'], target=e['target'],
                                          region=e.get('region', 0))
                                    for e in events if e['kind'] == 'EDGE']
            record['predictions'] = [dict(region=e.get('region', 0), pair=e['pair'],
                                           expectation_id=e['expectation_id'],
                                           channel=e['channel'], signal=e['signal'])
                                     for e in events if e['kind'] == 'EXPECTATION']
            record['actions'] = [e['action'] for e in events
                                 if e['kind'] in ('ACTION', 'ACTION_SIGNAL')]
            record['events'] = events
    finally:
        try:
            offset = len(model.events)
            model.receive(None, learn=learn)
            if records:
                records[-1]['events'].extend(model.events[offset:])
        finally:
            model.reset_episode()
            windows.close()
    return dict(frames=records, frame_count=len(records), bytes_consumed=consumed,
                stop_reason=('frame_limit' if frame_limit is not None and
                             len(records) == frame_limit else 'end_of_stream'),
                resolved_actions=resolutions)
