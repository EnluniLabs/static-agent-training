"""An animal-owned finite receptor bank for arbitrary-length byte stimuli.

The fixed, nonlearned transducer preserves shared bit components rather than
hashing each complete stimulus into an unrelated identity. Finite receptors
necessarily lose information for sufficiently long inputs; no injectivity or
semantic similarity is claimed.
"""
from dataclasses import asdict, dataclass, replace
from typing import Protocol

from .signals import SignalSpec, Step, resolve_actions


def receptor_regions(receptors, partitions):
    """Fixed contiguous regions of a receptor surface, independent of content."""
    if type(receptors) is not int or receptors < 1:
        raise ValueError('receptors must be a positive integer')
    if type(partitions) is not int or not 1 <= partitions <= receptors:
        raise ValueError('partitions must be a positive integer no greater than receptors')
    width, extra = divmod(receptors, partitions)
    return tuple((region * width + min(region, extra),
                  (region + 1) * width + min(region + 1, extra))
                 for region in range(partitions))


@dataclass(frozen=True)
class FixedSenses:
    receptors: int = 256
    actions: int = 9
    partitions: int | None = None

    def __post_init__(self):
        if type(self.receptors) is not int or self.receptors < 16 or self.receptors % 16:
            raise ValueError('receptor count must be a positive multiple of 16')
        if type(self.actions) is not int or self.actions < 1:
            raise ValueError('action count must be positive')

        if self.partitions is not None:
            receptor_regions(self.receptors, self.partitions)

    @property
    def interface(self):
        return "legacy-bytes"

    @property
    def raw_window_bytes(self):
        return self.receptors // 16

    def bind_partitions(self, count):
        receptor_regions(self.receptors, count)
        if self.partitions is not None and count != self.partitions:
            raise ValueError("sensory partitions are immutable")
        return replace(self, partitions=count) if self.partitions is None else self

    def _partition_count(self, partitions):
        count = self.partitions if partitions is None else partitions
        count = 1 if count is None else count
        if self.partitions is not None and count != self.partitions:
            raise ValueError("sensory partitions are immutable")
        receptor_regions(self.receptors, count)
        return count

    @property
    def spec(self):
        # One generic sensory surface, with no position/goal-specific channels.
        return SignalSpec(channels=1, bytes_per_signal=self.receptors // 8, actions=self.actions)

    def frame(self, stimulus):
        """Convert any length of bytes into the same fixed receptor pattern.

        Every bit position has a zero receptor and a one receptor (dual rail).
        Positions beyond capacity wrap around and superimpose with bitwise OR.
        Missing input differs from an empty but present stimulus.
        """
        if stimulus is None:
            return (None,)
        if not isinstance(stimulus, (bytes, bytearray, memoryview)):
            raise ValueError('sensory input must be bytes, bytearray, memoryview, or None')
        stimulus = bytes(stimulus)
        pattern = bytearray(self.receptors // 8)
        bit_slots = self.receptors // 2
        for offset, byte in enumerate(stimulus):
            for bit_index in range(8):
                bit = (byte >> (7 - bit_index)) & 1
                receptor = 2 * ((offset * 8 + bit_index) % bit_slots) + bit
                pattern[receptor // 8] |= 1 << (7 - receptor % 8)
        return (bytes(pattern),)

    def regions(self, partitions=None):
        """Return half-open receptor bounds, without assigning object meanings."""
        return receptor_regions(self.receptors, self._partition_count(partitions))

    def partition_indices(self, partitions=None):
        return tuple(tuple(range(start, end)) for start, end in self.regions(partitions))

    def active_features(self, frame, region=None, partitions=None):
        """Keep original feature coordinates and shared presence in every region."""
        partitions = self._partition_count(partitions)
        bounds = self.regions(partitions)
        if region is not None and (type(region) is not int or not 0 <= region < partitions):
            raise ValueError('region must identify one of the configured partitions')
        encoded = self.spec.encode(frame)
        if region is None:
            return [i for i, value in enumerate(encoded) if value]
        start, end = bounds[region]
        return [i for i, value in enumerate(encoded)
                if value and (i == 0 or start < i <= end)]

    def stream(self, window_bytes=None):
        """Create a bounded, chunk-independent receptor input for byte streams."""
        from .stream_input import ReceptorStream
        return ReceptorStream(self, window_bytes=window_bytes)

    def to_data(self):
        data = asdict(self)
        if self.partitions is None:
            del data['partitions']
        return dict(**data, transducer='dual-rail-position-fold-or-v1')

    @classmethod
    def from_data(cls, data):
        if data.get('transducer') == 'dual-rail-signal-surface-v1':
            return SignalSenses.from_data(data)
        if data.get('transducer') == 'packed-binary-retina-v1':
            return RetinaSenses.from_data(data)
        data = dict(data)
        if data.pop('transducer') != 'dual-rail-position-fold-or-v1':
            raise ValueError('unknown sensory transducer')
        return cls(**data)


@dataclass(frozen=True)
class RetinaSenses:
    """Immutable spatial eye consuming MSB-first packed binary pixels."""
    width: int = 16
    height: int = 16
    partitions: int = 4
    actions: int = 9
    scheme: str = 'tiles'

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.width, self.height)):
            raise ValueError('retina dimensions must be positive integers')
        if self.receptors < 16 or self.receptors % 16:
            raise ValueError('retina pixel count must be a multiple of 16')
        receptor_regions(self.receptors, self.partitions)
        if type(self.actions) is not int or self.actions != 9:
            raise ValueError('retina requires nine actions')
        if self.scheme != 'tiles':
            raise ValueError('unknown retina partition scheme')

    @property
    def interface(self):
        return 'retina'

    @property
    def receptors(self):
        return self.width * self.height

    @property
    def raw_window_bytes(self):
        return self.receptors // 8

    @property
    def spec(self):
        return SignalSpec(channels=1, bytes_per_signal=self.raw_window_bytes, actions=self.actions)

    def bind_partitions(self, count):
        self._partition_count(count)
        return self

    def _partition_count(self, partitions):
        count = self.partitions if partitions is None else partitions
        if type(count) is not int or count != self.partitions:
            raise ValueError('sensory partitions are immutable')
        return count

    @property
    def partition_rectangles(self):
        def split(x, y, w, h, count):
            if count == 1:
                return ((x, y, w, h),)
            candidates = []
            left_count = count // 2
            for axis, length, other in ((0, w, h), (1, h, w)):
                for cut in range(1, length):
                    if cut * other >= left_count and (length-cut) * other >= count-left_count:
                        error = abs(cut * count - length * left_count) / length
                        candidates.append((error, -length, axis, cut))
            _, _, axis, cut = min(candidates)
            if axis == 0:
                return split(x,y,cut,h,left_count) + split(x+cut,y,w-cut,h,count-left_count)
            return split(x,y,w,cut,left_count) + split(x,y+cut,w,h-cut,count-left_count)
        return split(0, 0, self.width, self.height, self.partitions)

    def regions(self, partitions=None):
        self._partition_count(partitions)
        return self.partition_rectangles

    def partition_indices(self, partitions=None):
        self._partition_count(partitions)
        return tuple(tuple(row*self.width+col for row in range(y,y+h) for col in range(x,x+w))
                     for x,y,w,h in self.partition_rectangles)

    def frame(self, payload):
        """Accept exactly one packed raster; transport adapters own any padding."""
        if payload is None:
            return (None,)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError('retina input must be packed bytes or None')
        payload = bytes(payload)
        if len(payload) != self.raw_window_bytes:
            raise ValueError('packed retina must match fixed pixel capacity exactly')
        return (payload,)

    def active_features(self, frame, region=None, partitions=None):
        count = self._partition_count(partitions)
        if region is not None and (type(region) is not int or not 0 <= region < count):
            raise ValueError('region must identify a configured partition')
        encoded = self.spec.encode(frame)
        indices = range(self.receptors) if region is None else self.partition_indices()[region]
        return ([0] if encoded[0] else []) + [i+1 for i in indices if encoded[i+1]]

    def stream(self, window_bytes=None):
        from .stream_input import ReceptorStream
        return ReceptorStream(self, window_bytes=window_bytes)

    def to_data(self):
        return dict(**asdict(self), version=1, transducer='packed-binary-retina-v1')

    @classmethod
    def from_data(cls, data):
        data = dict(data)
        if data.pop('transducer') != 'packed-binary-retina-v1' or data.pop('version', 1) != 1:
            raise ValueError('unsupported retina transducer version')
        return cls(**data)


@dataclass(frozen=True)
class SignalSenses(RetinaSenses):
    """Immutable tiled binary signal approximation; arbitrary bytes use dual rails."""
    @property
    def interface(self):
        return 'signals'

    @property
    def raw_window_bytes(self):
        return self.receptors // 16

    @property
    def spec(self):
        return SignalSpec(channels=1, bytes_per_signal=self.receptors // 8, actions=self.actions)

    def frame(self, stimulus):
        return FixedSenses.frame(self, stimulus)

    def to_data(self):
        return dict(**asdict(self), version=1, transducer='dual-rail-signal-surface-v1')

    @classmethod
    def from_data(cls, data):
        data = dict(data)
        if data.pop('transducer') != 'dual-rail-signal-surface-v1' or data.pop('version', 1) != 1:
            raise ValueError('unsupported signal transducer version')
        return cls(**data)


def stimulus_payload(environment, senses, fallback=None):
    """Select source bytes without interpreting their contents or modality."""
    if isinstance(senses, SignalSenses):
        receptor = getattr(environment, 'receptor_stimulus', None)
        if callable(receptor):
            return receptor()
    elif isinstance(senses, RetinaSenses):
        renderer = getattr(environment, 'render_retina', None)
        if not callable(renderer):
            raise ValueError('retina environment must implement render_retina(senses)')
        return renderer(senses)
    return fallback if fallback is not None else environment.stimulus()


def compatible_senses(owned, supplied):
    """Historical unbound byte adapters may serve an equivalent bound animal."""
    if owned == supplied:
        return True
    return (type(owned) is FixedSenses and type(supplied) is FixedSenses
            and supplied.partitions is None
            and (owned.receptors, owned.actions) == (supplied.receptors, supplied.actions))


@dataclass(frozen=True)
class StimulusStep:
    stimulus: bytes | None
    done: bool = False
    success: bool = False
    reason: str | None = None


class StimulusEnvironment(Protocol):
    def reset(self, task: object) -> StimulusStep: ...
    def step(self, action: object) -> StimulusStep: ...


class SensoryAdapter:
    """Connect different byte-producing worlds to the same animal-owned senses."""
    def __init__(self, environment, senses):
        self.environment = environment
        self.senses = senses
        self.spec = senses.spec

    def _convert(self, result):
        if isinstance(self.senses, SignalSenses):
            frame = (None if result.stimulus is None else
                     self.senses.frame(stimulus_payload(self.environment, self.senses, result.stimulus)))
        elif isinstance(self.senses, RetinaSenses):
            frame = self.senses.frame(stimulus_payload(self.environment, self.senses))
        else:
            frame = None if result.stimulus is None else self.senses.frame(result.stimulus)
        return Step(frame, result.done, result.success, result.reason)

    def reset(self, task):
        return self._convert(self.environment.reset(task))

    def step(self, action):
        return self._convert(self.environment.step(action))

    def step_batch(self, actions, policy='agreement'):
        self.last_resolution = resolve_actions(actions, policy, action_count=self.spec.actions)
        return self.step(self.last_resolution.action)
