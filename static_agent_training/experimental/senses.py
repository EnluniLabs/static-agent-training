"""An animal-owned finite receptor bank for arbitrary-length byte stimuli.

The fixed, nonlearned transducer preserves shared bit components rather than
hashing each complete stimulus into an unrelated identity. Finite receptors
necessarily lose information for sufficiently long inputs; no injectivity or
semantic similarity is claimed.
"""
from dataclasses import asdict, dataclass
from typing import Protocol

from .signals import SignalSpec, Step


@dataclass(frozen=True)
class FixedSenses:
    receptors: int = 256
    actions: int = 9

    def __post_init__(self):
        if type(self.receptors) is not int or self.receptors < 16 or self.receptors % 16:
            raise ValueError('receptor count must be a positive multiple of 16')
        if type(self.actions) is not int or self.actions < 1:
            raise ValueError('action count must be positive')

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
        if type(stimulus) is not bytes:
            raise ValueError('environment transducers must supply bytes')
        pattern = bytearray(self.receptors // 8)
        bit_slots = self.receptors // 2
        for offset, byte in enumerate(stimulus):
            for bit_index in range(8):
                bit = (byte >> (7 - bit_index)) & 1
                receptor = 2 * ((offset * 8 + bit_index) % bit_slots) + bit
                pattern[receptor // 8] |= 1 << (7 - receptor % 8)
        return (bytes(pattern),)

    def to_data(self):
        return dict(**asdict(self), transducer='dual-rail-position-fold-or-v1')

    @classmethod
    def from_data(cls, data):
        data = dict(data)
        if data.pop('transducer') != 'dual-rail-position-fold-or-v1':
            raise ValueError('unknown sensory transducer')
        return cls(**data)


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
        frame = None if result.stimulus is None else self.senses.frame(result.stimulus)
        return Step(frame, result.done, result.success, result.reason)

    def reset(self, task):
        return self._convert(self.environment.reset(task))

    def step(self, action):
        return self._convert(self.environment.step(action))
