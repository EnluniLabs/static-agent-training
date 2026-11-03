"""Fixed-capacity opaque signal boundary. No numeric or token-name semantics."""
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class SignalSpec:
    channels: int = 2
    bytes_per_signal: int = 1
    actions: int = 9

    def __post_init__(self):
        for value in (self.channels, self.bytes_per_signal, self.actions):
            if type(value) is not int or value < 1:
                raise ValueError('interface dimensions must be positive integers')

    @property
    def bits(self):
        return 8 * self.bytes_per_signal

    @property
    def features(self):
        return self.channels * (self.bits + 1)

    def validate(self, frame):
        if not isinstance(frame, tuple) or len(frame) != self.channels:
            raise ValueError('frame must have exactly one slot per declared channel')
        if any(value is not None and (type(value) is not bytes or
               len(value) != self.bytes_per_signal) for value in frame):
            raise ValueError('signals must be fixed-width bytes or absent')

    def encode(self, frame):
        """MSB-first bits in fixed channel slots, each preceded by a presence bit."""
        self.validate(frame)
        features = []
        for value in frame:
            features.append(int(value is not None))
            features.extend((byte >> bit) & 1 for byte in (value or bytes(self.bytes_per_signal))
                            for bit in range(7, -1, -1))
        return tuple(features)

    def to_data(self):
        return dict(**asdict(self), encoding='raw-bytes-msb-first-channel-slots-presence-v1')

    @classmethod
    def from_data(cls, data):
        data = dict(data)
        if data.pop('encoding') != 'raw-bytes-msb-first-channel-slots-presence-v1':
            raise ValueError('unknown signal encoding')
        return cls(**data)


@dataclass(frozen=True)
class Action:
    index: int


@dataclass(frozen=True)
class Prediction:
    channel: int
    signal: bytes


@dataclass(frozen=True)
class Step:
    frame: tuple | None
    done: bool = False
    success: bool = False
    reason: str | None = None


class SignalEnvironment(Protocol):
    """Tasks and success belong to the runner; only frames reach the model."""
    spec: SignalSpec

    def reset(self, task: object) -> Step: ...
    def step(self, action: Action) -> Step: ...
