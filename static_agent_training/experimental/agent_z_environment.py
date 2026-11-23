"""Agent Z environment adapter for signed relative-position measurements."""
import random
from types import MappingProxyType

from ..environment import MOVES
from .agent_z import Feedback


class NumberLine:
    """Map exact byte patterns to actions, then measure once after the action batch."""

    def __init__(self, *, encoding_seed=0, action_limit=12):
        if action_limit < 1:
            raise ValueError('action limit must be positive')
        moves = list(MOVES)
        random.Random(f'moves:{encoding_seed}').shuffle(moves)
        self.actions = MappingProxyType({bytes([0x70 + i]): move for i, move in enumerate(moves)})
        self.action_limit = action_limit
        self.done = True

    def measurement(self):
        return (self.goal - self.position).to_bytes(1, 'big', signed=True)

    def reset(self, task):
        start, goal = task
        if any(type(value) is not int or not -20 <= value <= 20 for value in task):
            raise ValueError('expected integer start and goal between -20 and 20')
        self.position, self.goal, self.remaining = start, goal, self.action_limit
        self.done = start == goal
        return self.measurement()

    def interpret(self, signals):
        if self.done:
            raise RuntimeError('reset required')
        recipients = frozenset(i for i, signal in enumerate(signals) if signal not in self.actions)
        # Order by wire pattern, retaining duplicates. Stop applying actions at
        # the first terminal state, then measure that state if any chain expects it.
        for signal in sorted(signal for signal in signals if signal in self.actions):
            self.position = max(-20, min(20, self.position + self.actions[signal]))
            self.remaining -= 1
            self.done = self.position == self.goal or self.remaining == 0
            if self.done:
                break
        return Feedback(self.measurement() if recipients else None, recipients, self.done,
                        self.position == self.goal)
