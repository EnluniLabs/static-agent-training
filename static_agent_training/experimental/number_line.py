"""Numeric meanings belong exclusively to this experimental environment adapter."""
import random
import json

from ..environment import MOVES
from .signals import Action, SignalSpec, Step
from .senses import StimulusStep


class ByteNumberLine:
    """One-byte signed integers, position in slot 0 and goal in slot 1.

    Expansion changes world bounds, never the model's interface dimensions.
    The agent never receives the conversion routines, bounds, or action meanings.
    """
    spec = SignalSpec(channels=2, bytes_per_signal=1, actions=len(MOVES))

    def __init__(self, *, lower=-20, upper=20, encoding_seed=0, action_limit=12):
        if (type(lower) is not int or type(upper) is not int or
                not -128 <= lower < upper <= 127):
            raise ValueError('world bounds must fit the signed one-byte interface')
        if type(action_limit) is not int or action_limit < 1:
            raise ValueError('action limit must be a positive integer')
        self.lower, self.upper, self.action_limit = lower, upper, action_limit
        self.moves = list(MOVES)
        random.Random(f'moves:{encoding_seed}').shuffle(self.moves)
        self.done = True

    def frame(self):
        return (self.position.to_bytes(1, 'big', signed=True),
                self.goal.to_bytes(1, 'big', signed=True))

    def reset(self, task):
        start, goal = task
        if any(type(v) is not int or not self.lower <= v <= self.upper for v in (start, goal)) or start == goal:
            raise ValueError('task must contain different in-range integer positions')
        self.position, self.goal, self.actions, self.done = start, goal, 0, False
        return Step(self.frame())

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        if not isinstance(action, Action) or type(action.index) is not int or not 0 <= action.index < self.spec.actions:
            raise ValueError('unknown action')
        self.actions += 1
        self.position = max(self.lower, min(self.upper, self.position + self.moves[action.index]))
        success = self.position == self.goal
        self.done = success or self.actions >= self.action_limit
        return Step(self.frame(), self.done, success,
                    'target' if success else 'action_limit' if self.done else None)


class StimulusNumberLine:
    """World emits bytes; it never declares or changes the animal's senses.

    JSON is this world's explicit serialization choice, not a model parser.
    Arbitrarily large integer bounds change neither sensory capacity nor weights.
    """
    def __init__(self, *, lower=-20, upper=20, encoding_seed=0, action_limit=12,
                 boundary_mode="clamp", goal_conditioned=True):
        if type(lower) is not int or type(upper) is not int or lower >= upper:
            raise ValueError('bounds must be ordered integers')
        if type(action_limit) is not int or action_limit < 1:
            raise ValueError('action limit must be positive')
        if boundary_mode not in ("clamp", "terminate"):
            raise ValueError("boundary mode must be clamp or terminate")
        self.boundary_mode, self.goal_conditioned = boundary_mode, goal_conditioned
        self.lower, self.upper, self.action_limit = lower, upper, action_limit
        self.moves = list(MOVES)
        random.Random(f'moves:{encoding_seed}').shuffle(self.moves)
        self.done = True

    def stimulus(self):
        return json.dumps([self.position, self.goal] if self.goal_conditioned else [self.position], separators=(',', ':')).encode('utf-8')

    def reset(self, task):
        start, goal = task
        if any(type(v) is not int or not self.lower <= v <= self.upper for v in (start, goal)) or start == goal:
            raise ValueError('task must contain different in-range integers')
        self.position, self.goal, self.actions, self.done = start, goal, 0, False
        return StimulusStep(self.stimulus())

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        if not isinstance(action, Action) or type(action.index) is not int or not 0 <= action.index < len(self.moves):
            raise ValueError('unknown action')
        self.actions += 1
        self.position += self.moves[action.index]
        if self.boundary_mode == "clamp":
            self.position = max(self.lower, min(self.upper, self.position))
        elif not self.lower <= self.position <= self.upper:
            self.done = True
            return StimulusStep(None, True, False, "out_of_bounds")
        success = self.position == self.goal
        self.done = success or self.actions >= self.action_limit
        return StimulusStep(self.stimulus(), self.done, success,
                            'target' if success else 'action_limit' if self.done else None)
