"""Number-line world and token interface, independent of agent implementation."""

from dataclasses import dataclass
import random


MOVES = (-5, -3, -2, -1, 0, 1, 2, 3, 5)
POSITIONS = tuple(range(-20, 21))


@dataclass(frozen=True)
class Observation:
    token: str
    goal: str | None = None


@dataclass(frozen=True)
class Output:
    token: str


@dataclass(frozen=True)
class Feedback:
    expectation_id: int
    source: str
    predicted: str
    observed: str
    correct: bool


@dataclass(frozen=True)
class Transition:
    observation: Observation | None
    feedback: tuple[Feedback, ...] = ()
    expired_expectations: tuple[int, ...] = ()
    done: bool = False
    success: bool = False
    reason: str | None = None


class NumberLine:
    """One episode; numeric state is available only through diagnostic_state().

    Token encoding is fixed for an experiment and independent of episode RNG.
    A fresh agent computation begins whenever a transition has an observation.
    """

    def __init__(self, *, encoding_seed=0, episode_seed=0, action_limit=12, internal_limit=16, goal_conditioned=False, boundary_mode="clamp"):
        if action_limit < 1 or internal_limit < 1:
            raise ValueError("limits must be positive")
        if boundary_mode not in ("clamp", "terminate"):
            raise ValueError("boundary mode must be clamp or terminate")
        self.boundary_mode = boundary_mode
        self.goal_conditioned = goal_conditioned
        self._goal = 0
        permutation = list(POSITIONS)
        random.Random(encoding_seed).shuffle(permutation)
        self._signals = ({x: f"obs_{x:+d}" for x in POSITIONS} if goal_conditioned else
                         {x: f"obs_{i:02d}" for i, x in enumerate(permutation)})
        shuffled_moves = list(MOVES)
        random.Random(f"moves:{encoding_seed}").shuffle(shuffled_moves)
        self._moves = {f"move_{i:02d}": a for i, a in enumerate(shuffled_moves)}
        self._expects = {"expect_" + token[len("obs_"):]: x for x, token in self._signals.items()}
        self.observation_tokens = tuple(Observation(token) for token in self._signals.values())
        if goal_conditioned:
            self.observation_tokens += tuple(Observation(f"goal_{x:+d}") for x in POSITIONS)
        self.output_tokens = tuple(Output(t) for t in (*self._moves, *self._expects))
        self._rng = random.Random(episode_seed)
        self.action_limit = action_limit
        self.internal_limit = internal_limit
        self._done = True

    def reset(self, *, start=None, goal=0):
        """Explicit numeric starts are for the experiment runner, never the agent."""
        if type(goal) is not int or goal not in POSITIONS or (goal != 0 and not self.goal_conditioned):
            raise ValueError("goal must be in [-20, 20]; nonzero goals require goal-conditioned observations")
        self._goal = goal
        if start is not None and (type(start) is not int or start not in POSITIONS or start == goal):
            raise ValueError("start must be in [-20, 20] and differ from goal")
        self._position = self._rng.choice([x for x in POSITIONS if x != goal]) if start is None else start
        self._actions = self._internal = self._next_expectation = 0
        self._pending = []
        self._done = False
        return Transition(self._observation())

    def _observation(self):
        return Observation(self._signals[self._position],
                           f"goal_{self._goal:+d}" if self.goal_conditioned else None)

    def _end(self, reason):
        expired = tuple(item[0] for item in self._pending)
        self._pending.clear()
        self._done = True
        return Transition(None, expired_expectations=expired, done=True, reason=reason)

    def stop(self):
        """Agent has no continuation capable of producing another output."""
        if self._done:
            raise RuntimeError("reset is required before another episode")
        return self._end("chain_terminated")

    def emit(self, output, *, source=""):
        """Consume one output; source identifies its emitting mathematical state."""
        if self._done:
            raise RuntimeError("reset is required before emitting outputs")
        token = output.token
        if token not in self._moves and token not in self._expects:
            raise ValueError("unknown output token")
        self._internal += 1
        if token in self._expects:
            self._pending.append((self._next_expectation, source, self._expects[token]))
            self._next_expectation += 1
            if self._internal >= self.internal_limit:
                return self._end("internal_limit")
            return Transition(None)

        self._actions += 1
        self._position += self._moves[token]
        self._internal = 0
        if self.boundary_mode == "clamp":
            self._position = max(-20, min(20, self._position))
        if self._position not in POSITIONS:
            # No input symbol exists outside the world. Commitments expire
            # without an observation against which to score them.
            return self._end("out_of_bounds")
        observed = self._signals[self._position]
        feedback = tuple(Feedback(i, source, self._signals[x], observed, x == self._position)
                         for i, source, x in self._pending)
        self._pending.clear()
        success = self._position == self._goal
        self._done = success or self._actions >= self.action_limit
        reason = "target" if success else "action_limit" if self._done else None
        return Transition(self._observation(), feedback, done=self._done, success=success, reason=reason)

    def diagnostic_state(self):
        """Privileged runner/viewer data; do not pass this to the agent."""
        return {"position": getattr(self, "_position", None),
                "actions": getattr(self, "_actions", 0),
                "goal": self._goal,
                "goal_conditioned": self.goal_conditioned,
                "boundary_mode": self.boundary_mode,
                "move_decoding": dict(self._moves),
                "expectation_decoding": dict(self._expects),
                "observation_encoding": dict(self._signals)}


def sample_starts(seed, count=20, *, allowed=tuple(x for x in POSITIONS if x)):
    """Common ordered starts for all agents in a generation; with replacement."""
    if count < 1 or not allowed or any(x not in POSITIONS or x == 0 for x in allowed):
        raise ValueError("require positive count and valid nonzero allowed starts")
    rng = random.Random(seed)
    return tuple(rng.choice(allowed) for _ in range(count))
