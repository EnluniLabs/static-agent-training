"""Lifetime traversal of fixed structure. This module has no reducer access."""

from dataclasses import dataclass
import random

from .environment import Output
from .learning import LocalLearning


@dataclass(frozen=True)
class Emission:
    output: Output
    source: str


class Agent:
    def __init__(self, repertoire, mappings, *, embedding_seed=0, output_seed=0,
                 learning=True, learning_config=None):
        if set(repertoire.pairs) != set(mappings.pair_ids):
            raise ValueError('mappings must match the supplied repertoire')
        self.repertoire = repertoire
        self.mappings = mappings
        self._embedding_rng = random.Random(embedding_seed)
        self._output_rng = random.Random(output_seed)
        self.events = []
        self.learning_enabled = learning
        self.learner = LocalLearning(repertoire, mappings, self.events, learning_config)
        self.timestamp = 0
        self.reset()

    def reset(self):
        """Reset active computation, retaining lifetime logits and RNG streams."""
        self.pair = None
        self.path = []
        self.signal = None
        self.goal = None
        self.initial_pair = None
        self.observation_index = 0
        self.learner.reset_episode()

    def observe(self, observation):
        self.signal = observation.token
        self.goal = observation.goal
        self.observation_index += 1
        self.pair = self.mappings.embed(observation.token, self._embedding_rng, context=self.goal)
        self.initial_pair = self.pair
        self.path = [] if self.pair is None else [self.pair]
        self.events.append(dict(kind='EMBED', observation=observation.token, goal=self.goal, pair=self.pair))

    def next_output(self):
        """Advance a stored edge before sampling an untyped output token."""
        successor = self.repertoire.successors.get(self.pair)
        if successor is None:
            self.events.append(dict(kind='TERMINATED', pair=self.pair))
            self.pair = None
            return None
        self.pair = successor
        self.path.append(successor)
        self.learner.encounter(successor, enabled=self.learning_enabled, context=self.goal)
        token = self.mappings.emit(successor, self._output_rng)
        self.events.append(dict(kind='OUTPUT', pair=successor, output=token))
        self.timestamp += 1
        self.learner.register(token, successor, self.signal, self.initial_pair,
                              self.path, self.timestamp, self.observation_index + 1)
        return Emission(Output(token), successor)

    def receive(self, transition):
        for feedback in transition.feedback:
            self.learner.feedback(feedback, enabled=self.learning_enabled)
        self.learner.expire(transition.expired_expectations)


def run_episode(agent, environment, *, start=None, learn=None, goal=None):
    """Runner owns the environment; the agent sees symbols and explicit feedback.

    `learn=False` freezes both update rules for assessment. Existing lifetime
    weights and learned revision history survive episode reset.
    """
    previous_learning = agent.learning_enabled
    if learn is not None:
        agent.learning_enabled = learn
    try:
        agent.reset()
        transition = environment.reset(start=start, **({} if goal is None else {"goal": goal}))
        transitions = [transition]
        while not transition.done:
            if transition.observation is not None:
                agent.observe(transition.observation)
            emission = agent.next_output()
            transition = (environment.stop() if emission is None else
                          environment.emit(emission.output, source=emission.source))
            agent.receive(transition)
            transitions.append(transition)
        agent.reset()
        return tuple(transitions)
    finally:
        agent.learning_enabled = previous_learning
