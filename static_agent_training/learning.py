"""Encounter-local empirical correction and delayed embedding correction."""

from dataclasses import asdict, dataclass
import math

from .mappings import softmax


@dataclass(frozen=True)
class LearningConfig:
    prediction_rate: float = 0.05
    embedding_rate: float = 0.05
    compatibility_threshold: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in
               (self.prediction_rate, self.embedding_rate, self.compatibility_threshold)):
            raise ValueError('learning parameters must be finite and nonnegative')
        if self.compatibility_threshold > 1:
            raise ValueError('compatibility threshold must not exceed one')


@dataclass(frozen=True)
class Expectation:
    expectation_id: int
    predicted: str
    source: str
    input_signal: str
    initial_pair: str
    path: tuple
    logits: tuple
    timestamp: int
    target_observation: int


class LocalLearning:
    def __init__(self, repertoire, mappings, events, config=None):
        self.repertoire = repertoire
        self.mappings = mappings
        self.events = events
        self.config = config or LearningConfig()
        self.expectations = tuple(o for o in mappings.outputs if o.startswith('expect_'))
        self.signals = {o: 'obs_' + o[len('expect_'):] for o in self.expectations}
        if any(s not in mappings.observations for s in self.signals.values()):
            raise ValueError('expectations must correspond to observation symbols')
        self.revised = {}  # Pair -> most recent empirical revision event.
        self.reset_episode()

    def reset_episode(self):
        self.pending = {}
        self.next_id = 0

    def register(self, token, source, signal, initial, path, timestamp, target_observation):
        if token not in self.signals:
            return
        record = Expectation(self.next_id, self.signals[token], source, signal, initial,
                             tuple(path), tuple(self.mappings.output[source].items()),
                             timestamp, target_observation)
        self.next_id += 1
        self.pending[record.expectation_id] = record
        self.events.append(dict(kind='EXPECTATION', **asdict(record)))

    def feedback(self, feedback, *, enabled=True):
        record = self.pending[feedback.expectation_id]
        if record.source != feedback.source or record.predicted != feedback.predicted:
            raise ValueError('feedback does not match pending expectation')
        observed_token = 'expect_' + feedback.observed[len('obs_'):]
        if observed_token not in self.signals:
            raise ValueError('unknown feedback observation')
        del self.pending[record.expectation_id]
        if not enabled:
            return
        row = self.mappings.output[record.source]
        logits = tuple(row[o] for o in self.expectations)
        probabilities = softmax(logits)
        # Stable log-sum-exp, even when the target probability underflows.
        largest = max(logits)
        loss = largest - row[observed_token] + math.log(math.fsum(math.exp(v - largest) for v in logits))
        before = dict(zip(self.expectations, logits))
        for token, probability in zip(self.expectations, probabilities):
            row[token] -= self.config.prediction_rate * (probability - (token == observed_token))
        revision = len(self.events)
        self.events.append(dict(kind='EMPIRICAL_UPDATE', expectation_id=record.expectation_id,
                                pair=record.source, predicted=record.predicted,
                                observed=feedback.observed, correct=feedback.correct,
                                loss=loss, before=before,
                                after={o: row[o] for o in self.expectations}))
        self.revised[record.source] = revision

    def expire(self, ids):
        for identity in ids:
            record = self.pending.pop(identity)
            self.events.append(dict(kind='EXPIRED', expectation_id=identity, pair=record.source))

    def encounter(self, pair, *, enabled=True):
        if not enabled or pair not in self.revised or not self.expectations:
            return
        successor = self.repertoire.successors.get(pair)
        if successor is None:
            return
        row = self.mappings.output[pair]
        token = max(self.expectations, key=lambda o: row[o])
        signal = self.signals[token]
        probability = self.mappings.embedding_probabilities(signal)[self.mappings.pair_ids.index(successor)]
        details = dict(pair=pair, signal=signal, successor=successor,
                       probability=probability, threshold=self.config.compatibility_threshold,
                       revision_event=self.revised[pair])
        if probability >= self.config.compatibility_threshold:
            self.events.append(dict(kind='COMPATIBLE', **details))
            return
        before = self.mappings.embedding[signal][successor]
        self.mappings.embedding[signal][successor] += self.config.embedding_rate
        self.events.append(dict(kind='CONFLICT', **details, before=before,
                                after=self.mappings.embedding[signal][successor]))
