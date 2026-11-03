"""Shared signal-to-repertoire map and factorized, full-width predictions.

Expectations are atomic outputs: a categorical action/prediction-channel gate,
then independent Bernoulli bits. All supported byte strings can be predicted.
This is a different output normalization and empirical objective from tables.
"""
import copy
import hashlib
import json
import math
import random

from ..learning import LearningConfig
from ..mappings import sample, softmax
from ..repertoire import Repertoire
from ..seeds import derive
from .signals import Action, Prediction, SignalSpec, Step
from .senses import FixedSenses


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def signal_bits(signal):
    return tuple((byte >> bit) & 1 for byte in signal for bit in range(7, -1, -1))


def pack_bits(bits):
    return bytes(sum(bit << (7 - j) for j, bit in enumerate(bits[i:i + 8]))
                 for i in range(0, len(bits), 8))


def validate_parameters(parameters, ids, spec):
    if set(parameters) != {'bias', 'weights', 'gates', 'prediction'}:
        raise ValueError('unknown or missing model parameters')
    if set(parameters['gates']) != set(ids) or set(parameters['prediction']) != set(ids):
        raise ValueError('parameter repertoire mismatch')
    matrices = [(parameters['bias'], (len(ids),)), (parameters['weights'], (spec.features, len(ids)))]
    for q in ids:
        matrices.extend([(parameters['gates'][q], (spec.actions + spec.channels,)),
                         (parameters['prediction'][q], (spec.channels, spec.bits))])
    def validate(value, shape):
        if not shape:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('parameters must be finite')
        else:
            if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
                raise ValueError('parameter shape mismatch')
            for child in value:
                validate(child, shape[1:])
    for value, shape in matrices:
        validate(value, shape)


class BitModel:
    def __init__(self, repertoire, spec=None, *, seed=0, learning_config=None, senses=None, parameters=None):
        self.repertoire = repertoire
        if senses is not None and spec is not None and spec != senses.spec:
            raise ValueError('sensory capacity and signal specification disagree')
        self.senses = senses
        self.spec = senses.spec if senses is not None else spec or SignalSpec()
        self.config = learning_config or LearningConfig()
        self.ids = tuple(repertoire.pairs)
        # Keyed initialization is independent of other parameter counts.
        def value(*coordinates):
            return random.Random(derive(seed, 'bit-model-v1', *coordinates)).gauss(0, .1)
        if parameters is None:
            self.bias = [value('bias', q) for q in self.ids]
            self.weights = [[value('input', i, q) for q in self.ids] for i in range(self.spec.features)]
            self.gates = {q: [value('gate', q, i) for i in range(self.spec.actions + self.spec.channels)] for q in self.ids}
            self.prediction = {q: [[value('prediction', q, c, i) for i in range(self.spec.bits)]
                                  for c in range(self.spec.channels)] for q in self.ids}
        else:
            validate_parameters(parameters, self.ids, self.spec)
            for name, values in copy.deepcopy(parameters).items():
                setattr(self, name, values)
        self.embedding_rng = random.Random(derive(seed, 'bit-embedding'))
        self.output_rng = random.Random(derive(seed, 'bit-output'))
        self.events = []
        self.revised = {}
        self.reset_episode()

    def reset_episode(self):
        self.pair = None
        self.frame = None
        self.pending = {}
        self.pending_logits = {}
        self.next_id = 0

    def probabilities(self, frame):
        x = self.spec.encode(frame)
        active = [self.weights[i] for i, value in enumerate(x) if value]
        return softmax(self.bias[j] + math.fsum(row[j] for row in active)
                       for j in range(len(self.ids)))

    def observe(self, frame):
        self.spec.validate(frame)
        self.frame = frame
        self.pair = sample(self.ids, self.probabilities(frame), self.embedding_rng)
        self.events.append(dict(kind='EMBED', frame=[v.hex() if v is not None else None for v in frame],
                                pair=self.pair))

    def structural_update(self, pair, *, enabled):
        successor = self.repertoire.successors.get(pair)
        if not enabled or pair not in self.revised or successor is None:
            return
        # The modal complete prediction includes both its gate and bit mass.
        channel = max(range(self.spec.channels), key=lambda c:
                      self.gates[pair][self.spec.actions + c] -
                      math.fsum(math.log1p(math.exp(-abs(v))) for v in self.prediction[pair][c]))
        signal = pack_bits([int(v >= 0) for v in self.prediction[pair][channel]])
        frame = list(self.frame)
        frame[channel] = signal
        frame = tuple(frame)
        j = self.ids.index(successor)
        probability = self.probabilities(frame)[j]
        event = dict(pair=pair, successor=successor, channel=channel, signal=signal.hex(),
                     probability=probability, threshold=self.config.compatibility_threshold,
                     revision_event=self.revised[pair])
        if probability >= self.config.compatibility_threshold:
            self.events.append(dict(kind='COMPATIBLE', **event))
            return
        # Normalize the local active-feature update: exactly embedding_rate
        # added to this frame's successor logit. Other frames share components.
        active = [i for i, bit in enumerate(self.spec.encode(frame)) if bit]
        delta = self.config.embedding_rate / (1 + len(active))
        self.bias[j] += delta
        for i in active:
            self.weights[i][j] += delta
        self.events.append(dict(kind='CONFLICT', **event, active_features=active,
                                component_delta=delta, logit_delta=self.config.embedding_rate))

    def emit(self, *, learn=True):
        successor = self.repertoire.successors.get(self.pair)
        if successor is None:
            self.events.append(dict(kind='TERMINATED', pair=self.pair))
            return None
        previous = self.pair
        self.pair = successor
        self.events.append(dict(kind='EDGE', source=previous, target=successor))
        self.structural_update(successor, enabled=learn)
        gate = sample(tuple(range(len(self.gates[successor]))), softmax(self.gates[successor]), self.output_rng)
        if gate < self.spec.actions:
            self.events.append(dict(kind='ACTION', pair=successor, action=gate))
            return Action(gate)
        channel = gate - self.spec.actions
        logits = self.prediction[successor][channel]
        signal = pack_bits([int(self.output_rng.random() < sigmoid(v)) for v in logits])
        identity = self.next_id
        self.next_id += 1
        self.pending[identity] = (successor, Prediction(channel, signal))
        self.pending_logits[identity] = list(logits)
        self.events.append(dict(kind='EXPECTATION', expectation_id=identity, pair=successor,
                                channel=channel, signal=signal.hex(), logits=list(logits)))
        return Prediction(channel, signal)

    def receive(self, frame, *, learn=True):
        if frame is not None:
            self.spec.validate(frame)
        for identity, (source, prediction) in list(self.pending.items()):
            observed = None if frame is None else frame[prediction.channel]
            if observed is None:
                self.events.append(dict(kind='EXPIRED', expectation_id=identity, pair=source))
                continue
            correct = prediction.signal == observed
            bits = signal_bits(observed)
            emitted_logits = self.pending_logits.get(identity, self.prediction[source][prediction.channel])
            prediction_nll = math.fsum(max(v, 0) - v * bit + math.log1p(math.exp(-abs(v)))
                                       for v, bit in zip(emitted_logits, bits))
            self.events.append(dict(kind='FEEDBACK', expectation_id=identity, pair=source,
                                    predicted=prediction.signal.hex(), observed=observed.hex(),
                                    channel=prediction.channel, correct=correct, prediction_nll=prediction_nll))
            if not learn:
                continue
            row = self.prediction[source][prediction.channel]
            before = list(row)
            loss = math.fsum(max(v, 0) - v * bit + math.log1p(math.exp(-abs(v)))
                             for v, bit in zip(row, bits))
            for i, bit in enumerate(bits):
                row[i] -= self.config.prediction_rate * (sigmoid(row[i]) - bit)
            self.revised[source] = len(self.events)
            self.events.append(dict(kind='EMPIRICAL_UPDATE', expectation_id=identity, pair=source,
                                    channel=prediction.channel, correct=correct, loss=loss,
                                    before=before, after=list(row)))
        self.pending.clear()
        self.pending_logits.clear()

    def parameters(self):
        return copy.deepcopy(dict(bias=self.bias, weights=self.weights, gates=self.gates,
                                  prediction=self.prediction))

    def to_data(self):
        if self.pair is not None or self.pending:
            raise ValueError('save model snapshots only between episodes')
        return dict(schema_version='crt-bit-model-v1', spec=self.spec.to_data(),
                    senses=self.senses.to_data() if self.senses is not None else None,
                    repertoire=self.repertoire.to_data(), learning_config=vars(self.config),
                    parameters=self.parameters(),
                    embedding_rng=self.embedding_rng.getstate(), output_rng=self.output_rng.getstate(),
                    revised=dict(self.revised), events=copy.deepcopy(self.events))

    @classmethod
    def from_data(cls, data):
        if data['schema_version'] != 'crt-bit-model-v1':
            raise ValueError('unsupported experimental model')
        senses = FixedSenses.from_data(data['senses']) if data.get('senses') is not None else None
        model = cls(Repertoire.from_data(data['repertoire']), SignalSpec.from_data(data['spec']),
                    learning_config=LearningConfig(**data['learning_config']), senses=senses,
                    parameters=data['parameters'])
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
        model.embedding_rng.setstate(tuples(data['embedding_rng']))
        model.output_rng.setstate(tuples(data['output_rng']))
        model.events = copy.deepcopy(data['events'])
        model.revised = dict(data['revised'])
        return model

    def structure_hash(self):
        return hashlib.sha256(json.dumps(self.repertoire.to_data(), sort_keys=True).encode()).hexdigest()


def run_episode(model, environment, task, *, learn=True, internal_limit=16):
    """Only opaque frames enter the model; task metadata and success stay outside."""
    if model.spec != environment.spec:
        raise ValueError('environment and model interface specifications differ')
    if model.senses is not None and getattr(environment, 'senses', None) != model.senses:
        raise ValueError('environment must use the animal\'s saved sensory transducer')
    if type(internal_limit) is not int or internal_limit < 1:
        raise ValueError('internal limit must be positive')
    model.reset_episode()
    transition = environment.reset(task)
    transitions = [transition]
    internal = 0
    while not transition.done:
        if transition.frame is not None:
            model.observe(transition.frame)
        emission = model.emit(learn=learn)
        internal += 1
        if emission is None:
            transition = Step(None, done=True, reason='chain_terminated')
        elif isinstance(emission, Prediction):
            transition = (Step(None, done=True, reason='internal_limit')
                          if internal >= internal_limit else Step(None))
        else:
            transition = environment.step(emission)
            internal = 0
            if transition.frame is None and not transition.done:
                raise ValueError('an action must return an observation or terminate')
            model.receive(transition.frame, learn=learn)
        transitions.append(transition)
    model.receive(None, learn=learn)
    model.reset_episode()
    return tuple(transitions)
