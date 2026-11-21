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
from .senses import FixedSenses, compatible_senses


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
    def __init__(self, repertoire, spec=None, *, seed=0, learning_config=None, senses=None, parameters=None, embedding_selection="argmax",
                 sensory_partitions=None, action_policy="agreement"):
        if embedding_selection not in ("sample", "argmax"):
            raise ValueError("embedding_selection must be sample or argmax")
        if sensory_partitions is None:
            sensory_partitions = getattr(senses, 'partitions', None) or 1
        if type(sensory_partitions) is not int or sensory_partitions < 1:
            raise ValueError('sensory partitions must be a positive integer')
        if action_policy not in ('agreement', 'vote'):
            raise ValueError('action policy must be agreement or vote')
        if sensory_partitions > 1 and (senses is None or embedding_selection != 'argmax'):
            raise ValueError('parallel activation requires fixed senses and argmax embedding')
        if senses is not None and sensory_partitions > senses.receptors:
            raise ValueError('sensory partitions cannot exceed receptor count')
        self._sensory_partitions, self.action_policy = sensory_partitions, action_policy
        self.embedding_selection = embedding_selection
        self.repertoire = repertoire
        if senses is not None and spec is not None and spec != senses.spec:
            raise ValueError('sensory capacity and signal specification disagree')
        self._senses = senses.bind_partitions(sensory_partitions) if senses is not None else None
        self._spec = senses.spec if senses is not None else spec or SignalSpec()
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

    @property
    def senses(self):
        return self._senses

    @property
    def spec(self):
        return self._spec

    @property
    def sensory_partitions(self):
        return self._sensory_partitions

    def reset_episode(self):
        self.pair = None
        self.frame = None
        self.pending = {}
        self.pending_logits = {}
        self.next_id = 0
        self.activations = []
        self.pending_regions = {}
        self._region = None
        self.round = 0

    def feature_indices(self, frame, region=None):
        if region is not None and self.sensory_partitions > 1:
            return self.senses.active_features(frame, region=region, partitions=self.sensory_partitions)
        x = self.spec.encode(frame)
        return [i for i, value in enumerate(x) if value]

    def probabilities(self, frame, region=None):
        active = [self.weights[i] for i in self.feature_indices(frame, region)]
        return softmax(self.bias[j] + math.fsum(row[j] for row in active)
                       for j in range(len(self.ids)))

    def observe(self, frame):
        self.spec.validate(frame)
        self.frame = frame
        if self.sensory_partitions > 1:
            self.activations = []
            self.round = 0
            for region in range(self.sensory_partitions):
                probabilities = self.probabilities(frame, region)
                maximum = max(probabilities) if probabilities else None
                tied = [q for q, probability in zip(self.ids, probabilities) if probability == maximum]
                pair = tied[0] if len(tied) == 1 else self.embedding_rng.choice(tied) if tied else None
                self.activations.append(dict(region=region, pair=pair))
                self.events.append(dict(kind='EMBED', region=region, pair=pair,
                                        frame=[v.hex() if v is not None else None for v in frame]))
            self.pair = None
            return
        probabilities = self.probabilities(frame)
        if self.embedding_selection == 'sample':
            self.pair = sample(self.ids, probabilities, self.embedding_rng)
        elif not probabilities:
            self.pair = None
        else:
            maximum = max(probabilities)
            tied = [q for q, p in zip(self.ids, probabilities) if p == maximum]
            self.pair = tied[0] if len(tied) == 1 else self.embedding_rng.choice(tied)
        self.events.append(dict(kind='EMBED', frame=[v.hex() if v is not None else None for v in frame],
                                pair=self.pair))

    def structural_update(self, pair, *, enabled, region=None):
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
        probability = self.probabilities(frame, region)[j]
        event = dict(pair=pair, successor=successor, channel=channel, signal=signal.hex(),
                     probability=probability, threshold=self.config.compatibility_threshold,
                     revision_event=self.revised[pair])
        if region is not None:
            event['region'] = region
        if probability >= self.config.compatibility_threshold:
            self.events.append(dict(kind='COMPATIBLE', **event))
            return
        # Normalize the local active-feature update: exactly embedding_rate
        # added to this frame's successor logit. Other frames share components.
        active = self.feature_indices(frame, region)
        delta = self.config.embedding_rate / (1 + len(active))
        self.bias[j] += delta
        for i in active:
            self.weights[i][j] += delta
        self.events.append(dict(kind='CONFLICT', **event, active_features=active,
                                component_delta=delta, logit_delta=self.config.embedding_rate))

    def emit(self, *, learn=True):
        if self.sensory_partitions > 1 and self._region is None:
            raise ValueError('parallel activations must use emit_round')
        successor = self.repertoire.successors.get(self.pair)
        if successor is None:
            self.events.append(dict(kind='TERMINATED', pair=self.pair))
            return None
        previous = self.pair
        self.pair = successor
        self.events.append(dict(kind='EDGE', source=previous, target=successor))
        self.structural_update(successor, enabled=learn, region=self._region)
        gate = sample(tuple(range(len(self.gates[successor]))), softmax(self.gates[successor]), self.output_rng)
        if gate < self.spec.actions:
            self.events.append(dict(kind='ACTION' if self._region is None else 'ACTION_SIGNAL', pair=successor, action=gate))
            return Action(gate)
        channel = gate - self.spec.actions
        logits = self.prediction[successor][channel]
        signal = pack_bits([int(self.output_rng.random() < sigmoid(v)) for v in logits])
        identity = self.next_id
        self.next_id += 1
        self.pending[identity] = (successor, Prediction(channel, signal))
        self.pending_logits[identity] = list(logits)
        if self._region is not None:
            self.pending_regions[identity] = self._region
        self.events.append(dict(kind='EXPECTATION', expectation_id=identity, pair=successor,
                                channel=channel, signal=signal.hex(), logits=list(logits)))
        return Prediction(channel, signal)

    def emit_round(self, *, learn=True):
        """Advance every activation once; retain duplicates and output order."""
        if self.sensory_partitions == 1:
            emission = self.emit(learn=learn)
            return () if emission is None else (emission,)
        emissions, survivors = [], []
        for activation in self.activations:
            region = activation['region']
            self._region, self.pair = region, activation['pair']
            offset = len(self.events)
            try:
                emission = self.emit(learn=learn)
            finally:
                self._region = None
            for event in self.events[offset:]:
                event.update(region=region, round=self.round)
            if emission is not None:
                survivors.append(dict(region=region, pair=self.pair))
                emissions.append(emission)
        self.activations = survivors
        self.pair = None
        self.round += 1
        return tuple(emissions)

    def receive(self, frame, *, learn=True):
        if frame is not None:
            self.spec.validate(frame)
        for identity, (source, prediction) in list(self.pending.items()):
            observed = None if frame is None else frame[prediction.channel]
            if observed is None:
                self.events.append(dict(kind='EXPIRED', expectation_id=identity, pair=source,
                                        **({'region': self.pending_regions[identity]} if identity in self.pending_regions else {})))
                continue
            correct = prediction.signal == observed
            bits = signal_bits(observed)
            emitted_logits = self.pending_logits.get(identity, self.prediction[source][prediction.channel])
            prediction_nll = math.fsum(max(v, 0) - v * bit + math.log1p(math.exp(-abs(v)))
                                       for v, bit in zip(emitted_logits, bits))
            self.events.append(dict(kind='FEEDBACK', expectation_id=identity, pair=source,
                                    predicted=prediction.signal.hex(), observed=observed.hex(),
                                    channel=prediction.channel, correct=correct, prediction_nll=prediction_nll,
                                    **({'region': self.pending_regions[identity]} if identity in self.pending_regions else {})))
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
                                    before=before, after=list(row),
                                    **({'region': self.pending_regions[identity]} if identity in self.pending_regions else {})))
        self.pending.clear()
        self.pending_logits.clear()
        self.pending_regions.clear()

    def parameters(self):
        return copy.deepcopy(dict(bias=self.bias, weights=self.weights, gates=self.gates,
                                  prediction=self.prediction))

    def to_data(self):
        if self.pair is not None or self.pending or self.activations:
            raise ValueError('save model snapshots only between episodes')
        return dict(schema_version='crt-bit-model-v1', spec=self.spec.to_data(),
                    senses=self.senses.to_data() if self.senses is not None else None,
                    embedding_selection=self.embedding_selection,
                    sensory_partitions=self.sensory_partitions, action_policy=self.action_policy,
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
                    parameters=data['parameters'], embedding_selection=data.get('embedding_selection','sample'),
                    sensory_partitions=data.get('sensory_partitions'), action_policy=data.get('action_policy','agreement'))
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
        model.embedding_rng.setstate(tuples(data['embedding_rng']))
        model.output_rng.setstate(tuples(data['output_rng']))
        model.events = copy.deepcopy(data['events'])
        model.revised = dict(data['revised'])
        return model

    def structure_hash(self):
        return hashlib.sha256(json.dumps(self.repertoire.to_data(), sort_keys=True).encode()).hexdigest()


def step_action_batch(model, environment, actions, *, learn=True):
    """The interface arbitrates actions only after the entire round emitted."""
    transition = environment.step_batch(actions, model.action_policy)
    resolution = environment.last_resolution
    model.events.append(dict(kind='BATCH_ACTION', policy=model.action_policy,
                             action=getattr(resolution.action, 'index', None),
                             conflict=resolution.conflict, counts=list(resolution.counts),
                             round=model.round - 1))
    if transition.frame is None and not transition.done:
        raise ValueError('an action must return an observation or terminate')
    model.receive(transition.frame, learn=learn)
    return transition


def run_episode(model, environment, task, *, learn=True, internal_limit=16):
    """Only opaque frames enter the model; task metadata and success stay outside."""
    if model.spec != environment.spec:
        raise ValueError('environment and model interface specifications differ')
    if model.senses is not None and not compatible_senses(model.senses, getattr(environment, 'senses', None)):
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
        if model.sensory_partitions > 1:
            emissions = model.emit_round(learn=learn)
            actions = [emission for emission in emissions if isinstance(emission, Action)]
            internal += 1
            if actions:
                transition = step_action_batch(model, environment, actions, learn=learn)
                internal = 0
            elif not model.activations:
                transition = Step(None, done=True, reason='chain_terminated')
            else:
                transition = Step(None, done=True, reason='internal_limit') if internal >= internal_limit else Step(None)
            transitions.append(transition)
            continue
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
