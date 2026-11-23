"""Agent Y: full-signal-conditioned, variable-count receptor partition learning."""
import copy
import math

from .bit_model import BitModel
from ..mappings import softmax, sample


class AgentY(BitModel):
    """Sequential learned grouping, with a new-group option at every receptor.

    Receptors may join any existing group or create a new one. There is no fixed
    group count, candidate menu, imposed boundary, or scheme-specific pair map.
    """
    def __init__(self, repertoire, *, senses, router_rate=.01, **kwargs):
        if not math.isfinite(router_rate) or router_rate <= 0:
            raise ValueError('router rate must be finite and positive')
        if kwargs.pop('embedding_selection', 'argmax') != 'argmax':
            raise ValueError('Agent Y requires argmax pair selection')
        kwargs.pop('sensory_partitions', None)
        self.partition = None
        self.router_rate = router_rate
        self.router_baseline = None
        self._routing = None
        self.router_weights = dict(
            affinity=[[0.] * senses.receptors for _ in range(senses.receptors)],
            split=[[0.] * (senses.spec.features + 1) for _ in range(senses.receptors)])
        # A complexity prior discourages one new group for every receptor. It is
        # a learnable score, not a cap or a fixed desired number of groups.
        for row in self.router_weights['split']:
            row[-1] = -math.log(senses.receptors)
        super().__init__(repertoire, senses=senses, sensory_partitions=senses.partitions or 1,
                         embedding_selection='argmax', **kwargs)

    @property
    def sensory_partitions(self):
        return len(self.partition) if self.partition is not None else self._sensory_partitions

    @staticmethod
    def validate_router(router, senses):
        weights = router['weights']
        if set(weights) != {'affinity', 'split'}:
            raise ValueError('unsupported Agent Y router; fixed-count routers require a new session')
        for key, width in [('affinity', senses.receptors), ('split', senses.spec.features + 1)]:
            matrix = weights[key]
            if len(matrix) != senses.receptors or any(len(row) != width or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) for v in row) for row in matrix):
                raise ValueError('invalid router weights')
        if not math.isfinite(router['rate']) or router['rate'] <= 0:
            raise ValueError('invalid router rate')
        baseline = router['baseline']
        if baseline is not None and (not isinstance(baseline, (int, float)) or not math.isfinite(baseline)):
            raise ValueError('invalid router baseline')

    def router_data(self):
        return dict(weights=copy.deepcopy(self.router_weights), baseline=self.router_baseline,
                    rate=self.router_rate)

    def reset_episode(self):
        super().reset_episode()
        self._routing = None
        self.partition = None

    def feature_indices(self, frame, region=None):
        active = super().feature_indices(frame, None)
        if region is None:
            return active
        allowed = {0, *(i + 1 for i in self.partition[region])}
        return [i for i in active if i in allowed]

    def observe(self, frame):
        self.spec.validate(frame)
        features = (*super().feature_indices(frame, None), self.spec.features)
        groups, decisions = [], []
        for receptor in range(self.senses.receptors):
            candidates = tuple(tuple(g) for g in groups)
            affinity = self.router_weights['affinity'][receptor]
            scores = [math.fsum(affinity[j] for j in g) / len(g) for g in candidates]
            scores.append(math.fsum(self.router_weights['split'][receptor][i] for i in features))
            probabilities = softmax(scores)
            chosen = sample(tuple(range(len(scores))), probabilities, self.embedding_rng)
            decisions.append((candidates, probabilities, chosen))
            if chosen == len(groups):
                groups.append([])
            groups[chosen].append(receptor)
        previous = self.partition
        self.partition = tuple(tuple(g) for g in groups)
        self._routing = (features, decisions)
        self.events.append(dict(kind='PARTITION_SELECT', partition=self.partition,
                                groups=len(groups), previous_groups=len(previous) if previous else None))
        super().observe(frame)

    def receive(self, frame, *, learn=True):
        offset = len(self.events)
        super().receive(frame, learn=learn)
        feedback = [e for e in self.events[offset:] if e['kind'] == 'FEEDBACK']
        if learn and feedback and self._routing is not None:
            loss = math.fsum(e['prediction_nll'] / self.spec.bits for e in feedback) / len(feedback)
            features, decisions = self._routing
            baseline = loss if self.router_baseline is None else self.router_baseline
            advantage = baseline - loss
            for receptor, (groups, probabilities, chosen) in enumerate(decisions):
                for k, probability in enumerate(probabilities):
                    delta = self.router_rate * advantage * (int(k == chosen) - probability)
                    if k == len(groups):
                        for i in features:
                            self.router_weights['split'][receptor][i] += delta
                    else:
                        for j in groups[k]:
                            self.router_weights['affinity'][receptor][j] += delta / len(groups[k])
            self.router_baseline = .95 * baseline + .05 * loss
            self.events.append(dict(kind='PARTITION_UPDATE', groups=len(self.partition),
                                    loss=loss, advantage=advantage))
        self._routing = None

    def to_data(self):
        data = super().to_data()
        # Physical apparatus metadata stays immutable; active group count varies.
        data.update(schema_version='crt-agent-y-v3', sensory_partitions=self.senses.partitions or 1,
                    router=self.router_data())
        return data

    @classmethod
    def from_data(cls, data):
        if data['schema_version'] != 'crt-agent-y-v3':
            raise ValueError('fixed-count Agent Y snapshots require a new session')
        base = BitModel.from_data(dict(data, schema_version='crt-bit-model-v1'))
        cls.validate_router(data['router'], base.senses)
        model = cls(base.repertoire, senses=base.senses, parameters=base.parameters(),
                    learning_config=base.config, action_policy=base.action_policy,
                    router_rate=data['router']['rate'])
        model.router_weights = copy.deepcopy(data['router']['weights'])
        model.router_baseline = data['router']['baseline']
        model.embedding_rng.setstate(base.embedding_rng.getstate())
        model.output_rng.setstate(base.output_rng.getstate())
        model.events, model.revised = base.events, base.revised
        return model


def main():
    """Run a fresh Agent Y on an arbitrary byte stream and save its snapshot."""
    import argparse
    from pathlib import Path
    from .senses import FixedSenses
    from .stream_input import run_stream
    from ..genome import develop
    from ..initialize import I, K
    from ..experiment import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--learn', action='store_true')
    parser.add_argument('--max-frames', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output path')
    if args.snapshot:
        import gzip
        from .. import serialization
        opener = gzip.open if args.snapshot.suffix == '.gz' else open
        with opener(args.snapshot, 'rt') as handle:
            model = AgentY.from_data(serialization.load(handle)['model'])
    else:
        senses = FixedSenses(receptors=256, partitions=1)
        repertoire = develop([(I, I), (I, K), (K, I)], ['input'], ['output'], budgets=32).repertoire
        model = AgentY(repertoire, senses=senses)
    with args.input.open('rb') as source:
        result = run_stream(model, source, frame_limit=args.max_frames, learn=args.learn)
    model.reset_episode()
    write_json(args.output, dict(model=model.to_data(), result=result))


if __name__ == '__main__':
    main()
