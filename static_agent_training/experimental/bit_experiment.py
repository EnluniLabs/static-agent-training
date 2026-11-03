"""Fixed-structure learning and range transfer using the opaque byte model."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

from ..candidates import load_saved_candidate
from ..engine import run
from ..experiment import write_json
from ..initialize import Config, initialize
from ..repertoire import Repertoire
from ..seeds import derive
from .bit_model import BitModel, run_episode
from .number_line import StimulusNumberLine
from .senses import FixedSenses, SensoryAdapter


def tasks(seed, count, lower, upper, *, exclude=()):
    rng = random.Random(seed)
    values = tuple(range(lower, upper + 1))
    excluded = set(exclude)
    pairs = [(s, g) for s in values for g in values if s != g and (s, g) not in excluded]
    if not pairs:
        raise ValueError('no tasks remain after exclusions')
    return [rng.choice(pairs) for _ in range(count)]


def block(model, world, task_list, *, learn):
    successes, reasons = 0, Counter()
    offset = len(model.events)
    for task in task_list:
        final = run_episode(model, world, task, learn=learn)[-1]
        successes += final.success
        reasons[final.reason] += 1
    events = model.events[offset:]
    tested = [e for e in events if e['kind'] == 'FEEDBACK']
    return dict(episodes=len(task_list), successes=successes,
                success_rate=successes / len(task_list) if task_list else None, reasons=dict(reasons),
                tested_predictions=len(tested),
                exact_prediction_accuracy=sum(e['correct'] for e in tested) / len(tested) if tested else None,
                prediction_nll=sum(e['prediction_nll'] for e in tested) / len(tested) if tested else None,
                event_counts=dict(Counter(e['kind'] for e in events)))


def experiment(repertoire, *, seed=48, episodes=100, assessment_episodes=200, receptors=256):
    if any(type(n) is not int or n < 1 for n in (episodes, assessment_episodes)):
        raise ValueError('episode counts must be positive integers')
    senses = FixedSenses(receptors=receptors)
    model = BitModel(repertoire, seed=seed, senses=senses)
    def world(lower=-20, upper=20):
        return SensoryAdapter(StimulusNumberLine(lower=lower, upper=upper), senses)
    initial = model.to_data()
    structure = model.structure_hash()
    training_tasks = tasks(derive(seed, 'bit-original-adaptation'), episodes, -20, 20)
    training = block(model, world(), training_tasks, learn=True)
    trained = model.to_data()
    adaptation_tasks = tasks(derive(seed, 'bit-expanded-adaptation'), episodes, -30, 30)
    excluded = set(training_tasks + adaptation_tasks)
    old = tasks(derive(seed, 'bit-original-test'), assessment_episodes, -20, 20, exclude=excluded)
    expanded = tasks(derive(seed, 'bit-expanded-test'), assessment_episodes, -30, 30, exclude=excluded)
    # Distinct task identities: this CLI tests new draws in old regions and new
    # regions, not a held-out-value partition or evolutionary selection.
    test = dict(old=old,
                new=[t for t in expanded if all(abs(v) > 20 for v in t)],
                crossing=[t for t in expanded if (abs(t[0]) > 20) != (abs(t[1]) > 20)],
                expanded=expanded)
    results, snapshots = {}, {}
    for name, source, learn in [('prior-frozen', trained, False),
                                 ('prior-adapting', trained, True),
                                 ('fresh-adapting', initial, True)]:
        candidate = BitModel.from_data(source)
        # Match adaptation and evaluation sampling streams across all arms.
        candidate.embedding_rng.seed(derive(seed, 'transfer-embedding'))
        candidate.output_rng.seed(derive(seed, 'transfer-output'))
        adaptation = block(candidate, world(-30, 30), adaptation_tasks, learn=learn)
        adapted = candidate.to_data()
        scores = {}
        for region, task_list in test.items():
            assessor = BitModel.from_data(adapted)
            assessor.embedding_rng.seed(derive(seed, 'test-embedding', region))
            assessor.output_rng.seed(derive(seed, 'test-output', region))
            before = assessor.parameters()
            scores[region] = block(assessor, world(-30, 30), task_list, learn=False)
            assert assessor.parameters() == before
        assert candidate.structure_hash() == structure
        results[name] = dict(adaptation=adaptation, frozen_evaluation=scores,
                             event_counts=dict(Counter(e['kind'] for e in candidate.events)))
        snapshots[name] = adapted
    report = dict(schema_version='crt-bit-experiment-v2', seed=seed, spec=model.spec.to_data(),
                  senses=senses.to_data(), world_serialization='utf8-json-array-position-goal-v1',
                  structure_hash=structure, training=training, results=results,
                  repertoire_nodes=len(repertoire.pairs), repertoire_edges=len(repertoire.successors),
                  task_lists=dict(training=training_tasks, adaptation=adaptation_tasks, evaluation=test),
                  scope='fixed-structure reference test; no evolution; fresh task draws, not held-out values',
                  exact_identity_feedback=True)
    return report, dict(initial=initial, trained=trained, **snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, help='reuse a saved run champion structure, not its tabular mappings')
    parser.add_argument('--seed', type=int, default=48)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--assessment-episodes', type=int, default=200)
    parser.add_argument('--receptors', type=int, default=256, help='fixed animal sensory capacity, chosen once at initialization')
    parser.add_argument('--label', default='', help='short experiment label saved in run.json')
    parser.add_argument('--notes', default='', help='purpose or hypothesis saved with this run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    metadata = dict(schema_version='crt-sensory-run-v1', status='running', pid=os.getpid(),
                    created=time.time(), seed=args.seed, receptors=args.receptors,
                    episodes=args.episodes, assessment_episodes=args.assessment_episodes,
                    label=args.label, notes=args.notes, argv=sys.argv,
                    candidate=str(args.candidate.resolve()) if args.candidate else None)
    write_json(args.output / 'run.json', metadata)
    try:
        if args.candidate:
            bundle = load_saved_candidate(args.candidate)
            repertoire = Repertoire.from_data(bundle['individual']['repertoire'])
            provenance = dict(run=str(args.candidate), generation=bundle['generation'])
        else:
            generated = initialize(Config(seed=args.seed, count=4))
            pairs = [(p['left']['ast'], p['right']['ast']) for p in generated['initial_pairs']]
            repertoire = Repertoire.from_data(run(pairs, 64, seed=args.seed)['repertoire'])
            provenance = dict(initialization_seed=args.seed, initial_pairs=generated)
        report, snapshots = experiment(repertoire, seed=args.seed, episodes=args.episodes,
                                       assessment_episodes=args.assessment_episodes, receptors=args.receptors)
        report['structure_source'] = provenance
        for name, snapshot in snapshots.items():
            write_json(args.output / f'{name}.json.gz', snapshot)
        write_json(args.output / 'report.json', report)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        updated=time.time(), error=f'{type(error).__name__}: {error}')
        write_json(args.output / 'run.json', metadata)
        raise
    metadata.update(status='completed', updated=time.time())
    write_json(args.output / 'run.json', metadata)
    print(json.dumps(report['results'], indent=2))
    print(f'Saved: {args.output / "report.json"}')


if __name__ == '__main__':
    main()
