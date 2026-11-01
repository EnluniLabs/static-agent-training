"""Read-only goal-input controls and stronger movement baselines at a checkpoint."""
import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics

from .accelerated import assert_equivalent, evaluate_batch, evaluate_frozen
from .evolution import TrainingConfig, generation_tasks
from .genome import Species
from .mappings import InheritedMappings
from .seeds import derive


def movement_baseline(tasks, moves, boundary_mode):
    """Exact success for a goal-independent uniform choice among supplied moves."""
    @lru_cache(maxsize=None)
    def probability(position, goal, remaining):
        if not -20 <= position <= 20:
            return 0.0
        if position == goal:
            return 1.0
        if not remaining:
            return 0.0
        destinations = [position + move for move in moves]
        if boundary_mode == 'clamp':
            destinations = [max(-20, min(20, x)) for x in destinations]
        return sum(probability(x, goal, remaining - 1) for x in destinations) / len(moves)
    return statistics.mean(probability(start, goal, 12) for start, goal in tasks)


def audit_checkpoint(source, *, sample=128, sample_seed=20261101, repeats=10, backend='native', workers=4):
    if type(sample) is not int or sample < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError('sample and repeats must be positive integers')
    source = Path(source)
    checkpoint = source / 'checkpoint.json.gz' if source.is_dir() else source
    # Keep one inode open while reading and hashing, even if live training replaces
    # its checkpoint atomically during the audit.
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='rb') as compressed:
            state = json.load(compressed)
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if state['schema_version'] != 'crt-training-v1':
        raise ValueError('unsupported checkpoint schema')
    config = TrainingConfig.from_data(state['config'])
    if not config.varying_goals:
        raise ValueError('goal-input audit requires a varying-goal run')
    generation = state['next_generation']
    indices = sorted(random.Random(sample_seed).sample(range(len(state['population'])), min(sample, len(state['population']))))
    population = [Species.from_data(state['population'][i]) for i in indices]
    seeds = [derive(config.seed, 'evaluation', generation, i) for i in indices]
    starts, assessment, goals, assessment_goals = generation_tasks(config, generation, state['partitions'])
    results, learned = evaluate_batch(
        population, starts, assessment, seeds=seeds, encoding_seed=config.encoding_seed,
        training_goals=goals, assessment_goals=assessment_goals, learning=config.learning,
        learning_config=config.learning_config, backend=backend, workers=workers,
        return_weights=True, boundary_mode=config.boundary_mode)
    frozen = []
    for lane, species in enumerate(population):
        inherited = species.inherited
        ids = inherited.pair_ids
        embedding = {s: {q: float(learned[0][lane, i, j]) for j, q in enumerate(ids)}
                     for i, s in enumerate(inherited.observations)}
        output = {q: {o: float(learned[1][lane, i, j]) for j, o in enumerate(inherited.outputs)}
                  for i, q in enumerate(ids)}
        mappings = InheritedMappings(inherited.observations, inherited.outputs, ids, embedding, output)
        frozen.append(replace(species, inherited=mappings))
    positions = state['partitions']['assessment']
    permutation = dict(zip(positions, positions[1:] + positions[:1]))
    exhaustive = [(s, g) for s in positions for g in positions if s != g] * repeats
    controls = {}
    for mode in ('correct_goal', 'permuted_goal_input', 'goal_input_removed'):
        members = []
        for species in frozen:
            data = species.inherited.to_data()
            old = data['embedding']
            if mode == 'permuted_goal_input':
                data['embedding'] = {s: dict(row) for s, row in old.items()}
                for goal in positions:
                    data['embedding'][f'goal_{goal:+d}'] = dict(old[f'goal_{permutation[goal]:+d}'])
            elif mode == 'goal_input_removed':
                for goal in positions:
                    data['embedding'][f'goal_{goal:+d}'] = {q: 0.0 for q in species.inherited.pair_ids}
            members.append(replace(species, inherited=InheritedMappings.from_data(data)))
        settings = dict(seeds=seeds, encoding_seed=config.encoding_seed, backend=backend,
                        workers=workers, boundary_mode=config.boundary_mode)
        scores = evaluate_frozen(members, assessment, assessment_goals, **settings)
        if mode == 'correct_goal':
            for actual, reference in zip(scores, results):
                assert_equivalent(actual, reference['assessment'])
        fresh = evaluate_frozen(members, [s for s, g in exhaustive], [g for s, g in exhaustive], **settings)
        rates = [r['fitness'] for r in scores]
        controls[mode] = dict(fixed_assessment_mean=statistics.mean(rates),
                              exhaustive_heldout_mean=statistics.mean(r['fitness'] for r in fresh),
                              per_animal_fixed=rates)
    policies = dict(uniform_9=(-5, -3, -2, -1, 0, 1, 2, 3, 5),
                    uniform_nonzero_8=(-5, -3, -2, -1, 1, 2, 3, 5),
                    always_plus_5=(5,), always_minus_5=(-5,))
    baselines = {name: dict(
        fixed_assessment_mean=movement_baseline(list(zip(assessment, assessment_goals)), moves, config.boundary_mode),
        exhaustive_heldout_mean=movement_baseline(exhaustive, moves, config.boundary_mode))
        for name, moves in policies.items()}
    return dict(generation=generation + 1, checkpoint_next_generation=generation,
                checkpoint_sha256=digest.hexdigest(), config=asdict(config), partitions=state['partitions'],
                population_sample=len(indices), total_population=config.population, sample_seed=sample_seed,
                repeats_per_heldout_pair=repeats,
                source='read-only checkpoint snapshot; lifetime training replayed before frozen controls',
                indices=indices, boundary_mode=config.boundary_mode, heldout_positions=positions,
                goal_permutation=permutation, controls=controls, movement_baselines=baselines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='run directory or retained checkpoint.json.gz')
    parser.add_argument('--sample', type=int, default=128)
    parser.add_argument('--sample-seed', type=int, default=20261101)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--backend', choices=('native', 'cuda'), default='native')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('runs/bias-audit.json'))
    args = parser.parse_args()
    result = audit_checkpoint(args.checkpoint, sample=args.sample, sample_seed=args.sample_seed,
                              repeats=args.repeats, backend=args.backend, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in result['controls'].items():
        print(f"{name}: fixed {row['fixed_assessment_mean']:.3%}; exhaustive {row['exhaustive_heldout_mean']:.3%}")
    print(f'Saved: {args.output}; source checkpoint was not modified.')


if __name__ == '__main__':
    main()
