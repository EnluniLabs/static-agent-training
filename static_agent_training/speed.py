"""Measure end-to-end population evaluation time on shared initial genotypes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from .evolution import TrainingConfig, evaluate_generation, initialize_population, partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--populations', type=int, nargs='+', default=[64, 256, 1024])
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--assessment-episodes', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--backends', nargs='+', choices=['python', 'native', 'cuda'], default=['python', 'native', 'cuda'])
    parser.add_argument('--python-limit', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('runs/speed-benchmark.json'))
    args = parser.parse_args()
    if args.repeats < 1 or min(args.populations) < 1:
        parser.error('require positive population sizes and repeats')
    config = TrainingConfig(population=max(args.populations), elites=min(5, max(args.populations)-1),
                            generations=1, episodes=args.episodes, assessment_episodes=args.assessment_episodes,
                            workers=args.workers, batch_size=args.batch_size, seed=42)
    print(f'Initializing {config.population} independent genotypes...', flush=True)
    started = time.perf_counter()
    population = initialize_population(config)
    initialization = time.perf_counter()-started
    split = partitions(config.seed)
    # Compile / initialize libraries outside steady-state measurements.
    for backend in args.backends:
        if backend != 'python':
            evaluate_generation(population[:2], replace(config, population=2, elites=1, episodes=2,
                                                        assessment_episodes=2, backend=backend), 0, split)
    measurements = []
    for size in args.populations:
        for backend in args.backends:
            if backend == 'python' and size > args.python_limit:
                continue
            times = []
            for repeat in range(args.repeats):
                started = time.perf_counter()
                record, _ = evaluate_generation(population[:size], replace(config, population=size,
                                                       elites=min(5, size-1), backend=backend), 0, split)
                times.append(time.perf_counter()-started)
                print(f'{size:4} animals | {backend:6} | {times[-1]:.3f}s | repeat {repeat+1}/{args.repeats}', flush=True)
            measurements.append(dict(population=size, backend=backend, seconds=times,
                                     median_seconds=statistics.median(times), summary=record['summary']))
    result = dict(seed=config.seed, episodes=args.episodes, assessment_episodes=args.assessment_episodes,
                  workers=args.workers, batch_size=args.batch_size, initialization_seconds=initialization,
                  scope='whole generation evaluation including packing, reference RNG generation, transfers, and champion replay; excludes reproduction and checkpoint writes; warm compilation',
                  measurements=measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
