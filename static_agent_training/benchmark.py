"""Compare a frozen exported animal with uniform random movement on shared tasks."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from .agent import Agent, run_episode
from .candidates import load_saved_candidate
from .environment import NumberLine, Output
from .baseline import random_success, random_task_mean
from .genome import Species
from .seeds import derive
from .tasks import sample_tasks


def interval(successes, count):
    """Wilson 95% interval for a binomial success rate."""
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count))/denominator
    width = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denominator
    return [center-width, center+width]


def compare(directory, *, episodes=2000, seed=20261101):
    if type(episodes) is not int or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    directory = Path(directory)
    bundle = load_saved_candidate(directory)
    destination = directory / 'demo'
    destination.mkdir(exist_ok=True)
    saved = json.loads((directory / 'config.json').read_text())
    config = saved['config']
    split = saved['partitions']['assessment']
    conditioned = bundle.get('varying_goals', False)
    distribution = config.get('distance_distribution', 'uniform')
    if distribution == 'normal':
        tasks = sample_tasks(derive(seed, 'benchmark-tasks'), episodes, starts=split,
                             goals=split if conditioned else (0,),
                             mean=config.get('distance_mean', 10.0), stddev=config.get('distance_stddev', 5.0))
    else:
        task_rng = random.Random(derive(seed, 'benchmark-tasks'))
        tasks = []
        for _ in range(episodes):
            start = task_rng.choice(split)
            goal = task_rng.choice([g for g in split if g != start]) if conditioned else 0
            tasks.append((start, goal))
    if bundle.get('model') == 'senses':
        from .experimental.sensory_evolution import benchmark
        return benchmark(directory, bundle, tasks, seed=seed)
    species = Species.from_data(bundle['individual'])
    maps = species.inherited.lifetime()
    maps.embedding = {k: dict(v) for k, v in bundle['learned_embedding'].items()}
    maps.output = {k: dict(v) for k, v in bundle['learned_output'].items()}
    agent = Agent(species.repertoire, maps, learning=False,
                  embedding_seed=derive(seed, 'benchmark-animal-embedding'),
                  output_seed=derive(seed, 'benchmark-animal-output'))
    animal_world = NumberLine(encoding_seed=bundle['encoding_seed'], goal_conditioned=conditioned, boundary_mode=config.get('boundary_mode','terminate'))
    random_world = NumberLine(encoding_seed=bundle['encoding_seed'], goal_conditioned=conditioned, boundary_mode=config.get('boundary_mode','terminate'))
    moves = tuple(random_world.diagnostic_state()['move_decoding'])
    move_rng = random.Random(derive(seed, 'benchmark-random-moves'))
    outcomes = []
    for start, goal in tasks:
        final = run_episode(agent, animal_world, start=start, goal=goal, learn=False)[-1]
        agent.events.clear()
        transition = random_world.reset(start=start, goal=goal)
        while not transition.done:
            transition = random_world.emit(Output(move_rng.choice(moves)), source='random')
        outcomes.append(dict(start=start, goal=goal, animal_success=final.success,
                             random_success=transition.success, animal_reason=final.reason,
                             random_reason=transition.reason))
    result = dict(schema_version='crt-benchmark-v1', seed=seed, episodes=episodes,
                  generation=bundle['generation'], genotype_id=species.genotype_id,
                  task_source='fresh draws from the saved held-out position partition; equal starts and goals excluded',
                  distance_distribution=distribution,
                  distance_mean=config.get('distance_mean', 10.0) if distribution == 'normal' else None,
                  distance_stddev=config.get('distance_stddev', 5.0) if distribution == 'normal' else None,
                  random_policy='uniform over nine movement buttons including zero; no expectations; same 12-move limit',
                  boundary_mode=config.get('boundary_mode','terminate'),
                  exact_random_success=random_task_mean([s for s,g in tasks], [g for s,g in tasks],boundary_mode=config.get('boundary_mode','terminate')),
                  outcomes=outcomes)
    for name in ('animal', 'random'):
        successes = sum(row[name + '_success'] for row in outcomes)
        result[name] = dict(successes=successes, success_rate=successes/episodes,
                            interval95=interval(successes, episodes),
                            reasons=dict(Counter(row[name + '_reason'] for row in outcomes)))
    differences = [int(row['animal_success']) - int(row['random_success']) for row in outcomes]
    mean = sum(differences)/episodes
    se = math.sqrt(sum((d-mean)**2 for d in differences)/(episodes*(episodes-1))) if episodes > 1 else 0
    result['paired_difference'] = dict(percentage_points=100*mean, interval95=[100*(mean-1.96*se),100*(mean+1.96*se)])
    (destination / 'benchmark.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    result = compare(args.run, episodes=args.episodes, seed=args.seed)
    for name in ('animal', 'random'):
        row=result[name]
        print(f"{name}: {row['success_rate']:.2%} ({row['successes']}/{result['episodes']}); 95% interval {row['interval95'][0]:.2%}–{row['interval95'][1]:.2%}")
    print(f"Exact random expectation on these tasks: {result['exact_random_success']:.2%}")
    print(f"Saved: {args.run / 'demo' / 'benchmark.json'}")


if __name__ == '__main__':
    main()
