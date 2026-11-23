"""Reproducible Agent Z evolution and held-out assessment."""
import argparse
from dataclasses import asdict
from pathlib import Path
import random
from time import perf_counter

from .. import serialization
from ..evolution import partitions
from ..experiment import write_json
from ..initialize import generate_term
from ..mutation import mutate_seed_pairs
from ..structure_profiles import apply_structure_arguments
from ..evolution import TrainingConfig
from .agent_z import AgentZ, Structure
from .agent_z_environment import NumberLine
from .agent_z_kernels import NativeKernels, PythonKernels


def build(seeds, budgets, rng, kernels, regions, config, *, parent=None,
          inherited_structure=None, mutate_weights=True):
    structure, stops = inherited_structure or Structure.develop(
        seeds, budget=budgets, seed=rng.getrandbits(64),
        max_nodes=config.development.max_nodes,
        max_transitions=config.development.max_transitions)

    def weights(key, name, size):
        if parent is not None and key in parent.index:
            row = getattr(parent, name)[parent.index[key]]
            return ([value + rng.gauss(0, config.mutation.weight_stddev) for value in row]
                    if mutate_weights else row)
        return [rng.gauss(0, 0.5) for _ in range(size)]

    agent = AgentZ(structure, [weights(key, 'embedding', 9) for key in structure.pairs],
                   [weights(key, 'continuation', 8) for key in structure.pairs],
                   regions=regions, kernels=kernels)
    return agent, stops


def evaluate(agent, tasks, *, seed, round_limit=64, learning='both', trace=False):
    rates = dict(empirical_rate=0.05 if learning in ('both', 'empirical') else 0,
                 structural_rate=0.05 if learning in ('both', 'structural') else 0)
    # A lifetime can learn across episodes. The inherited genome is untouched.
    agent = AgentZ(agent.structure, agent.embedding, agent.continuation,
                   regions=agent.regions, kernels=agent.kernels)
    rng, world = random.Random(seed), NumberLine()
    successes = rounds = 0
    records = []
    for task in tasks:
        chains = agent.activate(world.reset(task), rng)
        for _ in range(round_limit):
            if not chains or world.done:
                break
            chains, feedback, emissions = agent.step(chains, world, rng, **rates)
            rounds += 1
            if trace:
                records.append([asdict(emission) | {
                    'signal': emission.signal.hex(),
                    'observation': (None if emission.observation is None else
                                    emission.observation.hex())} for emission in emissions])
            if feedback.done:
                break
        successes += world.position == world.goal
    return dict(successes=successes, episodes=len(tasks), rounds=rounds,
                **({'trace': records} if trace else {}))


def sample_tasks(seed, count, goals):
    rng = random.Random(seed)
    tasks = []
    for _ in range(count):
        goal = rng.choice(goals)
        start = rng.choice([value for value in range(-20, 21) if value != goal])
        tasks.append((start, goal))
    return tasks


def run(*, seed=7, backend='python', generations=3, population=8, episodes=30,
        regions=2, learning='both', candidate=None, structure_preset='baseline', progress=None):
    if min(generations, population, episodes) < 1 or not 1 <= regions <= 8:
        raise ValueError('positive generations, population and episodes required; regions must be 1–8')
    kernels = NativeKernels() if backend == 'native' else PythonKernels()
    settings = dict(seed=seed, generations=generations, population=population,
                    episodes=episodes, regions=regions, learning=learning, structure_preset=structure_preset)
    rng = random.Random(seed)
    config = apply_structure_arguments(TrainingConfig(), {'structure_preset': structure_preset})
    mutation = config.mutation
    init = mutation.generator
    split = partitions(seed, include_zero=True)
    train_goals, test_goals = split['training'], split['assessment']

    def pair():
        return generate_term(rng, init), generate_term(rng, init)

    start = perf_counter()
    held_out = sample_tasks(f'{seed}:test', episodes, test_goals)
    if candidate is not None:
        agent = AgentZ.from_data(candidate, kernels=kernels)
        assessment = evaluate(agent, held_out, seed=f'{seed}:assessment', learning=learning)
        return dict(settings=settings, assessment=assessment, seconds=perf_counter() - start)

    agents = []
    for _ in range(population):
        seeds = [pair() for _ in range(config.initial_pairs)]
        budgets = [config.chain_budget] * len(seeds)
        agent, stops = build(seeds, budgets, rng, kernels, regions, config)
        agents.append((agent, seeds, budgets, stops))
    history = []
    for generation in range(generations):
        tasks = sample_tasks(f'{seed}:train:{generation}', episodes, train_goals)
        results = [evaluate(agent, tasks, seed=f'{seed}:learn:{generation}', learning=learning)
                   for agent, _, _, _ in agents]
        ranks = sorted(range(population), key=lambda index: -results[index]['successes'])
        champion, _, _, stops = agents[ranks[0]]
        assessment = evaluate(champion, held_out, seed=f'{seed}:assessment', learning=learning)
        frozen = evaluate(champion, held_out, seed=f'{seed}:assessment', learning='off')
        history.append(dict(generation=generation + 1,
                            mean=sum(r['successes'] for r in results) / (episodes * population),
                            best=results[ranks[0]]['successes'] / episodes,
                            held_out=assessment['successes'] / episodes,
                            held_out_frozen=frozen['successes'] / episodes,
                            development_stops=stops, repertoire=len(champion.ids)))
        if progress is not None:
            progress(history[-1])
        if generation + 1 == generations:
            break
        parents = [agents[i] for i in ranks[:max(1, population // 4)]]
        agents = [parents[0]]
        while len(agents) < population:
            parent, inherited_seeds, inherited_budgets, stops = rng.choice(parents)
            operators = {'subtree': mutation.subtree_probability, 'term': mutation.term_probability,
                         'insert': mutation.insertion_probability, 'delete': mutation.deletion_probability}
            selected = {name: rng.random() < probability for name, probability in operators.items()}
            seeds, budgets, _ = mutate_seed_pairs(
                inherited_seeds, inherited_budgets, config=mutation, selected=selected,
                seeds={name: rng.getrandbits(64) for name in operators})
            unchanged = seeds == inherited_seeds and budgets == inherited_budgets
            agent, stops = build(seeds, budgets, rng, kernels, regions, config, parent=parent,
                                 inherited_structure=(parent.structure, stops) if unchanged else None,
                                 mutate_weights=rng.random() < mutation.weight_probability)
            agents.append((agent, seeds, budgets, stops))
    return dict(settings=settings, history=history, champion=champion.to_data(),
                seconds=perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('python', 'native'), default='python')
    parser.add_argument('--learning', choices=('both', 'off', 'empirical', 'structural'), default='both')
    parser.add_argument('--structure-preset', choices=('baseline', 'exploratory'), default='baseline')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--regions', type=int, default=2)
    parser.add_argument('--candidate', type=Path, help='assess a saved Agent Z champion')
    parser.add_argument('--output', type=Path, help='save the result and champion')
    args = parser.parse_args()
    candidate = None
    if args.candidate is not None:
        candidate = serialization.loads(args.candidate.read_bytes())
        candidate = candidate.get('champion', candidate)
    print(f'Agent Z: number line -20 to 20, {args.structure_preset}, '
          f'{args.population} agents, {args.generations} generations, '
          f'{args.episodes} learning episodes per agent per generation', flush=True)

    def progress(row):
        print(f"Generation {row['generation']}/{args.generations} | "
              f"mean {row['mean']:.1%} | best {row['best']:.1%} | "
              f"held-out {row['held_out']:.1%}", flush=True)

    result = run(**{key: value for key, value in vars(args).items()
                    if key not in ('output', 'candidate')}, candidate=candidate, progress=progress)
    if args.output is not None:
        write_json(args.output, result)
    print(serialization.dumps({k: v for k, v in result.items() if k != 'champion'}, indent=2))


if __name__ == '__main__':
    main()
