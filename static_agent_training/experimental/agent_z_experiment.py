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
from ..repertoire import freeze
import gzip
from .agent_z_environment import NumberLine
from .agent_z_kernels import NativeKernels, PythonKernels


def read_z_artifact(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as handle:
        return serialization.load(handle)


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


def evaluate(agent, tasks, *, seed, round_limit=64, learning='both', trace=False, return_weights=False):
    rates = dict(empirical_rate=0.05 if learning in ('both', 'empirical') else 0,
                 structural_rate=0.05 if learning in ('both', 'structural') else 0)
    if isinstance(agent.kernels, NativeKernels):
        return agent.kernels.evaluate(agent, tasks, random.Random(seed), round_limit=round_limit,
                                      trace=trace, return_weights=return_weights, **rates)
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
                **({'trace': records} if trace else {}),
                **({'embedding': agent.embedding, 'continuation': agent.continuation}
                   if return_weights else {}))


def sample_tasks(seed, count, goals):
    rng = random.Random(seed)
    tasks = []
    for _ in range(count):
        goal = rng.choice(goals)
        start = rng.choice([value for value in range(-20, 21) if value != goal])
        tasks.append((start, goal))
    return tasks


def run(*, seed=7, backend='native', generations=3, population=8, episodes=30,
        regions=2, learning='both', candidate=None, structure_preset='baseline', progress=None, directory=None, resume=False, initial_population=None, speciation=False):
    if min(generations, population, episodes) < 1 or not 1 <= regions <= 8:
        raise ValueError('positive generations, population and episodes required; regions must be 1–8')
    kernels = NativeKernels() if backend == 'native' else PythonKernels()
    settings = dict(seed=seed, generations=generations, population=population,
                    episodes=episodes, regions=regions, learning=learning, structure_preset=structure_preset)
    if speciation:
        settings['speciation'] = True
    if initial_population is not None:
        settings['initial_population'] = str(initial_population)
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

    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

    niches = None

    def checkpoint():
        if directory is not None:
            write_json(directory / 'checkpoint.json.gz', dict(
                schema_version='crt-agent-z-checkpoint-v1', settings=settings, history=history,
                **({'niches': niches.to_data()} if niches is not None else {}),
                rng=rng.getstate(), population=[
                    dict(individual=agent.to_data(), seeds=seeds, budgets=budgets, stops=stops)
                    for agent, seeds, budgets, stops in agents]))

    history, agents = [], []
    if speciation:
        from ..agent_z_speciation import Niches
    if resume:
        saved = read_z_artifact(directory / 'checkpoint.json.gz')
        if saved.get('schema_version') != 'crt-agent-z-checkpoint-v1':
            raise ValueError('Expected an Agent Z training checkpoint')
        if settings | {'generations': saved['settings']['generations']} != saved['settings']:
            raise ValueError('Resume must retain the saved Agent Z settings')
        if speciation:
            niches = Niches(population, saved['niches'])
        history = saved['history']
        version, state, gaussian = saved['rng']
        rng.setstate((version, tuple(state), gaussian))
        agents = [(AgentZ.from_data(row['individual'], kernels=kernels),
                   [tuple(map(freeze, pair)) for pair in row['seeds']], row['budgets'], row['stops'])
                  for row in saved['population']]
    else:
        if initial_population is not None:
            saved = read_z_artifact(initial_population)
            if saved.get('schema_version') != 'crt-agent-z-population-v1':
                raise ValueError('Choose a saved Agent Z population')
            for member in saved['members'][:population]:
                if 'seeds' not in member or 'budgets' not in member:
                    raise ValueError('This legacy Agent Z champion can be assessed but has no development seeds')
                seeds = [tuple(map(freeze, pair)) for pair in member['seeds']]
                budgets = member['budgets']
                agent = AgentZ.from_data(member['individual'], kernels=kernels)
                if agent.regions != regions:
                    raise ValueError('Agent Z population must use the requested number of regions')
                if any(len(term) < 2 for pair in seeds for term in pair):
                    raise ValueError('Saved Agent Z seed term is malformed')
                agents.append((agent, seeds, budgets, member.get('stops', ())))
        for _ in range(len(agents), population):
            seeds = [pair() for _ in range(config.initial_pairs)]
            budgets = [config.chain_budget] * len(seeds)
            agent, stops = build(seeds, budgets, rng, kernels, regions, config)
            agents.append((agent, seeds, budgets, stops))
        if speciation:
            niches = Niches(population)
            niches.seed(agent.structure for agent, *_ in agents)
        checkpoint()
    if len(history) >= generations:
        raise ValueError('Choose a target greater than the saved generation count')
    for generation in range(len(history), generations):
        tasks = sample_tasks(f'{seed}:train:{generation}', episodes, train_goals)
        results = [evaluate(agent, tasks, seed=f'{seed}:learn:{generation}', learning=learning)
                   for agent, _, _, _ in agents]
        ranks = sorted(range(population), key=lambda index: -results[index]['successes'])
        champion, _, _, stops = agents[ranks[0]]
        assessment = evaluate(champion, held_out, seed=f'{seed}:assessment', learning=learning)
        frozen = assessment if learning == 'off' else evaluate(
            champion, held_out, seed=f'{seed}:assessment', learning='off')
        history.append(dict(generation=generation + 1,
                            mean=sum(r['successes'] for r in results) / (episodes * population),
                            best=results[ranks[0]]['successes'] / episodes,
                            held_out=assessment['successes'] / episodes,
                            held_out_frozen=frozen['successes'] / episodes,
                            development_stops=stops, repertoire=len(champion.ids)))
        if niches is not None:
            groups = niches.group(ranks)
            history[-1]['species_count'] = sum(bool(group) for group in groups)
            history[-1]['species_sizes'] = [len(group) for group in groups if group]
        if progress is not None:
            progress(history[-1])
        if directory is not None:
            write_json(directory / 'snapshots' / f'{generation + 1:05d}.json.gz', champion.to_data())
            write_json(directory / 'summaries' / f'{generation + 1:05d}.json', history[-1])
            write_json(directory / 'best-population.json.gz', dict(
                schema_version='crt-agent-z-population-v1', name=directory.name,
                config=dict(model='agent-z', environment='original', sensory_partitions=regions),
                source_type='generation', members=[
                    dict(individual=agents[i][0].to_data(), generation=generation,
                         training_success=results[i]['successes'] / episodes,
                         seeds=agents[i][1], budgets=agents[i][2], stops=agents[i][3])
                    for i in ranks]))
        if generation + 1 == generations and directory is None:
            break
        def offspring(parent_record):
            parent, inherited_seeds, inherited_budgets, stops = parent_record
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
            return agent, seeds, budgets, stops

        if niches is None:
            parents = [agents[i] for i in ranks[:max(1, population // 4)]]
            agents = [parents[0]] + [offspring(rng.choice(parents)) for _ in range(population - 1)]
        else:
            next_agents, assignments = [], []
            for species, members in enumerate(niches.group(ranks)):
                if not members:
                    continue
                survivors = members[:max(1, len(members) // 2)]
                next_agents.extend(agents[i] for i in survivors)
                assignments.extend([species] * len(survivors))
                for _ in range(len(members) - len(survivors)):
                    parent = agents[rng.choice(survivors)]
                    child = offspring(parent)
                    next_agents.append(child)
                    assignments.append(species if child[0].structure is parent[0].structure else
                                       niches.classify(child[0].structure, parent=(parent[0].structure, species)))
            agents, niches.assignments = next_agents, assignments
        checkpoint()
    return dict(settings=settings, history=history, champion=champion.to_data(),
                seconds=perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('python', 'native'), default='native')
    parser.add_argument('--learning', choices=('both', 'off', 'empirical', 'structural'), default='both')
    parser.add_argument('--structure-preset', choices=('baseline', 'exploratory'), default='baseline')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--generations', type=int)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--regions', type=int, default=2)
    parser.add_argument('--speciation', action='store_true', help='Cull and reproduce within structural species')
    parser.add_argument('--initial-population', type=Path)
    parser.add_argument('--candidate', type=Path, help='assess a saved Agent Z champion')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--workspace-directory', type=Path)
    mode.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, help='save the result and champion')
    args = parser.parse_args()
    directory = args.workspace_directory or args.resume
    if args.resume:
        saved = read_z_artifact(directory / 'checkpoint.json.gz')
        if saved.get('schema_version') != 'crt-agent-z-checkpoint-v1':
            parser.error('Expected an Agent Z training checkpoint')
        target = args.generations
        for key, value in saved['settings'].items():
            setattr(args, key, value)
        args.generations = target if target is not None else args.generations
        args.backend = 'native'
    if args.generations is None:
        args.generations = 3
    candidate = None
    if args.candidate is not None:
        candidate = read_z_artifact(args.candidate)
        candidate = candidate.get('champion', candidate)
    print(f'Agent Z: number line -20 to 20, {args.structure_preset}, '
          f'{args.population} agents, {args.generations} generations, '
          f'{args.episodes} learning episodes per agent per generation', flush=True)

    def progress(row):
        print(f"Generation {row['generation']}/{args.generations} | "
              f"mean {row['mean']:.1%} | best {row['best']:.1%} | "
              f"held-out {row['held_out']:.1%}" +
              (f" | species {row['species_count']}" if 'species_count' in row else ''), flush=True)

    keys = ('seed', 'backend', 'generations', 'population', 'episodes', 'regions', 'learning', 'structure_preset', 'initial_population', 'speciation')
    settings = {key: getattr(args, key) for key in keys}

    def session_report(status):
        if directory is not None:
            import os
            write_json(directory / 'report.json', dict(schema_version='crt-agent-z-run-v1',
                       status=status, pid=os.getpid(), config=settings | {'environment': 'original', 'initial_population':
                                  str(args.initial_population) if args.initial_population is not None else None}))

    session_report('running')
    try:
        result = run(**settings, candidate=candidate, progress=progress,
                     directory=directory, resume=bool(args.resume))
    except KeyboardInterrupt:
        session_report('stopped')
        raise SystemExit(130)
    except Exception:
        session_report('failed')
        raise
    if directory is not None:
        write_json(directory / 'result.json.gz', result)
    session_report('completed')
    if args.output is not None:
        write_json(args.output, result)
    print(serialization.dumps({k: v for k, v in result.items() if k != 'champion'}, indent=2))


if __name__ == '__main__':
    main()
