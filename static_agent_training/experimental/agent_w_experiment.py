"""Agent W number-line evolution with native lifetime evaluation."""
import argparse
from pathlib import Path
import random

from ..agent_z_speciation import Niches
from ..evolution import TrainingConfig, partitions
from ..experiment import write_json
from ..initialize import generate_term
from ..mutation import mutate_seed_pairs
from ..structure_profiles import apply_structure_arguments
from .agent_w import AgentW
from .agent_w_environment import NumberLine
from .agent_z import Structure
from .agent_z_experiment import sample_tasks


def run(directory, *, seed=7, population=1000, generations=1000, episodes=500, regions=2,
        structure_preset='baseline', speciation=False, action_rate=.05, structural_rate=.05,
        model='agent-w', initial_nodes=None):
    if min(population, generations, episodes) < 1 or not 1 <= regions <= 8:
        raise ValueError('positive population, generations and episodes and 1–8 regions required')
    from math import isfinite
    if any(not isfinite(rate) or rate < 0 for rate in (action_rate, structural_rate)):
        raise ValueError('learning rates must be finite and nonnegative')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'checkpoint.json.gz').exists():
        raise ValueError('choose a fresh output directory')

    settings = dict(seed=seed, population=population, generations=generations, episodes=episodes,
                    regions=regions, structure_preset=structure_preset, speciation=speciation,
                    action_rate=action_rate, structural_rate=structural_rate, backend='native')
    config = apply_structure_arguments(TrainingConfig(), {'structure_preset': structure_preset})
    if model == 'agent-d':
        from .agent_d import AgentD, RandomGraph
        default_nodes = 15 if structure_preset == 'baseline' else 55
        initial_nodes = default_nodes if initial_nodes is None else initial_nodes
        if type(initial_nodes) is not int or not 1 <= initial_nodes <= config.development.max_nodes:
            raise ValueError('initial node count must be positive and within the graph capacity')
        model_type = AgentD
        settings.update(model=model, initial_nodes=initial_nodes)
    elif model == 'agent-w' and initial_nodes is None:
        model_type = AgentW
    else:
        raise ValueError('invalid model or initial node count')
    mutation = config.mutation
    actions = NumberLine().action_signals
    rng = random.Random(seed)
    goals = partitions(seed, include_zero=True)
    held_out = sample_tasks(f'{seed}:test', episodes, goals['assessment'])
    rates = dict(action_rate=action_rate, structural_rate=structural_rate)
    frozen = dict(action_rate=0, structural_rate=0)

    def build(seeds, budgets, *, parent=None, existing=None, perturb=False):
        structure = existing if existing is not None else Structure.develop(
            seeds, budget=budgets, seed=rng.getrandbits(64),
            max_nodes=config.development.max_nodes,
            max_transitions=config.development.max_transitions)[0]

        def weights(key, name, width):
            if parent is None or key not in parent.index:
                return [rng.gauss(0, .5) for _ in range(width)]
            row = getattr(parent, name)[parent.index[key]]
            return ([value + rng.gauss(0, mutation.weight_stddev) for value in row]
                    if perturb else row)

        ids = model_type.node_ids(structure)
        agent = model_type(structure, [weights(key, 'embedding', 9) for key in ids],
                           [weights(key, 'continuation', len(actions)) for key in ids],
                           actions=actions, regions=regions)
        return agent, seeds, budgets

    def offspring(record):
        parent, old_seeds, old_budgets = record
        if model == 'agent-d':
            graph = parent.structure.mutate(rng, mutation, config.development.max_nodes)
            return build(None, None, parent=parent, existing=graph,
                         perturb=rng.random() < mutation.weight_probability)
        probabilities = dict(subtree=mutation.subtree_probability, term=mutation.term_probability,
                             insert=mutation.insertion_probability, delete=mutation.deletion_probability)
        selected = {name: rng.random() < probability for name, probability in probabilities.items()}
        seeds, budgets, _ = mutate_seed_pairs(
            old_seeds, old_budgets, config=mutation, selected=selected,
            seeds={name: rng.getrandbits(64) for name in probabilities})
        existing = parent.structure if seeds == old_seeds and budgets == old_budgets else None
        return build(seeds, budgets, parent=parent, existing=existing,
                     perturb=rng.random() < mutation.weight_probability)

    agents = []
    for _ in range(population):
        if model == 'agent-d':
            agents.append(build(None, None, existing=RandomGraph.generate(initial_nodes, rng)))
            continue
        seeds = [tuple(generate_term(rng, mutation.generator) for _ in range(2))
                 for _ in range(config.initial_pairs)]
        agents.append(build(seeds, [config.chain_budget] * len(seeds)))

    niches = Niches(population) if speciation else None
    if niches is not None:
        niches.seed(agent.structure for agent, _, _ in agents)

    history = []

    def save_checkpoint():
        write_json(directory / 'checkpoint.json.gz', dict(
            schema_version=f'crt-{model}-checkpoint-v1', settings=settings, history=history,
            rng=rng.getstate(), niches=niches.to_data() if niches is not None else None,
            population=[dict(individual=agent.to_data(), seeds=seeds, budgets=budgets)
                        for agent, seeds, budgets in agents]))

    save_checkpoint()
    for generation in range(generations):
        tasks = sample_tasks(f'{seed}:train:{generation}', episodes, goals['training'])
        seed_label = f'{seed}:learn:{generation}'
        scores = [agent.evaluate(tasks, seed=seed_label, **rates)['successes'] for agent, _, _ in agents]
        ranks = sorted(range(population), key=lambda i: -scores[i])
        champion = agents[ranks[0]][0]
        evaluation = champion.evaluate(held_out, seed=f'{seed}:assessment', **rates)
        fixed = champion.evaluate(held_out, seed=f'{seed}:assessment', **frozen) if any(rates.values()) else evaluation
        row = dict(generation=generation + 1, mean=sum(scores) / (episodes * population),
                   best=scores[ranks[0]] / episodes, held_out=evaluation['successes'] / episodes,
                   held_out_frozen=fixed['successes'] / episodes,
                   actions_per_episode=evaluation['actions'] / episodes, repertoire=len(champion.ids))
        groups = niches.group(ranks) if niches is not None else None
        if groups is not None:
            row['species_sizes'] = [len(group) for group in groups if group]
        history.append(row)
        write_json(directory / 'snapshots' / f'{generation + 1:05}.json.gz', champion.to_data())
        write_json(directory / 'summaries' / f'{generation + 1:05}.json', row)
        if groups is None:
            parents = [agents[i] for i in ranks[:max(1, population // 4)]]
            agents = [parents[0]] + [offspring(rng.choice(parents)) for _ in range(population - 1)]
        else:
            next_agents, assignments = [], []
            for species, members in enumerate(groups):
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
        save_checkpoint()
        print(f"Generation {generation + 1}/{generations} | mean {row['mean']:.1%} | "
              f"best {row['best']:.1%} | held-out {row['held_out']:.1%} | "
              f"frozen {row['held_out_frozen']:.1%}" +
              (f" | species {len(row['species_sizes'])}" if groups is not None else ''), flush=True)
    write_json(directory / 'result.json.gz', dict(settings=settings, history=history,
                                                  champion=champion.to_data()))


def main(model='agent-w'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    for name, default in (('seed', 7), ('population', 1000), ('generations', 1000),
                          ('episodes', 500), ('regions', 2)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--speciation', action='store_true')
    parser.add_argument('--structure-preset', choices=('baseline', 'exploratory'), default='baseline')
    parser.add_argument('--action-rate', type=float, default=.05)
    parser.add_argument('--structural-rate', type=float, default=.05)
    if model == 'agent-d':
        parser.add_argument('--initial-nodes', type=int)
    options = vars(parser.parse_args())
    directory = options.pop('output')
    print(f'{model} native CPU | number line -20 to +20 | {options}', flush=True)
    run(directory, model=model, **options)


if __name__ == '__main__':
    main()
