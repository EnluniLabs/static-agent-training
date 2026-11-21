"""Read-only structural summaries; these never enter selection or mutation."""
from collections import Counter
from . import serialization as json

from .initialize import depth, node_count


def population_structure(population, champion):
    convergence, term_nodes, term_depths = [], [], []
    for animal in population:
        incoming = Counter(animal.repertoire.successors.values())
        convergence.append(sum(count > 1 for count in incoming.values()))
        terms = [term for pair in animal.initial_pairs for term in pair]
        term_nodes.append(sum(node_count(term) for term in terms) / len(terms))
        term_depths.append(sum(depth(term) for term in terms) / len(terms))
    developed = json.loads(population[champion].development_json)
    return dict(mean_convergence_states=sum(convergence) / len(population),
                mean_seed_term_nodes=sum(term_nodes) / len(population),
                mean_seed_term_depth=sum(term_depths) / len(population),
                champion_development_stops=dict(Counter(group['status'] for group in developed['groups'])),
                champion_development_accounting=developed['accounting'])
