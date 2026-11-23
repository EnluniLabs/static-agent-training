"""Fixed-representative structural speciation for Agent Z."""
from .speciation import Structure, _jaccard


def distance(left, right):
    return (2 * _jaccard(left.pairs, right.pairs) + _jaccard(left.edges, right.edges)) / 3


class Niches:
    """Assign developed structures by pair and directed-link overlap."""

    threshold = .5

    def __init__(self, population, saved=None):
        self.limit = min(64, max(1, population // 2))
        self.representatives = [] if saved is None else [
            Structure.from_data(row) for row in saved['representatives']]
        self.assignments = [] if saved is None else saved['assignments']
        self.cache = {}

    def seed(self, structures):
        # Leave niches available for innovations after initialization.
        initial_limit = min(8, max(1, self.limit // 2))
        self.assignments = [self.classify(structure, limit=initial_limit) for structure in structures]
        self.cache.clear()

    def classify(self, structure, *, parent=None, limit=None):
        digest, descriptor = structure.fingerprint
        if parent is not None and digest == parent[0].fingerprint[0]:
            return parent[1]
        if digest in self.cache:
            return self.cache[digest]

        separation, species = min(((distance(descriptor, representative), index)
                                 for index, representative in enumerate(self.representatives)),
                                default=(float('inf'), None))
        if separation > self.threshold and len(self.representatives) < (self.limit if limit is None else limit):
            species = len(self.representatives)
            self.representatives.append(descriptor)
            # A new representative can change the closest match for past structures.
            self.cache.clear()
        if len(self.cache) >= 4096:
            self.cache.pop(next(iter(self.cache)))
        self.cache[digest] = species
        return species

    def group(self, ranked):
        groups = [[] for _ in self.representatives]
        for index in ranked:
            groups[self.assignments[index]].append(index)
        return groups

    def to_data(self):
        return dict(representatives=[rep.to_data() for rep in self.representatives],
                    assignments=self.assignments)
