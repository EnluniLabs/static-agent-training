"""Generate inherited explanation seeds; no reduction or physical mapping yet."""

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random


# Immutable De Bruijn ASTs: ("var", index), ("abs", body), ("app", fn, arg).
def canonical_bytes(term):
    encoded = bytearray()
    pending = [term]
    while pending:
        node = pending.pop()
        tag = node[0]
        if tag == "var":
            encoded.append(ord("v"))
            value = node[1]
            while value >= 128:
                encoded.append((value & 127) | 128)
                value >>= 7
            encoded.append(value)
        elif tag == "abs":
            encoded.append(ord("l"))
            pending.append(node[1])
        elif tag == "app":
            encoded.append(ord("a"))
            pending.extend((node[2], node[1]))
        else:
            raise ValueError(f"Unknown node: {tag}")
    return bytes(encoded)


def term_id(term):
    return hashlib.sha256(b"crt-term-v1" + canonical_bytes(term)).hexdigest()


def pair_id(left, right):
    return hashlib.sha256(
        b"crt-pair-v1" + bytes.fromhex(term_id(left)) + bytes.fromhex(term_id(right))
    ).hexdigest()


def node_count(term):
    count = 0
    pending = [term]
    while pending:
        node = pending.pop()
        count += 1
        if node[0] != "var":
            pending.extend(reversed(node[1:]))
    return count


def depth(term):
    maximum = 0
    pending = [(term, 1)]
    while pending:
        node, level = pending.pop()
        maximum = max(maximum, level)
        if node[0] != "var":
            pending.extend((child, level + 1) for child in reversed(node[1:]))
    return maximum


def is_closed(term, binders=0):
    pending = [(term, binders)]
    while pending:
        node, bound = pending.pop()
        if node[0] == "var":
            if not 0 <= node[1] < bound:
                return False
        else:
            pending.extend((child, bound + (node[0] == "abs"))
                           for child in reversed(node[1:]))
    return True


def display(term):
    pieces = []
    pending = [term]
    while pending:
        node = pending.pop()
        if isinstance(node, str):
            pieces.append(node)
        elif node[0] == "var":
            pieces.append(str(node[1]))
        elif node[0] == "abs":
            pieces.append("λ.")
            pending.append(node[1])
        else:
            pieces.append("(")
            pending.extend((")", node[2], " ", node[1]))
    return "".join(pieces)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    count: int = 4
    max_depth: int = 6
    max_nodes: int = 31
    combinator_probability: float = 0.5

    def validate(self):
        if self.count < 1:
            raise ValueError("count must be positive")
        if self.max_depth < 2 or self.max_nodes < 2:
            raise ValueError("closed terms require max_depth and max_nodes >= 2")
        if not 0 <= self.combinator_probability <= 1:
            raise ValueError("combinator_probability must be between 0 and 1")


I = ("abs", ("var", 0))
K = ("abs", ("abs", ("var", 1)))
S = ("abs", ("abs", ("abs", ("app", ("app", ("var", 2), ("var", 0)), ("app", ("var", 1), ("var", 0))))))


def generate_term(rng, config, *, binders=0):
    library = [t for t in (I, K, S) if depth(t) <= config.max_depth and node_count(t) <= config.max_nodes]
    if library and rng.random() < config.combinator_probability:
        return rng.choice(library)

    def build(binders, levels, nodes):
        if levels == 1 or nodes == 1:
            return ("var", rng.randrange(binders))
        choices = ["var", "abs"] if binders else ["abs"]
        # A closed child needs at least two nodes and two levels.
        minimum = 1 if binders else 2
        if nodes >= 1 + 2 * minimum and levels >= (2 if binders else 3):
            choices.append("app")
        kind = rng.choice(choices)
        if kind == "var":
            return ("var", rng.randrange(binders))
        if kind == "abs":
            return ("abs", build(binders + 1, levels - 1, nodes - 1))
        left_nodes = rng.randint(minimum, nodes - 1 - minimum)
        return ("app", build(binders, levels - 1, left_nodes), build(binders, levels - 1, nodes - 1 - left_nodes))

    return build(binders, config.max_depth, config.max_nodes)


def initialize(config):
    config.validate()
    pairs = []
    for index in range(config.count):
        # Independent per-chain streams keep existing seeds stable as count grows.
        chain_seed = int.from_bytes(hashlib.sha256(f"crt-init-v1:{config.seed}:{index}".encode()).digest(), "big")
        rng = random.Random(chain_seed)
        left, right = generate_term(rng, config), generate_term(rng, config)
        pairs.append({
            "chain_id": index, "generation_seed": chain_seed,
            "pair_id": pair_id(left, right),
            "left": {"term_id": term_id(left), "ast": left, "display": display(left)},
            "right": {"term_id": term_id(right), "ast": right, "display": display(right)},
        })
    return {"schema_version": "crt-initialization-v1", "generator_version": "closed-mixture-v1",
            "config": asdict(config), "initial_pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=31)
    parser.add_argument("--combinator-probability", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = initialize(Config(**{k: v for k, v in vars(args).items() if k != "output"}))
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
