# Static agent training

Current milestone: a working lambda engine and training environment interface. See
[ENVIRONMENT_INTERFACE.md](ENVIRONMENT_INTERFACE.md) for the contract and
`static_agent_training/environment.py` for the independent number-line world.

An existing initializer creates the mathematical seeds of a set of explanations.
Each seed is an ordered pair `(A, B)` of closed untyped lambda terms. Chain
development attempts `C = NF(B A)` and visit `(B, C)`.
These seeds are not yet developed explanations or physical interpretations.

Run with Python 3.10 or later; no dependencies are required:

```sh
python3 -m static_agent_training.initialize --seed 42 --count 4 --output explanations.json
python3 -m static_agent_training.engine explanations.json --budget 64 --seed 42 --output developed.json
python3 -m unittest discover -s tests
```

The JSON records configuration, independent per-chain generation seeds, raw
De Bruijn ASTs, readable terms, and stable SHA-256 term/pair IDs. Duplicate
pairs are retained with distinct chain provenance; sampling is independent,
not sampling without replacement. No inherited terms are beta-normalized.

The provisional generator chooses uniformly from the fitting I, K, and S
combinators with probability 0.5. Otherwise it recursively chooses uniformly
among feasible variable, abstraction, and application constructors, with a
uniform valid bound-variable index and uniform feasible application node-budget
split. Depth includes the root; node limits are upper bounds, not target sizes.
Defaults are depth 6 and 31 nodes. Configure these using `--max-depth`,
`--max-nodes`, and `--combinator-probability`. This distribution is an explicit
implementation choice implementing the report's small-combinator bias.

The engine develops multiple starting pairs with separate contraction budgets,
stochastic reduction, and budget pooling at shared ordered pairs. Fixed-repertoire
physical mappings, agent traversal, and local lifetime learning are implemented.
Evolutionary mutation operators and the population training loop follow later.

The stochastic reducer is available as `static_agent_training.reduction.normalize`.
It accepts a closed tuple AST, a beta-contraction budget, a seed, and an AST-size
cap. It returns `NORMAL`, `BUDGET`, or `SIZE`, the final term, charged steps,
remaining budget, and selected redex paths. It reduces beneath abstractions and
uses capture-avoiding De Bruijn substitution. A failed attempt does not certify
divergence.

Use `static_agent_training.engine.run(pairs, budgets, seed=42)` from Python.
`pairs` is a list of `(left_ast, right_ast)` seeds; `budgets` is a list of
individual budgets, or one integer assigned to every starting pair. JSON ASTs
from the initializer are also accepted.

Execution is logically parallel: each scheduler round permits one contraction
per active execution. This is deterministic interleaving, not CPU parallelism.
A shared single term does not merge executions. Reaching an exact ordered pair
already visited by another execution pools their remaining budgets and retains
all source indices. Arrival times need not match. A later arrival forwards its
budget to the shared execution's current position, retaining any partial
reduction, and resumes it if budget exhaustion had stopped it. Work spent before
the merge is never refunded; pooled downstream work is charged once. The earlier
execution's RNG stream continues after a merge.

Cycles are recorded and continue spending budget. `max_transitions` (default
1000) bounds each surviving execution, including zero-contraction cycles;
`max_nodes` (default 256) bounds intermediate applications. Budget pooling does
not override `SIZE` or `DEPTH` stops. Results include initial ASTs and budgets,
configuration, source provenance, contraction paths, normal-form edges, merges,
cycles, terminal statuses, partial terms, and conserved budget accounting.
Identical inputs, seed, and configuration reproduce the complete event log.


Every development result now includes a `repertoire` export: unique immutable
pairs and certified successor IDs. Only completed normalizations create edges;
partial reductions do not. Load it without evaluating any lambda terms:

```python
import json
from static_agent_training.repertoire import Repertoire
from static_agent_training.mappings import InheritedMappings
from static_agent_training.agent import Agent, run_episode
from static_agent_training.environment import NumberLine

result = json.load(open("developed.json"))
repertoire = Repertoire.from_data(result["repertoire"])
env = NumberLine(encoding_seed=0, episode_seed=1)
inherited = InheritedMappings.initialize(
    repertoire,
    [s.token for s in env.observation_tokens],
    [o.token for o in env.output_tokens],
    seed=42,
)
agent = Agent(repertoire, inherited.lifetime(), embedding_seed=2, output_seed=3)
transitions = run_episode(agent, env)
```

Embedding logits have one row per observation and one column per unique pair.
Output logits have one row per unique pair and one column per output symbol.
Each row is softmax-normalized when used. Initial logits use independent,
keyed Gaussian draws with standard deviation 0.1 (a configurable implementation
choice). Sampling has separate embedding and output RNG streams. Terminal pairs
remain eligible entries; an empty repertoire or missing successor terminates
without inventing an output.

At each observation the agent samples an existing entry pair, follows a stored
edge, and samples an output at the successor. The environment interprets the
output. Expectations continue traversal; actions trigger re-embedding whenever
a new observation arrives, including a repeated symbol. The environment owns
action and output limits. The runner passes no numeric decoding to the agent.

The repertoire, inherited embedding logits, and inherited output logits are
immutable. Each lifetime receives independent mutable copies of the logits;
episode resets preserve those copies. The runner applies local learning by default. Use
`run_episode(agent, env, learn=False)` for assessment with both update rules
disabled, or construct `Agent(..., learning=False)` for a fixed baseline.

Reduction occurs only during initial development or mutation redevelopment.
To reconcile mappings after external redevelopment, call
`InheritedMappings.initialize(new_repertoire, observations, outputs,
previous=inherited, seed=...)`. Surviving pair IDs keep their inherited logits,
new IDs receive seeded logits, and removed IDs disappear. Pass inherited weights,
not learned lifetime weights. No mutation operators are implemented yet.
The repertoire, mapping, and lifetime-agent modules never import the reducer.


Local learning follows report sections 6.1–6.4. Every explicitly emitted and
tested expectation updates only its source pair's expectation-token logits,
using conditional expectation cross entropy and learning rate 0.05. Correct
predictions also update; action logits and other pairs do not. Expired
expectations receive no empirical update. With multiple pending expectations,
each is scored and updated sequentially in emission order against the same
next observation. Each loss uses the current logits; emission snapshots remain
in the event records. The total action/expectation probability split is allowed
to change through softmax renormalization, as agreed.

Empirical updates mark only the emitting pair as revised. No immediate embedding
update or global scan occurs. When a revised pair is subsequently encountered,
the most-supported expectation's observation embedding is checked against the
stored successor. If its probability is below 0.1, that observation's embedding
logit for the successor increases by 0.05. A missing successor skips the check.
Revised pairs remain eligible for checks on future encounters; corrections can
repeat until compatible. Episodes clear pending commitments and active traversal,
but retain lifetime weights and revision history. Configurable rates and threshold
are provided by `learning.LearningConfig`, passed as `Agent(...,
learning_config=...)`.

Agent events record expectation IDs, physical symbols, origin and emitting pairs,
traversal paths, emission logit snapshots, timestamps, empirical loss and updates,
expiration, compatibility checks, and local embedding conflicts linked to their
preceding empirical revision. Neither update rule changes the fixed repertoire
or inherited weights, or invokes lambda reduction.
