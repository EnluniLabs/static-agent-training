# Implementation and experiment boundaries

This describes the implementation as of 2026-11-01. It separates current behavior
from historical experiment settings and hypotheses in [future work](../FUTURE_WORK.md).

## What the agent receives

In varying-goal experiments, the inputs are the displayed current number and target
number, represented by separate categorical tokens. Their independently learned
embedding rows are added before sampling an entry pair. No numeric subtraction,
relative-distance feature, route, suggested move, or reward gradient is supplied.
The numeral labels do not give this lookup-table architecture an arithmetic model.

The fixed repertoire is a directed graph of closed lambda-term pairs. Each output
requires following a stored successor edge. The output distribution jointly covers
nine movement tokens and 41 next-observation expectations. Missing continuations
end an episode. An animal can emit expectations before choosing a movement.

## What learning changes

Empirical learning updates only the expectation-token logits of the pair that
emitted an explicitly tested expectation. It minimizes conditional prediction
cross-entropy, using the actual next observation. It does not directly optimize
movement logits or receive a movement-reward gradient. Because all output tokens
share a sampling softmax, changing expectation logits can nevertheless change the
total probability of movement versus expectation.

Empirical revisions make a pair eligible for a delayed compatibility check on a
later encounter. The strongest supported prediction identifies an observation
embedding row; that row's successor logit is increased if compatibility is below
the threshold. Goal context participates in this check, but its separate embedding
row is not updated by that local correction. Goal rows can change through inherited
weight mutation and selection. The error-update conjecture has **not** been implemented.

Lifetime updates never alter the repertoire. Episodes retain learned weights and
revision history while clearing pending expectations and active traversal. Assessment
uses a separate frozen copy of learned weights and independent RNG streams.
Offspring use inherited weights, not learned lifetime weights.

## Evolution and selection

Seed pairs, budgets, and inherited mappings form a genotype. Structural mutation
can regenerate the repertoire; surviving pair identities retain inherited mapping
cells. Weight perturbation mutates inherited logits. Fitness is the fraction of
training episodes that reach the goal; it is not prediction accuracy.

Selection and elite preservation use training fitness exclusively. The default
has no speciation or crossover. Optional speciation groups structural repertoires,
shares fitness, and confines offspring tournaments to niches; its global elite
budget remains equal to the control. This is the custom treatment documented in
the README, not full NEAT.

The legacy `Species` class names an inherited genotype, not a speciation cluster.

## Environment and task sampling

Current new-run defaults are uniform valid ordered start/goal pairs, all 41 positions
participating in a 31/10 train/assessment partition, hard walls, and speciation off.
Sampling is with replacement. Each allowed goal has equal probability within its
partition, but different goals and distances need not have equal physical difficulty.
Zero is included in the varying-goal partition; fixed-zero tasks exclude starting at zero.

Movement choices are −5, −3, −2, −1, 0, +1, +2, +3, +5. An overshoot stops at −20
or +20 and consumes a movement. There are at most 12 movements and at most 16 outputs
per observation. A sixteenth expectation ends the episode; a movement as output 16
is valid. Observations and empirical feedback reflect the actual bounded position.

Normal-distance sampling remains optional. It samples distance first, then a
feasible pair. This avoids multiplying distance probability by pair counts, but
does not guarantee uniform exposure to target values near the center and ends.

Training tasks change each generation and are shared across animals. The assessment
list is fixed across animals and generations. Held-out *start and goal values* are
disjoint from their training values. Training episodes may still pass through
held-out positions, so this is not a guarantee that all held-out states are unseen.

## Backends and reproducibility

The original Python evaluator produces full event traces. Optional compiled CPU
and CUDA evaluators execute independent lifetimes while preserving episode order,
local-update order, separate RNG streams, and immutable inherited state. They use
Python-derived MT19937 states and double precision. They perform no lambda reduction.
Every accelerated generation's champion is replayed through Python and checked
before its detailed trace is saved. Tests compare sampled behavior, update counts,
and learned weights; floating-point loss/logit roundoff is tolerated.

Checkpoints atomically save the next inherited population, configuration, partitions,
history, and optional speciation representatives. Resume restores saved settings.
Old configs without boundary settings use historical out-of-bounds termination;
without distance settings they use historical uniform-pair sampling; without
zero-inclusion settings they keep the old 40-position partition. No old run is
silently converted to new rules on resume.

A checkpoint does not contain learned mappings for every evaluated individual.
Detailed learned mappings are currently retained for each generation's champion.
The collection of multiple top performers is planned, not implemented.

## Scope of changes during this session

Earlier work added goal conditioning to the agent's input embedding and compatibility
check, as well as population evolution, mutation, assessment, and acceleration.
The later hard-wall and uniform-task changes altered the environment, sampling,
baseline calculations, and corresponding evaluator simulations; they did not add a
new agent decision rule or change the local-learning objectives. Documentation-only
updates and read-only audits do not change an already running experiment.

Live playground attempts always use hard walls. Recorded scores retain their saved
rules, which the UI labels; exported historical replays retain original semantics.
Raw run folders, checkpoints, caches, and virtual environments are excluded from Git.
