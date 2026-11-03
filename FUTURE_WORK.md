# Future work

For commands and current local runs, see [USE.md](USE.md) and `./project runs`.
Status paragraphs below are dated research records, not a live process registry.
The GitHub coherence-control and signal-interface proposal sections were restored
locally on 2026-11-02; implemented prototype behavior and remaining gaps are in
the [interface design](docs/AGNOSTIC_INTERFACE.md) and
[audit](docs/AGNOSTIC_INTERFACE_AUDIT.md).

Recorded 2026-11-01. These are plans and hypotheses, not instructions to change
an active experiment. Speciation stays off for the next studies unless explicitly
included as a treatment.

See the [learning experiment plan](docs/LEARNING_EXPERIMENT_PLAN.md) for the
three evolutionary conditions, original-versus-learned mapping controls, and
deployment-only adaptation follow-up.

## Central experiment: coherent versus arbitrary-node repertoires

Status: **primary proposed hypothesis test** (discussion 2026-11-02).
This takes priority conceptually over generalized sensory interfaces and future
repertoire expansion. It is a plan, not authorization to interrupt or change
ongoing experiments.

**Question:** Does an inherited repertoire generated under independently defined
lambda-calculus reduction rules guide lifetime learning better than an otherwise
comparable repertoire of arbitrary nodes and successor links?

The coherent animal develops pair states and certified successors by lambda
reduction. The control uses arbitrary node identities and assigned successor
links (simple node chains as a first baseline). After development, both are
traversed as nodes and links: lambda expressions are not reduced during
lifetime learning. Both retain the **same** sensory interface, expectation and
movement outputs, empirical-error rule, delayed structural-error rule,
initialization protocol, evolution/fitness procedure, and evaluation tasks.
The random control still receives structural-error updates toward its own
assigned successors; it is **not** an empirical-only or anchor-free animal.
Keep successor links fixed during a lifetime in both arms.

The distinction is the *generative restrictions on admissible repertoires*:
lambda-derived links must follow a computational rule independent of the
environment and fitness, while arbitrary links need not. Merely relabeling
the same graph with random node IDs is mathematically equivalent under
matched parameters and random choices, and cannot establish an advantage.
A simple arbitrary linked-list baseline is a useful first test, but its
topology may differ from the lambda-derived repertoire; if an effect appears,
follow up with random graphs matched for node/edge counts, chain-length
distribution, branching, merges, cycles, and terminal/successor availability
where feasible. Report remaining mismatches rather than calling them
coherence effects.

### Is the current number-line task informative enough?

Begin in the existing hard-wall number-line environment without redesigning
the agent. Its simple regularity might be enough to expose a difference, or
animals may solve it using only a few nodes or local associations. Before
interpreting null or positive results, instrument the degree to which
inherited structure is actually exercised:

- Number and fraction of unique repertoire nodes selected, by episode and run.
- Number and fraction of unique successor edges exercised, repeated edge use,
  and coverage of chains, merges, branches, or cycles where present.
- Eligibility, count, and magnitude of structural-error updates; separate
  these from empirical-error updates.
- Effect of structural updates on later mappings, expectations, and behavior,
  using a matched structural-learning-disabled control where practical.
- Relationship between utilization, structural-update activity, prediction
  accuracy, adaptation speed, and held-out task success.

**Utilization is not dependence.** Visiting many nodes does not show that
structural guidance helps, and frequent structural updates do not prove
benefit. A learner may exploit only isolated edges and never engage the
higher-order organization of its repertoire. Check whether the present local
successor correction can produce effects spanning multiple connected states,
rather than assuming it does.

### Comparison and interpretation

Use multiple independent repertoire generations and evolutionary seeds;
match population sizes, experience budgets, selection criteria, initial
parameter distributions, and held-out trials. Compare learning curves and
sample efficiency, frozen prediction accuracy, final task success, retention,
and transfer if supported by the unchanged interface. Include empirical-only
(structural correction off) as a diagnostic, but keep coherent vs arbitrary
nodes with **both** learning mechanisms as the central contrast. Separate
effects of repertoire generation from differences in graph size, topology,
compute, and evolutionary selection.

If the number-line task engages little structure, a similar performance
result is weak evidence about the central hypothesis; consider a task that
requires reuse of more structural relationships, with its requirements
specified before testing. If the coherent animal wins, establish whether
the advantage comes from lambda-specific generative constraints or from
ordinary graph statistics. If neither wins despite demonstrably substantial
structural dependence, that is meaningful negative evidence under the tested
conditions. Do not infer that lambda-derived structure necessarily models
physical laws, or that its origin alone changes the behavior of an otherwise
identical unlabeled graph.

The proposed role of coherent structure is **scaffolding acquisition**:
it may guide representations during learning without needing to remain
computationally important after those representations are acquired. Future
repertoire expansion could add new structure when existing learned
interpretations prove insufficient; that extension is not part of this test.

## Lifetime learning on/off benchmark

Status: launched on 2026-11-01 as `lifetime-learning-20261101-153325`: 256 animals
per arm, 20 generations, seeds 42/43/44, 100 training and 200 assessment episodes,
and 2,000 final shared held-out trials per candidate. Both arms have speciation off.
Stopped at the user’s request before all seeds completed; this is not a completed benchmark.

- Compare lifetime learning enabled against both local learning rules disabled.
- Match initial genotypes, population sizes, task sequences, mutation settings,
  generation counts, and evaluation budgets across paired seeds.
- Use uniform valid start/target pairs and the same hard-wall environment in
  both arms. Keep speciation disabled.
- Freeze weights during final assessment in both arms. Select candidates using
  training performance only, then assess on fresh shared held-out trials.
- Report per-seed effects, uncertainty across seeds, and random-movement baselines.
- Preserve the current rule that offspring inherit genotype weights rather than
  the parent's learned lifetime weights, so inheritance is not another variable.

## Collections of top performers

Save collections rather than only one generation champion, to investigate which
structures emerge consistently within and across independent runs.

- Make collection size and capture frequency configurable.
- Preserve complete inherited genotypes, developed repertoires and successor
  edges, learned lifetime mappings, and the settings needed to replay assessment.
- Record run, seed, generation, individual identity, training and held-out scores,
  development budgets, ancestry, and mutation history or references to its logs.
- Define collection selection and tie-breaking using training results only;
  held-out scores remain analysis data.
- Retain provenance when the same structure appears in several individuals or
  runs. Distinguish exact duplicates, inherited persistence, and independent
  emergence rather than counting every saved copy as a separate discovery.
- Analyze recurring seed terms, pair/edge patterns, cycles, mergers, repertoire
  sizes, and their relationships to performance and lifetime learning.
- Separate the inherited structure from learned physical mappings when comparing
  performers. Repeated structures alone do not establish that they cause success.

## Error-update conjecture

User conjecture, retained for later testing:

> Structural error may apply during any transition. Empirical error may apply
> only to continuation distributions during expectation.

Before implementing an experiment, specify what “structural error,”
“continuation distributions,” and “during expectation” mean operationally. In
particular, distinguish expressing an expectation from receiving feedback that
tests it; this note does not settle that timing question or assume that
“continuation distributions” means the current output-logit table.

Possible controlled comparison:

- Current implementation as the reference.
- Broader structural-error eligibility alone.
- Restricted empirical-error updates alone.
- Both proposed changes together.

Hold the rest of training fixed across matched seeds. Record where each rule is
eligible, the evidence available at that transition, which distributions change,
and update counts and magnitudes. Evaluate task success, prediction quality,
structural/mapping consistency, and computational cost. Treat this as a hypothesis,
not an established improvement.


## Follow-up controls from the bias audit

See [the audit](docs/BIAS_AUDIT.md) for the measurements and their limits.

- Make stronger movement controls (including random nonzero movement) standard
  alongside the uniform-nine baseline; the read-only audit already computes them.
- Repeat goal-input permutation/removal controls across independent runs and on
  trained versus held-out goal values, with per-goal and per-distance results.
- Separate an immediate same-genotype learning-on/off counterfactual from the full
  matched evolutionary learning-on/off study. The full matched study was stopped before completion; the immediate same-genotype study remains planned.
- Fix design choices before evaluating a fresh final test partition, rather than
  repeatedly adapting the implementation to the same visible held-out curve.


## Inherited lifetime learning benchmark

Authorized on 2026-11-01 after stopping the learning on/off study. Compare learned
weight inheritance off/on with lifetime learning on and speciation off in both
arms: 256 animals, 20 generations, seeds 42/43/44, 100 training and 200 frozen
assessment episodes, and 2,000 fresh final held-out trials. Inheritance includes
end-of-training embedding/output weights, including elite parents, before normal
mutation. It excludes assessment updates and transient learner state. Current error
rules remain unchanged. The 20-generation study completed: mean effect +1.12
percentage points, seed-paired 95% interval [-2.63, +4.86]. The subsequent
200-generation inheritance comparison was stopped before all seeds completed.
Neither result establishes an aggregate benefit or harm from inheritance; see
[experiment records](docs/EXPERIMENTS.md).

## Fixed-structure lifetime learning and transfer

Hypothesis: lifetime learning will be more useful across a broader range of tasks,
allowing an animal to reuse existing structure rather than build new structure.
This is planned work, not authorization to change the current benchmark.

- Freeze each selected animal's seed terms, development budgets, repertoire, and
  successor graph. Disable reproduction, structural mutation, and redevelopment
  during this test; only the existing lifetime mapping updates may adapt.
- Expose animals to several task families, then introduce a new family. Define
  those families and compatible observation/action interfaces before testing.
- Compare lifetime learning on/off from identical starting genotypes and mappings,
  using matched task sequences, random seeds, and interaction budgets. Repeat
  across independently obtained animals rather than relying on one champion.
- Compare prior-task experience against a fresh copy learning the new family from
  the original mappings. This separates transfer from simply learning anew.
- Measure adaptation speed, success after a fixed interaction budget, and retention
  on earlier families. Verify structure hashes remain unchanged throughout.
- Keep adaptation trials separate from frozen evaluation trials. Reserve unseen
  tasks for final evaluation and do not use their scores to select animals.

Success would demonstrate adaptation or transfer with fixed structure; it would
not by itself establish that any particular structural feature caused the benefit.


## Longer learning on/off follow-up

Now running as `lifetime-learning-long-20261101-161824`: 200 generations, 256 animals per
arm, three matched seeds, lifetime learning off/on during evolution, learned-weight
inheritance off, and speciation off. The longer inheritance test was stopped
before all seeds completed. Deployment-only adaptation is not part of this run;
retain the fixed-structure transfer test above for that question.

## Expanded number line with fixed structure

Once candidates reliably solve the original puzzle, expand the number line from
[-20, 20] to [-30, 30] without evolutionary retraining. Test whether lifetime
learning can adapt existing structure to the larger world. This is planned work;
it does not change or interrupt the current benchmark.

- Freeze the selected candidate's seed terms, development budgets, repertoire,
  and successor graph. Disable reproduction, mutation, and redevelopment, and
  verify that structure hashes remain unchanged.
- Preserve existing mappings. Initialize mappings for new position and goal
  tokens using the original initialization rule and matched seeds. Specify how
  the expectation-output vocabulary expands, preserving existing logits and
  documenting any resulting normalization changes. New mappings are interface
  expansion, not newly evolved internal structure.
- Compare identical expanded candidates with lifetime learning enabled versus
  disabled, using matched tasks and interaction budgets. Record immediate
  performance before adaptation as well as improvement through interaction.
- Keep movement rules and hard walls consistent; define trial sampling and move
  budgets in advance so longer journeys do not confound the comparison.
- Report success separately within the original range, in the newly added
  regions, and on journeys crossing between them. Recheck the original task to
  measure retention or forgetting.
- Allow updates only during adaptation interactions. Freeze mappings for
  separate evaluation trials, reserve fresh tasks for final evaluation, and
  repeat across independently obtained candidates.

Improvement over the learning-disabled copy would support adaptation through
existing structure. This involves learning from new experience, even though no
new structure is evolved. A later 2D grid test would additionally require an
explicit coordinate and movement interface and should be treated separately.

## Goal-free exploration and self-correction

The expectation-and-correction loop can operate without an externally assigned
goal: act, express an expectation, observe the outcome, and apply the existing
local updates. Local learning uses prediction and structural consistency;
evolutionary selection currently uses externally measured task success. Treat
these as separate mechanisms rather than assuming local learning requires a goal.

- Let saved animals explore without target-location inputs or task rewards,
  holding their structure fixed while allowing lifetime mapping updates.
- Specify the goal-free input interface and initialization before comparison;
  do not encode a hidden target through replacement inputs.
- Compare exploration experience against identical animals without that
  experience, then test later task adaptation with matched interaction budgets.
  Include a learning-disabled exploration control to distinguish local updates
  from other effects of interaction or transient state.
- Measure prediction quality, local update activity, position coverage, movement
  diversity, and later transfer. Track stationary or repetitive behavior: accurate
  predictions in a narrow, predictable region need not imply useful exploration.
- Separate fixed-structure exploration from any future evolution under a
  goal-free fitness criterion. Define that criterion explicitly if introduced;
  expectation correction alone does not specify reproductive selection.

## Food finding without a supplied goal location

Replace the explicit target with food at an undisclosed location. Provide current
position and a local food-encounter signal, with no remote food-location input or
distance hint. Evolve animals using food collected or a defined survival measure,
while retaining the existing local expectation-and-correction rules as the
lifetime learning mechanism. Food-based fitness is distinct from a new reward
gradient; introducing the latter would require a separate experiment.

- Define food detection, consumption, replenishment, episode termination, energy
  costs (if any), and how food observations enter the existing interface before
  running the experiment.
- Compare food fixed across generations (which may favor an innate route), food
  varied between lifetimes (general search), and food stable within a lifetime
  then relocated (discovery, return, and adaptation). Use separate conditions so
  their interpretations remain clear.
- Test whether existing traversal state and mapping updates retain enough
  information to exploit a discovery. Neither prediction accuracy nor a food
  signal alone guarantees attraction to food or memory of its location.
- Compare lifetime learning on/off with matched initial animals, food placements,
  and interaction budgets. Keep learned-weight inheritance off unless explicitly
  introduced as another treatment.
- Report discovery time, food collection or survival, return behavior, exploration
  coverage, and response to relocation. Include random-movement controls and
  reserve unseen food placements for evaluation, without selecting on those scores.

These are future experiments, not instructions to change the active benchmark.

## Proposed type-agnostic signal interface and generalization filter

Status: **design hypothesis / future work only** (discussion recorded 2026-11-02).
Do not change the current agent, checkpoints, backends, or active comparisons
without a separately scoped implementation and validation plan.

### Motivation and boundary

The present interface allocates one observation-embedding row per categorical
observation and one expectation-output logit per possible predicted observation.
Moving from the number-line world [-20, 20] to [-30, 30] therefore requires
expanding both the input vocabulary and prediction vocabulary before the same
animal can even be tested. This confounds genuine transfer with human-provided
interface adaptation. The aim is **type-agnostic and robust**, not an unlimited
universal representation: supported unseen stimuli must pass through the same
fixed-capacity interface without allocating a new learned row or changing the
repertoire. Position, goal, letter, and other environment-specific *meanings*
must not be supplied to the agent by the interface.

### One shared mapping, not an embedding stacked on another embedding

Candidate minimal replacement: encode each supported input signal into a
fixed-width bit vector x, then directly produce logits over repertoire entries
using a shared learned map:

    entry_logit_j(x) = bias_j + sum_i x_i * W[i,j]

The existing observation-ID lookup would be **replaced**, not preceded by a
second trainable encoder. The matrix W associates signal components with
repertoire states; all inputs use the same parameters. Position and goal remain
distinct roles in the task interface, but neither requires a vocabulary row per
possible value. The representation must include role/channel framing if the
same byte sequence can appear as position versus goal; such framing is not a
computed distance, goal direction, or semantic feature. Specify signed-integer
encoding (e.g. fixed-width two's complement), byte order, length/framing, and
supported range explicitly. Eight-bit ASCII tokens are a convenient bounded
test case; signed integers spanning [-30, 30] also fit in eight bits. Variable
length inputs, overflow, collisions, and wider-world support require separate
design decisions. Finite capacity is acceptable; silently changing dimensions
on transfer is not.

Shared component weights mean that updating the interpretation of one signal
can change the interpretation of another signal that activates some of the same
components. This permits **inductive transfer**, but does not establish semantic
similarity: adjacent numbers can have very different binary patterns (15 vs 16),
and nearby bit patterns need not describe related situations. Binary encoding
is an inductive bias, not a built-in theory of number, alphabet, or physical
closeness. The animal may learn to exploit or suppress component overlap based
on its experience. Compare ordinary binary, alternative fixed encodings, and
randomized identity-preserving codes to measure encoding dependence. Do not
claim that any shared-bit update necessarily improves generalization.

### Expectations and empirical refutation must use the same interface

Replacing only the observation lookup is insufficient. The current output
vocabulary has one expectation token per possible next position; an animal
could observe an unseen position but still could not explicitly predict it.
Develop an output/expectation representation that can express a supported
signal without allocating a new categorical output per value. Keep movement
commands separate and unchanged for the first experiment. Candidate approaches
include bitwise or sequential signal emission; each has different timing,
normalization, decoding, and credit-assignment consequences, so no choice is
approved yet. The agent must not receive a privileged numerical decoder.

A prediction of one exact signal and an observation of another are objectively
different, regardless of binary distance. There is also a proposed *graded*
alternative: let empirical credence change according to compatibility between
the predicted and observed signals in the animal's **learned** representation,
rather than a hand-coded numerical or Hamming-distance metric. This is an open
design choice, **not** a decision to implement graded matching. Exact identity
and learned functional equivalence should be distinguishable in traces and
evaluation. A system that makes all signals compatible could evade informative
refutation; whether task survival/evolutionary selection sufficiently penalizes
that collapse is a hypothesis, not a guarantee. If empirical and structural
correction both use compatibility, specify whether they share the same
definition and how it is learned without adding another embedding subsystem.

### Preserve the two local learning mechanisms

The empirical update currently revises expectation-token logits after explicit
predictions are tested. The delayed structural update checks the compatibility
of a revised pair's supported expectation with its fixed repertoire successor,
and strengthens an observation-to-successor mapping if below threshold. A
shared bitwise mapping would require new, local credit assignment to component
weights (for example a sparse update on active bits), while retaining the
separation between empirical evidence and structural compatibility. Such a
rule would propagate some effects across inputs, including potentially harmful
interference. It must be specified and tested, not assumed equivalent to the
existing per-token update. Neither rule should silently change repertoire
edges or inherited structure during lifetime learning.

### Why retain a coherent lambda-derived repertoire?

The signal interface supplies candidate interpretations; it must **not**
independently solve the full explanation problem. The research hypothesis is
that empirical pressure anchors predictions to reality while the inherited
coherent repertoire constrains how interpretations of successive experiences
can fit together into a consistent story. Formally, if q_t is the animal's
distribution over internal states after a history of observations and T is the
fixed successor operator, a possible structural-consistency criterion is
agreement between q_(t+1) and the distribution obtained by propagating q_t
through T. This is a conceptual target, not the implemented loss or a proven
guarantee of global coherence.

**Coherent ingredients are not automatically coherent explanations.** The
current lambda-derived repertoire certifies local successor edges; a
probabilistic mapping to those states may fail to preserve relations across an
entire observed history. Existing delayed corrections provide local pressure
toward compatibility, not a proof that all learned interpretations constitute
a globally coherent explanation. Moreover, a random fixed graph also imposes
constraints. To attribute any advantage specifically to lambda-derived
coherence, identify and test the computational/compositional relationships
preserved by the repertoire, not just its graph size or the fact that its
individual edges are valid. The broader conjecture—that coherent physical
dynamics admit coherent representations, and empirical plus structural
learning can discover them—remains a research hypothesis. Hidden state,
stochasticity, limited observations, finite repertoires, and local optimization
all limit what can be guaranteed.

### Proposed experiments and controls

1. Preserve the existing implementation as the baseline. Implement the new
   interface as an explicit experimental variant with a fixed shape, saved
   encoding specification, deterministic initialization, and traceable updates.
2. Train in [-20, 20], then evaluate and adapt in [-30, 30] **without resizing
   either sensory or expectation parameters**, changing repertoire, or
   reinitializing learned mappings. Keep the movement set and boundary
   semantics explicit; report old-region, new-region, crossing, and retention
   performance separately.
3. Test unfamiliar symbols (e.g. ASCII tokens) with fresh signal identities,
   not merely human-visible relabeling of the same internal token IDs. A simple
   alphabetical relabeling has no inherent ordinal meaning to this agent; a
   genuine new signal encoding is what makes transfer testable.
4. Compare transfer from prior experience against an otherwise matched fresh
   animal using the same interface and repertoire, with equal adaptation
   interactions, frozen evaluations, and matched seeds. Distinguish prior
   learned mappings from a repertoire-only transfer.
5. Isolate the source of generalization: coherent lambda-derived repertoire,
   randomized repertoire matched for size and relevant graph statistics, and
   a direct associative mapping baseline of comparable capacity. Compare
   empirical-only, structural-only where well-defined, both updates, and no
   lifetime learning. Evaluate not just task success but prediction quality,
   compositional transfer, mapping collapse, consistency across trajectories,
   forgetting, and computation/parameter cost.
6. Repeat under alternative encodings and across independently evolved animals.
   No result should be attributed to coherent structure if it is explained by
   the shared input map, encoding bias, task geometry, or extra compute.

The central design requirement is **a stable, type-agnostic signal boundary
that permits learned generalization without pre-imposing the relationships
the coherent repertoire is supposed to explain**. The simplest bitwise
shared-weight map is a candidate starting point, not a settled architecture.
