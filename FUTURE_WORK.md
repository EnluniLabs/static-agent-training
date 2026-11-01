# Future work

Recorded 2026-11-01. These are plans and hypotheses, not instructions to change
an active experiment. Speciation stays off for the next studies unless explicitly
included as a treatment.

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
