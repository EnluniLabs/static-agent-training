# Three central experiments: coherence, continuous learning, and exploratory knowledge

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

Status: research plan — **not implemented or executed**
Date: 2026-11-02

This file consolidates three major experiments for the static-agent research
program. These are distinct questions, not three versions of one benchmark.
See [the detailed coherence test specification](COHERENT_VS_RANDOM_TEST_SPEC.md)
and [implementation boundaries](IMPLEMENTATION.md).

## Shared architectural premise

The animal's lifetime operation is **always on**. It may act, traverse its
inherited repertoire, and **choose when or whether to emit expectations**.
An emitted expectation is criticized when a corresponding observation becomes
available; empirical and structural learning then operate continuously.
There is no intrinsic train/inference switch.

A goal can be supplied **explicitly**, conveyed **indirectly** by environmental
conditions, or **not supplied at all**. Neither expectation emission nor
ordinary action requires a goal token. Goal-free operation is permitted, but
effective curiosity or exploration is **not guaranteed** by the interface.

Frozen-weight assessments remain useful as *counterfactual measurement
instruments*, not as the intended deployed behavior. Prefer read-only
observation of continuously learning animals and isolated cloned-state probes
when comparisons require a common starting point. Never let diagnostic
evaluation affect evolutionary selection or the live animal's state.

## Test 1 — Computational coherence versus arbitrary anchoring

**Question:** Does a repertoire developed through lambda-calculus constraints
support more efficient learning than an arbitrary-node repertoire?

**Design:** Compare COHERENT and RANDOM_CHAIN animals using the **same**
input/output interface, expectation emission, empirical learning, structural
learning, task sequences, evaluation budget, and evolutionary selection.
Random anchors are nodes with arbitrary fixed successor links; they are **not**
an anchor-free neural-network control. Start with the existing number line.
Add topology-/size-matched random controls to isolate what accounts for
differences. A graph-isomorphic relabeling is a sanity check, not an
independent structural treatment.

**Measure:** held-out task success versus interactions, prediction quality,
structural-update effects, nodes/edges exercised, coverage of connected
substructures, search/mutation efficiency, and compute cost. Evaluate
structural-learning-off as a **diagnostic ablation on isolated copies**, not
the animal's normal operating mode.

**Interpret carefully:** If the learner rarely exercises successor structure,
a null result may mean the environment/learner did not expose the hypothesis.
Any advantage is initially attributable to the *distribution of graphs
produced by the generator*, not to invisible lambda labels after reduction.

**Implementation detail:** [Full test specification](COHERENT_VS_RANDOM_TEST_SPEC.md).

## Test 2 — Always-on lifetime learning and long-horizon adaptation

**Question:** Can a single animal continue learning across an extended
lifetime, retaining useful knowledge while adapting to changing conditions,
without distinct training and inference phases?

**Design:**
1. Instantiate an animal with its inherited repertoire and mappings.
2. Run a continuous sequence of interactions and episodes; preserve learned
   mapping weights and revision state across ordinary environment resets.
   Keep empirical and structural criticism active throughout.
3. Introduce predeclared environmental changes at known interaction counts:
   e.g. different goal distributions, movement constraints, or a compatible
   changed world. Clearly separate changes the existing signal vocabulary
   can represent from those requiring the future generalized interface.
4. Track recovery after changes and retention on earlier conditions.
   For controlled comparison, branch **copies of the same learned state**
   into no-learning or structural-off probes; do not freeze the main animal.
5. Include matched animals with different inherited structures to see
   whether coherence affects long-horizon retention and adaptability.

**Measure:** online rolling success and prediction loss, adaptation/recovery
interactions after a change, retention on prior tasks, catastrophic
interference, update counts, prediction-emission frequency, repertoire
utilization, and state continuity. Report time series, not only end scores.

**Critical checks:** Some animals may emit very few expectations, preventing
meaningful empirical criticism. Measure this explicitly rather than assuming
learning is occurring. Distinguish uninterrupted learned-state persistence
from checkpoints, replay, and ordinary episode resets.

**Success criterion:** Demonstrable adaptation during uninterrupted learning,
with quantified retention and recovery. No claim of indefinite stability
from a finite run.

## Test 3 — Goal-free exploration followed by an unforeseen goal

**Question:** Does an animal acquire useful, transferable knowledge by
exploring *before it knows what it will be asked to accomplish*?

**Design:**
1. Place the animal in an unfamiliar environment **without a goal input**.
   Allow it to choose actions and expectations, and keep lifetime learning on.
   Record the environmental observations available to it.
2. After a fixed exploration budget, introduce a **new explicit goal** that
   was not supplied during exploration. Keep learning enabled during goal
   pursuit; this is not a train/test mode switch.
3. Compare with animals that receive the same goal but **no prior exploration**,
   animals with equal time/interaction budgets spent in a matched
   noninformative control environment, and (where feasible) animals whose
   exploration histories are matched but structurally different.
4. Repeat with multiple withheld goals and environments; randomize goal
   assignment **after** exploration or precommit the goal set without
   exposing it to the animal. Use paired seeds and equal goal-pursuit budgets.
5. Extend to **indirect goals** conveyed by cues (e.g. food/resource signals)
   rather than a target token. Treat this as a separate condition from
   completely goal-free exploration.

**Measure:** subsequent goal-acquisition success and interactions-to-success,
advantage over no-exploration and noninformative-exposure controls, predictive
accuracy on previously encountered states, state/transition coverage,
diversity of actions, voluntary expectation emission, and whether previously
learned mappings are reused.

**Important confounds:** Prior exploration also supplies additional
interactions and weight updates. Match exposure budgets and isolate useful
environmental information from mere time or stochastic movement. Do not
mistake broad random wandering for systematic exploration. If no expectations
are emitted, prediction-based learning may be minimal even if the animal
moves extensively.

**Implementation consideration:** The current number-line interface is
goal-conditioned in varying-goal mode. A valid goal-free condition must
**remove** the goal input rather than pass a fake goal that reveals the
future task. The existing zero-target interface is not automatically a
goal-free benchmark if its environment or selection pressures implicitly
encode zero as the target. Ensure the no-goal adapter exposes only legitimate
observations and that the learner does not receive a hidden goal hint.
The generalized signal interface is useful future work, but a carefully
scoped goal-free experiment may be possible before it.

**Success criterion:** A reproducible advantage on *previously undisclosed*
goals attributable to informative prior exploration, with continuous
learning in both exploration and pursuit.

## Relationship and recommended order

- **First:** instrument prediction emission, empirical/structural updates,
  node/edge use, and uninterrupted state persistence. All three tests
  require these observables.
- **Then:** execute the coherence-vs-random pilot on the current number line,
  with the controls in the detailed specification.
- **In parallel or next:** validate uninterrupted lifetime adaptation on
  changing conditions using the existing interface.
- **Next:** build a genuinely goal-free exploration condition and test
  exploration-before-goal. Repeat the latter across coherent/random
  repertoires if feasible: this connects Tests 1 and 3 directly.
- **Later:** generalize the input/output interface to unseen signal types
  and expand to environments where reusable relational knowledge matters.

## Cross-cutting validity rules

- Keep the animal's choice of when to emit expectations intact.
- Do not insert a mandatory prediction schedule or task-reward gradient.
- Do not turn off lifetime learning in the main operating trajectory.
- Distinguish online measurements from frozen cloned-state diagnostics.
- Use independent seeds, predeclared budgets, common task lists where
  meaningful, and untouched final evaluation conditions.
- Preserve the existing distinction between evolutionary selection on task
  success and local lifetime learning from prediction/structural criticism.
- Report failures, missing predictions, insufficient repertoire use, and
  ambiguous null results rather than claiming a decisive answer.

These experiments address three complementary claims: **whether coherent
structure helps**, **whether learning can accumulate continuously**, and
**whether knowledge acquired without a known objective is useful later**.
