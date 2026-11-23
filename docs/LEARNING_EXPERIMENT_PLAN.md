# Separating lifetime learning, selection, and inheritance

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

Discussion note, 2026-11-01. This is a plan for later analysis, not authorization
to launch experiments or change the active benchmark.

## Question

Does lifetime learning help because it changes which inherited genotypes
evolution selects, because the tested animal retains useful learned mappings,
or because learned mappings improve descendants when inherited?

The working conjecture is that learning changes the fitness landscape: selection
can favor genotypes that learn effectively without inheriting the resulting
weight changes. This is a Baldwin-effect interpretation, not an established
explanation of our results. Copying learned weights might instead preserve
adjustments specific to a parent's experience; that explanation also needs testing.

## Three evolutionary conditions

| Condition | Lifetime learning during evolution | Learned-weight inheritance |
|---|---|---|
| A: no lifetime learning | Off | Off |
| B: lifetime learning, original inheritance | On | Off |
| C: lifetime learning, learned inheritance | On | On |

In B, each animal starts from its original inherited mappings, learns during its
training lifetime, and receives training fitness after that interaction. Selected
parents reproduce from their original inherited mappings, followed by mutation.
In C, end-of-training learned embedding and output logits instead enter the
reproductive parent, including elites, before the same mutation rules.
No condition inherits assessment updates or transient learner state.

Hold initial genotypes, tasks, seeds, population size, generation count, mutation,
selection, environment, and interaction budgets fixed across matched runs.
Disable speciation. Keep both local update formulas unchanged. Select candidates
using training performance only, never held-out scores. Compare A versus B for
the total effect of learning during evolution; compare B versus C for inheritance.
Use independently repeated seed triplets before treating this as a unified
three-condition experiment; separate interrupted studies are not a complete
matched comparison.

## What the current endpoint includes

The existing final benchmark loads the selected candidate's saved learned
mappings and disables further learning. Frozen testing does not reset those
mappings to their original inherited values. Therefore a B-versus-A advantage
can reflect both changed evolutionary selection and retained lifetime adaptation.
It does not isolate the effect on selection alone.

The 20-generation B-versus-C pilot completed with an inconclusive seed-paired
interval. The longer inheritance study was stopped before all seeds completed.
The 200-generation A-versus-B study was still running when this note was written.
See [experiment records](EXPERIMENTS.md) for provenance and results; do not infer
a general inheritance penalty or learning benefit from one seed or a live mean.

## Additional test: original versus learned mappings of the same candidate

For each saved candidate in each condition, keep its exact structure fixed and
construct independent evaluation copies:

1. **Original mappings, frozen:** use the candidate's inherited genotype mappings
   at the start of its evaluated lifetime, with all learning disabled.
2. **Saved learned mappings, frozen:** use its end-of-training mapping snapshot,
   with all further learning disabled. This reproduces the existing endpoint.
3. **Original mappings, fresh adaptation:** start from the same original mappings,
   permit lifetime updates on a separate, matched adaptation block, then freeze
   mappings for evaluation on fresh trials.

For C, "original" already includes learning inherited from ancestors; it means
before this candidate's own lifetime, not before all ancestral learning.
Reset pending expectations, traversal, and revision state consistently when
constructing evaluation copies. Preserve mapping snapshots without modifying
the saved candidate. Verify structure hashes and record snapshot identities.

Use identical final tasks and paired sampling seeds across copies. Keep final
tasks out of adaptation and candidate selection. Repeat fresh adaptation blocks
to avoid attributing a fortunate experience sequence to reliable adaptability.
For A, saved learned and original mappings should coincide; use this as a control.

Within a candidate, comparison 2 minus 1 measures the retained effect of its saved
lifetime updates. Comparison 3 minus 1 measures the benefit of fresh adaptation.
Across A and B, comparison 1 examines evolved starting performance; comparison 3
examines evolved adaptability. A remaining difference is consistent with learning
having changed evolutionary outcomes, but does not uniquely identify a structural
cause: inherited mappings, structure, and selection trajectories can all differ.
Do not add these differences as if they were independent causal contributions.

## Deployment and transfer follow-ups

The fresh-adaptation test also addresses the proposal to evolve with learning off
and enable learning only in deployment. Cross evolutionary learning off/on with
deployment adaptation off/on, with learned-weight inheritance off throughout.
Separate adaptation interactions from frozen evaluation and report performance
against interaction count, as well as final success and prediction quality.

Extend this to the fixed-structure tests in [future work](../FUTURE_WORK.md):

- Transfer across task families, comparing prior experience with learning the new
  family from the original mappings; measure retention on earlier families.
- Expand the number line from [-20, 20] to [-30, 30], preserve existing mappings,
  and initialize new observation, goal, and expectation entries consistently.
  Compare adaptation on/off within the old range, new regions, and crossings.
- Explore without goals before later task adaptation, with controls for mere
  interaction and predictable stationary behavior.

## Reporting

Predeclare the final endpoint and candidate selection rule. Report each seed and
the paired mean difference with uncertainty across independent seeds, including
random-movement controls. Distinguish population means, selected-candidate scores,
and frozen versus adapting performance. Retain full configurations, task hashes,
candidate IDs, mapping provenance, and structure hashes so results are replayable.
Final success is the primary result here; computation time is a separate measure.
