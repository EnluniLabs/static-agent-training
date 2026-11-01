# Future work

Recorded 2026-11-01. These are plans and hypotheses, not instructions to change
an active experiment. Speciation stays off for the next studies unless explicitly
included as a treatment.

## Lifetime learning on/off benchmark

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
  matched evolutionary learning-on/off study. Neither has been completed yet.
- Fix design choices before evaluating a fresh final test partition, rather than
  repeatedly adapting the implementation to the same visible held-out curve.
