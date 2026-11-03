# Enluni Labs — Artificial Animal Research Roadmap

**Recorded:** 2026-11-02  
**Status:** Proposed research plan, not a claim of demonstrated capability.  
**Repository:** `EnluniLabs/static-agent-training`  
**Planning horizon:** November 2026 onward; dates are estimates and evidence gates govern progression.

## Objective and organizing principle

Develop and experimentally evaluate artificial animals that inherit independent, immutable lambda-derived coherent structures, learn representations through lifetime interaction, explore, predict, and act. Investigate whether useful repertoires can be preserved and gifted across evolutionary projects to accumulate increasingly general animal competence. Only after a credible inherited animal foundation exists should research prioritize autonomous construction and criticism of genuinely novel coherent structures (the universal explainer).

Organize research by *dependencies and falsifiable evidence*, not just conceptual ambition. Preserve negative findings, reproducible configurations, matched controls, and exact distinctions between implemented mechanisms and hypotheses.

## Current baseline and caveats

The implemented system develops successor repertoires from closed untyped lambda-term pairs, traverses fixed successors during lifetimes, learns observation/output mappings through local prediction-error and delayed compatibility updates, and evolves populations under task success. Reduction occurs during development or mutation, **not** during lifetime traversal. The current number-line interface supplies categorical observations and optional goals. Inherited learned weights are optional and off by default.

Existing curated results are preliminary: the goal-input audit found little difference when held-out goal input was permuted or removed; this does not establish meaningful goal conditioning. A completed 20-generation learned-weight inheritance pilot estimated +1.12 percentage points mean held-out success with a 95% interval spanning zero; the longer inheritance study was stopped. A longer lifetime-learning comparison was recorded as pending. No completed coherent-versus-random test or goal-free exploration demonstration is documented. Verify current run status before making claims.

References: `README.md`, `docs/IMPLEMENTATION.md`, `docs/EXPERIMENTS.md`, `docs/BIAS_AUDIT.md`, `FUTURE_WORK.md`, `docs/COHERENT_VS_RANDOM_TEST_SPEC.md`, `docs/THREE_CORE_EXPERIMENTS.md`, `docs/REPERTOIRE_LIBRARIES_AND_INHERITED_ANIMAL.md`, `ENVIRONMENT_INTERFACE.md`.

## Schedule overview

| Phase | Provisional window | Primary result | Dependencies |
|---|---|---|---|
| 1. Learning validity | Nov–Dec 2026 | Credible controlled learning baseline | Current implementation |
| 2. Generalized signal interface | Nov 2026–Jan 2027, overlaps 1/3 | Type-agnostic input/output interface and parity tests | Defined encoding, feedback semantics |
| 3. Coherent vs arbitrary structure | Dec 2026–Feb 2027 | Controlled structural-bias comparison | Phase 1; existing interface first |
| 4. Goal-free exploration | Jan–Mar 2027 | Surprise-goal transfer experiment | Phases 1–2; baseline controls |
| 5. Cross-environment representation transfer | Mar–Apr 2027 | Fixed-repertoire transfer evidence | Phases 2–4 |
| 6. Repertoire population analysis | Mar–May 2027 | Structural statistics, ablations, package candidates | Reproducible evolved populations |
| 7. Repertoire library and inheritance | Apr–Jun 2027 | Versioned packages and clean transfer ablations | Phases 2, 5–6 |
| 8. Independent package interactions | May–Jul 2027 | A/B/A+B and matched composite-task tests | Phase 7 |
| 9. Cumulative animal development | Jul–Oct 2027 | Multi-stage accumulation vs monolithic control | Phase 8 |
| 10. Lifetime structural innovation | 2027+ / undated | Novel coherent structure created within a lifetime | Credible animal competence; new mechanism |

Windows are *targets*, not deadlines or guarantees. Parallelize only when an experiment can preserve its controls. Avoid changing interface and repertoire generation simultaneously within a comparison.

## Phase 1 — Validate learning and goal dependence

**Questions:** Do lifetime updates improve future performance? Does the animal use goal inputs? Which structural links and compatibility corrections materially influence learning?

**Work:** Reconcile pending/stopped studies and their saved artifacts; run same-genotype frozen-repertoire learning-on/off assessments; separate empirical prediction updates from structural compatibility updates; repeat goal permutation/removal controls across seeds, trained/held-out values, and distances; instrument repertoire node/edge utilization, structural correction frequency, prediction calibration and voluntary prediction rates. Include goal-independent movement policies and fair random baselines. Freeze assessment weights.

**Exit gate:** Repeatable learning effects or well-characterized null results with uncertainty; documented goal dependence or explicit limitations; replayable manifests. A null result triggers diagnosis rather than premature scaling.

## Phase 2 — Generalized type-agnostic signal interface (FOUNDATIONAL)

The generalized interface is not a cosmetic refactor. It enables testing whether representations are learned rather than hard-coded by input type, and whether repertoires remain useful across environments.

**Design:** Specify one consistent encoded signal interface for observations, optional goals, actions and voluntarily emitted expectations. Consider a universal bitwise representation, but compare incremental introduction against direct adoption. A shared mapping should operate on encoded signals rather than independent semantic-specific embedding tables or a second embedding layer stacked on the first. Define framing and length, output validity, temporal feedback and matching of predictions to observations, and how the environment distinguishes actions from predictions. Avoid computed distance, goal direction, hidden labels or privileged semantic hints. Preserve the two local learning rules and fixed repertoire. Ensure absent goals are genuinely absent, not replaced with a sentinel that reveals the task. Specify how identical payload bytes used in different roles are framed without leaking semantic interpretations.

**Engineering stages:** (a) formal interface contract and encoding invariants; (b) compatibility adapter for current number line; (c) reference and accelerated backend parity, replay, serialization and test suite; (d) new signal/environment types; (e) ablations versus existing categorical interface and capacity-matched direct associative controls.

**Exit gate:** A tested, reproducible interface that supports the existing benchmark and at least one genuinely distinct signal setting, without hidden task information. The initial coherence-vs-random experiment should still use the unchanged old interface first; repeat relevant findings on the new interface separately.

## Phase 3 — Coherent versus arbitrary-node repertoires

**Question:** Does lambda-development-induced repertoire topology improve learning relative to arbitrary successor graphs under matched learning and evolution?

Follow `docs/COHERENT_VS_RANDOM_TEST_SPEC.md`: fixed-repertoire lifetime-learning comparison first; random chains as initial control, then topology-matched graphs if an effect appears; isomorphic relabeling sanity check; same learning code, mapping initialization, tasks, evaluation budgets and evolutionary selection; structural-off and all-learning-off diagnostics; matched seeds, independent repertoire samples and frozen held-out assessment. Measure utilization and structural dependence before interpreting nulls. Graph-isomorphic repertoires are behaviorally equivalent to the lifetime agent with correspondingly permuted mappings: do not claim invisible lambda semantics cause differences.

**Exit gate:** Reproducible estimate with uncertainty and a precise attribution statement. Negative or ambiguous evidence may redirect generator/learning design.

## Phase 4 — Goal-free exploration and unforeseen goals

Implement genuinely goal-free observation and action, voluntary expectations, and informative exploration. Compare neutral/no-task selection, exploration-based selection, predictive-improvement selection and explicit-goal evolution as distinct regimes. After exploration, introduce a previously withheld goal (e.g. resource/food location) without leakage. Compare no-exploration, uninformative exposure, random exploration, coherent and arbitrary repertoires, and matched experience budgets. Guard against no predictions, trivial predictions, and aimless movement. Move beyond number line only when necessary to make the hypothesis testable.

**Exit gate:** Measured held-out adaptation advantage or informative failure, with controls for prior exposure and selection.

## Phase 5 — Cross-environment transfer of representations

With coherent structures held fixed, change signal encoding, task or environment. Compare fresh representations against inherited learned representations, with equal adaptation budgets. Measure retention, interference, sample efficiency and whether learned meaning transfers or must be reconstructed. Keep interface version and signal semantics in manifests.

**Exit gate:** Clear conditions under which transfer works or fails.

## Phase 6 — Population-scale repertoire analysis

Evolve multiple populations under varied evaluations, preserve lineages, and compare successful and unsuccessful individuals. Examine recurring structure statistics, chain lengths, branching/merges/cycles, utilization and relation to performance. Perform ablations of intact structures and subsets with and without relearning; identify useful, redundant and dependent subsets. Control for repertoire size, graph topology, environment, mapping initialization, lineage and compute. Analyze representative cohorts (potentially 1,000 animals) without presuming all high-performing structures are causal.

**Exit gate:** Reproducibly characterized candidate packages with provenance and ablation evidence.

## Phase 7 — Versioned repertoire library and inheritance

Create a package manifest containing immutable coherent states and successors, generator and lineage, development constraints, selection/evaluation history, interfaces and encoding versions, optional learned mapping snapshot and performance evidence. Distinguish (i) evolved structure with fresh mappings, (ii) structure plus learned representations, and (iii) matched fresh/random structure. Ensure no graph splicing or structural merging: packages enlarge the set of independent available structures. Test loading, replay, mapping reconciliation and isolation of package effects.

**Exit gate:** Reusable packages with demonstrated transfer behavior and reproducible provenance.

## Phase 8 — Interactions among independent packages

Evolve A and B independently (e.g. exploration/navigation and interaction). Compare fresh animals given neither, A, B, A+B, and capacity-matched controls, on constituent tasks and a previously unseen composite task. Test structure-only and structure-plus-mappings separately. Track retention, adaptation speed, interference, node/edge use and performance beyond capacity alone. Never merge or splice coherent structures; coordination occurs through learned selection, representation and behavior. Shared formal coherence does not guarantee package interoperability.

**Exit gate:** Replicated evidence for useful coexistence, synergy, neutral interaction or interference, with transparent baselines.

## Phase 9 — Cumulative animal development

Iterate: independently evolve useful packages, analyze and preserve them, gift multiple packages to new populations, and introduce genuinely novel evaluations. Measure marginal capability gain, prior-skill retention, package count, interference, total compute and adaptation time. Compare against monolithic evolution given comparable total resources. This tests **researcher-mediated accumulation of inherited competence**, not autonomous cultural innovation.

**Exit gate:** Multiple documented stages of increasing transferable competence under fair controls, or an identified scaling bottleneck.

## Phase 10 — Lifetime structural innovation and eventual universal explanation

Only after animal competence is demonstrated, introduce a mechanism for an individual to construct a genuinely novel coherent structure during its own lifetime, rather than merely reinterpret an inherited one: `(C,T) -> (C,T')` is analogy; `C(t+1)=C(t) ∪ {C*}` with new `C*` is structural innovation. Test novelty, coherence, criticism/error correction, independent usefulness, and later communication/reconstruction by another animal. Distinguish genuine autonomous cumulative innovation from researcher-mediated package transfer.

**Exit gate:** Explicitly defined later; no date or success claim is justified now.

## Parallel documentation and publication track

- **Nov–Dec 2026:** Draft architecture and reproducibility report, with mathematical definitions, actual interfaces, implemented update rules, limitations and reproducible baselines.
- **Dec 2026–Feb 2027:** Generalized-interface technical specification and controlled coherence-vs-random experimental paper, conditional on results.
- **Mar–May 2027:** Goal-free exploration and representation-transfer report, conditional on results.
- **Jun–Aug 2027:** Repertoire library, independent-package interaction and ablation report, conditional on results.
- **Late 2027+:** Cumulative inherited cognitive development report, only if multi-stage evidence warrants it.

Maintain a dated hypothesis register and experiment registry: ID, preregistered comparison, status (proposed/implemented/running/completed/stopped), config and commit SHA, seeds, train/test separation, compute cost, metrics, failure criteria, uncertainty, artifacts, and decision. Do not treat publication targets as promises.

## Near-term execution queue

1. Verify whether `lifetime-learning-long-20261101-161824` has completed and reconcile local run status with committed records.
2. Produce a small, reproducible same-genotype learning diagnostic and goal-input controls.
3. Draft the generalized signal interface contract, encoding/framing tests, and number-line parity adapter; **do not** silently replace the current benchmark interface.
4. Implement and validate random-node repertoire parity and isomorphic relabeling tests.
5. Run fixed-repertoire coherence-vs-random pilots and utilization diagnostics.
6. Only then decide whether to scale evolution or redesign the environment/learner.
7. In parallel, draft the technical architecture paper and establish experiment manifests.

## Decision rules

- Prefer the smallest discriminating experiment to a large ambiguous run.
- Record negative results and residual confounds.
- Distinguish coherent structure, learned representation, selection pressure and inherited acquired weights.
- Keep structures independent and immutable during lifetime learning; repertoire packaging never implies structural merging.
- Require held-out, frozen evaluation and matched resource budgets.
- Revisit the roadmap after each evidence gate; a negative foundational result may reorder or cancel downstream work.
