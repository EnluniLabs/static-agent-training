# Experiment record

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

This is a dated historical record. Use `./project runs`, `./project status`, and
`./project show RUN` for current local artifacts and process status; see
[USE.md](../USE.md). Pending/running descriptions below reflect when they were
written and are not a live registry.

All results below were collected locally on 2026-11-01. They are implementation
experiments, not proof of the underlying theory. Machine-readable summaries are
in [results](results/). Raw run directories and checkpoints are not committed.

## Early integration runs

These were development runs, not a matched causal comparison. All used the earlier
out-of-bounds termination rule and uniform start/goal sampling within the old
40-position partition. They must not be compared directly with current hard-wall
scores.

| Population | Generations completed | Task | Highest training-selected score | That candidate's saved held-out score |
|---|---:|---|---:|---:|
| 32 | 50 | Fixed goal zero | 50% (10/20, generation 48) | 25% |
| 32 | 50 | Varying goals | 45% (generation 20) | 20% |
| 64 | 150 | Varying goals | 30% (generation 95) | 13% |
| 1,024 | 48 of 150, then stopped | Varying goals | 32% (generation 23) | 9% |

The 32-animal varying-goal candidate subsequently scored 9.85% on 2,000 fresh
held-out task draws, compared with 11.20% for simulated uniform random movement
(exact random expectation 11.53%). The earlier 20-episode score was not a reliable
estimate of its broader success. [Early run histories](results/early-runs.json)
and [fresh comparison summary](results/early-varying-goal-random-comparison.json).

## Evaluation throughput

Hardware: Intel Core i7-14700KF, 28 logical CPUs; NVIDIA RTX 4070 SUPER, approximately
12 GB VRAM, accessible through WSL. Measurements used three repeats, eight native
workers, batches up to 1,024, seed 42, and 50 training plus 50 assessment episodes
per animal. They used the **earlier normal-distance, terminating-boundary** task.
They must not be presented as measurements of the current uniform/hard-wall task.

| Population | Python median | Native CPU median | CUDA median |
|---|---:|---:|---:|
| 64 | 3.904 s | 0.205 s | 0.789 s |
| 256 | Not measured | 0.464 s | 1.095 s |
| 1,024 | Not measured | 1.447 s | 2.014 s |

Timings cover evaluation, packing, RNG setup, transfers, and champion replay after
compilation warmup. They exclude initialization, reproduction, and checkpoint writes.
They cover initial populations; evolved populations can have different costs.
Native CPU was selected for later large runs because it was faster in this test.
[Samples and summaries](results/evaluation-speed.json).

## Speciation pilot

Three paired seeds (42, 43, 44), 256 animals per arm, eight generations, 40 training
and 80 assessment episodes per animal, then 1,000 fresh held-out trials of each
run's training-selected best candidate. Both arms used the same inputs, initial
genotypes, mutation settings, elite counts, and evaluator. Task rules were normal
distances centered at 10 with standard deviation 5 and **out-of-bounds failure**.

| Seed | Control | Speciation | Difference |
|---|---:|---:|---:|
| 42 | 12.50% | 12.90% | +0.40 pp |
| 43 | 13.20% | 12.90% | −0.30 pp |
| 44 | 13.60% | 13.60% | 0.00 pp |

Mean difference: +0.03 percentage points; 95% t interval over paired seeds:
−0.84 to +0.91 points. No evidence of benefit in this pilot.
[Full summary](results/speciation-pilot.json).

## Larger speciation comparison

Three paired seeds, 1,024 animals per arm, 30 generations, 100 training and 200
assessment episodes per animal, then 2,000 fresh held-out trials per selected
candidate. Same historical normal-distance and terminating-boundary rules as the
pilot. Speciation used threshold 0.6 and at most 16 niches. Initial population
hashes and every paired generation's task hashes matched.

| Seed | Control | Speciation | Difference |
|---|---:|---:|---:|
| 42 | 16.80% | 15.30% | −1.50 pp |
| 43 | 15.55% | 14.90% | −0.65 pp |
| 44 | 16.90% | 14.50% | −2.40 pp |

Means: 16.42% control versus 14.90% speciation. Mean difference: −1.52 points;
95% paired-seed interval: −3.69 to +0.66 points. This configuration lost on all
three seeds, but the interval still includes zero. Speciation remains optional
and is off for subsequent studies; this is not a conclusion about all possible
speciation schemes. [Full summary](results/speciation-benchmark.json).

These historical studies must not be merged with later uniform/hard-wall scores.
The comparison command's defaults now use those newer rules; recreating the old
studies requires their original configuration/manifest and original partitions.

## Target location and boundary diagnostic

Three saved no-speciation candidates from the larger study were evaluated with
learned mappings frozen. Distances 1, 5, and 10 were held fixed, with 100 repeats
per feasible ordered pair, under both old and new boundary rules. Aggregate rates
weight target locations equally rather than weighting targets with two approach
directions twice. Exact random-movement probabilities provide the control.

For distance 1 under old rules, mean animal success was 35.1% for |goal| ≤ 5 and
24.4% for |goal| from 13 to 20. Random movement likewise fell from 45.5% to 34.4%.
At distance 10, animal rates were approximately 17% in both regions. With hard
walls, distance-1 rates became 35.9% centrally and 48.6% near the ends. This shows
that environment geometry contributed substantially to the earlier location effect;
it does not establish improved learning from changing the boundary rule.

The original normal-distance sampler also gave each trained goal with |goal| 1–5
about 4.1% probability on average, versus 2.5% for |goal| 13–20 in these partitions.
Sampling distance normally does not imply equal target exposure. New defaults
therefore use uniform valid start/goal pairs and hard walls, including zero in the
varying-goal partition. [Diagnostic data](results/target-location-diagnostic.json).

## Stopped long run and bias audit

The earlier long run was launched with 1,024 animals, a 500-generation target, 100
training and 200 frozen assessment episodes, seed 42, native CPU evaluation,
uniform pairs, zero included, hard walls, and speciation off. It was stopped at generation 446/500; its checkpoint remains resumable.
The audit notes are snapshots, not a completed 500-generation result.

The [bias audit](BIAS_AUDIT.md) records the generation-88 baseline comparison and
a later read-only goal-input control. Neither establishes that local gradient
descent caused the population's improvement. The matched lifetime-learning
studies below address that question; deployment adaptation remains
[future work](../FUTURE_WORK.md).


## Lifetime learning on/off study (stopped)

`lifetime-learning-20261101-153325` was launched after stopping the 1,024-animal
long run gracefully at generation 446/500. The saved checkpoint remains resumable.
The new study uses 256 animals per arm, 20 generations, seeds 42/43/44, 100 training
and 200 assessment episodes, and 2,000 fresh shared held-out trials per selected
candidate. Both arms use uniform pairs, hard walls, zero included, and no speciation.

The control disables both existing lifetime update rules; the treatment enables
both without changing their formulas. Assessment is frozen in both arms. Input
and initial-population hashes are checked as in the speciation studies. This is a
comparison of evolutionary runs with/without lifetime learning, not a same-genotype
instantaneous ablation and not a test of a new error-function implementation.
The live report is under `runs/comparisons/lifetime-learning-20261101-153325/` and
visible at `/comparisons`. The user stopped this study before all seeds completed; no aggregate conclusion is claimed.


## Inherited lifetime learning study (completed)

`inherited-learning-20261101-154322` used the same small configuration above, with lifetime
learning on in both arms and inherited learned weights off/on. Speciation remains
off. Only end-of-training embedding and output logits enter reproductive parents;
assessment is frozen and does not influence inheritance or selection. Elites also
receive learned weights in the inheritance arm. Initial population and task hashes
are checked, with evaluated/reproductive parent identities recorded. This tests
inheritance without changing the existing error rules. All three seeds completed;
see the [curated result](results/inherited-learning-pilot.json) and summary below.


## Longer inherited lifetime learning benchmark (stopped)

The 20-generation pilot completed across all three seeds. Inheritance on improved
fresh held-out success by a mean 1.12 percentage points, with a seed-paired 95%
interval of [-2.63, +4.86] points; this is not conclusive evidence of a benefit.

A fresh rerun, `inherited-learning-long-20261101-155447`, targeted 200 generations per arm
(ten times the pilot). It retains 256 animals, seeds 42/43/44, 100 training and
200 frozen assessment episodes, 2,000 final shared held-out trials, native CPU
evaluation with four workers per arm, uniform pairs, hard walls, lifetime learning
on, and speciation off. Only learned-weight inheritance differs between arms.
The pilot is preserved separately; this rerun starts from the same seeded initial
populations rather than selecting a pilot winner. No learning or mutation rules
were changed. The user stopped this study after seed 42 completed and seed 43
reached four generations per arm; seed 44 had not started. Seed 42 alone favored
inheritance off by 4.10 percentage points on the final endpoint. One completed
seed pair does not establish a general effect.


## Longer lifetime learning on/off benchmark

At the user’s request, the longer inheritance comparison was stopped before all
three seeds completed; its partial results do not establish an aggregate winner.
A fresh study, `lifetime-learning-long-20261101-161824`, compares lifetime learning
off/on during evolution for 200 generations, with learned-weight inheritance and
speciation disabled in both arms. It retains 256 animals per arm, seeds 42/43/44,
100 training and 200 frozen assessment episodes, 2,000 final shared held-out
trials, uniform pairs, hard walls, and native CPU evaluation (four workers per arm).
Both existing local update rules are disabled/enabled together; their formulas
are unchanged. Final assessment freezes weights in both arms. This study tests
learning during evolution; deployment-time adaptation remains a separate future
test. Results are pending in `/comparisons`; no final winner is claimed.


## Inheritance comparison rerun

At the user’s request, `inherited-learning-rerun-20261101-170747` starts a fresh
200-generation learned-weight inheritance off/on comparison across seeds 42/43/44.
Both arms enable lifetime learning and disable speciation. The configuration
matches the earlier longer inheritance study: 256 animals per arm, 100 training
and 200 frozen assessment episodes, 2,000 final shared held-out trials, uniform
pairs, hard walls, and native CPU evaluation with four workers per arm.
The earlier stopped study is preserved separately. Reusing its seeds makes this
a repeat of the same seeded trajectories, not additional independent replication.
This run is intended to complete all three seed pairs before interpreting the
aggregate effect. Results are pending in `/comparisons`; no agent, learning, or
mutation rules were changed.
