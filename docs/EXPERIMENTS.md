# Experiment record

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

## Current long run and bias audit

The current run was launched with 1,024 animals, a 500-generation target, 100
training and 200 frozen assessment episodes, seed 42, native CPU evaluation,
uniform pairs, zero included, hard walls, and speciation off. It continues in the
background; the notes below are snapshots, not its final result.

The [bias audit](BIAS_AUDIT.md) records the generation-88 baseline comparison and
a later read-only goal-input control. Neither establishes that local gradient
descent caused the population's improvement. The full matched lifetime-learning
on/off study remains [future work](../FUTURE_WORK.md).
