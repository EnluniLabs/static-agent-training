# Genome caches and parallel offspring development

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](../ARCHITECTURE.md), reviewed 2026-11-22.

Measured 2026-11-22. These changes supplement the
[checkpoint saving improvements](CHECKPOINT_SAVE_SPEED.md).

## Implemented

- Per-Species cached genotype IDs. Immutable merger summaries avoid reparsing
  development logs repeatedly; weight-only mutations skip that parsing entirely.
- Per-sensory-mapping read-only NumPy source arrays, copied into independent
  writable evaluation batches. The public parameter API still returns copies.
- Spawned offspring workers, bounded pending jobs and two-entry parent caches.
  Selection and per-child seeds remain in the main process; collection follows
  the original child order. Speciation and lifetime inheritance use the same path.
  ASTs use stack-safe transport. Equal genotype IDs with different development
  provenance do not share a worker parent cache entry.

Caches belong to immutable objects, are excluded from saved formats, and are
not carried into replacement/mutated mappings. Existing hashes and mutation logs
are preserved. Spawning avoids forking an initialized native/OpenMP runtime.

## Phase measurements

Source: a read-only snapshot of
`runs/ui-20261121-131148-session-b43354/checkpoint.json.gz`, population 32,
generation 50. Reproduction used the same synthetic training fitness ordering and
seeds for all modes. Assessment scores did not enter selection. Genotype IDs were
warmed before timing. Table values are medians of two trials. The pre-change
baseline was run from an isolated copy of commit `15a86c3213d5e550a06466ffeee9382e8d92b4c0`.

| Reproduction mode | Seconds, no speciation | Seconds, speciation |
| --- | ---: | ---: |
| Before caching, serial | 0.4973 | not measured |
| Cached, serial | 0.3620 | 0.3139 |
| Cached, 2 processes | 0.4115 | 0.3568 |
| Cached, 4 processes | 0.3123 | 0.2824 |

Cached serial reproduction took 27.2% less time than the pre-change serial
baseline on this sample. That baseline overlapped the larger benchmark below, so
this comparison is indicative rather than an isolated speedup measurement. Four processes took 37.2% less time than that baseline,
including process startup and transport. Two processes were slower than cached
serial, demonstrating that adding workers is not automatically beneficial.

Child and mutation-log digests matched across the pre-change baseline and all
non-speciated modes. Across worker counts, children, logs, normalized config,
speciation reports and persistent species state matched in both treatments.

A second saved population from
`runs/ui-20261121-130828-session-aa15d0/checkpoint.json.gz` (10 MiB compressed,
generation 19, 32 parents and 27 children) measured:

| Mode | Median reproduction seconds |
| --- | ---: |
| Cached serial | 4.3167 |
| Cached, 2 processes | 2.9633 |
| Cached, 4 processes | 1.6799 |

Four processes provided a 2.57× reproduction speedup on this larger case. All
completed modes produced identical children, logs and reports. Aggregate parent
and child packets totaled approximately 173 MB; parallel transport is included
in the timings. The large speciation case exceeded the bounded benchmark budget,
so no large-speciation timing is claimed.

A separate warm `_pack` benchmark on eight saved individuals measured
6.906 ms before versus 0.121 ms after (57.2×), with exactly equal arrays. This
measures repeated packing of the same immutable objects, not first-time packing
of newly mutated genomes. It is not a whole-generation speedup claim.

## Worker controls and limits

`--reproduction-workers 0` is automatic; `1` is serial; `2` or more requests
parallel workers. The workspace exposes the same setting as **Offspring
processes**. Automatic mode requires at least eight children and an estimated
700,000 parameter-cell operations across the schedule. It then uses at most
four processes, limited by available CPUs and evaluation workers. The small
population above stays serial, while the larger measured schedule (717,880 cells)
uses parallel workers in automatic mode. This threshold is a conservative
heuristic informed by these two samples, not a universal crossover point. Explicit workers allow
benchmarking other workloads.

Source packets and working offspring are bounded by in-flight worker count;
large ASTs and mappings still add transient memory compared with serial execution.
No training jobs were restarted or changed for these measurements. New code takes
effect when training starts or resumes.

## Validation

Golden pre-change identity and mutation-log checks, immutable source isolation,
Python/native evaluation parity, serial/process child and log equality,
speciation, lifetime inheritance, deep-term process transport and exact checkpoint
resume are covered by regression tests. The full suite passed with 325 tests.
