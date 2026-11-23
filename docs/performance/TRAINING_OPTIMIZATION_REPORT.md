# Faster training while preserving the studied architecture

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](../ARCHITECTURE.md), reviewed 2026-11-22.

Reviewed 2026-11-05 UTC (2026-11-04 America/Chicago). Base commit:
`34f34ac594cedce66ed716d35041f959b079b854`, with the current uncommitted puzzle
implementation. This is a report and an isolated measurement, not an optimization
implementation. No production code or existing run data was changed.

## Recommendation

Optimize checkpoint writing and repeated immutable-data conversion first. Then
parallelize independently seeded offspring construction if measurement justifies
it. Keep the learning algorithm, sensory representation, per-generation reference
verification, and complete saved records intact. Introduce asynchronous saving only
after explicit crash/recovery tests establish a reliable generation commit boundary.

The objective is less wall-clock time for the **same experiment**, not a CPU usage
percentage. Eight native workers on 28 logical CPUs cannot occupy the entire CPU;
serial stages reduce average occupancy further. Logical CPUs are not equivalent
to independent physical cores. Raising worker count does not accelerate Python
serialization or reproduction.

## Evidence and its limits

Earlier live samples found no competing CPU, memory, or disk pressure. Training
averaged about 1.2–1.3 CPU cores despite eight evaluator workers. Recent generation
times varied substantially as populations evolved; comparisons between different
generations or fresh and resumed runs do not isolate the worker-count effect.

A bounded stage probe loaded an atomic checkpoint snapshot from
`runs/ui-20261104-220724-Model_A_1-eb173e`, next generation 91. Configuration:
fixed-senses, original number line, 256 receptors, 100 animals, 100 training and 200
assessment episodes, eight native workers, lifetime inheritance and speciation off.
It evaluated and reproduced one population in memory and wrote artifacts only in a
temporary directory. The user's live run continued independently.

| Stage | Seconds | Approximate share |
|---|---:|---:|
| Evaluation, including packing, results and Python champion replay | 0.832 | 15.7% |
| Update saved best candidates | 0.267 | 5.0% |
| Selection and reproduction | 1.437 | 27.1% |
| Generation record write | 0.162 | 3.1% |
| Champion trace write | 0.133 | 2.5% |
| Convert next population into checkpoint data | 0.186 | 3.5% |
| Checkpoint JSON encoding, gzip, write and sync | 2.292 | 43.2% |
| Total measured span | **5.309** | **100%** |

Nested inside evaluation: packing 0.176 s, native phase wrappers 0.120 s, Python
champion replay 0.239 s. These are **not additional times**. Phase wrappers include
RNG/array preparation and metrics allocation as well as the C++ call. The checkpoint
was 8,239,926 compressed bytes.

This is one instrumented sample under concurrent live training, not a controlled
speedup benchmark. It excludes startup/checkpoint loading, uses reduced orchestration
around the same stage functions, and omits some inexpensive provenance assembly.
It is not a replacement trainer or an exact replica of its output records. Native
libraries may already be cached. The timing is specific to this population; larger
repertoires and more episodes can shift the balance.

Even eliminating the measured native phase entirely would remove only about 2.3%
of this probe's wall time. That explains why more native workers alone have little
effect. It does not establish a universal upper bound for other workloads.

The [raw measurement](TRAINING_STAGE_SAMPLE.json) includes the checkpoint hash.
The [probe script](profile_training_stages.py) documents the method; it reads the
latest checkpoint at the named path, so a later invocation uses a different population.
The exact sampled population is identified by hash but is not archived here.

## Scientific invariants

These are acceptance requirements, not optional performance tradeoffs:

1. **Sensory apparatus:** preserve receptor count, dual-rail positional folding,
   presence handling, byte encoding and action mapping. Keep all 256 prediction
   outputs, including constant/inactive receptors. No solver, semantic shortcuts,
   environment-specific learned embedding, or new learned lookup table.
2. **Within-lifetime order:** observation → entry sampling → repertoire traversal
   → eligible delayed structural correction → output → outcome/feedback. Preserve
   empirical update order, revision eligibility across episodes, expiration and
   terminal feedback. Repeated predictions from the same source update sequentially.
3. **Learning mechanism:** preserve rates, compatibility threshold, shared input
   weights and bias corrections, bit-prediction losses, and every updated parameter.
   Lifetime learning must never modify repertoire edges.
4. **Randomness and arithmetic:** preserve stream seeds, draw count/order, pair and
   token ordering, threshold/tie behavior, double precision, compensated summation,
   and existing compiler arithmetic restrictions. No fast-math or RNG replacement.
5. **Evolution:** preserve tasks, training-only fitness selection, elite ordering,
   tournament draws, tie breaks, mutation attempts and per-offspring seeds, development
   budgets, retained-state parameters and speciation behavior. Preserve configured
   inheritance semantics; frozen assessment must not leak into reproduction.
6. **Isolation and records:** no shared mutable learner arrays between animals,
   threads or parent/child genomes. Preserve frozen assessments, champion replay and
   its assertions, candidate ranking, genotype IDs, provenance and resumable state.

The [interface contract](../AGNOSTIC_INTERFACE.md),
[implementation description](../IMPLEMENTATION.md), and
[error-propagation audit](../audits/ERROR_PROPAGATION_AUDIT.md) define the relevant
mechanisms. Preserve observed interference as well as useful transfer; changing
which inputs are affected would change the architecture under study.

## Proposed changes, in order

### 1. Faster synchronous saving — narrowest first change

In `experiment.write_json`, test gzip level 1 against the current default level 9.
Keep JSON schema/content, per-generation cadence, temporary-file replacement, file
sync and directory sync. Compare decompressed content, not gzip bytes: gzip headers
and compression level can differ legitimately. Expect larger files; measure both
write and resume-read costs. This cannot affect model computations.

Separately test a single standard-library JSON encoding followed by a bulk write
against incremental `json.dump`. Preserve serialization options and canonical
hashing behavior. A full encoded string increases peak memory, especially on large
populations; use a measured memory bound or retain streaming for large snapshots.
Do not bundle a new serializer, binary format or reduced precision into this change.

### 2. Remove redundant immutable-data work

`Species.genotype_id` repeatedly serializes and hashes inherited mappings;
`sensory_accelerated._pack` thaws immutable parameters into Python containers before
copying them into NumPy. `populations.update_best` reloads and validates saved genomes
every generation. Reuse validated immutable data within a run and cache genotype
IDs on immutable objects, with bounded ownership and explicit reload on resume.

A cached packed genome must be read-only. Each lifetime needs a separate mutable
working copy. Mutation must produce a new cache identity even when repertoire
structure stays the same. Preserve IDs bit-for-bit and never cache learned state
under an inherited-genome key. Measure retained memory before adopting caches.

### 3. Parallel offspring construction

`evolution.reproduce` already derives mutation seeds from generation and child
index. Generate tournament selections and an ordered list of offspring jobs on the
coordinator using the existing RNG order. Build/mutate independent offspring in a
persistent process pool, then assemble children and ancestry by original index,
regardless of completion order. Ordinary Python threads will not remove the GIL
bottleneck for this Python-heavy work.

Keep elites unchanged. Do not select parents by worker completion time or allocate
random streams by worker ID. Worker failures must fail/retry the same indexed job,
not substitute a different child. Process startup, genome transfer and serialization
may erase the benefit for 100 small animals; measure them. Bound worker count and
memory. Speciation needs its own equivalence coverage; preserve its allocation and
selection ordering before dispatching mutation jobs.

### 4. Native caches and scheduling

These are secondary for the measured run but can matter as evaluation grows:

- Cache sensory action/prediction gate cumulative probabilities: gate weights are
  fixed within a lifetime. Reproduce existing divisions, summation order, comparisons
  and fallback behavior; consume the same random draw per output.
- During frozen assessment, cache observation distributions by complete sensory
  frame, scoped to the current animal and frozen parameter version. Bound the cache;
  the number-line and puzzle frame domains have very different sizes. This is
  memoization of the existing computation, not new learned memory. Discard it when
  parameters change. Do not initially cache training observations.
- Precompute active feature indices for fixed frames, retaining their order. Improve
  contiguous memory access while preserving the accumulation order for each logit.
- Reuse duplicate prediction-loss calculations only when the source row is unchanged.
  Preserve emission-time losses and sequential updates for repeated source rows.
- Test dynamic/guided OpenMP scheduling for unequal animal workloads. Animals must
  retain their own RNG state, arrays and result index. Scheduling overhead may make
  small workloads slower.

Do not reduce receptor count, skip prediction bits or delayed corrections, shrink
assessment counts, or change learning rates as performance optimizations.

### 5. Background saving — useful, but higher engineering risk

A writer can overlap generation N's save with generation N+1's computation only
from an immutable snapshot. Python JSON encoding in a thread may still compete for
the GIL; a process introduces transfer/copy costs. Faster synchronous writes should
be measured first.

Use a bounded queue and backpressure, preserve save order, propagate writer errors,
and drain pending saves at normal exit/pause. Never allow a late older save to
replace a newer checkpoint. Publish an explicit durable generation boundary after
its required records, traces and checkpoint are complete. Existing file-level
atomic replacement alone is not a multi-file transaction.

Distinguish computed progress from durable progress. A crash must resume from a
consistent saved boundary and deterministically rerun later work; readers must not
mistake partial artifacts for committed generations. Keep every-generation saving.
This is orchestration work requiring design and fault injection, not a quick thread
around the current writer.

### Preserve champion verification initially

The Python champion replay costs about 0.239 s in this sample and provides both
reference verification and full learned-model/event traces. Keep it every generation
in the first optimizations. Replaying it periodically changes audit coverage and
artifact availability, even if the model's equations stay the same. Treat that as a
separate explicit product/research decision, not a hidden speed improvement.

## Validation and rollout gates

Use a frozen baseline checkout and identical saved population/task snapshots. Run
old and proposed implementations sequentially on the same machine, warm native
compilation, alternate comparison order, repeat measurements, and report median,
spread, peak RSS, file size, CPU time and wall time. Start with the measured workload,
then cover puzzle environments, larger repertoires and population sizes. Keep timing
and diagnostics out of selection and canonical identity inputs.

| Change | Required evidence before adoption |
|---|---|
| Compression/encoding | Old/new readers load identical data; original snapshot remains readable; write failure before replacement preserves the previous checkpoint; resume is identical to uninterrupted training. |
| Immutable caches | Identical genotype IDs, packed inputs, candidates and all learned arrays; no aliasing; bounded memory; mutation and resume cannot reuse stale values. |
| Parallel reproduction | Identical selected parents, competitors, mutation logs, child ordering, full genomes and ancestry across worker counts, ties, retries, speciation and inheritance modes. |
| Native optimization | Same actions, predictions, termination, correction timing/counts and RNG consumption; compare all parameter groups and repertoire edges, including repeated-source feedback and frozen phases. |
| Background writer | Kill/fault injection before/after encoding, sync, replacement and generation commit; bounded queue, ordered commits, error propagation, clean shutdown, correct UI durable progress and deterministic resume. |

For changes wholly outside floating-point computation, demand exact logical results
and parameter equality, not merely similar fitness. Native/reference validation
currently uses `rel_tol=1e-11`, `abs_tol=1e-10`; do not relax it. Compare old/native
versus new/native exactly where operation order is preserved, and require identical
discrete trajectories. Passing aggregate success rates alone is insufficient.

Rerun local correction, shared-component transfer with unrelated control, and delayed
structural correction checks for computational changes. Include observations with
varying lengths/folding, terminal feedback, chain termination, inheritance on/off,
and frozen assessment. Preserve the existing native/Python equivalence and resume
suite. A failed equivalence check blocks rollout even if the change is faster.

Make one reversible change at a time; retain old checkpoint compatibility and the
baseline comparison. Adopt only measured improvements. Apply to newly started or
explicitly resumed processes, not by patching a live process. First deliverable:
a compression-only change with equality, interruption/resume tests and before/after
write-time/file-size measurements. Then remeasure the full generation before choosing
the next intervention. No target speedup is promised before those measurements.
