# Current E3 training benchmark — 2026-11-22

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](../ARCHITECTURE.md), reviewed 2026-11-22.

The active run was `runs/ui-20261122-113151-E3-065d76` (E3), PID 143337.
Its trainer was not stopped, restarted or modified. A completed checkpoint was
copied through one open file descriptor to a private `/tmp` directory, then one
generation was replayed there using the production training loop. Every replay
write targeted that private directory. No benchmarking changes to training code
were needed.

## Configuration

32 fixed-senses agents, 2×2 Lights Out, 100 training and 200 assessment episodes
per agent, 256 binary signal receptors, four sensory partitions, vote policy,
native evaluation with four workers, automatic offspring processes, no speciation
and no lifetime-weight inheritance. The imported population had grown beyond its
original seed count: the preceding saved generation averaged roughly 461
repertoire states and 146 unique starting pairs per agent.

The checkpoint's `next_generation` was 8: the replay measured zero-based generation
8, displayed as generation 9. Its compressed input was 104.7 MiB. The source
checkpoint SHA-256 and complete configuration are in the
[raw measurement](CURRENT_RUN_SAMPLE_20261122.json).

## Results

Measured generation span: **75.51 seconds**, excluding checkpoint restoration and
startup. Recent live generation-record intervals were **70.78, 72.57, 74.80 and
74.63 seconds**. These are one-generation observations under concurrent training,
not isolated estimates of a universal throughput limit.

| Stage | Seconds | Share of measured generation |
| --- | ---: | ---: |
| Checkpoint conversion, JSON encoding, gzip, write and sync | 38.21 | 50.6% |
| Best-candidate loading, validation, selection and saving | 20.63 | 27.3% |
| Selection and offspring reproduction, including process transport | 6.54 | 8.7% |
| Evaluation, including reference verification | 6.16 | 8.2% |
| Champion trace write | 3.21 | 4.3% |
| Generation record write | 0.29 | 0.4% |
| Other orchestration | 0.46 | 0.6% |

Within evaluation, native phase wrappers took 1.59 s, packing 0.19 s, and the
Python champion reference 4.20 s. Within best-candidate maintenance, writing alone
took 15.88 s. These nested measurements are already included above. Best-candidate
loading took 2.07 s, including validation inside `read_population`; further
validation/selection accounts for additional time before its write.

Checkpoint saving already used gzip level 1 and batched JSON writes. Its 38.21 s
includes per-agent `to_data()` conversion, serialization, compression, file sync
and directory sync, not just disk throughput. The next compressed checkpoint was
105.0 MiB. Ordinary best-population and trace archives still use gzip level 9.

Saved-artifact maintenance accounts for approximately **82.6%** of this measured
generation. Additional evaluator workers alone would address only a small part
of the measured time. The highest-priority follow-up is profiling and reducing
repeated serialized development provenance and best-candidate read/validate/write
work, while preserving all required records and crash-safe checkpoint behavior.

## Live resource observation

A concurrent 40-second `/proc` sample measured about **1.11 CPU cores** for the
trainer and **0.77 CPU cores** for the visualizer. These are process CPU-time
increments divided by wall time; the trainer figure excludes subprocess CPU.
Offspring workers run only during reproduction, so that figure is not total
training process-tree consumption. The available CPU/memory/I/O pressure reports
had zero recent stall averages during the sample. The visualizer therefore also
warrants separate profiling for repeated expensive reads.

The replay's peak main-process RSS was **2,403 MiB**, excluding offspring workers.
A 5 GiB virtual-address ceiling bounded the benchmark process. Loading its
checkpoint JSON took 5.00 s; restoration and warm-up are excluded from the
per-generation table. Genotype IDs were warmed during restoration to match the
already-constructed offspring in a steady-state generation. Native parameter
packing was not artificially warmed.

## Correctness and limits

The replayed generation record exactly matched the corresponding live generation
record, including evaluation results, task/population identifiers, child identities
and mutation logs. Both raw-file SHA-256 values are recorded. Native/reference
verification ran normally in the production evaluation path.

The source checkpoint was pinned atomically, but separately copied candidate and
historical files were not a joint filesystem transaction; they affect archival
bookkeeping, not the replay's training/selection tasks. The live run continued
into later generations throughout. Concurrent training, visualizer work and the
benchmark itself may influence timing. No success-rate conclusion follows from
this runtime benchmark.
