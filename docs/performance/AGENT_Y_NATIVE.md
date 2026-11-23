# Agent Y compiled CPU evaluation

Agent Y now has a compiled C++/OpenMP evaluator, not only Python multiprocessing.
`native/evaluate_agent_y.cpp` implements full-frame-conditioned group construction,
variable group counts, shared argmax embeddings, chain traversal, both action
policies, prediction feedback, structural corrections and join/new-group updates.
It uses the reference RNG streams. Native sources and compiler options key the
cached shared library. Agent X's kernel and dispatch remain unchanged.

Training batches use OpenMP workers; a Python champion replay checks metrics and
retains full inspection traces. Learned pair/router parameters can be inherited.
Frozen native assessment retains those parameters without updates. Native runs
support checkpoint/resume. Explicit Python-to-native Agent Y resume records a
backend migration rather than silently changing an existing run's evaluator.

## Measurement

A copied initial checkpoint from ui-20261122-142351-session-869d63 supplied its
first four 256-receptor agents, measured on 20 training and 40 assessment tasks
per agent. Lifetime inheritance was enabled to validate learned parameters too.

| Path | Seconds |
| --- | ---: |
| Serial Python reference | 6.417 |
| Native, first call including compilation | 1.180 |
| Native, cached compilation | 0.207 |

Native used four threads. The warmed evaluation was about 31× faster than serial
Python on this slice. Timings include packing and learned-mapping reconstruction,
not checkpoint saving, reproduction or the extra champion replay. This is one
pass pair under concurrent system load, not a whole-generation speedup guarantee.
Full result metrics and learned pair/router parameters matched within reference
parity tolerances. See AGENT_Y_NATIVE_SAMPLE.json for raw timings.

## Verification

The final full regression suite passed 344 tests. Browser checks confirm native
CPU is the Agent Y creator default and the fixed partition-count field is hidden.

Tests cover agreement/vote on Number Line, Lights Out 2×2 and switchboard; mixed,
single-group and all-singleton partitions; frozen and learning-enabled signals;
all learned mapping/router parameters; native uninterrupted/resumed equality;
and explicit Python-to-native migration. A real session API test trains in Python,
loads/replays its champion, and resumes with native CPU.

Floating-point parity uses the existing numerical tolerance. Exact genotype hashes
across different evaluator implementations are not promised when tiny learned
floating-point differences occur. A fixed backend's checkpoint replay remains
exact. No task-success or structural-shielding claim follows from these checks.
