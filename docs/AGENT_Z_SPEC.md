# Agent Z — Technical Implementation Specification

Status: **Required design specification; implementation pending** (November 22, 2026).

Build Agent Z as a separate implementation in `EnluniLabs/static-agent-training`, preserving Agent X as a runnable historical baseline. Inspect `docs/MAJOR_CORRECTIONS.md`, `static_agent_training/engine.py`, `repertoire.py`, `experimental/bit_model.py`, `experimental/signals.py`, `native/evaluate_senses.cpp`, existing runners, tests, and environment adapters before implementing.

## 1. Fixed coherent structure

For ordered pair `(B,A)`, calculate `C = NF(B A)`, applying B to A, and transition deterministically to `(C,B)`. The current engine uses reversed application and successor orientation; correct it for Agent Z and audit validation, traversal, serialization and tests. Use asymmetric lambda terms to test orientation. Successors are fixed mathematical facts, never learned. Handle unavailable normal forms or exhausted development budgets explicitly. Do not silently reinterpret old serialized Agent X repertoires.

## 2. Learned input and universal output

An opaque sensory signal is mapped through a learned input embedding `E_theta(q | s)` to lambda pairs. Each active pair `q` samples an opaque signal `s` from a learned continuation distribution `D_phi(s | q)`. Output signals are not internally typed as actions or expectations. Remove the categorical action/prediction gate and separate action-policy and prediction distributions from Agent Z, including native CPU code. Use a common signal representation for sampled outputs and environmental measurements, or a documented fixed, nonlearned conversion. Preserve sensory partitioning and multiple active chains.

## 3. Environment interpretation and feedback

The environment defines exact signal-to-physical-action mappings; the agent does not. For each emitted signal:

- If the signal exactly matches a physical action mapping, execute that action, return **no error-feedback measurement**, and advance the existing lambda chain without immediate empirical or structural correction. Do **not** automatically re-embed a new sensory observation after the action. Several actions may occur consecutively before measurement.
- Otherwise interpret the emitted signal as an expectation, return the environment's **current sensory measurement**, and trigger both error calculations. The measurement reflects physical changes already executed.

Feedback presence, not an internal action/expectation flag, determines whether correction occurs. Represent absent feedback distinctly from a valid all-zero measurement. Distinguish ordinary initial sensory activation from feedback returned for learning. Keep task success and fitness outside the agent's signal semantics.

## 4. Two independent errors from one pre-update snapshot

For pair `q` with deterministic successor `q'`, sampled expectation `s`, and returned observation `o`:

- **Empirical:** compare the emitted expectation against the actual measurement and update the predecessor's learned continuation distribution `D_phi(. | q)` toward the observed signal, using an explicitly documented distribution-appropriate objective (e.g. bitwise negative log likelihood).
- **Structural:** evaluate compatibility of **the same exact sampled expectation `s`** with the successor's **pre-update** input embedding `E_theta(q' | s)`. Correct the successor's embedding when incompatible. The structural target is not the observed `o`, the modal output, an average over the output distribution, or a resampled output.

Calculate both errors from a shared pre-update parameter snapshot, then apply corrections. An empirically false expectation **still** supplies the structural target. Do not add direct goal-reward/action-policy gradients to lifetime learning. If no feedback was returned, neither update occurs.

## 5. Simultaneous chains — mandatory shared feedback

When multiple chains emit expectations in the same evaluation round, the environment returns **one current sensory measurement `o` shared by every expectation-producing chain**. Each chain independently uses its own sampled signal, originating pair, and deterministic successor:

```
chain 1: q1 emits s1 ─┐
                     ├── environment current measurement o
chain 2: q2 emits s2 ─┘

empirical 1: compare s1 with o; update D(.|q1)
empirical 2: compare s2 with o; update D(.|q2)
structural 1: evaluate E(q1'|s1); update successor q1' embedding
structural 2: evaluate E(q2'|s2); update successor q2' embedding
```

**The measurement is shared; numerical errors need not be equal.** Never average sampled expectations before computing errors or substitute another chain's signal. Capture every chain's error from the **same round-wide pre-update snapshot** before applying any updates. If updates touch shared parameters, aggregate their contributions using a deterministic, documented, order-independent rule. Preserve provenance for each emitted signal and feedback calculation.

A physical-action emission does not itself receive an error target or immediate correction. For mixed rounds with actions and expectations, define deterministic environmental ordering and the precise measurement instant, ensuring all expectations in that round share one observation. Do not silently inherit the old categorical action-voting policy. Specify how simultaneous action signals are resolved externally, without making the agent internally classify outputs.

## 6. Execution and learning boundaries

Preserve evolution and lifetime learning as distinct mechanisms; fitness may drive evolutionary selection but not a new lifetime reward-gradient rule. Keep seeded reproducibility, learning-enabled/disabled evaluation, held-out tasks, and empirical-only/structural-only ablations. Preserve existing multi-region architecture where compatible. Keep Agent X runnable and unchanged.

## 7. Deliverables and acceptance tests

Implement a Python reference Agent Z, a semantically matching native CPU path, an environment adapter, versioned serialization, documentation, and reproducible benchmarks. Do not report completion until tests run.

Tests must verify: asymmetric lambda orientation; fixed successor; universal output sampling without categorical gates; exact environment action mapping; action execution with no feedback or immediate correction and no automatic re-observation; expectation feedback including zero-valued observations; empirical updates to predecessor continuation; structural updates to actual successor from the **sampled** expectation even when empirically wrong; independent pre-update error calculation; shared round observation for multiple expectation chains with distinct errors; deterministic aggregation of shared parameter updates; mixed-round handling; Python/native parity; and preserved Agent X baseline.

Document unresolved implementation choices (especially simultaneous physical-action arbitration and structural-loss form) explicitly rather than silently inventing architectural commitments. Report actual test results, performance changes, migration needs, and limitations.
