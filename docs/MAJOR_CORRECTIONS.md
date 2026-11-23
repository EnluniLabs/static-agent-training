# Major Corrections

Status: **Required; implementation pending**. This records the three immediate architecture corrections identified during the November 22, 2026 Agent X code review. Do not mark an item complete until both the Python and native CPU paths, serialization, and tests agree.

## 1. Correct lambda-pair transition orientation

**Intended convention:** for an ordered pair `(B, A)`, apply `B` to `A` to obtain `C = NF(B A)`, then construct the next pair `(C, B)`. In general, `(f, x) -> (NF(f x), f)`.

**Current code:** `static_agent_training/engine.py` constructs `('app', pair[1], pair[0])` and `successor = (pair[1], expression)`, yielding `(B, A) -> (A, NF(A B))`. `static_agent_training/repertoire.py` validates `pairs[source][1] == pairs[target][0]`, which encodes that reversed convention. The native evaluator follows the precomputed successor IDs and does not beta-reduce terms.

**Required:** fix construction and validation together; audit all assumptions about pair order, serialization, generated repertoires, and tests. Existing serialized repertoires may require regeneration or explicit migration. Add a test with asymmetric lambda terms for which `NF(B A)` and `NF(A B)` differ, so reversal cannot pass accidentally.

## 2. Replace hard-coded action/prediction gates with universal output signals

**Intended:** the agent emits uninterpreted output signals through a universal interface, analogous to sensory input. The external environment/interface interprets signal patterns as physical effects. The agent must not internally classify its outputs as actions versus predictions or sample a hard-coded categorical action/prediction gate.

**Current code:** `static_agent_training/native/evaluate_senses.cpp` samples ten categorical `gates` entries (nine action tokens plus a prediction token). `static_agent_training/experimental/bit_model.py` similarly separates action tokens and sensory prediction channels.

**Required:** design a fixed-shape, type-agnostic output-signal representation and environment adapter; replace categorical gate semantics in Python and native CPU implementations. Preserve multi-region output aggregation and a way to evaluate predictions without making 'prediction' an intrinsic output-token type. Specify how output signals and subsequently observed sensory signals are paired for empirical error measurement before implementing the learning rule.

## 3. Make empirical learning update the actual universal output mapping

**Intended (refined November 22):** empirical error corrects the continuation distribution only when the environment interprets a sampled output as an expectation that can be checked against an observation. A physical-action output causes no immediate empirical or structural correction. Structural error adjusts the deterministic successor's input embedding for compatibility with the original sampled expectation, not with the observed signal. No direct goal-reward/action-policy gradient is introduced.

**Current code:** lifetime empirical learning changes `prediction` logits only. `gates` do not receive lifetime updates. Structural learning modifies `weights` and `bias` for the successor embedding.

**Required:** after correction 2, train the universal output-signal distribution using a precisely specified empirical target and error rule for expectation outputs. Do not silently add reinforcement learning of action rewards or empirically update physical-action outputs merely because they occurred. Verify appropriate expectation-output dimensions receive updates when empirical evidence exists. Keep evolution distinct from within-lifetime learning.

## Structural consistency: preserve and reassess

The existing structural check asks whether a modal predicted sensory frame would select the current pair's fixed successor with probability above a threshold, then increases successor embedding logits if not. This is a local compatibility rule, not full-distribution equality. Its basic principle may be retained, but its inputs and interpretation must be revisited after corrections 1–3. Do not change the structural error rule solely on the assumption that it is already proven wrong.

## 4. Sampled-expectation structural correction and simultaneous error calculation

**Required correction (not an optional hypothesis):** for a deterministic transition `(B,A) -> (C,B)`, let the predecessor sample a concrete output signal `s` from its pre-update continuation distribution. The environment adapter, not the agent's internal architecture, determines whether the emitted signal is an expectation or a physical action.

- **Physical action:** proceed to the deterministic successor without an immediate empirical or structural error update.
- **Expectation:** obtain the corresponding environmental observation `o`. Calculate empirical error between the sampled expectation `s` and `o`, targeting the predecessor's continuation distribution. Independently calculate structural error from compatibility between the **same original sampled expectation `s`** and the successor's pre-update input embedding.
- **Snapshot and apply:** calculate both errors using the same pre-update parameter state, then apply the respective corrections. Empirical correction updates the predecessor's continuation distribution; structural correction updates the successor's embedding. Never replace `s` with the observation, a newly sampled prediction, the modal output, or the entire continuation distribution for the structural comparison.
- **Even if wrong:** an empirically inaccurate sampled expectation still supplies the structural target. This deliberately separates grounding of the continuation distribution in observations from enforcing internal compatibility along coherent lambda transitions.

**Acceptance tests:** verify structural correction uses the exact sampled expectation (not a distribution average or argmax), is unaffected by applying the empirical update, targets the actual deterministic successor, and is skipped for action outputs. Verify empirical and structural corrections use pre-update snapshots, including when the expectation is false. Apply consistently to Python and native CPU paths.

**Interface nuance:** the agent must not contain a hard-coded action-versus-prediction gate. The environment adapter may interpret signals and decide whether a prediction can be evaluated. The precise signaling protocol remains to be designed.

## Acceptance checks

- Asymmetric reduction test proves `(B,A) -> (NF(B A), B)` in development, repertoire validation, and native traversal.
- Python and native CPU emit the same type-agnostic output-signal representation; no special action/prediction gate remains.
- Empirical learning demonstrably changes appropriate output-signal probabilities during lifetime, without goal-reward gradients.
- Structural learning still checks compatibility with the actual successor, and tests distinguish empirical from structural updates.
- End-to-end benchmark and parity tests run; report any performance changes and migration requirements.
