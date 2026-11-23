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

**Intended:** empirical error corrects learned signal representations, including signal patterns the environment may interpret as physical actions, without a direct goal-reward/action-policy gradient. Structural error adjusts input/representational mappings to maintain compatibility along coherent transitions.

**Current code:** lifetime empirical learning changes `prediction` logits only. `gates` do not receive lifetime updates. Structural learning modifies `weights` and `bias` for the successor embedding.

**Required:** after correction 2, train the universal output-signal distribution using a precisely specified empirical target and error rule. Do not silently add reinforcement learning of action rewards. Verify that all output-signal dimensions intended to learn actually receive gradients/updates when empirical evidence exists. Keep evolution distinct from within-lifetime learning.

## Structural consistency: preserve and reassess

The existing structural check asks whether a modal predicted sensory frame would select the current pair's fixed successor with probability above a threshold, then increases successor embedding logits if not. This is a local compatibility rule, not full-distribution equality. Its basic principle may be retained, but its inputs and interpretation must be revisited after corrections 1–3. Do not change the structural error rule solely on the assumption that it is already proven wrong.

## Acceptance checks

- Asymmetric reduction test proves `(B,A) -> (NF(B A), B)` in development, repertoire validation, and native traversal.
- Python and native CPU emit the same type-agnostic output-signal representation; no special action/prediction gate remains.
- Empirical learning demonstrably changes appropriate output-signal probabilities during lifetime, without goal-reward gradients.
- Structural learning still checks compatibility with the actual successor, and tests distinguish empirical from structural updates.
- End-to-end benchmark and parity tests run; report any performance changes and migration requirements.
