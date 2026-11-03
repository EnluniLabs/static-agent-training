# One complete learning event: categorical versus fixed senses

This record audits the actual Python reference learning functions. It separates
shared update timing from changes to the prediction objective and credit assignment.
It is a deterministic mechanism experiment, not a training-performance result.
No learning-rule changes were made to make this trace pass.

Full before/after distributions, parameters, event payloads, and fixture data:

- [Categorical trace](categorical.json)
- [Fixed-senses trace](senses.json)
- [Reproduction code](../../../static_agent_training/learning_trace.py)
- [Assertions](../../../tests/test_learning_trace.py)

Reproduce from the repository root:

```sh
./project trace-learning --output docs/records/learning-event
./project check tests.test_learning_trace tests.test_learning tests.test_bit_model tests.test_sensory_accelerated
```

## Fixture and method

Both models use the same four-state repertoire developed from `(I,K)` with budget
20. Start is 5, goal is 0; the first explicit expectation is position 6 (or its
sensory pattern). The following action moves zero, so the outcome remains 5.
The source state receives one empirical update. On its next encounter it performs
one delayed structural correction, then emits another prediction.

The fixture sets initial embeddings to equal distributions and calibrates the
sensory gate so both models initially assign 0.820036173331 total probability to
prediction. This calibration is **not the training default**. Scripted inverse-CDF
draws choose the matched path without modifying parameters during the event.
A 0.5 Bernoulli draw selects modal bits. The fixture uses weak initial logits so
one update changes the modal prediction from 6 to 5. Actual stochastic outcomes
are not guaranteed to follow this path.

Observation, expectation, action/outcome, feedback, and correction use production
implementations. For the second encounter, traversal and the encounter hook are
split explicitly so snapshots can bracket structural correction; output sampling
then runs without repeating that correction. The categorical next-output sampling
and expectation registration are likewise split to expose the intermediate state.
This is instrumented stepping, not an unmodified end-to-end runner trace. Native
CPU and CUDA implementations are separately checked against Python episode metrics
and all learned parameters in `tests/test_sensory_accelerated.py`.

JSON `before` and `after` entries include full distributions and parameter snapshots.
Categorical prediction probabilities are conditional on choosing an expectation;
sensory prediction is factorized, so all 256 Bernoulli probabilities and all ten
gate probabilities are recorded rather than enumerating 2^256 patterns. Complete
observed-pattern probability is also recorded. Repertoire state columns follow
sorted pair IDs; `pair_aliases` identifies the checked successor. No probabilities
from the two different sample spaces should be treated as the same quantity.

## Stage-by-stage observations

| Stage | Categorical | Fixed senses |
|---|---|---|
| Observation | Embeds position 5 plus goal context; parameters unchanged | Embeds the fixed receptor pattern for `[5,0]`; parameters unchanged |
| Expectation | Emits `expect_+6`; registers a pending prediction | Emits the receptor pattern of `[6,0]`; saves pending bits and emission logits |
| Outcome | Zero move returns position 5; feedback waits for `receive` | Same action and physical outcome; feedback waits for `receive` |
| Empirical update | Conditional probability of position 5: 0.0243842666840 → 0.0256024740776 | Probability of the full observed pattern: 2.97479149411e-77 → 6.93696364472e-76 |
| Structural correction | Checked successor probability: 0.00606816552012 → 0.00637730290328 | Same checked successor probability before and after |
| Next prediction | Emits `expect_+5` under the controlled draw | Emits the exact receptor pattern of `[5,0]` under the controlled draws |

The empirical losses are 3.71381716269 (41-way categorical cross entropy) and
176.208878210 (summed 256-bit binary cross entropy). These losses have different
sample spaces and scales. Both use prediction rate 0.05, but that does **not**
make the learning dynamics equivalent.

## Preserved learning behavior

- Explicit predictions wait for the next physical observation; an expectation alone does not update weights.
- Feedback updates only the emitting source state's prediction parameters; embeddings wait for structural correction.
- Revised-state eligibility precedes structural correction; the check runs on a later encounter, against the fixed successor.
- Structural correction triggers below threshold 0.1 and raises the checked successor logit by 0.05.
- No lifetime operation changes repertoire nodes or edges. Correct predictions also receive empirical updates; frozen evaluation disables both rules (covered separately by regression tests).

## Representation changes and learning-rule changes

| Difference | Classification and consequence |
|---|---|
| Per-token embedding → fixed receptors with shared weights | Intended interface change. New values require no additional learned input rows. Different stimuli can share components or collide. |
| Position-only prediction plus separately retained goal → prediction of the entire sensory pattern | Additional prediction-target change. The sensory model must also predict unchanged goal/format/padding receptors. This is not just a new input encoding. |
| 41-way prediction softmax → 256 independent Bernoulli bits | Material objective change. Bitwise BCE provides component-level credit even when exact pattern correctness is false. It does not implement learned semantic equivalence. |
| One 50-way action/expectation softmax → ten-way action/prediction gate plus bit sampling | Material output-policy change. At zero logits, prediction probability is 41/50 = 82% versus 1/10 = 10%. The fixture calibrates this away initially to isolate updates. |
| Empirical update changes expectation logits inside the full output softmax; sensory gate is frozen | Material learning-policy coupling change. Even after calibration, categorical prediction mass changes 0.820036173331 → 0.820040587678, while sensory mass stays 0.820036173331. |
| One embedding cell +0.05 → bias and active weights each +0.05/(1+active count) | New component credit rule. Here 41 active features receive +0.00119047619048 each, together with bias. The checked logit still rises exactly 0.05, but unrelated inputs also move through shared features and global bias. |

For example, the structural step leaves the categorical position-6 and position-7
embedding distributions unchanged. In the sensory model, their successor
probabilities change from 0.00606816552012 to 0.00636223340026 and 0.00636976372394.
That is directly measured interference, not evidence of beneficial generalization.

## Verdict and unresolved design decisions

The models preserve the two-stage local-learning schedule, **not identical learning
rules**. The changes above were explicit implementation choices already described
as provisional in `docs/AGNOSTIC_INTERFACE.md`; the project notes leave prediction
normalization and credit assignment open. Documentation does not establish user
approval or demonstrate that these choices are required by the agnostic interface.
In particular, action/prediction odds and empirical gate coupling changed beyond
what fixed sensory capacity alone requires. Previous statements about feature
parity refer to training/tooling support and must not be read as mathematical
learning-rule equivalence.

Before attributing benchmark differences to the interface alone, specify the
intended prediction prior, whether empirical learning should update prediction
versus action propensity, how to scale component loss, and which shared structural
credit rule is intended. Then compare controlled variants. This record leaves
those design choices visible instead of silently selecting new rules.
