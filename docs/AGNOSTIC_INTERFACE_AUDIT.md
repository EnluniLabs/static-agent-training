# Comparison with the GitHub signal-interface notes

Reviewed 2026-11-02 against `FUTURE_WORK.md` on GitHub, section
"Proposed type-agnostic signal interface and generalization filter", content blob
`7d39a11bfa80d23bd880d6cbaacc4f8e847427e5`, and the user's subsequent clarification:
the animal has a fixed sensory capacity across changes of environment and input
vocabulary size. This audit changes no model or experimental result.

Source: https://github.com/EnluniLabs/static-agent-training/blob/main/FUTURE_WORK.md

Subsequent integration on 2026-11-02: the sensory model is now connected to the
shared evolutionary trainer, including structural/weight mutation, selection,
elites, optional speciation and learned inheritance, checkpoint/resume, champion
collections, frozen benchmarks and generation controls in the visualizer.
The original audit below describes the earlier fixed-structure prototype;
its converter and retention findings still apply. Evolutionary integration
does not by itself establish useful transfer or resolve the encoding concerns.

## Verdict

The core fixed-capacity boundary and shared map are implemented. The current
code is a candidate reference variant, not a fully validated realization of the
notes. In particular, the long-input transducer is weak and original-world
retention is not measured correctly. Successful shape checks and unit tests
must not be presented as evidence of useful adaptation or robust generalization.

## Requirement comparison

| Requirement | Finding |
|---|---|
| Fixed sensory capacity across new values and environments | Implemented by animal-owned `FixedSenses`; adapters use its saved specification. |
| No new learned rows or resizing for unseen values | Implemented; fixed input matrix and bit-prediction arrays depend on receptors and repertoire, not vocabulary size. |
| Direct shared map, without a second learned encoder | Implemented: bias plus receptor features times W; sensory conversion is fixed and nonlearned. |
| Similar components share update effects | Implemented and mechanically tested. Dual-rail coding shares matching zero and one bits; the shared state bias also affects all inputs. This does not prove helpful transfer. |
| No supplied numeric meaning, distance, direction or suggested action | Implemented at the model boundary. The world serializes position and goal as ordered JSON values; the model does not parse them. |
| Distinguish input roles without value-specific rows | Ordered JSON supplies framing within the tested short-input domain. Folding can erase distinctions for longer inputs. |
| Predictions use the stable sensory representation | Implemented: factorized predictions over the same receptor pattern. No new prediction category is needed for a new stimulus. |
| Empirical and structural learning remain separate; repertoire stays fixed | Implemented with traceable new bit-loss and active-component correction rules. These are candidate rules, not unchanged tabular objectives. |
| Exact identity versus learned equivalence | Only exact receptor-pattern identity is implemented. Different raw inputs that collide are indistinguishable. Learned equivalence is an optional open design in the notes, not an implemented feature. |
| Explicit encoding, capacity and collision behavior | Saved and documented, but robustness and acceptable operating limits are insufficiently characterized. See measured saturation below. |
| Expansion without changing sensory/output shapes or structure | Implemented and tested for number-line expansion. This is interface compatibility, not demonstrated beneficial adaptation. |
| Original-region and retention results | Old-region tasks exist, but all evaluations run in the expanded world. Original-world retention is missing. |
| Separate variant preserving the existing baseline | Implemented in the experimental package; production evaluator and checkpoints are untouched. |
| Alternative encodings, associative/random-repertoire controls, learning ablations, independent animals/seeds | Not implemented as a comparative experiment suite. No causal or generalization claims are justified yet. |
| Unfamiliar symbols | Mechanically tested through the same animal and sensory adapter; no measured cross-environment task-learning comparison yet. |

## Finding 1: long-input folding saturates

`FixedSenses.frame` maps source bit positions modulo half the receptor count and
ORs dual-rail activations. At the default 256 receptors, the first 16 bytes have
distinct positions. Longer data superimposes and can quickly erase distinctions.

Checks with Python `random.Random(48)` and independently generated uniform bytes:

| Input bytes | Active receptors |
|---:|---:|
| 16 | 128 / 256 |
| 64 | 245 / 256 |
| 512 | 256 / 256 |

The two distinct 32-byte inputs `00` repeated 16 times followed by `ff` repeated
16 times, and the reverse ordering, produce exactly the same all-active pattern.
This is a converter failure mode, not learned functional equivalence. Every
ordered position/goal combination in [-30,30] was also checked: 3,721 stimuli
yielded 3,721 distinct patterns. Thus this issue does not invalidate sensory
identity in that bounded number-line test, but prevents claiming robust arbitrary
long-input support. The notes permit finite capacity and collisions; they do not
establish that this particular folding rule is adequate.

Before broader experiments, choose an explicit admissible input regime or revise
how finite senses receive larger inputs, and report collision/saturation behavior.
Do not silently replace this converter under existing saved weights.

## Finding 2: old-region evaluation is not original-world retention

`bit_experiment.experiment` creates old-region task pairs within [-20,20], but
evaluates every region with walls at [-30,30]. Reusing the original coordinates
does not restore original dynamics. For start 20 and goal 19, actions +5 then -1
reach 19 in the original world but 24 in the expanded world. A changed score can
therefore reflect wall geometry rather than forgetting.

Keep the expanded-world old-region score, but add frozen pre/post-adaptation
evaluation on the same reserved tasks with the original [-20,20] walls. Also
report immediate expanded-world performance before adaptation explicitly.

## Provisional architectural choices

The notes leave output timing, normalization and credit assignment open. The
implementation chose atomic complete-pattern emission, independent Bernoulli
receptor bits, a separate action/prediction gate, exact receptor identity for
correctness, summed bitwise cross entropy, and a normalized sparse structural
increment. Those choices are documented and testable, but are not uniquely
specified by the notes. Bit-loss factorization and dual-rail encoding introduce
inductive biases. There is no learned-representation graded matching and no
trajectory-wide structural-consistency objective; the latter is explicitly a
conceptual target rather than a required implemented loss in the notes.

## Validation limits

The targeted fixed-senses and bit-model tests pass (10 tests). They exercise
shape invariance, vocabulary growth, new byte stimuli, checkpoint replay,
local prediction learning, shared-component structural updates and frozen
evaluation. The full project suite passed 115 tests before this read-only audit.
The fixed-senses smoke run exercised both learning mechanisms on an 18-node
repertoire. None of these checks establish useful adaptation, semantic similarity,
or an advantage of lambda-derived structure over ordinary graph statistics.

## Learning-rule trace audit

The [paired learning-event record](records/learning-event/README.md) contains complete before/after
distributions for observation, expectation, outcome, empirical feedback, delayed
structural correction and the next prediction. The sensory model preserves the
two-stage learning schedule but changes the prediction objective, output-policy
coupling and shared structural credit assignment. Tooling parity does not imply
identical learning dynamics. Reproduce with `./project trace-learning`.
