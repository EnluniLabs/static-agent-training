# Experimental fixed sensory apparatus

The experimental model lives in `static_agent_training.experimental`. Its default
experiment now uses an animal-owned sensory apparatus, rather than the earlier
position/goal byte-slot prototype. The sensory model now uses the shared
evolutionary trainer as well as the standalone adaptation experiment. Existing
categorical checkpoints and native CPU evaluation remain supported.

## Evolutionary training

`./project train --model senses --population 32 --generations 50 --receptors 256`
runs actual population evolution. Sensory weights, biases, action/prediction
gates and prediction-bit logits form immutable inherited mappings in the shared
`Species` genotype. The existing structural mutations, tournaments, elites and
optional speciation operate on these genotypes. Retained state IDs preserve their
parameters after redevelopment; newly discovered states receive deterministic
parameters. Receptor capacity does not change. Learned-weight inheritance is
optional, disabled by default. Generational checkpoints resume or extend through
the same trainer; champion traces retain the trained sensory model for frozen
playback and benchmarks. Both models support Python reference and native CPU evaluation. Accelerated training replays the selected champion through Python and checks its results.

The visualizer exposes **Fixed-senses model → Evolutionary training** and
**Compare both models → Evolutionary comparison**, both with generations.
`./project compare --treatment model --generations 50` uses matched initial
structures and tasks and freezes training-selected champions for final testing.
The separate `senses` and `compare-models` commands retain their explicit
fixed-structure lifetime experiments.

## Stable boundary across environments

`FixedSenses` fixes an animal's receptor count at construction (256 by default).
Environments emit arbitrary-length byte stimuli through `StimulusEnvironment`.
`SensoryAdapter` sends them through that animal's saved, nonlearned transducer.
The model receives one fixed-width sensory pattern, regardless of environment,
input vocabulary size, input length or the meanings of the external data.
It allocates no learned rows for new values and performs no semantic parsing.

Worlds still need physical adapters: a number, text string, image or audio sample
must become bytes. Such conversion does not imply that different modalities or
serializations are perceptually equivalent. This is an explicit finite sensory
bottleneck, not a lossless universal representation or a human sensory simulation.

### Receptors and similar components

The prototype uses `dual-rail-position-fold-or-v1`. Each source bit position
activates one of two receptors, for zero or one. Source bit positions wrap at
half the receptor count; overlapping activations combine by OR. Both matching
zeros and matching ones therefore share receptors. Similar components retain
shared activations, unlike hashing the complete stimulus into an unrelated ID.

With 256 receptors, up to 16 input bytes fit without positional folding. Longer
stimuli still pass through the same bank, but may collide or saturate. The finite
capacity, loss and encoding bias are deliberate prototype choices, not claims
that this is the optimal transducer. Variable-length input is accepted without
resizing, truncation or allocating per-value parameters. Missing input and empty
but present input are distinguished. Changing receptor count creates a different
animal; it is not a transfer operation.

The same direct shared map computes:

    entry_logit[j] = bias[j] + sum(receptor[i] * W[i,j])

There is also a fixed presence feature. There is no second learned encoder.
Updating shared components changes the interpretation of other stimuli that
activate them. Such interference can help or harm; it does not establish that
similar sensory patterns have similar physical meaning.

### Number-line adapter

`StimulusNumberLine` serializes its current position and goal together as a UTF-8
JSON array, e.g. `[12,-3]`. Those meanings and serialization belong to the world,
not to model input slots. The model sees only the generic receptor pattern and
does not parse JSON or numbers. Larger bounds and longer integer representations
require no model changes. UTF-8 JSON is an encoding choice whose biases need
controls; changing the serialization is a changed sensory experience.

Other worlds can emit text, raw bytes or other serialized measurements through
the same adapter and saved senses. There is no requirement that they expose a
position, a goal, or the same number of possible observations. The first experiment
retains nine opaque action indices; changing motor capacity is a separate design
question. Each world must interpret those actions explicitly.

## Expectations and local learning

After following a fixed repertoire successor, a categorical gate chooses an action
or a sensory prediction. A prediction independently samples each receptor bit,
emitting a complete sensory pattern atomically. It predicts what the animal will
sense, not a decoded external object. Any pattern in the fixed sensory space is
expressible without adding output categories. Some generated patterns will not
correspond to realizable stimuli; actual feedback still tests them.

Predictions do not advance physical time. Multiple predictions can precede an
action; the next observation tests them in emission order. Missing observations
expire them without refutation; terminal observations still test predictions.
A sixteenth consecutive prediction ends an episode under the default internal
limit. Movement as the sixteenth output remains valid.

Empirical feedback records exact sensory-pattern identity. Tested predictions
update only their source state's bit logits using summed Bernoulli cross entropy.
Correct predictions also update. Traces retain emission-time prediction NLL and
sequential current-logit training loss. Gates are not updated by this objective.
Exact pattern identity is not necessarily external stimulus identity when the
sensory transducer has collisions. Learned functional equivalence and graded
semantic refutation are not implemented.

After an empirical revision, later encounters check whether the source state's
modal sensory prediction maps compatibly to its fixed successor. If probability
is below threshold, its successor bias and active receptor weights each increase
by `embedding_rate / (1 + active_feature_count)`. The predicted pattern's successor
logit increases by exactly `embedding_rate`; other stimuli change according to
shared activations. Other state columns remain untouched. This is a specified new
local credit rule, not a claim of equivalence to the tabular model's rule.

Neither learning mechanism changes repertoire nodes or edges. Resets clear pending
predictions and traversal, retaining parameters, RNG streams and revision history.
Frozen evaluation disables both updates. Between-episode snapshots save the sensory
transducer, fixed capacity, model parameters, repertoire, RNG streams and traces.
A restored animal requires its saved transducer when connected to a world.

## Reference experiment

```sh
.venv/bin/python -m static_agent_training.experimental.bit_experiment \
  --candidate runs/your-run --episodes 100 --assessment-episodes 200 \
  --receptors 256 --seed 48 --output runs/fixed-senses-test
```

Omit `--candidate` to develop a fresh four-seed repertoire. A saved champion
supplies only structure: its categorical learned weights are not imported.
The command performs fixed-structure lifetime learning, not evolution.

It adapts in [-20,20], then compares prior experience plus adaptation, prior
experience without additional adaptation, and a fresh model adapting in [-30,30].
All arms share adaptation tasks and sampling seeds. Frozen evaluation excludes
exact task pairs used in either adaptation block. These are new task pairs, not
held-out values. Reports separate old-region, new-region, crossing and expanded
performance, prediction metrics and update activity. Empty regions report null.
All stages use the same receptor count, weights shape and repertoire.

This checks operation, not useful transfer. In particular, the earlier two-slot
smoke run used only three repertoire nodes and triggered no structural correction.
It did not demonstrate shared-input learning or a transfer advantage. Dedicated
tests force structural conflicts and check cross-stimulus effects. Substantial
structural dependence, independent animals/seeds, encoding controls, matched random
repertoires and associative baselines are still needed for research conclusions.

The earlier `ByteNumberLine` and explicit `SignalSpec` path remain available to
read and test the original byte-slot prototype; the default experiment uses
`FixedSenses`, `SensoryAdapter` and `StimulusNumberLine`.

## Shared model capabilities

The sensory model is an alternative mapping implementation in the same evolutionary
system. It supports population training, structural and parameter mutation,
tournament selection, elites, speciation, optional learned-weight inheritance,
checkpoint resume/extension, saved champion collections, frozen benchmarks,
standalone replay exports, goal-input audits, and the visualizer.

Both models support `--backend python` and `native`. Each accelerated
sensory lane owns its bias, shared input weights, action/prediction gates and
prediction logits. Episodes within a lifetime remain sequential. Training weights
continue into frozen assessment; assessment updates never enter reproduction.
Only the selected champion receives a full Python trace, independently checked
against the accelerated results before saving. Receptor counts above 16384 or
repertoires above 4096 states require the Python backend. Batch allocations retain
the shared 512 MiB guard.

`--boundary-mode clamp|terminate` and fixed/varying goals work with either model.
Removing the goal in a sensory audit removes it from the environment's serialized
packet; permuting goals changes the presented goal while retaining the physical
target. These interventions operate on the sensory input, rather than editing
categorical embedding rows.

Examples:

```sh
./project train --model senses --backend native --receptors 256 --population 32 --generations 50
./project compare --treatment model --backend native --population 32 --generations 50
./project compare --model senses --treatment learning --backend native --population 32 --generations 50
./project audit latest-sensory-training --sample 8 --repeats 1
./project speed --models categorical senses --backends python native --populations 32 --repeats 2
```

Receptor encoding and factorized prediction remain intentional differences from
the categorical model. Larger runs do not by themselves establish that either
representation learns better. The sensory bank can still lose distinctions when
long inputs fold onto its fixed capacity.

## Learning-rule trace audit

The [paired learning-event record](records/learning-event/README.md) contains complete before/after
distributions for observation, expectation, outcome, empirical feedback, delayed
structural correction and the next prediction. The sensory model preserves the
two-stage learning schedule but changes the prediction objective, output-policy
coupling and shared structural credit assignment. Tooling parity does not imply
identical learning dynamics. Reproduce with `./project trace-learning`.
