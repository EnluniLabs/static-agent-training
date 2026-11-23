# Agent X: partitioned sensory activation

Current system context and model boundaries: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

New Agent X sessions default to the immutable `signals` interface described in
[Receptor signals](RECEPTOR_SIGNALS.md): a 16×16 layout of 256 binary receptors,
four fixed tiles, and one shared generic dual-rail byte transducer for all sources.
The layout indexes receptors; it is not a camera. Historical `retina` packed-pixel
snapshots and `legacy-bytes` controls remain supported with their saved interfaces.

Agent X is an experimental extension of the fixed-senses bit model. It lets
several regions of one receptor surface activate repertoire pairs concurrently.
It is a mechanism experiment, not evidence of better learning or task success.
New Agent X sessions use four fixed tiles; the single-partition baseline remains available.

Construct `BitModel` with saved `SignalSenses` (or historical `FixedSenses`), `sensory_partitions=N`,
`embedding_selection='argmax'`, and `action_policy='agreement'` or `'vote'`.
More than one partition requires fixed senses and argmax selection. The count
must be an integer from one through the receptor count. Signals receptors use fixed tiles in the saved indexing layout. Historical
`legacy-bytes` receptors use contiguous balanced regions, with extra receptors
assigned to earlier regions. Presence feature zero is shared by every region. Each region embeds
independently using its original feature indices in the shared weight matrix.
Selection ties use the model's seeded embedding RNG. Duplicate activations are
retained: two regions selecting the same pair are still two activations.

`observe(frame)` establishes the regional activations. `emit_round(learn=True)`
advances each live activation by one repertoire successor and returns a tuple
of `Action` and `Prediction` emissions; terminated activations contribute no
emission. Region identity follows the successor. `run_episode` uses rounds when
there is more than one partition. The existing single-partition `emit` path
remains supported.

All predictions in a round are registered before the environment processes its
actions. Each prediction has an independent pending identity, including
identical predictions from the same pair. Feedback compares the complete
fixed-width signal, not just the originating region. Duplicate feedback applies
empirical updates sequentially to the shared prediction row. Prediction loss
for feedback uses the logits saved at emission, even when an earlier pending
prediction has already changed the row. The pending prediction payload retains
its existing pair/prediction shape; regional bookkeeping is separate metadata.

The environment resolves a round's actions through `step_batch(actions, policy)`:

| Policy | Resolution |
| --- | --- |
| `agreement` | Execute once if all proposed action indices agree; otherwise no-op. |
| `vote` | Execute the unique plurality winner once; a tie is a no-op. |

Multiplicity counts as votes. A resolved batch consumes one environment action,
including a conflicting or tied no-op, and supplies the resulting observation
for feedback. A prediction-only round does not itself step the environment.
`BATCH_ACTION` records the resolution; regional model events make concurrent
activation and prediction attribution inspectable.

Snapshots preserve partition count and action policy. Historical snapshots
without these fields load with one partition and agreement; the existing
historical embedding-selection fallback remains sampling. Frozen evaluation
can emit events and advance rollout RNG streams but must leave learned
parameters, revision metadata, and repertoire structure unchanged. Reading
parameters or the structure hash does not consume RNG.

This extension does not change repertoire development, mutation, or the
single-partition learning baseline. Shared weights and shared prediction rows
still couple regions; region count is not an independent increase in parameter
capacity. The finite receptor transducer is lossy, and deterministic argmax
selection still has seeded random tie resolution. Performance claims require
matched seeds, repertoire, sensory capacity, action budget, learning budget,
and assessment tasks, with agreement and vote reported separately.

The historical mechanism fixtures in `tests/test_agent_x.py` use a developed real repertoire,
fixed senses, and forced gates. Run them with:

```sh
.venv/bin/python -m unittest tests.test_agent_x -v
```

They cover balanced original-index masks and shared presence, regional successor
execution, duplicate predictions and sequential updates, argmax tie streams,
prediction/action ordering, environment arbitration, frozen replay, snapshot
compatibility, and invalid configurations. Playback learning is compared with
`run_episode`, including subsequent frozen playback. Native parity checks compare
metrics and all learned parameters with Python for uneven three-way partitions,
both policies, Number Line, and Lights Out. A separate puzzle check verifies
that a tied batch consumes one action without changing the board. These checks
establish behavioral contracts and backend agreement; they are not a learning
benchmark. The native parity test requires the native backend build dependencies.

These historical checks do not establish outcome tests or learning success for
the new default `signals` interface.
