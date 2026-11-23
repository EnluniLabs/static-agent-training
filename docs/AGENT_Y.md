# Agent Y: learned partition structure

Agent Y (`experimental/agent_y.py`) learns both receptor membership and the number
of groups. No partition schemes are supplied, and there is no four-group cap.
One shared signal-to-pair mapping is used for every generated grouping.

## Construction

For every observation, a full-frame-conditioned sequential policy processes each
receptor. It can join any group already constructed for that observation, or
create a new group. Every receptor is assigned once. Groups are nonempty, may
contain nonadjacent receptors, and range in number from one to receptor count.
Joining combines receptors; choosing new groups separates them. No geometric
boundary or desired group count is imposed. An initial learnable new-group score
of −log(receptor count) discourages immediate fragmentation; it is not a group
count constraint. The first receptor necessarily creates the first group.

Existing-group scores use learned receptor affinities averaged over group members.
New-group scores use learned weights over the entire receptor signal and a bias.
Shared presence and original feature coordinates are retained in each group's
pair embedding. Pair selection remains argmax; partition construction samples to
explore structural alternatives.

## Learning and persistence

Prediction feedback supplies mean negative log likelihood per bit. Relative to a
moving baseline, better predictions reinforce sampled join/new-group decisions
and worse predictions weaken them. Each score-function update uses its actual
construction probabilities. Initial feedback establishes the baseline. Expired or
absent predictions do not train the router. Frozen assessment updates neither the
policy nor its baseline, although it advances sampling RNGs.

The learned partition policy persists across episodes and is conditioned on the
incoming signal, so returning signals can recover previous grouping behavior.
There is no separately stored finite library of schemes. The partition itself is
reconstructed each observation and need not be identical on repeated input.
Snapshots (`crt-agent-y-v3`) preserve the policy, baseline and RNG state. Prior
fixed-count v1/v2 prototypes are rejected; start a new Agent Y session.

## Workspace and evolution

Select **Agent Y · learned sensory partitions** in the session creator. The fixed
partition-count control is hidden; group count is learned. New sessions use the
signals apparatus. Native CPU evaluation and argmax are selected by default.
The compiled evaluator includes partition construction, variable group counts,
shared embeddings, prediction/structural updates and router learning. OpenMP
workers evaluate independent lifetimes. Python remains the reference evaluator;
its workers use bounded spawned processes. A champion Python replay checks native
metrics and supplies complete inspection traces. Workers use additional RAM.
Session details and logs show completed-agent progress during evaluation.
The inherited apparatus partition field remains compatibility metadata, not Agent
Y's active group count. Do not interpret that field as a learned-group limit.

Router parameters are part of inherited mappings and genotype identity. Training,
mutation, reconciliation, checkpoints, resume, candidate export/import and
playground replay preserve them. Lifetime inheritance determines whether learned
router and pair mappings reach reproductive parents. Without it, each new
lifetime starts from inherited parameters. Playback frames expose actual sampled
groupings; brain input labels identify learned grouping instead of fixed tiles.
Paired population comparisons remain unavailable.

```sh
./project train --model senses --agent-y --backend native --sensory-interface signals
.venv/bin/python -m static_agent_training.experimental.agent_y \
  --input README.md --output /tmp/agent-y.json --max-frames 100 --learn
```

## Limits and evidence

Tests cover complete nonempty generated partitions, input-dependent changes from
one group to one group per receptor, return to earlier grouping behavior, join/new
group learning credit, controlled-feedback learning toward more or fewer groups,
frozen state, snapshot replay, evolutionary inheritance,
export/import and checkpoint resume. These verify the implemented mechanism, not
successful generalization or reduced catastrophic forgetting.

The policy uses O(R² + R×features) parameters. Extremely fragmented partitions
increase chain execution and may change voting behavior; one vote per active
chain is retained from Agent X. The learner has no temporal context, protected
scheme archive, or consolidation of signal-to-pair weights. Shared mappings remain
plastic. Prediction credit can favor easy predictions and is absent in action-only
rollouts. Structural shielding is still an experimental objective.

## Existing Python sessions

A stopped Agent Y Python session offers **Resume with native CPU**. This explicitly
migrates only the evaluator, retains the inherited population, and records the
backend change and generation in checkpoint metadata. Interrupted generations rerun.
The CLI equivalent is `python -m static_agent_training.experiment --resume RUN
--backend native`. Other configuration changes remain prohibited on resume.

See [native performance and parity](performance/AGENT_Y_NATIVE.md).
