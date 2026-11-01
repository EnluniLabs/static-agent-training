# Training environment and agent interface

New training runs use the varying-goal interface. The environment shows the
current number and the requested target separately, for example `obs_+12` and
`goal_-3`. The `Observation.goal` field carries the target token alongside
`Observation.token`. Both are inputs to the inherited mapping: their learned
embedding logits are added before sampling an entry pair. Neither the environment
adapter nor the agent computes a distance, path, or recommended action.

There are 82 input symbols (41 current numbers plus 41 target numbers), nine
movement outputs, and 41 next-number expectation outputs. Predictions concern
the next absolute current number, not a distance. World boundaries remain −20
to 20 for every target. `reset(start=..., goal=...)` chooses the task; success
means reaching that goal within 12 moves. The generation runner supplies a fresh
ordered training task list per generation and a fixed disjoint assessment list.
All animals share each list. New experiments sample valid ordered start/goal pairs
uniformly within each partition, so each allowed goal has equal probability. All
41 positions, including zero, participate in the new varying-goal partitions.
Normal-distance sampling remains optional. The agent receives only current and
target numbers, without a computed distance. Lifetime learning survives changes
of task.

New worlds have hard walls: movement is clamped to [−20, 20], consumes one action,
and receives normal observation/prediction feedback at the actual resulting
position. An edge target can be reached by an overshooting move that stops there.
A move into a wall from that same edge leaves the position unchanged and still
counts toward the action limit.

`NumberLine()` retains the zero-target symbol vocabulary but now defaults to hard
walls. `NumberLine(goal_conditioned=True)` supplies separate current and target
inputs. Use `boundary_mode="terminate"` for faithful historical behavior.
Saved older checkpoints and replays automatically select that historical rule.

## Original zero-target interface


The first task uses the report's number-line world. Positions are −20 through
20, the target is 0, and moves are −5, −3, −2, −1, 0, +1, +2, +3, +5.
Starts are sampled uniformly from nonzero positions. Reaching 0 succeeds;
leaving the range or using 12 moves without reaching 0 fails. Reaching 0 on
move 12 succeeds.

### Agent boundary

The agent chooses whether and when to make an expectation. Following report
sections 5.1–5.3 and 7, each successful mathematical continuation samples a
single output from its learned distribution over all 50 symbols. That sample
determines whether the agent expresses an expectation or acts. There is no
required prediction before a move and no separate prediction schedule.
An agent can emit zero, one, or several expectations before acting. After an
expectation it continues the same mathematical traversal without re-embedding.
Only explicitly emitted expectations are eligible for empirical criticism.

`NumberLine.reset()` returns a `Transition` with an opaque `Observation`.
The agent receives 41 observation symbols and a vocabulary of 50 output symbols:
nine moves and 41 expectations. Numeric meanings are private to the environment;
token permutations are fixed by a recorded encoding seed. An expectation symbol
corresponds to the same observation symbol with the same suffix.

`emit(Output(token), source=pair_id)` consumes one sampled output. The physical
interface interprets its type after emission. A move changes position; an
expectation commits to the next observed position after the next move, without
advancing physical time. Multiple expectations may precede a move. Move-only
behavior creates no implicit prediction.

A transition with an observation requires the agent to end its previous traversal
and re-embed the new observation, including when a zero move repeats the symbol.
A transition without an observation permits continued internal traversal unless
`done` is true. At most 16 outputs are permitted per observation: a move on
output 16 is valid; a sixteenth expectation ends the episode unsuccessfully.
`stop()` records failure when the agent runs out of coherent continuations.

After an in-range move, each pending expectation receives independent feedback
with its ID, source pair, predicted symbol, actual symbol, and correctness.
Feedback enables local learning; it does not directly add fitness. Pending
expectations expire without refutation when no subsequent observation exists.
In particular, an out-of-bounds move ends the episode without inventing a 42nd
observation symbol. This terminal convention is an explicit implementation
decision. Terminal in-range observations still score pending expectations.

### Original training contract

Evaluate each animal on the same ordered list of 20 starts within a generation.
`sample_starts()` produces this reproducibly with replacement; the runner must
record both the seed and actual list. Keep token encoding fixed across animals,
training, and assessment. Keep environment RNG independent of agent RNG.

Clear active computation and pending expectations at episode reset. Retain
lifetime learned mappings between episodes. Fitness is the fraction of episodes
that reach the target. Report a separate assessment with learning disabled and
explicitly disjoint held-out start positions; pass those partitions through
`allowed` when sampling starts. The current population runner implements selection, learning, and evaluation
with the newer configurable episode counts described in the README.

`diagnostic_state()` exposes numeric coordinates and decoding for tests and a
future viewer. The agent adapter must never receive this privileged data.
The environment provides correctness records; update rules belong to the agent.

See [implementation boundaries](docs/IMPLEMENTATION.md) for the current training
contract and [bias audit](docs/BIAS_AUDIT.md) for evaluation limitations.
