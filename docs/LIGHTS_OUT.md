# Lights Out environment

Select **Lights Out · 3×3 puzzle** in training setup, starting-population
benchmarks, frozen assessments or the playground. CLI identifier: `lights-out`.
Both categorical and fixed-senses models support Python reference and native CPU
evaluation, normal evolutionary selection/mutation, candidate reuse, and checkpoints.

## Rules

Nine binary lights form a 3×3 board. Pressing a cell flips that cell and its
horizontal/vertical neighbors. Diagonals and wrapping are excluded. The goal is
always all lights off. An episode ends on success, after 12 presses, on a terminal
repertoire edge, or after 16 consecutive predictions without a press. Presses have
no walls or movement distances. Number-line boundary settings do not affect this
world. Multiple presses compose by XOR; repeating a press cancels its effect.

The runner identifies a board with a 9-bit integer mask: bit `row*3+column`
represents that cell. This is an identifier for logging and task storage, not a
coordinate or distance shown to the animal. Action indices map to physical cells
through the saved action-encoding seed. The playground decodes those indices for
human-readable row/column press labels and board replay.

## Input, predictions and learning

The sensory adapter emits exactly nine bytes, one per cell, each 0 or 1. The
animal-owned fixed receptor converter processes those bytes as usual. It receives
no solution, shortest path, numeric board distance or suggested press. Its receptor
count, shared input weights, gate and bit-prediction shapes are unchanged when
loading number-line sensory candidates into this environment. Predictions describe
the next whole sensory pattern; the existing empirical and delayed structural
rules are unchanged. Winning observations still test pending predictions.

The categorical model uses 512 opaque board-observation symbols and 512 matching
expectation symbols, plus nine action symbols. With the legacy goal-context option
enabled it also receives a single constant goal symbol; this does not create varying
puzzle targets. Loading categorical candidates into a new training session retains
matching entries and initializes unseen board entries, as for other vocabulary
changes. Frozen categorical assessment requires existing puzzle mappings and never
initializes entries. These differing parameter counts and output normalizations
mean a categorical-versus-sensory puzzle comparison is a whole-model comparison.

Neither model invokes a solver or changes repertoire edges during a lifetime.
Native CPU and Python execute the same board transitions and learning schedule;
reference tests compare episodes, losses and all learned parameter groups.

## Tasks and checks

The world enumerates every subset of the nine buttons pressed from the all-off
board. This certifies solvability without supplying a solution to the agent.
Exhaustive enumeration gives 512 distinct boards, each with a solution of at most
nine presses. Training and assessment starts exclude the already-solved board.

For an evolutionary run, the seed partitions the 511 nonzero starting boards
into 409 training and 102 assessment boards. Training draws vary by generation;
assessment draws are fixed. Both sides of a starting-population benchmark share
these draws. Trajectories may pass through a reserved starting board, so disjoint
starts do not establish that every intermediate assessment state was unseen.
Tasks sample boards uniformly, including when a reused config contains number-line
distance settings. Records store solution-length diagnostics instead of treating
board IDs as distances; those diagnostics never enter the model.

Frozen assessments sample the same ordered nonzero boards for every candidate,
with goal 0 and no warmup or learning. Their tasks may overlap earlier training.
The random control is uniform over the nine buttons, with an exact 12-press success
probability computed from the board transition graph. It has no no-op button.

## Use

```sh
./project train --model senses --environment lights-out --backend native \
  --population 32 --generations 50

./project train --model senses --environment lights-out --backend native \
  --initial-population runs/populations/number-line-candidates.json.gz \
  --population 32 --generations 50

./project assess --populations runs/populations/puzzle-candidates.json.gz \
  --environment lights-out --episodes 200 --output runs/puzzle-frozen-check
```

In the playground, click cells under **Starting board** to choose a task, then
run the saved animal. **Current board** replays actual presses and predictions.
The goal is locked to all-off. The optional playground-learning toggle continues
to operate on an isolated copy of the selected animal.

Verification covers all board/press transitions in both adapters, all solution
certificates, native/reference learning and learned weights, frozen assessment,
checkpoint resume, candidate transfer, and browser execution. These checks verify
implementation, not that a trained population already solves the puzzle reliably.

## Simpler option: independent 2×2 switches

**Simple switches · 2×2** (`switches-2x2`) is a separate environment. The original
3×3 Lights Out environment remains available with its existing rules.

Each press flips only its own light, with no neighbor effects. All 16 boards are
reachable; any board needs at most four useful presses to switch off its lit cells.
The goal is all-off and the action limit remains 12. The four row-major input bytes
are each 0 or 1 and use the same fixed receptor converter. Categorical models use
16 distinct switch observations, 16 expectations and nine action symbols, plus the
optional constant goal symbol. Switch symbols differ from 3×3 board symbols.

Saved models retain their nine action signals: two signals control each of the four
lights, and the ninth does nothing. The action-encoding seed shuffles this mapping.
Idle actions count toward the limit. Exact random-control probabilities include
both aliases and the idle signal; for example, one lit cell has a 2/9 chance of
being solved by one random action. No learning rules or repertoire edges change.

The 15 nonzero starting boards split into 12 training and three assessment starts.
The same caveat about intermediate-state overlap applies. Training, population
comparisons, candidate save/load, frozen assessments and the playground all accept
this option. For a simpler starting task, select it before creating a session:

```sh
./project train --model senses --environment switches-2x2 --backend native \
  --population 32 --generations 50
```

## Middle option: 2×2 Lights Out

Select **Lights Out · 2×2 puzzle** (`lights-out-2x2`) for a small board with
interacting switches. A press flips its own cell and both orthogonal neighbors;
the diagonally opposite cell stays unchanged. In row-major order, the four press
masks are `7`, `11`, `13`, and `14`. Repeating a press cancels its effect.

All 16 boards are reachable, and any board can be solved in at most four useful
presses. Unlike independent switches, solving it may require temporarily turning
other lights on. It retains the 12-press limit, all-off goal, four input bytes,
12/3 partition of nonzero training/assessment starts, two action signals per cell
and one idle signal. The action-encoding seed gives the same cell mapping as simple
2×2 switches; the effects of pressing those cells differ. Categorical board tokens
have their own `board2` prefix so the different mechanics do not silently reuse
simple-switch prediction symbols. Sensory mappings and learning rules are unchanged.

This third puzzle option supports both models, native and reference evaluation,
evolutionary training, candidate save/load, population comparisons, frozen
assessments and playground learning/replay. The original two puzzle options remain
available. Exact random controls include the aliases and idle action: a board with
mask `7` has a one-press random success probability of `2/9`, while a board with
just one light on cannot be solved in one press.

```sh
./project train --model senses --environment lights-out-2x2 --backend native \
  --population 32 --generations 50
```
