# Switchboard

Current system context and model boundaries: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

New Agent X sessions default to the immutable `signals` interface described in
[Receptor signals](RECEPTOR_SIGNALS.md): a 16×16 layout of 256 binary receptors,
four fixed tiles, and one shared generic dual-rail byte transducer for all sources.
The layout indexes receptors; it is not a camera. Historical `retina` packed-pixel
snapshots and `legacy-bytes` controls remain supported with their saved interfaces.

`switchboard` is three independent panels of three binary lights, displayed as
three rows of a 3×3 grid. Panel 0 contains cells 0–2, panel 1 cells 3–5, and
panel 2 cells 6–8. Bit `row*3+column` of the board mask is that light. Each of
nine action signals toggles exactly one cell (`1 << cell`); the action-encoding
seed shuffles the signal-to-cell mapping. There are no aliases or idle buttons.
The goal is always all-off, with the standard 12-action budget. A resolved no-op
consumes an action without changing the board. Standard repertoire termination
and internal-prediction limits still apply.

All 512 boards are reachable. The exact shortest solution presses each initially
lit cell once, requiring the board's population count, at most nine presses.
Task starts exclude zero; zero is the terminal goal. Solver enumeration certifies
solvability for task generation and diagnostics only. Neither model receives a
solution, suggested action, panel labels, distance, or other puzzle hints.

The stimulus is exactly `bytes(board_bits(board, 'switchboard'))`: nine bytes,
each 0 or 1, in row-major order. New Agent X uses the same generic dual-rail conversion with its fixed 256-receptor,
four-tile layout. The **historical `legacy-bytes` control** uses 144 receptors
and three sensory partitions with argmax embedding selection. The following
panel-alignment description applies only to that control. The
existing dual-rail transducer assigns 16 receptors per input byte, so each
48-receptor partition corresponds exactly to one panel's three input bytes.
All 72 input bit positions have distinct rail pairs: there is no folding and no
unused receptor capacity. Each stimulus activates one rail of every pair (72
active receptors). The seven high zero bits in each byte activate constant zero
rails; their opposite rails are always zero on this task. Thus full capacity does
not mean all receptors vary or fire on these stimuli. Changing a panel affects
only its 48-receptor region. The apparatus is selected by configuration;
registering the environment does not automatically change receptor or partition
counts on an existing animal.

Categorical observations are 512 unique opaque `obs_panel_...` tokens, with
matching `expect_panel_...` expectations and nine `move_...` outputs. They do not
reuse the other puzzles' board vocabularies. The optional goal context is a
constant goal-zero token. Both adapters retain existing prediction and learning
rules; neither invokes a solver during an episode. Registration as kind
`lights-out` lets the generic native kernels use packed switchboard move masks
and board indices. Reference/native tests cover both sensory action policies,
144 receptors with three partitions, categorical evaluation, and learned weights.

The normal seeded task split reserves 409 nonzero boards for training starts and
102 for assessment starts. These pools are disjoint full-board configurations;
they are not disjoint sets of local panel patterns. For seeds 0, 1, 19, 42, and
2026, tests confirm that each pool contains all eight possible three-bit patterns
at each of the three panel positions. This is a compositional held-out-start task:
the reserved complete configurations combine local components also available in
the training pool. This coverage is verified for those seeds, not enforced by the
partition algorithm for arbitrary seeds. Training episodes sample from their pool;
a finite lifetime is not guaranteed to encounter every local pattern or every
training board. Trajectories can also visit held-out starting boards, so reserved
starts do not prove those states were never observed.

Independent mechanics and aligned input partitions do not guarantee an advantage
for a partitioned animal. Repertoire selection, prediction learning, and the
shared nine-action interface still determine performance. In particular, the
agreement policy can resolve different panel proposals to no-op; conflicts and
consumed action budget matter. Vote and agreement must be compared on matched
tasks and budgets rather than inferred from panel independence. Whole-board
categorical and sensory models also differ in parameterization.

The random control samples uniformly among the nine actual toggle buttons.
Its exact success probability uses the transition graph and the remaining budget;
with one action left a one-lit-cell board succeeds with probability `1/9`.
Frozen assessment has no warmup or learning, but separately sampled frozen tasks
can overlap prior training unless the harness explicitly reserves them.
