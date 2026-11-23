# Current architecture

Reviewed against the working tree on 2026-11-22. This describes implemented code,
including local changes that may not yet be committed. Saved configurations remain
attached to their runs; editing code does not update an already running process.

## Two model paths

The categorical baseline (`agent.py`, `mappings.py`) embeds discrete observation
and goal tokens into an immutable repertoire of ordered lambda-term pairs. Its
outputs are discrete actions or expectation tokens. Its full-coordinate number-line
inputs differ from Agent X's local sensors; results are not information-matched.

Agent X (`experimental/bit_model.py` and the sensory modules) uses a fixed receptor
apparatus and learned numerical mappings. New workspace Agent X sessions select
`signals`, normally 256 binary receptors indexed in a 16×16 layout and four fixed
tiles. These are workspace defaults, not universal Python API defaults:
`TrainingConfig` retains `legacy-bytes` as its compatibility default. Explicitly
choose the interface when constructing experiments outside the workspace.

## Agent Y

[Agent Y](AGENT_Y.md) adds learned full-signal-conditioned receptor-to-group
assignments above the same shared embedding. It generates partitions directly,
with a learned number of nonempty groups and no premade scheme menu. Each
receptor can join an existing group or create a new group.
`TrainingConfig(model='senses', agent_y=True, backend='native')` selects it.
The workspace creator exposes Agent Y, defaults to native CPU and locks argmax, and supports normal
evolutionary sessions, checkpoints, resume and champion playback. Router state is
part of inherited sensory mappings and genotype identity. Lifetime inheritance
controls whether learned router changes pass to offspring. Compiled native CPU and Python reference evaluation are supported.
Paired starting-population comparisons remain unsupported. This is implemented
partition learning, not demonstrated structural shielding.

## Input, parallel chains and output

Environment adapters supply bytes to `SignalSenses`. A generic dual-rail converter
maps each source bit to a zero or one receptor rail. A default bank accepts 16
source bytes without folding; its resulting receptor pattern contains 256 bits.
The presence feature is additional. Width and height index the fixed apparatus;
they do not require a rendered picture. Tile count can range from one through
receptor count and does not change receptor capacity.

Each region independently selects an entry pair from the same repertoire.
Activations remain distinct even if they choose the same pair. Regional chains
follow stored successors. Predictions concern the complete next receptor frame;
each prediction receives its own pending identity and feedback. Multiple regions
share learned mappings. Agreement executes a common action once; vote executes
a unique plurality winner once. Disagreement or a voting tie consumes a no-op
step in supported worlds. See [Agent X](AGENT_X.md) for the precise round contract.

Number Line signals encode 16 local readings at offsets −8 through +7: outside
0, ordinary location 64, self 128, and visible conditioned goal 255. There is no
absolute-coordinate or offscreen-goal channel. Puzzle adapters supply board bytes
through the same converter. Historical `retina` packed drawings and `legacy-bytes`
interfaces load explicitly with their saved geometry and semantics.

`ReceptorStream` windows arbitrary byte sources without folding; `run_stream`
advances on incoming observations. World episodes are action-paced. Stream
outputs need a consumer-defined motor interpretation. Binary input support does
not establish arbitrary-data understanding or cross-modal transfer.

## Development, lifetime learning and evolution

`engine.py` develops closed ordered pairs by reducing `B A` and progressing to
`(B, NF(B A))`. Exact ordered-pair intersections merge executions and pool remaining
budgets. Logical scheduler interleaving is distinct from CPU multiprocessing.
Certified completed transitions become repertoire successors. No lambda reduction
occurs during a lifetime rollout. Structural mutation may redevelop offspring;
weight-only mutation reuses structure.

`Species` holds immutable initial pairs, budgets, repertoire, inherited mappings,
development configuration and JSON provenance. Lifetime agents own mutable learned
weights. Episode resets retain lifetime learning; frozen held-out assessment does
not learn. Training fitness drives tournaments and elites. Learned mappings enter
reproductive parents only with explicit lifetime inheritance. Categorical and
sensory update rules differ; consult their model documents rather than applying
the categorical expectation-token equations to sensory predictions.

Training tasks derive from generation-specific seed streams. Assessment tasks stay
fixed. Reproduction derives child seeds independently of scheduling. Optional
speciation restricts selection and shares fitness; it does not introduce crossover.
The workspace exploratory preset increases seed count, development budgets and
structural mutation rates; it is distinct from the conservative dataclass defaults.
See [structural settings](EXPLORATORY_STRUCTURE.md).

## Execution, caching and persistence

Python is the reference evaluator; registered worlds also support native CPU
batch evaluation. Native workers and offspring processes are separate controls.
Offspring processing uses spawn, bounded outstanding jobs and deterministic result
ordering. Automatic reproduction parallelism applies only above workload thresholds
and uses at most four processes. Successive generations depend on their predecessors.
CUDA is removed; historical CUDA configurations migrate to native CPU on resume.

Genotype IDs, merger summaries and validated development JSON fragments are cached
per immutable Species. New agents start fresh caches. Native numerical source
arrays are immutable and copied into writable evaluation batches. Parent transport
caches retain at most two parents per worker, keyed by identity and provenance.
These caches consume memory; bounded entry counts do not establish a fixed byte
ceiling or proven long-run memory stability.

Each completed generation saves the next inherited population for resume. Interrupted
generations rerun. Checkpoints stream one agent at a time; development JSON is
validated once per immutable agent and streamed without repeated tree conversion.
Default `to_data()` still returns ordinary decoded data. Checkpoints and the
repeatedly overwritten best-population file use gzip level 1; other archives use
level 9. Writes close gzip, sync the temporary file, atomically replace the target,
and sync the directory. Schemas remain readable by existing loaders; whitespace
and compressed bytes can differ while decoded values match.

`best-population.json.gz` retains distinct historical best candidates selected by
training fitness, not only the latest generation. It is an export collection,
separate from reproductive selection. Historical candidates are still read and
validated each generation. Champion traces retain generation champions; the small
champion index is a separate historical collection. Checkpoints do not retain all
evaluated agents' learned lifetime weights.

## Inspection and evidence

The workspace launches and tracks runs, imports populations, replays snapshots and
shows a 3D mapping graph. Connected nodes are localized by deterministic swaps among
brain-shaped sites; inputs and outputs are pinned visual banks. Spatial proximity
and brain shape are display choices, not learned anatomy. Red sensory edges show
negative weights; graph edges must be interpreted by their mapping type.

Performance reports are dated measurements, not guarantees. The E3 stage benchmark
predates provenance streaming; the [save benchmark](performance/SAFE_SAVE_OPTIMIZATION.md)
measures copied historical checkpoints, not the new whole-generation runtime.
Passing parity/resume tests establishes implementation consistency, not successful
navigation, reasoning, biological equivalence or generalization.
