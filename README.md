# Static agent training

Start with [USE.md](USE.md) for commands and saved artifacts, or run
`./project help`. [Documentation map](docs/README.md) separates current behavior,
experimental designs, audits, and historical results.

Open the model workspace with `./project ui`: choose a model, configure and
launch sessions, monitor live charts and logs, inspect saved champions or sensory
snapshots, and replay frozen attempts. Both categorical and fixed-senses models
support evolutionary generations, mutation, selection, checkpoints and resume.
Fixed-structure lifetime learning remains a separate experiment mode.

A lambda-repertoire evolutionary training system with local lifetime learning,
replayable checkpoints, optional native CPU evaluation, and a WSL/Windows
browser playground.

Start with `./train`, inspect background runs with `./train status`, and open the
playground with `./demo`. The Python reference needs no third-party dependencies;
optional acceleration setup is documented below. New-run defaults are uniform
start/goal pairs, hard walls, and speciation disabled. A current number and target
are separate inputs; the agent receives no computed distance or suggested move.

Local lifetime updates use tested expectations and structural consistency rather
than a task-reward gradient. Evolutionary selection separately uses task success.
The current number-line experiment supplies a goal; goal-free exploration, food
finding without location hints, and fixed-structure transfer to a larger world
are proposed experiments in [future work](FUTURE_WORK.md), not demonstrated
capabilities. Learned-weight inheritance is optional and disabled by default.

Documentation:

- [Implementation and experiment boundaries](docs/IMPLEMENTATION.md)
- [Environment interface](docs/specs/ENVIRONMENT_INTERFACE.md)
- [Experimental type-agnostic signal model](docs/AGNOSTIC_INTERFACE.md)
- [Experiment record and result snapshots](docs/EXPERIMENTS.md)
- [Bias audit, attribution limits, and reproducible controls](docs/BIAS_AUDIT.md)
- [Future work and hypotheses](FUTURE_WORK.md)
- [Learning, selection, and inheritance experiment plan](docs/LEARNING_EXPERIMENT_PLAN.md)

Quick commands:

```sh
./train --population 1024 --generations 500 --backend native --batch-size 1024 --episodes 100 --assessment-episodes 200
./train status --watch
./demo
```

Install the optional native dependencies before the accelerated command, or use
`--backend python` for the dependency-free reference. `./train` automatically uses
`.venv/bin/python` when present. Training artifacts stay under ignored `runs/`;
curated historical summaries are committed under `docs/results/`.

An existing initializer creates the mathematical seeds of a set of explanations.
Each seed is an ordered pair `(A, B)` of closed untyped lambda terms. Chain
development attempts `C = NF(B A)` and visit `(B, C)`.
These seeds are not yet developed explanations or physical interpretations.

Run with Python 3.10 or later; no dependencies are required:

```sh
mkdir -p runs
python3 -m static_agent_training.initialize --seed 42 --count 4 --output runs/explanations.json
python3 -m static_agent_training.engine runs/explanations.json --budget 64 --seed 42 --output runs/developed.json
python3 -m unittest discover -s tests
```

The JSON records configuration, independent per-chain generation seeds, raw
De Bruijn ASTs, readable terms, and stable SHA-256 term/pair IDs. Duplicate
pairs are retained with distinct chain provenance; sampling is independent,
not sampling without replacement. No inherited terms are beta-normalized.

The provisional generator chooses uniformly from the fitting I, K, and S
combinators with probability 0.5. Otherwise it recursively chooses uniformly
among feasible variable, abstraction, and application constructors, with a
uniform valid bound-variable index and uniform feasible application node-budget
split. Depth includes the root; node limits are upper bounds, not target sizes.
Defaults are depth 6 and 31 nodes. Configure these using `--max-depth`,
`--max-nodes`, and `--combinator-probability`. This distribution is an explicit
implementation choice implementing the report's small-combinator bias.

The engine develops multiple starting pairs with separate contraction budgets,
stochastic reduction, and budget pooling at shared ordered pairs. Fixed-repertoire
physical mappings, agent traversal, and local lifetime learning are implemented.
Inherited species state and offspring mutation are implemented. The population
training loop and live terminal fitness graph are also implemented.

The stochastic reducer is available as `static_agent_training.reduction.normalize`.
It accepts a closed tuple AST, a beta-contraction budget, a seed, and an AST-size
cap. It returns `NORMAL`, `BUDGET`, or `SIZE`, the final term, charged steps,
remaining budget, and selected redex paths. It reduces beneath abstractions and
uses capture-avoiding De Bruijn substitution. A failed attempt does not certify
divergence.

Use `static_agent_training.engine.run(pairs, budgets, seed=42)` from Python.
`pairs` is a list of `(left_ast, right_ast)` seeds; `budgets` is a list of
individual budgets, or one integer assigned to every starting pair. JSON ASTs
from the initializer are also accepted.

Execution is logically parallel: each scheduler round permits one contraction
per active execution. This is deterministic interleaving, not CPU parallelism.
A shared single term does not merge executions. Reaching an exact ordered pair
already visited by another execution pools their remaining budgets and retains
all source indices. Arrival times need not match. A later arrival forwards its
budget to the shared execution's current position, retaining any partial
reduction, and resumes it if budget exhaustion had stopped it. Work spent before
the merge is never refunded; pooled downstream work is charged once. The earlier
execution's RNG stream continues after a merge.

Cycles are recorded and continue spending budget. `max_transitions` (default
1000) bounds each surviving execution, including zero-contraction cycles;
`max_nodes` (default 256) bounds intermediate applications. Budget pooling does
not override `SIZE` or `DEPTH` stops. Results include initial ASTs and budgets,
configuration, source provenance, contraction paths, normal-form edges, merges,
cycles, terminal statuses, partial terms, and conserved budget accounting.
Identical inputs, seed, and configuration reproduce the complete event log.


Every development result now includes a `repertoire` export: unique immutable
pairs and certified successor IDs. Only completed normalizations create edges;
partial reductions do not. Load it without evaluating any lambda terms:

```python
import json
from static_agent_training.repertoire import Repertoire
from static_agent_training.mappings import InheritedMappings
from static_agent_training.agent import Agent, run_episode
from static_agent_training.environment import NumberLine

result = json.load(open("developed.json"))
repertoire = Repertoire.from_data(result["repertoire"])
env = NumberLine(encoding_seed=0, episode_seed=1)
inherited = InheritedMappings.initialize(
    repertoire,
    [s.token for s in env.observation_tokens],
    [o.token for o in env.output_tokens],
    seed=42,
)
agent = Agent(repertoire, inherited.lifetime(), embedding_seed=2, output_seed=3)
transitions = run_episode(agent, env)
```

Embedding logits have one row per observation and one column per unique pair.
Output logits have one row per unique pair and one column per output symbol.
Each row is softmax-normalized when used. Initial logits use independent,
keyed Gaussian draws with standard deviation 0.1 (a configurable implementation
choice). Sampling has separate embedding and output RNG streams. Terminal pairs
remain eligible entries; an empty repertoire or missing successor terminates
without inventing an output.

At each observation the agent samples an existing entry pair, follows a stored
edge, and samples an output at the successor. The environment interprets the
output. Expectations continue traversal; actions trigger re-embedding whenever
a new observation arrives, including a repeated symbol. The environment owns
action and output limits. The runner passes no numeric decoding to the agent.

The repertoire, inherited embedding logits, and inherited output logits are
immutable. Each lifetime receives independent mutable copies of the logits;
episode resets preserve those copies. The runner applies local learning by default. Use
`run_episode(agent, env, learn=False)` for assessment with both update rules
disabled, or construct `Agent(..., learning=False)` for a fixed baseline.

Reduction occurs only during initial development or mutation redevelopment.
To reconcile mappings after external redevelopment, call
`InheritedMappings.initialize(new_repertoire, observations, outputs,
previous=inherited, seed=...)`. Surviving pair IDs keep their inherited logits,
new IDs receive seeded logits, and removed IDs disappear. Pass inherited weights,
not learned lifetime weights. The offspring builder below performs this
reconciliation automatically.
The repertoire, mapping, and lifetime-agent modules never import the reducer.


Local learning follows report sections 6.1–6.4. Every explicitly emitted and
tested expectation updates only its source pair's expectation-token logits,
using conditional expectation cross entropy and learning rate 0.05. Correct
predictions also update; action logits and other pairs do not. Expired
expectations receive no empirical update. With multiple pending expectations,
each is scored and updated sequentially in emission order against the same
next observation. Each loss uses the current logits; emission snapshots remain
in the event records. The total action/expectation probability split is allowed
to change through softmax renormalization, as agreed.

Empirical updates mark only the emitting pair as revised. No immediate embedding
update or global scan occurs. When a revised pair is subsequently encountered,
the most-supported expectation's observation embedding is checked against the
stored successor. If its probability is below 0.1, that observation's embedding
logit for the successor increases by 0.05. A missing successor skips the check.
Revised pairs remain eligible for checks on future encounters; corrections can
repeat until compatible. Episodes clear pending commitments and active traversal,
but retain lifetime weights and revision history. Configurable rates and threshold
are provided by `learning.LearningConfig`, passed as `Agent(...,
learning_config=...)`.

Agent events record expectation IDs, physical symbols, origin and emitting pairs,
traversal paths, emission logit snapshots, timestamps, empirical loss and updates,
expiration, compatibility checks, and local embedding conflicts linked to their
preceding empirical revision. Neither update rule changes the fixed repertoire
or inherited weights, or invokes lambda reduction.


Mutation is implemented in `genome.py` and `mutation.py`. `Species` stores closed
initial pairs, their individual development budgets, immutable inherited mapping
weights, the fixed repertoire, development limits, and a saved development log.
Species JSON and genotype IDs support replay; loading a species performs no
lambda reduction.

```python
from static_agent_training.genome import develop
from static_agent_training.mutation import mutate

# pairs is a sequence of inherited (left_ast, right_ast) starting pairs.
parent = develop(pairs, [s.token for s in env.observation_tokens],
                 [o.token for o in env.output_tokens], budgets=64,
                 development_seed=1, mapping_seed=2)
child, mutation_log = mutate(parent, seed=3)
# Training uses child.repertoire and child.inherited.lifetime().
```

The operator draws are independent. Defaults per offspring: weight perturbation
0.8, subtree replacement 0.15, complete seed-term replacement 0.10, initial-pair
insertion 0.05, and initial-pair deletion 0.05. Operators execute in that order
except weight perturbation, which occurs last, after structural redevelopment and
mapping reconciliation. Selected weight mutation adds independent Gaussian noise
with standard deviation 0.1 to every embedding and output logit, including newly
initialized entries. No crossover or intermediate-term mutations occur.

Subtree replacements are generated in their actual binder context, preserving
closedness and the configured whole-term depth and node limits (defaults 6 and
31). New subtrees may refer to surrounding binders using De Bruijn indices;
no existing subtree is moved between binder contexts. Operators retry up to 32
times to obtain a different term, then log a no-op if impossible. Deletion leaves
at least one initial pair. Inserted pairs receive a configurable fresh budget
(default 64); other initial-pair budgets are retained.

If the final starting pairs or budgets changed, the builder reruns the entire
repertoire with fresh budgets and a recorded fresh development seed. This
recomputes shared execution, including lost and new mergers. Otherwise the fixed
repertoire is reused with no reduction. All surviving pair IDs retain their
inherited weights before any selected weight noise; new IDs receive seeded
weights and removed IDs lose theirs. By default, lifetime weights are not consulted. With `--inherit-lifetime`, end-of-training learned mappings are first committed to reproductive parents, then the same mutation rules apply.

Each mutation log records parent and offspring genotype IDs, configuration,
independent RNG stream seeds, selected operators, edits and no-ops, added/removed
pair IDs, merger changes, and development contraction cost. Merger changes are
compared by junction pair ID and the multiset of participating starting-pair IDs.
A new development seed can change reduction costs and discovery in unedited
chains as well as in the mutated chain; those execution changes are recorded.

Save with `json.dump(parent.to_data(), file)`, and load using
`Species.from_data(json.load(file))`. A saved species can also be mutated via:

```sh
python3 -m static_agent_training.mutation species.json --seed 3 --output offspring.json
```

The output bundle contains `species` and `mutation`; it can itself be passed as
the input for the next mutation. Species construction and reproduction are
separate from the lifetime agent. Population selection uses fitness tournaments. Optional structural speciation
restricts parent tournaments to niches and shares fitness between their members;
crossover is not performed.


Start a new run with a live terminal graph using one command:

```sh
./train
```

The launcher defaults to 32 animals, 50 generations, and seed 42. Each run gets
its own timestamped folder under `runs/`, so repeating the command never
reuses a previous folder. Press Ctrl-C to stop. Resume the most recently
checkpointed run with:

```sh
./train resume
```

You can also choose settings or a specific checkpoint:

```sh
./train --population 64 --generations 100
./train resume runs/my-run --generations 100
./train --help
```

The executable can be invoked by absolute path from any directory; relative
run paths are resolved from the project root. The full experiment command
remains available:

Run a complete evolutionary experiment with a live terminal graph:

```sh
python3 -m static_agent_training.experiment --population 32 --generations 50 --seed 42 --output runs/first-run
```

The graph redraws in an interactive terminal while the current evaluation count
and provisional fitness update. It plots population mean training success (`M`),
best training success (`B`), and population mean frozen held-out success (`H`).
An asterisk marks overlapping series. The vertical axis is explicitly labeled
in percentages and auto-scales with a zero baseline to show small changes. All
completed generations remain represented as the horizontal axis compresses to
fit. Training starts change between generations, so scores can fluctuate.
Outside an interactive terminal, or with `--no-live`, one summary line prints
per completed generation without ANSI control sequences.

Default settings are population 200, 500 generations, 20 lifetime episodes,
20 frozen assessment episodes, five inherited elites, tournaments of size five,
four starting pairs, and 64 contractions per initial pair. Small populations
set the default elite count to `min(5, population - 1)`. All non-elite offspring
are constructed through mutation; unchanged elites begin fresh lifetimes.
Training fitness alone determines selection. Assessment never updates weights
or influences parent choice. `--no-learning` disables both local learning rules
for a baseline run. Episode counts, elite count, and tournament size are also
configurable through the CLI; all nested development, mutation, and learning
settings are available through `evolution.TrainingConfig` in Python.

New runs vary both the starting number and target number. The agent receives
those two displayed numbers separately (`obs_+12` and `goal_-3`, for example),
and chooses movement outputs itself. It receives no relative distance, suggested
move, or path. Separate learned embedding rows for the current number and target
contribute additively to the entry-pair logits; this supplies task context without
a hand-coded movement rule. There are 82 input symbols (41 current numbers and
41 goals), and 50 outputs (nine moves and 41 next-number predictions).

New varying-goal runs split all 41 positions, including zero, once using the run
seed: 31 training positions and ten held-out positions. The default sampler
chooses starts uniformly and goals uniformly from the other positions in the same
partition. Every valid ordered pair therefore has equal probability, and every
allowed goal has equal marginal probability. Training resamples each generation;
assessment uses a fixed task list with disjoint start and goal values. All task
lists and realized distances are recorded. The agent receives no computed distance.

The physical world has hard walls at −20 and +20. An overshooting move stops at
the boundary, still consumes a move, and produces an observation and prediction
feedback at the actual bounded position. Reaching the supplied target within 12
moves succeeds. There is no out-of-bounds failure under these new rules.

Normal-distance sampling remains available with `--distance-distribution normal`.
That sampler can expose central targets more often because there are more feasible
approaches to them; uniform pairs equalize target exposure, not task difficulty.
Use `--fixed-goal --distance-distribution uniform --boundary-mode terminate` for
the original zero-target experiment. Older checkpoints retain their original task
sampling, partitions, and boundary rules automatically when resumed.

Separate seed streams drive population generation, development, initial weights,
training starts, embedding sampling, output sampling, frozen assessment,
tournament selection, and offspring mutation. Episode resets retain lifetime
learning; assessment uses separate copies of learned weights and separate RNG
streams. Reproduction takes an immutable genotype; by default its weights are the original inherited weights. The optional `--inherit-lifetime` mode commits learned mappings before reproduction. The legacy
`Species` class denotes one inherited genotype, not a cluster or speciation rule.

Each run directory contains:

- `config.json`: complete configuration and actual position partitions.
- `generations/00000.json`, etc.: actual episode starts and order, per-animal
  fitness, prediction statistics, failures, repertoire sizes, and reproduction
  ancestry including mutation logs. Stored generation indices start at zero;
  the terminal displays them starting at one.
- `traces/00000-champion.json.gz`, etc.: detailed training and assessment events,
  inherited genotype, and learned mappings of that generation's best training
  animal. This is a generation champion, not a claim of improvement or correctness.
- `checkpoint.json.gz`: the next inherited population, configuration, generation
  index, and complete summary history. A next population is reproduced even at
  the requested endpoint so extending the run follows the same sequence.

A complete checkpoint is atomically replaced after every generation. Ctrl-C
leaves the last complete checkpoint usable; a partially evaluated generation is
rerun on resume. Resume or extend the same experiment with:

```sh
python3 -m static_agent_training.experiment --resume runs/first-run
python3 -m static_agent_training.experiment --resume runs/first-run --generations 100
```

Resumption restores the recorded configuration; only the generation target and
display mode may change. Seed streams are derived from saved experiment coordinates,
so uninterrupted and resumed runs produce identical results. Output directories
are never silently overwritten. Run artifacts under `runs/` are ignored by Git.

Watch the best saved training performer in a browser:

```sh
python3 -m static_agent_training.demo runs/my-run
```

Open the printed `demo/index.html` path in a browser. The standalone replay has
attempt selection, play/pause, step, speed, and a step slider. It shows the
animal moving along the number line toward zero, decoded predictions, and the
actual outcome of every held-out attempt, including failures. No server or
additional dependencies are needed.

Selection takes the highest training score among saved generation champions,
with ties going to the earliest generation. The exporter restores that animal's
saved learned weights and reproduces its original frozen assessment with the
recorded RNG streams and episode order; learning remains disabled. It checks
outcomes against the saved results. The accompanying `demo/champion.json`
contains the inherited animal, learned mappings, assessment seed, encoding seed,
and visual replay frames, so the selected performer can be kept separately from
the training population. Exporting does not resume or modify training.


Launch the interactive playground on Windows from WSL:

```sh
./demo
# Or select a run / exported animal explicitly:
./demo runs/my-run
./demo runs/my-run/demo/champion.json
```

This starts a local server at `http://localhost:8765` and opens the Windows
default browser using PowerShell when running in WSL. Choose your own start,
goal, and sampling seed, then run an attempt. Play/pause, step, speed, and a slider
control the animation. “Try again” changes the sampling seed. Training history
is omitted. Every attempt uses fresh copies of the saved learned weights with
learning disabled; the tester never trains or searches for a successful path.
Use `--no-open` to suppress browser launch or `--port` to choose another port.
Ctrl-C stops the server. Old zero-target animals cannot accept custom goals;
train a new varying-goal run to use them.

Compare a saved performer with uniform random movement on fresh held-out tasks:

```sh
python3 -m static_agent_training.benchmark runs/my-run --episodes 2000
```

Both policies receive the same ordered start/goal tasks and the same world
boundaries and 12-move limit. The random policy samples uniformly from all nine
movement buttons, including zero, and emits no predictions. The benchmark freezes
the selected animal's learned weights, records every outcome, and reports success
rates and 95% intervals. It also calculates the exact random success probability
for the sampled tasks. The task values come from the saved held-out partition;
these are fresh task draws, not a new partition. Results are saved to
`demo/benchmark.json` and shown in the playground when that animal is loaded.
The playground's Policy selector also lets you watch random movement on your
own tasks. Every new training generation records exact random expectations for
its training and assessment tasks alongside the animal scores; those expectations
do not influence selection.

A larger training session can be started with:

```sh
./train --population 64 --generations 150 --episodes 100 --assessment-episodes 100
```

Check background runs without finding log files:

```sh
./train status
./train status --watch
```

`./status` and `./status --watch` are equivalent shortcuts. Status lists active
runs first, then saved runs, with completed generations, percentage, latest
training and held-out scores, the random baseline where available, elapsed time,
and an estimated remaining time based on recent completed generations. Stopped
runs are distinguished from completed runs using live process discovery. Watching
status never starts or stops training; Ctrl-C only exits the status display.

While the playground is running, open `http://localhost:8765/runs` for the same
information in an automatically refreshing browser view. The playground links
to this page and displays a compact summary of active background runs. Status
refreshes every five seconds and reports completed generations; ongoing work
within the next generation is reflected when that generation finishes.

Optional acceleration preserves sequential lifetime learning while evaluating
independent animals in parallel. Choose `--backend python` (the original
reference) or `--backend native` (compiled C++ with CPU workers). The default remains `python`. The launcher uses the project's
`.venv/bin/python` when that environment exists.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-accelerated.txt

./train --backend native --workers 8 --batch-size 1024 --population 1024 --generations 150 --episodes 100 --assessment-episodes 100
```

The native backend needs `g++` with OpenMP support; compiled libraries are cached
under `~/.cache/static-agent-training`. NumPy is the only optional accelerated
Python dependency. New sessions support Python and native CPU evaluation.
Saved configurations from the retired CUDA backend load using native CPU with an
explicit migration notice; the model weights and historical results are retained.
Resumed runs record this backend migration in their configuration and checkpoint.

The compiled evaluator uses double precision, separate per-animal MT19937 streams
initialized directly from the reference Python RNG states, fixed repertoire edges,
sequential empirical updates, delayed compatibility correction, and the same
world limits. No target distance or movement policy is inserted into evaluation.
Inherited weights are copied before a lifetime; assessment uses the learned
weights with further learning disabled. Each generation's training champion is
rerun through the reference evaluator, its episode outcomes and statistics are
checked against the accelerated result, and its full reference trace is saved for
existing demo/export tools. Floating-point losses and learned logits can differ
by small rounding amounts; sampled behavior is checked by tests, with the champion
additionally checked during every accelerated generation.

Measure throughput on this machine:

```sh
.venv/bin/python -m static_agent_training.speed --populations 64 256 1024 --episodes 50 --assessment-episodes 50 --repeats 3
```

This benchmarks whole generation evaluation, including batch packing, RNG setup,
metrics and champion replay, after compilation warmup. It excludes
population initialization, reproduction, and checkpoint writes from the timed
samples (initialization time is recorded separately). The Python reference is
measured only up to 64 animals by default; change `--python-limit` to measure
larger populations. Genotypes are independently initialized with common seeds;
results represent the initial population, and evolved populations may take longer.
The JSON report includes all samples, medians, and resulting fitness summaries.

The playground's **Best candidate from run** selector loads the strongest saved
training performer from a chosen run, including runs still in progress. Scores,
generation, goal support, and the corresponding random comparison update for that
candidate. Selection ranks saved generation champions by training success, with
ties going to the earliest generation. **Refresh best** reloads the latest winner
for the selected run and discovers new runs. Runs without a published champion
are listed but unavailable until a generation has been saved.

Each attempt includes the selected candidate's run and generation identifier, so
new training results or selections in another browser tab cannot silently change
the animal being tested. Candidate loading reads saved traces and learned weights
without resuming training or rewriting its files. Large generation logs are scanned
for their small summaries rather than decoded in full for run selection/status.


Configure the trial-distance distribution for a new experiment:

```sh
./train --backend native --population 1024 --distance-mean 10 --distance-stddev 5
```

The optional `--distance-distribution normal` assigns each feasible nonzero integer
distance `d` a probability proportional to `exp(-0.5 * ((d - mean) / stddev)**2)`.
Weights are normalized over distances that can actually occur in the relevant
position partition. Then a pair at the chosen distance is sampled uniformly.
This avoids accidentally overweighting short distances just because more pairs
realize them. Bounds and sparse held-out partitions truncate the distribution, so
the realized mean and spread can differ from the configured center and scale.
Both directions are equally likely when both orientations are feasible. Mean must
be between 1 and 40 and standard deviation must be finite and positive.

Training, frozen assessment, and fresh random-comparison benchmarks use the saved
distribution settings. Assessment tasks stay fixed throughout a run; training
resamples each generation. Every generation stores its actual training and
assessment distances in `task_sampling`. Manual playground tasks still use the
start and goal you enter. Use `--distance-distribution uniform` to retain the
previous independent start/goal sampling. Older checkpoints automatically retain
that uniform sampling and their original RNG streams when resumed; sampling
settings cannot be changed on resume.


Enable optional structural speciation for a training run:

```sh
./train --backend native --population 1024 --speciation
# Tunable compatibility threshold and maximum niche count:
./train --backend native --population 1024 --speciation --speciation-threshold 0.6 --max-species 16
```

Speciation is off by default, including for older checkpoints. This implementation
groups lambda repertoires using a weighted Jaccard distance: 25% initial seed-pair
IDs, 50% repertoire pair IDs, and 25% stored successor edges. Mapping weights do
not affect membership. Compatible representatives retain persistent species IDs
across generations and checkpoints. New incompatible structures create species
until the configured cap is reached; excess unmatched candidates join the closest
species, with those forced assignments recorded. The cap is also limited by the
number of non-elite offspring slots.

The global elite count, tournament size, mutation operators, and mutation seed
coordinates match the control. Within a species, adjusted fitness divides each
member's training fitness by species size, making the species' total shared
fitness equal to its mean raw fitness. Each species reserves one mutated
offspring; remaining offspring slots are allocated in proportion to shared fitness
using deterministic largest-remainder rounding. All-zero fitness uses equal
species allocation. Parent tournaments then run within each assigned species.
With one species this reproduces the ordinary selection behavior. Assessment
fitness never influences clustering, allocation, parent choice, or elite choice.

This is a bounded structural-niche treatment for this project's lambda graphs,
inspired by [explicit fitness sharing](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/node3.html),
not a full NEAT implementation: it has no crossover, innovation-number machinery,
or adaptive compatibility threshold. Generation records include membership,
representatives, shared-fitness statistics, offspring quotas, new/extinct species,
and forced assignments. Summaries report species sizes and unique structures;
checkpoints preserve representative state for deterministic resume. The legacy
`Species` Python class still denotes one inherited genotype, not a niche.

Run a matched speciation/control benchmark:

```sh
./train compare
# Smaller exploratory study:
./train compare --population 256 --generations 8 --episodes 40 --assessment-episodes 80 --benchmark-episodes 1000
# Resume an interrupted study without changing its saved settings:
./train compare --resume runs/comparisons/my-study
```

Default comparison: seeds 42, 43, 44; 1,024 animals per arm; 30 generations;
100 training and 200 assessment episodes per animal; native evaluation with eight
workers per arm; uniform valid start/goal pairs and hard walls;
2,000 final fresh held-out trials per arm. Both arms for a seed run concurrently,
then the runner advances to the next seed. Each arm has its own ordinary run
folder, checkpoint, trace files, and log. `runs/comparisons/<study>/manifest.json`
records the complete study, and `report.json` is atomically refreshed throughout.
A study lock prevents two controllers from resuming the same comparison together.
Ctrl-C on the controller stops its active arms and leaves checkpoints resumable.

The comparison verifies initial population hashes and task hashes for every
paired generation. Both arms use the same population size, elite budget, learning,
mutation settings, input partitions, ordered tasks, evaluation seeds, and generation
budget; only the speciation switch differs. This compares equal evaluation budgets,
not equal wall-clock time. The primary endpoint selects each run's highest-training
saved champion without using assessment scores, then freezes its learned mappings
and evaluates both candidates on fresh shared draws from the held-out partition.
The random-movement baseline uses those same final tasks and is also recorded.
Per-seed differences and their mean are reported, with a 95% t interval across
paired seed differences; episodes are not treated as independent experimental
seeds. With three seeds, results remain exploratory. A long-run result is reported
only after the corresponding final evaluations are complete; live population
comparisons use matching completed generations.

Open `http://localhost:8765/comparisons` while the playground is running for live
study progress, per-seed results, confidence intervals, and links to test each
arm's best candidate. Training status and playground navigation link to this page.


Live playground attempts always use hard walls for both the saved animal and
random movement. The display identifies overshooting moves as stopping at a wall
and shows the exact random success probability for the entered start and goal.
Saved training/assessment scores and independent benchmark results retain the
rules used when they were collected; the view identifies those original boundary
rules. Faithful exported historical replays continue to reproduce their saved
rules. New native CPU training and random baselines use the same hard-wall semantics.

A controlled target-location diagnostic is available:

```sh
.venv/bin/python -m static_agent_training.goal_diagnostics runs/my-goal-aware-run
```

It freezes learned mappings, evaluates fixed distances 1, 5, and 10 for every
feasible target, compares against exact random-movement probabilities, and repeats
under the original and hard-wall rules. Rates are averaged per target so targets
with two feasible approach directions do not receive extra weight. The JSON output
also records expected target exposure from each candidate's original normal-distance
training distribution. This diagnostic changes no training state.


A lifetime-learning treatment is available in the same matched comparison runner:

```sh
./train compare --treatment learning --population 256 --generations 20 --episodes 100 --assessment-episodes 200
```

This runs learning off versus learning on with speciation disabled in both arms.
Learning off disables both empirical expectation updates and delayed embedding
compatibility corrections during the lifetime. It does not remove expectation
outputs, change the repertoire, alter mutation settings, or inherit lifetime
weights. Learning on retains the current error rules. Both arms use the same
initial genotypes, tasks, evaluation RNG coordinates, and evaluation budgets;
assessment and final testing freeze weights in both. Evolution may subsequently
diverge because training fitness differs. This tests the lifetime-learning package
as a whole; it does not separately attribute effects to its two update rules.

The default `./train compare` still tests speciation; `--treatment learning` selects
the new experiment. Reports store the treatment and explicit labels. Resume reads
the saved treatment, and historical speciation reports remain readable. The live
`/comparisons` view labels learning on/off and links to each arm's saved candidate.


To compare inherited lifetime learning against the default inheritance rule:

```sh
./train compare --treatment inheritance --population 256 --generations 20 --episodes 100 --assessment-episodes 200 --workers 4 --batch-size 256
```

Both arms enable lifetime learning and disable speciation. Inheritance on commits
end-of-training embedding and output logits to each reproductive parent, including
elites, before ordinary mutation and redevelopment. Inheritance off retains the
original genotype weights. Assessment never supplies inherited updates. Pending
expectations, traversal state, and learner revision flags are not inherited.
Generation records retain evaluated and reproductive parent IDs for provenance;
checkpoints preserve the resulting next population. Existing runs default to
inheritance off. A standalone run can enable this with `--inherit-lifetime`.

CLI training and comparison runs use a default **4096 MiB address-space limit
per process** (`--memory-limit-mb` changes it). This includes virtual mappings,
so a run can hit the ceiling before resident RAM reaches that size. Comparison
children inherit the parent's ceiling and apply their own default ceiling.
Evaluator array requests above 512 MiB fail with guidance to reduce batch size
or episode counts. Completed generations release their serialized state before
the next evaluation.

For an output directory `runs/example`, diagnostics are appended to
`runs/example-diagnostics/crash.log` (Python exceptions and fatal signal stacks)
and `resources.jsonl` (PID, command, memory ceiling, current RSS/virtual size and
peak RSS sampled every five seconds). Comparison children have separate logs.
A SIGKILL or WSL/host shutdown cannot produce a final traceback; the last resource
sample remains useful. Checkpoint and generation writes sync file contents and
the containing directory before reporting success. Progress reads avoid memory
mapping, which can produce SIGBUS if a file is truncated.

Each run also writes `champions.json`, an index of its top ten distinct saved
generation-champion genotypes, ranked only by training success (earliest generation
breaks ties). Use `--champion-count 20` to change the collection size. Entries link
to complete existing champion snapshots with inherited structure and mappings,
learned mappings, and training/assessment traces. All generation champions remain
saved, even when outside this index; repeated genotypes keep their best-scoring
lifetime. Resume rebuilds the index from saved generations. This collection does
not select on held-out scores or imply independent structural discoveries.

## Learning-rule trace audit

The [paired learning-event record](docs/records/learning-event/README.md) contains complete before/after
distributions for observation, expectation, outcome, empirical feedback, delayed
structural correction and the next prediction. The sensory model preserves the
two-stage learning schedule but changes the prediction objective, output-policy
coupling and shared structural credit assignment. Tooling parity does not imply
identical learning dynamics. Reproduce with `./project trace-learning`.

Training setup now includes original `[-20,20]` and shifted `[30,70]` environments,
saving top candidates and loading them into another ordinary evolutionary session.
See [the environment and population guide](USE.md#shifted-environment-and-reusable-populations)
for the choose environment → train → save candidates → load candidates → train workflow.
