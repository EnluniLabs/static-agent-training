# Using this project

Use `./project` for the command menu. It selects `.venv/bin/python` when present
and resolves relative paths from the project root, even when invoked elsewhere.
Opening the menu or asking for help starts no experiment. The older `./train`,
`./status`, and `./demo` entry points remain supported.

## Find and inspect work

Open the model workspace with:

```sh
./project ui
```

It opens `http://localhost:8765`. Use `--no-open` to print the address without
opening a browser, or `--port 8766` for another local port. No saved animal is
required to start the workspace. `./project demo` and `./demo` open it too.

The **Sessions** tab lists all model families, plots saved performance, displays
per-seed comparisons, and shows logs, configuration and downloadable artifacts.
The **New session** tab offers **Evolutionary training** or **Compare starting populations**. Both run full evolution. Choose the
model, **Environment**, and **Load saved candidates** (or **Start fresh**), then
set population, generations and episode budgets. Fixed senses also exposes receptor
capacity. Both models support native CPU and Python evaluation. Standalone lifetime/range
transfer, learning-treatment comparisons and other research procedures are not session
types in this form; existing saved reports and CLI research tools remain available.

The main workflow is:

1. Pick a model and environment and run evolutionary training.
2. On that session's overview, enter a candidate-set name and count, then click
   **Save candidates**.
3. Open **New session**, select the same model, choose the set under **Load saved
   candidates**, and pick the next environment.
4. Set the new generation target and start normal evolutionary training again.

Clicking **Start session** starts a separate background process. Closing the
browser or stopping the workspace server does not stop that training process.
Workspace-launched sessions expose **Stop session**; stopping preserves the last
completed checkpoint, so an interrupted generation reruns on resume.
Saved evolutionary runs for either model can be resumed or extended using the generation-target
field. Resumable stopped comparisons can also be restarted. Fixed-senses
adaptation currently has no resume implementation. Existing external live
processes are monitored without taking ownership or exposing stop controls.

The **Animal playground** tab selects the categorical or fixed-senses model,
source run, and an exact saved snapshot. It retains task start/goal/seed controls,
random movement, replay, stepping, restart, speed and scrubbing. Sensory models
also show their observed/predicted receptors and support the expanded world.
Learning is off by default. Enable **Lifetime learning across attempts** to keep
prediction and structural updates in a separate playground copy for either model.
Turning learning off freezes that current copy; **Reset learned state** restores
the selected saved snapshot. Random-policy attempts do not train the animal.
The status shows cumulative prediction updates and structural corrections.
Changes stay in visualizer memory across attempts, including world changes;
selecting a different snapshot, reloading the page, or restarting the server
starts a fresh playground lifetime. Saved mappings and training runs are never
modified. The attempt seed controls sampling while weights persist.
Use **Download model** to export the original selected snapshot. Original views remain
at `/playground`, `/runs`, and `/comparisons`.

New browser-launched sessions record identity, configuration, PID, process
identity, command and lifecycle under `runs/ui-sessions/`. Session control uses
same-origin local requests and an instance token; only validated option lists
are passed to subprocesses. The server binds to localhost, not a public network.

```sh
./project help
./project doctor
./project runs
./project runs --kind training
./project runs --kind sensory-training
./project runs --kind senses --all
./project runs --kind comparison --all
./project runs --kind interfaces --all
./project runs --all --json
./project status --watch
./project show latest-training
./project show latest-senses
./project show latest-comparison
./project show latest-interfaces
./project champions latest-training
```

`runs` lists the 20 most recently updated recognized runs by default. It reads
existing artifacts; it does not maintain another potentially stale registry.
Training, sensory models and paired comparisons are separate kinds. `status`
is the existing detailed live baseline-training monitor; Ctrl-C exits monitoring
without stopping training. A process that is absent is not inferred to be alive
from an old sensory `running` flag.

Commands accept a saved directory name, its path, or a typed latest alias:

| Alias | Selects |
|---|---|
| `latest-training` | Most recently updated baseline training run |
| `latest-sensory-training` | Most recently updated fixed-senses evolutionary run |
| `latest-model-comparison` | Most recently updated evolutionary old-versus-sensory study |
| `latest-senses` | Most recently updated sensory run |
| `latest-comparison` | Most recently updated paired comparison |
| `latest-interfaces` | Most recently updated old-versus-sensory benchmark |

Latest means latest artifacts, not best performance or necessarily completed.
Use an explicit run name when reproducibility matters. `show` displays settings,
compact results, champion references, and log/diagnostic locations. It does not
load the population checkpoint or print every task and episode.

## Baseline evolutionary model

```sh
./project train --help
./project train --population 32 --generations 50 --seed 48
./project train --population 1024 --generations 1000 --seed 48 \
  --episodes 100 --assessment-episodes 200 --backend native \
  --workers 4 --batch-size 256 --champion-count 10 \
  --no-inherit-lifetime --no-speciation --output runs/baseline-example
./project resume latest-training --generations 1000
```

The launcher defaults to 32 animals, 50 generations and seed 42. New baseline
runs use uniform start/goal pairs, hard walls, learning on, learned-weight
inheritance off, and speciation off. Plain `python -m ...experiment` has different
population/generation defaults; prefer the documented launcher or explicit flags.
Native CPU evaluation requires the optional dependencies and a C++ compiler.

Every completed generation checkpoints the next population. Interrupted partial
generations are rerun on resume. Resume preserves saved model/environment settings;
it does not turn a baseline run into a sensory-model run. A completed run can be
extended with a larger generation target.

## Fixed-senses alternative

For evolutionary population training, use the same trainer as the old model:

```sh
./project train --model senses --receptors 256 --population 32 --generations 50 \
  --episodes 100 --assessment-episodes 200 --backend native --seed 48
./project resume latest-sensory-training --generations 100
./project champions latest-sensory-training
./project benchmark latest-sensory-training --episodes 2000
```

This evolves initial lambda seeds and all sensory parameter groups using the
existing mutation, selection, elites and optional speciation machinery. Surviving
repertoire IDs retain sensory weights after structural mutation; new IDs receive
deterministic initial values. Each lifetime learns from a mutable copy. By default,
offspring inherit original genotype weights; `--inherit-lifetime` enables copying
end-of-training sensory parameters before reproduction. Assessment stays frozen
and never supplies inherited updates. Receptor capacity remains fixed across
generations and resume. Both models support Python and native CPU evaluation,
and clamp or terminate boundary modes. Checkpoints, generation records and champion collections use the shared
training artifact layout with tagged sensory mappings.

For the separate fixed-structure adaptation/transfer experiment:

```sh
./project senses --help
./project senses --receptors 256 --seed 48 --episodes 100 \
  --assessment-episodes 200 --label range-transfer \
  --notes "Check fixed-capacity operation across expanded input values" \
  --output runs/senses-example
./project senses --candidate latest-training --receptors 1024 \
  --seed 48 --output runs/senses-from-champion
```

These examples launch experiments only when you execute them. With no `--output`,
`senses` creates a unique `runs/senses-TIMESTAMP` directory. Existing nonempty
outputs are rejected. Receptor capacity is selected once for a new animal and
saved; it does not change when the environment expands. `--candidate` imports
only a baseline champion's repertoire, not its categorical mappings.

The `senses` command is the fixed-structure lifetime-learning/transfer runner;
evolution uses `train --model senses` above. Adaptation saves between-episode model snapshots; that experiment
does not yet provide CLI resume. `run.json` records PID, command, seed, capacity,
label, notes and running/completed/failed/interrupted status. `show` reads those
alongside the report. Historical sensory reports without `run.json` are recognized.

The current converter can saturate on long inputs. The old-region evaluation is
not yet a clean original-world retention test. Read the [interface design](docs/AGNOSTIC_INTERFACE.md)
and [audit](docs/AGNOSTIC_INTERFACE_AUDIT.md) before interpreting results. Ordinary
binary/alternative encoding controls and robust multi-environment comparisons
are not implemented. Passing interface checks is not evidence of useful transfer.

## Comparisons, assessment and browser

To evolve both models over matched generations:

```sh
./project compare --treatment model --population 32 --generations 50 \
  --episodes 100 --assessment-episodes 200 --benchmark-episodes 2000 \
  --receptors 256 --seeds 48 49 50 --backend python
./project show latest-model-comparison
./project compare --resume latest-model-comparison
```

Both arms begin with identical ordered lambda seeds, budgets and developed
repertoires; every generation's task hash is verified. Selection uses training
fitness. The final endpoint compares each arm's highest-training saved champion
on fresh common held-out tasks, while generation records separately report
population means. Both arms use the selected evaluator. Parameter counts, encodings and prediction targets differ, so this is a
whole-model comparison. The browser's training-population table opens either arm's
generation chart and saved champions.

For a direct old-model versus fixed-senses lifetime comparison:

```sh
./project compare-models --population 16 --seeds 48 49 50 \
  --episodes 100 --assessment-episodes 2000 --receptors 256
./project show latest-interfaces
./project compare-models --resume latest-interfaces
```

This develops 16 independent repertoires for each seed and gives each model the
same repertoire, training tasks and frozen held-out tasks. Both start with fresh
mapping parameters; neither imports evolved categorical weights. There is no
evolution, champion selection or expanded-world evaluation in this benchmark.
Its primary endpoint is population mean frozen task success, with effects and
uncertainty over independent seed means. Parameter counts, encoding and output
distributions differ; this is not a capacity-matched interface-only ablation.
Prediction accuracy targets also differ and must not be compared directly.

Runs are saved under `runs/interfaces/`. `plan.json` records settings and source
hashes, `tasks/` records shared tasks and exact random baselines, and each
`seed-N/animal-NNNN/` holds repertoire provenance, initial/trained snapshots and
`result.json`. The top-level `report.json` updates after each completed pair.
Resume skips completed pairs and deterministically reruns interrupted pairs
using the saved plan. `show` summarizes the report without dumping every pair.

```sh
./project compare --help
./project compare --treatment learning --population 256 --generations 200 \
  --seeds 42 43 44 --workers 4 --batch-size 256
./project compare --treatment inheritance --population 256 --generations 200
./project compare --resume latest-comparison
./project benchmark latest-training --episodes 2000
./project audit latest-training --sample 128 --output runs/my-audit.json
./project demo latest-training
./project speed --help
```

`compare --treatment model` and `benchmark` support sensory evolutionary runs.
The existing learning/inheritance/speciation CLI comparisons and `audit` retain
their baseline defaults. `benchmark` evaluates the training-selected best
saved champion with learning frozen against random movement. `audit` evaluates
isolated copies. `demo` serves the model workspace; an optional baseline source
initializes its legacy playground. Throughput benchmarks are separate from learning
experiments. The comparison tool defaults to the historical speciation treatment
unless `--treatment` is given; specify the treatment intentionally.

## Champion collections

```sh
./project champions latest-training
./project champions RUN_NAME --count 5
./project champions RUN_NAME --rebuild --count 20
```

Each baseline run saves its champion from every completed generation. The index
keeps the top ten distinct champion genotypes by default, ranked using training
fitness only, with earliest-generation ties. Each entry points to a complete
compressed trace with genotype and learned mappings. It is not the top ten
animals from each generation. Structural duplicates can have different genotype
IDs; repeated copies are not independent discoveries. `--rebuild` explicitly
recreates the index from retained generation records and traces and may take time.

## Artifact map

| Location | Contents |
|---|---|
| `runs/NAME/config.json` | Baseline configuration and partitions |
| `runs/NAME/checkpoint.json.gz` | Resumable next baseline population |
| `runs/NAME/generations/` | Generation metrics, tasks and ancestry |
| `runs/NAME/traces/` | Full saved generation champions |
| `runs/NAME/champions.json` | Champion collection index |
| `runs/NAME/demo/benchmark.json` | Fresh frozen baseline champion assessment |
| `runs/NAME/run.json` | New sensory experiment identity and lifecycle |
| `runs/NAME/report.json` | Sensory results, task lists and source provenance |
| `runs/NAME/*.json.gz` | Sensory initial, trained and transfer-arm snapshots |
| `runs/comparisons/NAME/` | Paired-study manifest, report and child logs |
| `runs/interfaces/NAME/` | Matched categorical-versus-sensory reference benchmark |
| `runs/ui-sessions/` | Browser-launched process identity and lifecycle records |
| `runs/NAME.log` | Background output when redirected to this convention |
| `runs/NAME-diagnostics/` | Baseline process resource samples and crash log |
| `docs/results/` | Curated historical summaries, tracked in Git |

Old runs stay at their original paths. Raw `runs/`, virtual environments and
generated bytecode are ignored by Git. Documentation, source and tests are tracked.
Descriptive output names plus sensory `--label`/`--notes` keep studies identifiable.
Comparison reports contain their last recorded status; use baseline live status
and child logs to establish whether a historical comparison is still active.

For background work, choose an explicit name and preserve the log:

```sh
nohup ./project train --population 1024 --generations 1000 --backend native \
  --output runs/my-study --no-live > runs/my-study.log 2>&1 < /dev/null &
./project status --watch
```

Run that shell example from the project root with an existing `runs/` directory.
Stopping the monitor does not stop the background process. Baseline CLI diagnostics
default to a 4096 MiB address-space ceiling; `--memory-limit-mb` adjusts it.
Sensory experiments currently record lifecycle metadata, not baseline resource
diagnostics or automatic recovery.

## Regression checks and cleanup

```sh
./project check
./project check tests.test_fixed_senses tests.test_bit_model
./project clean --dry-run
./project clean
```

`check` runs regression tests using temporary test fixtures, not a persistent
benchmark study. `clean` removes only Python bytecode under source and tests.
It never deletes run artifacts, checkpoints, reports, virtual environments,
reference documents or native compiler caches. Removing or archiving real runs
is a separate deliberate operation, not part of routine cleanup.

## Where to edit

| Area | Location |
|---|---|
| Shared command interface and catalog | `static_agent_training/cli.py`, `catalog.py` |
| Local browser API and session manager | `play.py`, `workspace.py`, `workspace_sessions.py`, `workspace_models.py` |
| Browser workspace | `static_agent_training/web/` |
| Existing evolutionary model | `evolution.py`, `agent.py`, `learning.py`, `mappings.py` |
| Existing environment and evaluation | `environment.py`, `evaluation.py`, `accelerated.py` |
| Experimental fixed senses and adapter | `experimental/senses.py` |
| Experimental learned mappings and predictions | `experimental/bit_model.py` |
| Experimental world and runner | `experimental/number_line.py`, `experimental/bit_experiment.py` |
| Regression tests | `tests/` |
| Technical specs and papers | `docs/specs/`, `docs/references/` |
| Example seed bundle | `examples/explanations.json` |

See the [documentation map](docs/README.md) for the roadmap, experiment history,
audits and distinction between implemented behavior and proposed research.

Both evolutionary models now offer the same evaluator choices (`python`, `native`), checkpoint/resume flow, mutation and selection options,
champion collections, frozen benchmarks and playback exports. For sensory
learning/inheritance/speciation comparisons, add `--model senses` to
`./project compare`. `./project audit latest-sensory-training` runs frozen
sensory goal-input controls. To measure both implementations on this machine:

```sh
./project speed --models categorical senses --backends python native --populations 32 --repeats 2
```

Accelerated sensory evaluation supports up to 16384 receptors and 4096 repertoire
states per animal, subject to the batch memory guard; use Python for larger
capacities. The visualizer exposes evaluator, CPU workers and batch size for
both evolutionary models. Earlier saved Python runs retain their saved backend
when resumed.

## Learning-rule trace audit

The [paired learning-event record](docs/records/learning-event/README.md) contains complete before/after
distributions for observation, expectation, outcome, empirical feedback, delayed
structural correction and the next prediction. The sensory model preserves the
two-stage learning schedule but changes the prediction objective, output-policy
coupling and shared structural credit assignment. Tooling parity does not imply
identical learning dynamics. Reproduce with `./project trace-learning`.

## Shifted environment and reusable populations

Before launching evolutionary training, choose **Environment**: original
`[-20,20]` or shifted `[30,70]`. Both models support both worlds with Python,
native CPU evaluation. The shifted world has identical movement buttons,
41 positions and wall rules, but sensory packets contain the actual shifted
coordinates. No centering or distance is supplied to the animal. Fixed-goal mode
uses the world's center: 0 in the original world, 50 in the shifted world.
The environment is saved in checkpoints and cannot change on resume; start a new
session to change environments.

```sh
./project train --model senses --environment shifted --backend native \
  --population 32 --generations 50 --population-save-count 10
```

New runs maintain `best-population.json.gz`, retaining distinct inherited genomes
ranked by their highest observed training fitness across completed evaluations.
Ties prefer earlier generation and animal index; held-out scores never select the
members. The **Top candidates to retain** field controls the collection size. Training
tasks can differ by generation, so these saved selection scores are not a matched
comparison by themselves.

On a training session's overview, **Save candidates** creates a separate,
reusable collection. It remains unchanged when the source session continues.
Older sessions can export only their saved generation champions; the collection
records this limitation instead of claiming to contain all top individuals.

Under **Load saved candidates**, select a saved collection for a new session. The
best ranked members fill the first slots; if more animals are requested, remaining
slots are deterministic mutated descendants. If fewer are requested, only the
highest ranked members are loaded. Collections contain inherited genotype weights,
not end-of-lifetime learned weights. This preserves ordinary evolutionary
inheritance semantics and makes the new lifetime-learning budget explicit.
Model type, receptor capacity, action encoding and goal-input mode must match.
Fixed-senses candidates retain all mappings when moving to the other environment.
Categorical candidates retain the repertoire and mapping entries whose opaque token
identities occur in both environments, including the movement columns. Unseen
observation/prediction tokens receive deterministic initial values using the usual
mapping initializer; no numeric relationship is supplied and no lifetime learning
rule changes. Thus categorical transfer does not preserve predictions for tokens
absent from the new vocabulary. Both models can start a new evolutionary session
in either environment. `candidate-import.json` records this preparation explicitly. Imported genomes are copied into
`initial-population.json.gz` in the new run for provenance; resume uses its saved
checkpoint and does not reread the source collection.

```sh
./project populations save runs/MY-RUN --count 10 \
  --output runs/populations/my-population.json.gz --name "My population"
./project train --model senses --environment shifted --backend native \
  --initial-population runs/populations/my-population.json.gz \
  --population 32 --generations 50
```

For separate research comparisons, the CLI can benchmark two or more collections
in a common environment; this is not a session type in the training setup. Every animal receives the same sampled training tasks and
learning budget, followed by the same frozen held-out tasks. Structure stays fixed;
there is no evolution in this comparison. The chart and table report population
means, with individual results in `report.json`. A newly named output session
preserves each comparison. Different source models/capacities are whole-model
comparisons, not isolated tests of the sensory interface. Old source selection
scores do not enter this benchmark's scoring.

```sh
./project populations benchmark \
  --populations runs/populations/first.json.gz runs/populations/second.json.gz \
  --environment shifted --episodes 100 --assessment-episodes 200 \
  --backend native --output runs/population-comparison
```

The benchmark uses clamp boundaries and default local learning rates for every
collection, reports exact random movement performance on the common assessment
tasks, and freezes both learning rules during assessment. Collections must have
matching goal-input modes; categorical vocabulary restrictions still apply.


CUDA support has been removed. New sessions offer native CPU and Python only.
Saved CUDA configurations remain readable and resume with native CPU after an
explicit notice; resumed runs record the migration while retaining weights and
historical results. GPU dependencies are no longer part of the project setup.
Historical audit/benchmark records remain unchanged descriptions of their original
implementations and measurements.


## Evolutionary benchmark of starting populations

In **New session**, select your model and **Compare starting populations**.
Choose **Arm A — starting candidates** and **Arm B — starting candidates**
independently: each can be **Start fresh** or any saved candidate set for that model.
This supports fresh vs saved, two different saved sets, or two fresh populations.
Choose one common environment, population size, generations and episode budgets,
then start the benchmark.

Both arms run normal evolutionary training, including reproduction, mutation,
elites, and the same selected lifetime-learning/inheritance/speciation options.
Training and assessment task hashes are verified to match at every completed
common generation. Fresh arms use separate initialization seeds while retaining a
common seed for tasks and evolutionary evaluation. Imported candidates use the
same loading and environment preparation as ordinary evolutionary sessions;
extra population slots follow the existing deterministic-mutant rule.

The benchmark displays both arms' population training and held-out curves by
**generation**, with links to their full training sessions and saved candidates.
After evolution, each arm's highest-training saved champion is tested frozen on
fresh common held-out tasks. This final champion comparison is distinct from the
population means on the training curves. A single paired seed describes that run;
it is not an estimate of reliable average advantage. The CLI supports multiple
paired seeds if independent repetitions are wanted.

The manifest records source names, original file hashes, frozen input copies and
initialization seeds. Original candidate files can be removed after launch;
resume uses the benchmark's immutable copies and rejects altered copies. Both
candidate sets must match the selected model and receptor capacity and share an
action encoding and goal-input mode; the UI carries those settings into fresh
arms as well. The common learning/evolution settings apply equally to both arms.

```sh
# Saved vs fresh: omit --population-b for a fresh B arm.
./project compare --treatment starting-population --model senses \
  --population-a runs/populations/candidates.json.gz \
  --environment shifted --population 32 --generations 50 \
  --episodes 100 --assessment-episodes 200 --benchmark-episodes 2000 \
  --seeds 48 --backend native --output runs/comparisons/saved-vs-fresh

# Saved vs saved: provide both --population-a and --population-b.
# Fresh vs fresh: omit both arguments.
./project compare --resume runs/comparisons/saved-vs-fresh
```

The workspace's **Stop session** stops the paired benchmark and its training
processes, preserving completed-generation checkpoints. Resume continues the
remaining training and then runs the common final tests.


## Frozen assessments

Open the **Assessments** tab to evaluate one or more saved candidate populations.
Select populations, an environment, episodes per candidate, task seed and wall
rules, then click **Run frozen assessment**. The tab shows population mean/best/worst
success, expandable individual scores, exact random-movement performance on the
same tasks, and a downloadable full report. Saved assessments remain in the tab's
history and the session library; closing the browser does not stop a running job.

Assessment uses each candidate's inherited weights **exactly as stored** in the
saved collection. It calls only frozen evaluators: no warmup, lifetime updates,
mutation, reproduction, vocabulary initialization, or candidate adaptation occurs.
Every candidate receives the same ordered tasks. Scores are reproducible for the
same saved inputs, configuration and seed. Tasks are sampled across the selected
world and may overlap earlier training tasks; this is not an automatic claim of
held-out generalization.

Fixed-senses candidates can be frozen-tested in either environment. Categorical
candidates require their existing environment vocabulary; select their saved
environment instead of creating new entries during assessment. Populations assessed
together must use the same goal-input mode. The results retain exact input copies,
source hashes, tasks, candidate identities and zero empirical/structural-update
counts. Stop an assessment from its results panel to retain partial results.

```sh
./project assess --populations runs/populations/first.json.gz runs/populations/second.json.gz \
  --environment shifted --episodes 200 --seed 48 --workers 4 \
  --output runs/frozen-assessment
```
