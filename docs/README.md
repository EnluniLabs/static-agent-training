# Documentation map

| Document | Purpose |
|---|---|
| [Agent Y](AGENT_Y.md) | Learned receptor grouping conditioned on full input over shared embeddings |
| [Current architecture](ARCHITECTURE.md) | Model boundaries, receptor flow, evolution, CPU execution, caching and persistence |
| [USE.md](../USE.md) | Command guide, run tracking, artifacts and cleanup |
| [Living glossary](JARGON.txt) | Static-agent concepts and related ML terminology |
| [Project README](../README.md) | Detailed baseline algorithms and options |
| [Implementation boundaries](IMPLEMENTATION.md) | Existing baseline behavior and attribution limits |
| [Three-panel switchboard](SWITCHBOARD.md) | Independent panels aligned with parallel sensory regions; matched comparisons |
| [Receptor signals](RECEPTOR_SIGNALS.md) | Default Agent X binary layout, shared dual-rail conversion and local sensors |
| [Historical fixed retina](FIXED_RETINA.md) | Packed-pixel drawing retained for snapshot compatibility |
| [Universal receptor input](UNIVERSAL_INPUT.md) | Generic byte streams, observation-driven processing and shared partitions |
| [Exploratory structure profile](EXPLORATORY_STRUCTURE.md) | Larger initialization, structural mutation controls and growth metrics |
| [Agent X parallel activation](AGENT_X.md) | Fixed sensory partitions, independent predictions, agreement and voting |
| [Lights Out environment](LIGHTS_OUT.md) | Puzzle rules, encodings, tasks and checks |
| [Baseline environment contract](specs/ENVIRONMENT_INTERFACE.md) | Number-line observations, actions and feedback |
| [Chain development specification](specs/CHAIN_MERGE_SPEC.md) | Lambda development and budget merging |
| [Fixed-senses design](AGNOSTIC_INTERFACE.md) | Implemented experimental sensory model |
| [Learning-event comparison](records/learning-event/README.md) | Full before/after distributions and learning-rule differences |
| [Fixed-senses audit](AGNOSTIC_INTERFACE_AUDIT.md) | Gaps between prototype and intended research design |
| [Research roadmap](RESEARCH_ROADMAP.md) | Staged research priorities |
| [Three core experiments](THREE_CORE_EXPERIMENTS.md) | Main controlled research questions |
| [Coherent versus random test](COHERENT_VS_RANDOM_TEST_SPEC.md) | Matched repertoire hypothesis test |
| [Inherited animal proposal](REPERTOIRE_LIBRARIES_AND_INHERITED_ANIMAL.md) | Reusable repertoire research proposal |
| [Future work](../FUTURE_WORK.md) | Hypotheses and proposed experiments; not live run status |
| [Learning experiment plan](LEARNING_EXPERIMENT_PLAN.md) | Selection, retained adaptation and inheritance controls |
| [Experiment history](EXPERIMENTS.md) | Dated historical studies, not a live registry |
| [Bias audit](BIAS_AUDIT.md) | Historical controls and limits on interpretation |
| [Curated results](results/) | Tracked summaries of selected historical measurements |

For current saved/running experiments, use `../project runs` or `../project
status` from this directory. Raw runs remain under `runs/` and are not tracked
in Git. Several historical documents preserve statuses recorded on their dates;
the run catalog and saved reports provide the current local evidence.

## Reference papers

- [Static Agent Evolution technical report](references/Static_Agent_Evolution_Tech_Report.pdf)
- [The Coherence Engine](references/MacDonald_The_Coherence_Engine.pdf)
- [The Universal Explainer](references/MacDonald_The_Universal_Explainer.pdf)

The PDFs and baseline specs were moved into these folders during organization;
their content was preserved. Model hypotheses in the papers are not demonstrated
merely by having an implementation or passing regression tests.

- [Training performance optimization report](performance/TRAINING_OPTIMIZATION_REPORT.md): measured stage costs, architecture-preserving changes, and validation gates.
- [Checkpoint save speed](performance/CHECKPOINT_SAVE_SPEED.md): measured compression and buffered-writing improvements with identical saved content.

- [Genome caches and offspring speed](performance/GENOME_AND_OFFSPRING_SPEED.md): immutable caching, bounded parallel reproduction, reproducibility, and measured tradeoffs.

- [Current E3 training benchmark](performance/CURRENT_RUN_BENCHMARK_20261122.md): a live-run snapshot replay and per-stage timing breakdown.
- [Safe save optimization](performance/SAFE_SAVE_OPTIMIZATION.md): validated provenance streaming, candidate compression, parity checks and measured tradeoffs.

- [Agent Y parallel evaluation](performance/AGENT_Y_PARALLEL.md): measured worker scaling, deterministic results and per-generation progress.

- [Agent Y native CPU](performance/AGENT_Y_NATIVE.md): compiled grouping and learning, Python parity and evaluator migration.

- [Workspace polling performance](performance/WORKSPACE_POLLING.md): metadata-only caches and avoiding full-model loads during discovery.
