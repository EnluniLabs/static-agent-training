# Documentation map

| Document | Purpose |
|---|---|
| [USE.md](../USE.md) | Command guide, run tracking, artifacts and cleanup |
| [Project README](../README.md) | Detailed baseline algorithms and options |
| [Implementation boundaries](IMPLEMENTATION.md) | Existing baseline behavior and attribution limits |
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
