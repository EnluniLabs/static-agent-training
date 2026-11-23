# Agent Z implementation

Agent Z is a separate experimental runner based on Agent X's budgeted lambda development, region-wise sensory activation, and fixed-successor traversal. The [Agent Z specification](https://github.com/EnluniLabs/static-agent-training/blob/a98dd4e/docs/AGENT_Z_SPEC.md) defines the corrected architecture. The existing Agent X and Agent Y commands still use their historical engine orientation and serialized repertoires.

## Run and assess

```sh
./project agent-z --seed 7 --backend python --generations 3 --population 8 --episodes 200 --output /tmp/agent-z-python.json
./project agent-z --seed 7 --backend native --generations 3 --population 8 --episodes 200 --output /tmp/agent-z-native.json
./project agent-z --seed 7 --candidate /tmp/agent-z-native.json --episodes 200 --learning off
.venv/bin/python -m unittest discover -s tests -p test_agent_z.py
```

Use `--structure-preset baseline` (the default) for the project baseline seed sizes, development budgets, and mutation settings.

Learning modes are `both`, `empirical`, `structural`, and `off`. Each evolutionary generation ranks agents on the same training tasks and measures its champion on held-out goals. Assessment also reports a frozen score. Within a lifetime, updates carry across episodes. Each evaluation starts from inherited parameters. Evolution selects parents using external task success, mutates inherited parameter rows, and sometimes replaces a seed pair.

The artifact stores the final inherited champion and generation summaries. Agent Z uses the version marker `crt-agent-z-function-first-v1`. Old repertoires encode the reverse orientation and must be redeveloped from seeds to use Agent Z; their transition tables and weights cannot be imported as Agent Z artifacts. Standalone runs with only `--output` save a final champion. Workspace training also writes resumable population checkpoints and retains each generation’s champion.

## Signals, rounds, and learning

Development applies the first term to the second and records `(NF(B A), B)` as the successor of `(B,A)`. Existing development limits still apply. A pair without a recorded successor emits once before its chain terminates. A measurable emission from a terminal pair can still produce an empirical update.

The number-line adapter emits one byte representing signed relative goal position. Each region uses a disjoint, interleaved subset of the eight bits for its initial embedding. All regions can reach the same repertoire and each active region draws a full-width signal from the pair's factorized Bernoulli continuation. There is no internal output classifier.

The adapter alone maps the exact bytes `0x70` through `0x78` to physical moves. When several actions occur in a round, it executes them in ascending byte order, including duplicates, until the world terminates. It then reads the current measurement once if any chain emitted an expectation. Every expectation receives that same measurement, including a valid zero byte. Action emissions receive no feedback and do not reactivate sensory chains.

Empirical learning records the bit mismatches of each sampled expectation and applies the gradient of bitwise negative log likelihood of the observation under the predecessor's original continuation distribution. Structural learning computes a region-local softmax over pairs using the exact sampled expectation, then applies a successor cross-entropy gradient when the successor probability falls below 0.1. The observation is used only for the empirical calculation. Round updates are calculated before changing either parameter table, summed per parameter using sorted contributions and an accurate floating-point sum, and then applied together.

The native CPU backend is the default. One C++ call evaluates an entire agent lifetime across all supplied episodes. C++ handles sensory activation, seeded signal sampling, successor traversal, the number-line action batch and shared measurement, simultaneous learning, and optional trace capture. Python supplies tasks, inherited parameters, initial RNG state, and the adapter's action table. Python also handles lambda development, evolutionary selection and reproduction, and artifact writing. Use `--backend python` for reference parity checks.

The native evaluator uses the one-byte number-line protocol above. New environments require their own native adapter. The workspace exposes Agent Z in **Setup**, **Sessions**, **Assessments**, and **Playground**. Workspace training runs on the native CPU and supports stopping, resuming, and extending a session. Each completed generation writes its metrics, champion snapshot, and next population checkpoint. Save Candidates exports the top scoring members of the latest generation. Exports with development seeds can start a new training session with the same region count. A legacy standalone result can be played and assessed using its final champion.

Frozen assessments use saved inherited weights and a shared set of number-line tasks. The playground uses a separate native lifetime, with resettable learning and a brain scan of the saved or current weights. Its stream inspector presents one file byte per frame to an isolated native evaluation. The passive stream records emitted signals and returns no environment feedback.

## Native evaluator benchmark

On WSL, a baseline run with seed 7, 2 generations, 8 agents, and 100 learning episodes per agent took a median of 3.72 seconds with the Python reference and 0.134 seconds with the native evaluator, excluding native compilation. The benchmark ran three times per backend and yielded identical generation metrics and champion data. Longer runs will also include the cost of Python development and reproduction as repertoires grow.


## Structural speciation

Enable **Structural speciation** in the Agent Z session setup or pass `--speciation` on the command line. Each species retains its top half by training fitness, rounded down with at least one survivor. Mutated offspring from those survivors refill the remaining slots. Held-out scores are excluded from selection.

Fingerprints use the existing ordered lambda-pair IDs and directed successor links. Unchanged children keep their parent's species. New fingerprints are compared with one fixed representative per species using a Jaccard distance weighted two-thirds toward pairs and one-third toward links. The distance threshold is 0.5. The population starts with at most eight species and leaves slots for later innovations. Total species are capped at 64 or half the population, whichever is smaller, with a minimum cap of one. When the cap is full, new structures join the closest representative. Checkpoints retain representatives and assignments.
