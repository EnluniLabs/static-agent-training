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

The artifact stores the final inherited champion and generation summaries. Agent Z uses the version marker `crt-agent-z-function-first-v1`. Old repertoires encode the reverse orientation and must be redeveloped from seeds to use Agent Z; their transition tables and weights cannot be imported as Agent Z artifacts. This runner provides candidate assessment and does not persist resumable population checkpoints.

## Signals, rounds, and learning

Development applies the first term to the second and records `(NF(B A), B)` as the successor of `(B,A)`. Existing development limits still apply. A pair without a recorded successor emits once before its chain terminates. A measurable emission from a terminal pair can still produce an empirical update.

The number-line adapter emits one byte representing signed relative goal position. Each region uses a disjoint, interleaved subset of the eight bits for its initial embedding. All regions can reach the same repertoire and each active region draws a full-width signal from the pair's factorized Bernoulli continuation. There is no internal output classifier.

The adapter alone maps the exact bytes `0x70` through `0x78` to physical moves. When several actions occur in a round, it executes them in ascending byte order, including duplicates, until the world terminates. It then reads the current measurement once if any chain emitted an expectation. Every expectation receives that same measurement, including a valid zero byte. Action emissions receive no feedback and do not reactivate sensory chains.

Empirical learning records the bit mismatches of each sampled expectation and applies the gradient of bitwise negative log likelihood of the observation under the predecessor's original continuation distribution. Structural learning computes a region-local softmax over pairs using the exact sampled expectation, then applies a successor cross-entropy gradient when the successor probability falls below 0.1. The observation is used only for the empirical calculation. Round updates are calculated before changing either parameter table, summed per parameter using sorted contributions and an accurate floating-point sum, and then applied together.

The native CPU backend uses compiled C++ kernels for embedding probabilities, continuation probabilities, and error gradients. Python and native share the scheduler, seeded draws, fixed successor traversal, environment, and update aggregation. The current native backend crosses the Python/C++ boundary per calculation. It is a parity implementation and has no batched evaluator or workspace UI.

## Benchmark

An earlier Agent Z prototype with smaller development budgets produced identical generation metrics and champion data on the WSL development machine. The timings below are historical and do not measure the current baseline preset. Across three runs per backend with the native library built before timing, median evolution and evaluation took **11.89 seconds in Python** and **17.95 seconds with native kernels**. Generation 3 reached **6.19% mean training success** and **5.0% held-out champion success**. This is a small parity benchmark with no evidence yet of a transfer advantage.
