# Bias and attribution audit

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

Recorded 2026-11-01. This audit reads saved artifacts and evaluates isolated copies.
It does not modify the live run, its inherited population, or its learning rules.

## Checks and limitations

Code inspection found training fitness alone used for tournaments and elites;
assessment copies have learning disabled, and generation 88 recorded zero empirical
updates and zero compatibility conflicts during assessment. Reproduction uses
inherited mappings. The numeric runner does not supply distance or an action hint.
Compiled champions are independently replayed by the Python evaluator each generation.
These checks found no identified held-out-to-selection leak; they are not a proof
that every implementation choice is unbiased or that no bug can exist.

The displayed random score is an exact dynamic-programming expectation for uniform
choice among all nine legal moves, including zero, on the same assessment task list
and with the same wall and 12-move rules. It is not a single noisy random run.

Uniform movement is only one baseline. Beating it does not demonstrate target use,
arithmetic reasoning, or a causal contribution from local gradient descent.

## Generation 88

| Policy | Success on the fixed 200 assessment tasks |
|---|---:|
| Evaluated population mean | 16.8706% |
| Uniform among nine moves | 15.9036% |
| Uniform among eight nonzero moves | 17.4351% |
| Always +5 | 17.0000% |
| Always −5 | 18.5000% |

The last four policies are goal-independent. The directional policies were examined
after observing the run, so these are diagnostic comparisons, not preregistered
hypothesis tests. They show that the small advantage over the displayed baseline
can be achieved by generic movement choices. They do not show that the animal
actually uses any one of these policies.

The fixed list contains 83 distinct ordered pairs among 200 trials, drawn from
90 possible ordered pairs of ten held-out positions. Exact uniform-nine success
over all 90 pairs is 15.9874%, close to the fixed-list expectation. Assessment goals
occur 16–23 times each; 49.5% of tasks require positive displacement.
[Task list, config, and exact baseline values](results/movement-baselines-generation-88.json).

## Goal-input control near generation 99

The audit read an atomic checkpoint containing the next inherited population and
randomly sampled 128 of its 1,024 members, using sample seed 20261101. It replayed
the next generation's lifetime training, copied each animal's learned mappings,
and froze them for three assessment conditions:

1. Correct goal input.
2. A cyclic permutation of the ten held-out goal embedding rows.
3. Held-out goal embedding rows replaced by zero logits, removing that input's
   contribution to entry-pair sampling.

Physical start positions and actual targets were unchanged. Sampling seeds were
matched across conditions. The correct-input frozen path was checked against the
original accelerated assessment. Evaluation covered both the fixed 200 tasks and
all 90 held-out ordered pairs repeated ten times per sampled animal.

| Frozen condition | Fixed-list population sample | All held-out pairs, repeated |
|---|---:|---:|
| Correct goal | 16.9219% | 17.0668% |
| Permuted goal input | 17.1406% | 17.1536% |
| Goal input removed | 16.9531% | 17.1042% |

This sample provides no positive evidence that the held-out advantage depends on
the correct goal input. It does not prove that every individual ignores goals,
that goals are unused on trained values, or that all later populations will behave
similarly. This was an exploratory single-checkpoint control, not a multi-seed
learning ablation. [Original audit output](results/goal-input-audit-generation-99.json).

The original audit was run with a temporary script; its output is retained here.
The exact checkpoint inode was not retained or fingerprinted at that time, so this
snapshot alone cannot reproduce every reported rollout. The reusable command now
records the checkpoint SHA-256, configuration, partitions, sample indices, and
sampling parameters to make subsequent audits more reproducible.

## Attribution concerns that remain open

- The movement distribution can improve through evolution without task-reward
  gradient descent. Local empirical gradients optimize prediction cross-entropy,
  not movement reward; shared softmax normalization can still affect action rates.
- Goal numbers use independent categorical embedding rows. Held-out goal values
  are never supplied as training goals, and local compatibility updates modify
  observation rows, not goal rows. Generalization to those goals needs its own test.
- A held-out position can be visited during a training episode. Holding out starts
  and goals is not equivalent to holding out all observations or transitions.
- One fixed validation list and one evolving lineage are not independent replicated
  training experiments. Related animals and repeated task pairs must not be counted
  as 204,800 independent demonstrations of broad generalization.
- Inspecting held-out curves and changing design choices in response creates a
  human-in-the-loop validation process. Final claims need fresh test tasks/partitions
  and independent seeds after design choices have been fixed.
- Hard walls favor some targets mechanically. Uniform target exposure does not
  equalize task difficulty; per-goal/per-distance controls remain useful.

A matched lifetime-learning on/off evolution study is needed to isolate its causal
contribution. A same-genotype counterfactual with learning disabled would answer a
narrower question about immediate lifetime benefit, not replace that full study.
Neither study was completed by this audit.

## Reproduce the audit procedure

```sh
.venv/bin/python -m static_agent_training.bias_audit runs/my-run \
  --sample 128 --sample-seed 20261101 --repeats 10 \
  --output runs/my-audit.json
```

A run directory reads its latest atomic checkpoint. A retained checkpoint file can
also be supplied for a repeatable fixed snapshot. Moving live checkpoints produce
new populations and therefore need not reproduce the historical table above.
The audit only changes copies and does not disable learning in the running trainer.
