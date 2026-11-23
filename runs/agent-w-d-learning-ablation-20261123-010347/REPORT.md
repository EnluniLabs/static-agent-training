# Agent W/D lifetime-learning ablation and forgetting

## Summary

Structural learning alone reproduced most of the number-line forgetting after Lights Out adaptation. W lost 18.02 percentage points and D lost 52.31. Action-only adaptation on Lights Out left W's number-line score essentially unchanged and improved D's by 0.60 points. Both update mechanisms could harm number-line performance when adaptation stayed on the original number line.

D reached near-perfect successor compatibility during frozen probes while goal success collapsed. Internal agreement can improve substantially while a useful policy is erased. W retained more number-line skill with structural learning enabled, but the cohorts came from different evolutionary checkpoints and differ in more than their graph-generation rule.

## Methods and validation

The study used the original 19 W candidates and 19 D candidates, selected in the earlier transfer assessment from checkpoints 472 and 932 respectively. Each candidate was copied into four learning conditions in each of two environments. Zero disables the corresponding update. The enabled rate is 0.05. The 304 trajectories each received 30,000 adaptation episodes. There was no evolutionary selection, mutation, remapping, or successor-graph change.

Lights Out used the original adapter mapping and 2,000 repeats of each of the 15 nonzero boards. Number-line adaptation used the original −20 to +20 world with 75 repeats of each of the 400 held-out start–goal pairs and a separately shuffled ordering. The frozen number-line probe used 50 repeats of each pair, so number-line adaptation and its probe share task support. Puzzle adaptation and the puzzle probe likewise share board support. The experiment measures same-distribution retention and puzzle performance with separate replay RNG seeds. It does not measure generalization to novel puzzles. The action cap was 12 and the round cap was 64.

At episodes 0, 100, 500, 1,000, 3,000, 5,000, 10,000, 20,000, and 30,000, the probes used the same matched seeds and 20,000 number-line or 30,000 puzzle episodes. Both probes start from copies of the checkpoint weights. Their RNGs do not consume adaptation draws. Native adaptation resumes from exact RNG state. Zero-rate probes at different checkpoints are identical and were reused. There are 2,736 candidate-condition checkpoint records.

Frozen baseline successes reproduced exactly for all 38 candidates. Split and uninterrupted native smoke calls gave identical final weights and RNG states. Disabled weight families remained unchanged throughout. All 38 combined puzzle-learning endpoints reproduced the prior experiment's learned weights and outcomes exactly. Source artifacts were verified unchanged by SHA-256. Restarting with the existing results directory also completed successfully.

The intervals use 4,000 paired candidate-bootstrap resamples and describe variability within each selected cohort. Candidates are correlated descendants of a single evolutionary run, rather than independent evolutionary seeds. No architecture-level inference follows from these intervals.

## Frozen endpoint after 30,000 episodes

Changes are mean paired differences from the original frozen candidates. Brackets are 95% candidate-bootstrap intervals, in percentage points.

| Agent | Adaptation | Updates | Number-line success | Number-line change (pp) | Puzzle success | Puzzle change (pp) |
|---|---|---|---:|---:|---:|---:|
| W | Number line | Frozen control | 40.62% | +0.00 [+0.00, +0.00] | 22.89% | +0.00 [+0.00, +0.00] |
| W | Number line | Action only | 19.87% | -20.75 [-23.92, -17.65] | 17.01% | -5.88 [-8.23, -3.45] |
| W | Number line | Structural only | 16.83% | -23.79 [-26.17, -21.87] | 33.03% | +10.14 [+4.17, +15.41] |
| W | Number line | Both | 11.57% | -29.05 [-31.76, -26.45] | 21.71% | -1.18 [-4.67, +2.88] |
| W | Lights Out | Frozen control | 40.62% | +0.00 [+0.00, +0.00] | 22.89% | +0.00 [+0.00, +0.00] |
| W | Lights Out | Action only | 40.64% | +0.02 [-0.90, +0.87] | 17.76% | -5.13 [-6.71, -3.78] |
| W | Lights Out | Structural only | 22.60% | -18.02 [-19.39, -16.48] | 23.47% | +0.58 [-1.00, +1.76] |
| W | Lights Out | Both | 21.50% | -19.12 [-20.33, -17.56] | 16.77% | -6.11 [-7.66, -4.91] |
| D | Number line | Frozen control | 70.83% | +0.00 [+0.00, +0.00] | 6.88% | +0.00 [+0.00, +0.00] |
| D | Number line | Action only | 64.73% | -6.10 [-9.49, -3.02] | 12.69% | +5.80 [+2.50, +9.59] |
| D | Number line | Structural only | 17.00% | -53.82 [-53.88, -53.76] | 6.74% | -0.14 [-0.17, -0.12] |
| D | Number line | Both | 16.58% | -54.25 [-54.93, -53.79] | 7.12% | +0.24 [-0.10, +0.74] |
| D | Lights Out | Frozen control | 70.83% | +0.00 [+0.00, +0.00] | 6.88% | +0.00 [+0.00, +0.00] |
| D | Lights Out | Action only | 71.43% | +0.60 [+0.16, +1.00] | 10.85% | +3.97 [+3.68, +4.26] |
| D | Lights Out | Structural only | 18.52% | -52.31 [-53.00, -51.20] | 6.74% | -0.14 [-0.17, -0.12] |
| D | Lights Out | Both | 17.86% | -52.96 [-53.25, -52.56] | 6.69% | -0.19 [-0.22, -0.17] |

## Observations and interpretation

1. **Which update causes most of the forgetting after puzzle adaptation?** Structural-only learning was sufficient. W fell from 40.62% to 22.60% and D from 70.83% to 18.52%. Action-only puzzle adaptation ended at 40.64% for W and 71.43% for D. Both updates together ended at 21.50% for W and 17.86% for D.
2. **Does harm require transfer?** No. On the original number line, structural-only adaptation dropped W by 23.79 points and D by 53.82 points. After only 100 episodes their frozen scores were already 20.49% and 16.92%, respectively. Action-only adaptation also harmed same-world retention at 30,000 episodes, by 20.75 points for W and 6.10 for D.
3. **Does an isolated update improve puzzle performance?** After puzzle adaptation, D action-only improved from 6.88% to 10.85% (+3.97 points, interval +3.68 to +4.26). W structural-only reached 23.47%, a +0.58-point estimate with an interval spanning zero. It was higher earlier, reaching 28.71% after 100 episodes. W action-only and D structural-only ended lower than baseline. In the original-world adaptation arms, W structural-only reached 33.03% on the frozen puzzle (+10.14 points), and D action-only reached 12.69% (+5.80 points). These increases remained below the prior 45.34% uniform random puzzle benchmark.
4. **What happens when both updates are combined?** After puzzle adaptation, both updates together hurt number-line retention more than either isolated update. The extra loss relative to structural-only was 1.10 points for W and 0.65 for D. For D puzzle success, combined learning erased the action-only gain, with a −4.02-point interaction. In same-world adaptation, the combined endpoint was also the worst for number-line retention. Positive same-world interaction values mean the combined loss was smaller than the sum of the isolated losses. Combined learning still produced the lowest retention scores.
5. **Did compatibility improve while task performance fell?** Yes. In D, after puzzle structural-only adaptation, mean compatibility on frozen puzzle feedback rose from 27.91% to 99.99%, while puzzle success slipped from 6.88% to 6.74% and frozen number-line success fell to 18.52%. With structural-only adaptation on the original number line, D's compatibility there rose from 36.99% to 99.99% while goal success fell from 70.83% to 17.00%. W's combined puzzle arm raised puzzle compatibility from 46.66% to 51.83% while puzzle success fell from 22.89% to 16.77%.
6. **Do action frequencies explain puzzle changes?** They account for much of several changes. D's puzzle action-only success rose 3.97 points while the frequency-matched control rose 6.50. W's combined puzzle adaptation lost 6.11 points while the frequency-matched score lost 8.18. W's structural-only puzzle arm gained 0.58 points despite a 2.94-point drop in the frequency-matched control, so a single pooled distribution does not explain every result. These comparisons use action counts from the same 30,000-episode frozen probes. Earlier control estimates used 3,000-episode traces.
7. **Does W retain more once the updates are isolated?** W lost fewer number-line points with structural-only learning and with both updates, in either adaptation environment. With action-only learning, D retained more relative to its baseline in both environments. W's apparent advantage therefore applies to the conditions with embedding updates in this cohort.
8. **What does this imply about the objective?** The compatibility objective rewards assigning feedback to the fixed successor even when that changes which states are useful for starting a task. In D's same-world structural arm, mean probe embedding entropy fell from 0.112 nats to 0.001 nats while compatibility approached one and navigation collapsed. This is consistent with loss of sensory discrimination. The ablation identifies the embedding update as sufficient for failure. It does not isolate a particular latent feature or establish an advantage from lambda-derived graphs.

## Interaction by checkpoint

Interaction is both − action-only − structural-only + control. Positive values indicate a higher score than additive changes would predict. Negative values indicate a lower score. The contrast is descriptive and does not identify a mechanism. Entries are percentage points with paired bootstrap intervals.

| Agent | Adaptation | Episodes | Number-line interaction (pp) | Puzzle interaction (pp) |
|---|---|---:|---:|---:|
| W | Number line | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| W | Number line | 100 | -0.86 [-2.38, +0.53] | -0.12 [-1.67, +1.40] |
| W | Number line | 500 | +0.72 [-0.58, +2.09] | -1.39 [-4.70, +1.87] |
| W | Number line | 1,000 | +0.59 [-0.60, +1.79] | -5.27 [-9.09, -1.33] |
| W | Number line | 3,000 | +6.92 [+3.29, +10.93] | -11.49 [-17.35, -5.73] |
| W | Number line | 5,000 | +14.57 [+9.89, +19.46] | -7.58 [-15.17, +0.06] |
| W | Number line | 10,000 | +16.11 [+11.51, +20.61] | -2.85 [-9.56, +3.42] |
| W | Number line | 20,000 | +14.72 [+9.86, +20.15] | -2.90 [-9.93, +4.37] |
| W | Number line | 30,000 | +15.49 [+10.31, +20.71] | -5.44 [-11.86, +1.12] |
| W | Lights Out | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| W | Lights Out | 100 | +0.38 [-1.22, +1.74] | -0.08 [-1.29, +1.03] |
| W | Lights Out | 500 | +0.78 [-0.96, +2.94] | +0.35 [-2.16, +2.51] |
| W | Lights Out | 1,000 | +1.15 [-0.48, +2.89] | -1.53 [-4.02, +0.58] |
| W | Lights Out | 3,000 | +1.51 [-1.19, +4.21] | -2.49 [-5.51, +0.23] |
| W | Lights Out | 5,000 | +1.39 [-0.36, +3.00] | -3.26 [-5.89, -0.52] |
| W | Lights Out | 10,000 | +0.56 [-1.51, +2.89] | -1.32 [-3.60, +1.84] |
| W | Lights Out | 20,000 | -1.18 [-3.24, +1.08] | +0.77 [-2.25, +4.42] |
| W | Lights Out | 30,000 | -1.12 [-3.58, +1.60] | -1.57 [-3.83, +0.97] |
| D | Number line | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| D | Number line | 100 | +0.05 [-0.02, +0.11] | +0.00 [-0.02, +0.02] |
| D | Number line | 500 | +0.03 [-0.04, +0.09] | +0.02 [-0.01, +0.05] |
| D | Number line | 1,000 | +0.05 [-0.00, +0.11] | +0.02 [-0.01, +0.04] |
| D | Number line | 3,000 | +0.04 [-0.01, +0.09] | +0.04 [+0.01, +0.06] |
| D | Number line | 5,000 | +0.06 [+0.02, +0.11] | +0.03 [+0.01, +0.05] |
| D | Number line | 10,000 | +0.10 [+0.05, +0.15] | +0.15 [+0.10, +0.21] |
| D | Number line | 20,000 | +0.25 [+0.14, +0.37] | -0.09 [-0.16, -0.03] |
| D | Number line | 30,000 | +5.68 [+2.64, +8.99] | -5.42 [-9.16, -2.23] |
| D | Lights Out | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| D | Lights Out | 100 | +0.11 [+0.04, +0.18] | +0.01 [-0.01, +0.03] |
| D | Lights Out | 500 | +0.02 [-0.05, +0.10] | +0.01 [-0.02, +0.04] |
| D | Lights Out | 1,000 | -0.05 [-0.36, +0.15] | -0.01 [-0.04, +0.02] |
| D | Lights Out | 3,000 | +0.17 [-0.26, +0.87] | -0.35 [-0.76, -0.09] |
| D | Lights Out | 5,000 | -0.40 [-1.84, +1.31] | -2.01 [-2.89, -1.17] |
| D | Lights Out | 10,000 | -1.02 [-2.10, +0.35] | -4.00 [-4.45, -3.48] |
| D | Lights Out | 20,000 | -1.57 [-2.43, -0.82] | -4.22 [-4.55, -3.89] |
| D | Lights Out | 30,000 | -1.25 [-2.61, -0.21] | -4.02 [-4.31, -3.73] |

## Frequency-matched puzzle results at 30,000 episodes

Each control draws independently from that candidate's physical toggle frequencies on the frozen puzzle probe. The chance of reaching zero was computed exactly on the 16 boards, with success absorbing and 12 actions available. The independent policy can use remaining actions after the agent's chains would have stopped. Actual minus frequency therefore also reflects state dependence and termination.

| Agent | Adaptation | Updates | Actual | Frequency control | Actual minus control (pp) |
|---|---|---|---:|---:|---:|
| W | Number line | Frozen control | 22.89% | 27.36% | -4.48 |
| W | Number line | Action only | 17.01% | 18.83% | -1.83 |
| W | Number line | Structural only | 33.03% | 33.71% | -0.68 |
| W | Number line | Both | 21.71% | 23.77% | -2.06 |
| W | Lights Out | Frozen control | 22.89% | 27.36% | -4.48 |
| W | Lights Out | Action only | 17.76% | 22.49% | -4.73 |
| W | Lights Out | Structural only | 23.47% | 24.42% | -0.95 |
| W | Lights Out | Both | 16.77% | 19.18% | -2.41 |
| D | Number line | Frozen control | 6.88% | 7.14% | -0.26 |
| D | Number line | Action only | 12.69% | 13.74% | -1.05 |
| D | Number line | Structural only | 6.74% | 6.73% | +0.00 |
| D | Number line | Both | 7.12% | 7.11% | +0.01 |
| D | Lights Out | Frozen control | 6.88% | 7.14% | -0.26 |
| D | Lights Out | Action only | 10.85% | 13.64% | -2.79 |
| D | Lights Out | Structural only | 6.74% | 6.74% | +0.00 |
| D | Lights Out | Both | 6.69% | 6.70% | -0.01 |

## Files and figures

- [Configuration, tasks, candidate IDs, and source hashes](config.json.gz)
- [Per-candidate statistics and paired changes](candidates.csv)
- [Group means, medians, SDs, paired intervals, and interactions for every checkpoint](summary.json)
- [Validation](validation.json), [execution log](experiment.log), and [restart validation](resume-validation.log)
- The checkpoints directory contains each candidate's learned weights, adaptation RNG state, interval and frozen diagnostics, success by starting state, and action frequencies. Cumulative touched-state fractions reflect actual parameter changes. Compatibility is measured only where the successor exists. Action-update signs refer to the sign of the compatibility-weighted coefficient. KL is original-to-current in nats. Sensory KL uses fixed signed feedback bytes from −40 to +40, across both regions.
- [01 Number-line retention](plots/01-number-line-retention.png)
- [02 Lights Out success](plots/02-lights-out-success.png)
- [03 Action KL](plots/03-action-kl.png)
- [04 Embedding KL](plots/04-embedding-kl.png)
- [05 Number-line compatibility](plots/05a-number-line-compatibility.png) and [Lights Out compatibility](plots/05b-puzzle-compatibility.png)
- [06 Action entropy](plots/06-action-entropy.png)
- [07 Actual and frequency-matched puzzle scores](plots/07-puzzle-frequency-control.png)
- [08 Learning interaction](plots/08-interaction.png)

Restart the native experiment module static_agent_training.experimental.wd_learning_ablation with the same output directory to resume. Each atomic checkpoint contains the weights and RNG state needed to continue. The original training processes remained paused.
