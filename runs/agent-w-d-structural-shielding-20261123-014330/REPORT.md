# Structural shielding in Agent W and Agent D

## Summary

**Evidence against the operationalized shielding prediction under the tested learning rules.** W retained more original number-line performance than D after learning on Lights Out. However, genuinely rewired W controls with comparable starting competence retained more than W in the structural-only condition. W also failed to improve on a physically equivalent number line with a new sensory code. The result does not rule out structure-dependent shielding under a different learning rule or model.

## Research and hypothesis

The repository's structural-constraints manuscript distinguishes coherent successor structure from the learned representation and proposes that structural relevance may restrict conflicting parameter updates. The Agent X proposal likewise cautions that shared embedding weights can produce interference. We operationalized an explanation as the combination of a fixed successor graph and the sensory and action distributions attached to its states. We tested whether incompatible feedback preserved original task behavior while the same learner could adapt its mapping to a structurally equivalent environment. Lights Out success was not required for evidence of protection.

Repository context: [structural constraints and continual learning manuscript](../../papers/structural-constraints-continual-learning/main.tex), [Agent X foliation hypothesis](../../agent_x_experimental.tex), and [prior W/D lifetime ablation](../agent-w-d-learning-ablation-20261123-010347/REPORT.md).

## Methods and validation

- W and D used the same original 19 saved candidates per architecture. The source cohorts came from W generation 472 and D generation 932. Candidate identities and all 38 original and Lights Out frozen baselines reproduced exactly. Source inputs were verified by SHA-256; prior training and candidate artifacts stayed unchanged. The original training processes remained paused.
- Each original and rewired candidate started fresh in each of 12 environment–update combinations. The full matrix executed 684 independent trajectories, each with 30,000 adaptation episodes, and saved 6,156 checkpoint records. Graphs, action mappings, and cohort membership stayed fixed during adaptation. No evolution or mutation ran. All evaluations used rates zero. Enabled action and structural rates were each 0.05.
- Adaptation used the original number line, 2×2 Lights Out with its existing mapping, and the remapped number line described below. The two number-line arms used identical physical tasks and RNG seeds. All arms used the 12-action cap and 64-round cap.
- Each checkpoint evaluated original competence and competence in the adaptation environment on independent frozen copies. Original and remapped probes used 20,000 episodes and Lights Out used 30,000. Both number-line conditions shared the same 400 physical start–goal pairs. Lights Out covered all 15 nonzero boards. Within each environment, adaptation and evaluation share task support and have separately ordered or seeded episodes.
- The evaluator saves and restores exact RNG state between adaptation chunks. A frozen evaluation consumes no adaptation RNG draws. Split and uninterrupted smoke trajectories matched exactly. Disabled weight families stayed unchanged, and all 38 previous combined-learning Lights Out endpoints matched exactly.
- Results use candidate-paired contrasts and 4,000 candidate-level bootstrap resamples. Candidate intervals describe uncertainty within these selected, genealogically correlated cohorts. There were no independent evolutionary replications.

### Physically equivalent remap

The adapter rotates every raw eight-bit sensory byte left by two bits before initial activation and before feedback reaches the learner. The model receives the rotated byte directly. World states, action signals, movement, clamping, goal checks, and timing remain identical. The map is bijective and preserves each of the two interleaved sensory regions. Permuting input weights within their regions can represent the exact old mapping without increasing capacity.

For validation only, we made a conjugate copy whose input weights undo the feature permutation. Using all 1,681 start–goal combinations, it produced the same sequence of actions and fixed successors as the original on paired seeds. Its feedback bytes matched the rotated original bytes and its compatibility matched to numerical tolerance. Actual adaptation started from untouched weights and did not get this compensation.

### Genuine rewiring and baseline matching

For each W candidate we shuffled the multiset of successor targets across source nodes, preserving each node's indegree and outdegree, terminal set, state IDs, and all initial weight rows. We rejected graphs isomorphic to the original even under arbitrary relabeling, using the canonical rooted-tree and directed-cycle decomposition of a functional graph. Thus these controls changed unlabeled topology, beyond permuting labels on an unchanged graph. They changed 7–23 edges per candidate, a median of 13 and a mean of 88.8% of edges.

We evaluated 128 eligible rewirings per candidate on a separate 2,000-episode calibration set and picked the closest to the source W score. The confirmatory 20,000-episode original probe was not used to choose the graph. On that probe, **18 of 19** rewired candidates were within **3 percentage points** of their source W candidate. The one outlier was +3.69 points and is omitted from the matched rewiring contrasts. The full-cohort W baseline was 40.62% and rewired W 40.48%, a difference of −0.14 points with a paired interval of −1.06 to +0.78. We also report all 19 trajectories in each cohort.

| Initial frozen probe | W | D | Rewired W |
|---|---:|---:|---:|
| Original number line | 40.62% | 70.83% | 40.48% |
| Remapped number line | 22.44% | 27.83% | 28.44% |
| Lights Out | 22.89% | 6.88% | 22.17% |

D is not baseline-matched to W. Rewired W is matched on the original task and parameter budget, but starts higher on the remapped line. The report uses within-candidate gains and treats differences in remapped starting competence as a limitation.

## Frozen performance at 30,000 episodes

Values are cohort means. Changes are paired against the same candidate before adaptation. Brackets give 95% candidate-bootstrap intervals in percentage points. The adaptation-world frequency baseline uses each candidate's observed physical-action frequencies on that frozen probe.

| Agent | Adaptation | Updates | Original success | Original change (pp) | Adaptation-world success | Adaptation-world change (pp) | Frequency baseline |
|---|---|---|---:|---:|---:|---:|---:|
| W | Original line | Frozen | 40.62% | +0.00 [+0.00, +0.00] | 40.62% | +0.00 [+0.00, +0.00] | 20.31% |
| W | Original line | Action only | 19.87% | -20.75 [-24.00, -17.63] | 19.87% | -20.75 [-24.00, -17.63] | 17.31% |
| W | Original line | Structural only | 16.83% | -23.79 [-26.01, -21.82] | 16.83% | -23.79 [-26.01, -21.82] | 18.88% |
| W | Original line | Both | 11.57% | -29.05 [-31.72, -26.47] | 11.57% | -29.05 [-31.72, -26.47] | 18.23% |
| W | Lights Out | Frozen | 40.62% | +0.00 [+0.00, +0.00] | 22.89% | +0.00 [+0.00, +0.00] | 27.36% |
| W | Lights Out | Action only | 40.64% | +0.02 [-0.87, +0.85] | 17.76% | -5.13 [-6.72, -3.76] | 22.49% |
| W | Lights Out | Structural only | 22.60% | -18.02 [-19.38, -16.38] | 23.47% | +0.58 [-0.98, +1.76] | 24.42% |
| W | Lights Out | Both | 21.50% | -19.12 [-20.39, -17.50] | 16.77% | -6.11 [-7.66, -4.90] | 19.18% |
| W | Remapped line | Frozen | 40.62% | +0.00 [+0.00, +0.00] | 22.44% | +0.00 [+0.00, +0.00] | 19.27% |
| W | Remapped line | Action only | 39.38% | -1.24 [-3.42, +0.42] | 22.26% | -0.18 [-1.17, +1.04] | 18.64% |
| W | Remapped line | Structural only | 21.58% | -19.04 [-20.81, -17.17] | 12.62% | -9.82 [-12.44, -7.22] | 19.28% |
| W | Remapped line | Both | 23.66% | -16.96 [-19.28, -14.63] | 6.45% | -15.99 [-18.45, -13.19] | 20.15% |
| D | Original line | Frozen | 70.83% | +0.00 [+0.00, +0.00] | 70.83% | +0.00 [+0.00, +0.00] | 15.66% |
| D | Original line | Action only | 64.73% | -6.10 [-9.65, -2.97] | 64.73% | -6.10 [-9.65, -2.97] | 16.93% |
| D | Original line | Structural only | 17.00% | -53.82 [-53.88, -53.76] | 17.00% | -53.82 [-53.88, -53.76] | 17.00% |
| D | Original line | Both | 16.58% | -54.25 [-54.90, -53.79] | 16.58% | -54.25 [-54.90, -53.79] | 16.60% |
| D | Lights Out | Frozen | 70.83% | +0.00 [+0.00, +0.00] | 6.88% | +0.00 [+0.00, +0.00] | 7.14% |
| D | Lights Out | Action only | 71.43% | +0.60 [+0.15, +1.01] | 10.85% | +3.97 [+3.68, +4.25] | 13.64% |
| D | Lights Out | Structural only | 18.52% | -52.31 [-53.01, -51.24] | 6.74% | -0.14 [-0.17, -0.12] | 6.74% |
| D | Lights Out | Both | 17.86% | -52.96 [-53.24, -52.58] | 6.69% | -0.19 [-0.22, -0.17] | 6.70% |
| D | Remapped line | Frozen | 70.83% | +0.00 [+0.00, +0.00] | 27.83% | +0.00 [+0.00, +0.00] | 13.60% |
| D | Remapped line | Action only | 70.78% | -0.05 [-0.11, +0.00] | 27.86% | +0.03 [-0.03, +0.09] | 13.60% |
| D | Remapped line | Structural only | 21.86% | -48.97 [-50.98, -47.12] | 3.87% | -23.96 [-25.32, -22.29] | 13.57% |
| D | Remapped line | Both | 21.72% | -49.10 [-51.01, -47.20] | 4.71% | -23.12 [-25.16, -20.61] | 13.65% |
| Rewired W | Original line | Frozen | 40.48% | +0.00 [+0.00, +0.00] | 40.48% | +0.00 [+0.00, +0.00] | 19.55% |
| Rewired W | Original line | Action only | 26.56% | -13.92 [-18.45, -9.23] | 26.56% | -13.92 [-18.45, -9.23] | 19.20% |
| Rewired W | Original line | Structural only | 22.10% | -18.37 [-20.37, -16.91] | 22.10% | -18.37 [-20.37, -16.91] | 22.67% |
| Rewired W | Original line | Both | 19.23% | -21.25 [-24.12, -18.69] | 19.23% | -21.25 [-24.12, -18.69] | 20.66% |
| Rewired W | Lights Out | Frozen | 40.48% | +0.00 [+0.00, +0.00] | 22.17% | +0.00 [+0.00, +0.00] | 23.65% |
| Rewired W | Lights Out | Action only | 41.16% | +0.68 [-1.10, +1.86] | 14.42% | -7.75 [-9.78, -5.66] | 16.67% |
| Rewired W | Lights Out | Structural only | 30.00% | -10.48 [-12.44, -8.48] | 18.23% | -3.94 [-7.54, -0.85] | 18.11% |
| Rewired W | Lights Out | Both | 24.96% | -15.52 [-18.96, -12.24] | 10.17% | -12.00 [-15.87, -8.73] | 10.25% |
| Rewired W | Remapped line | Frozen | 40.48% | +0.00 [+0.00, +0.00] | 28.44% | +0.00 [+0.00, +0.00] | 18.59% |
| Rewired W | Remapped line | Action only | 36.55% | -3.92 [-8.45, +0.21] | 26.76% | -1.68 [-4.77, +1.19] | 15.52% |
| Rewired W | Remapped line | Structural only | 21.05% | -19.43 [-20.57, -18.40] | 12.02% | -16.42 [-19.11, -13.69] | 18.18% |
| Rewired W | Remapped line | Both | 23.86% | -16.61 [-18.76, -14.54] | 10.75% | -17.70 [-21.33, -13.94] | 20.24% |

## Selectivity and baseline-matched comparison

Selective retention is original score change after Lights Out minus original score change after the reference adaptation condition. A positive value means less forgetting after incompatible adaptation. It does not indicate learning of the remap. The last column counts candidates that both showed positive selectivity and improved over their own frozen remapped-line baseline after remapped adaptation.

| Agent | Update | Reference adaptation | Selective retention (pp) | Positive selectivity and remap gain |
|---|---|---|---:|---:|
| W | Action only | Original line | +20.77 [+17.53, +24.06] | 4/19 |
| W | Action only | Remapped line | +1.26 [-0.69, +3.56] | 1/19 |
| D | Action only | Original line | +6.70 [+3.57, +10.33] | 10/19 |
| D | Action only | Remapped line | +0.65 [+0.18, +1.09] | 8/19 |
| Rewired W | Action only | Original line | +14.60 [+9.94, +19.11] | 7/19 |
| Rewired W | Action only | Remapped line | +4.60 [+0.33, +9.34] | 3/19 |
| W | Structural only | Original line | +5.77 [+3.30, +8.43] | 1/19 |
| W | Structural only | Remapped line | +1.02 [-0.77, +2.63] | 1/19 |
| D | Structural only | Original line | +1.52 [+0.81, +2.59] | 0/19 |
| D | Structural only | Remapped line | -3.34 [-5.50, -1.19] | 0/19 |
| Rewired W | Structural only | Original line | +7.89 [+5.24, +10.54] | 0/19 |
| Rewired W | Structural only | Remapped line | +8.95 [+6.34, +11.49] | 0/19 |
| W | Both | Original line | +9.92 [+7.33, +12.43] | 1/19 |
| W | Both | Remapped line | -2.16 [-4.91, +0.70] | 1/19 |
| D | Both | Original line | +1.28 [+0.71, +1.95] | 0/19 |
| D | Both | Remapped line | -3.86 [-5.90, -1.76] | 0/19 |
| Rewired W | Both | Original line | +5.73 [+1.70, +9.61] | 0/19 |
| Rewired W | Both | Remapped line | +1.10 [-3.22, +5.53] | 0/19 |

For the 18 baseline-matched W–rewired pairs, the following are **W minus rewired W**. A positive retention difference favors W. A positive selectivity difference means W showed a stronger retention advantage for Lights Out over the remapped environment.

| Update | Difference in Lights Out retention (pp) | Difference in selectivity (pp) | Difference in remapped gain (pp) |
|---|---:|---:|---:|
| Action only | -0.67 [-2.34, +1.52] | -3.56 [-8.55, +1.16] | +1.70 [-1.75, +5.37] |
| Structural only | -7.70 [-9.64, -5.96] | -7.99 [-10.36, -5.58] | +5.75 [+2.35, +9.08] |
| Both | -2.87 [-6.53, +1.04] | -1.68 [-7.07, +3.43] | +1.32 [-3.60, +6.32] |

## Representational change and alternative explanations

- **Different learning sensitivity and baseline levels.** W lost 18.02 points after Lights Out structural learning and D lost 52.31, retaining 55.6% and 26.2% of their own baselines respectively. But W also lost much less than D on the remapped number line and the original number line. The W–D difference therefore also appears outside incompatible adaptation. D began at 70.83% on the original task versus W's 40.62%; normalization helps describe the losses but does not match candidate competence or evolutionary history.
- **Structural learning collapse.** After structural-only Lights Out adaptation, D's original task score fell to 18.52% while puzzle successor compatibility rose from 27.91% to 99.99%. Its sensory information on the common probe set fell by 0.67 nats. In W's remapped structural-only condition, remapped success fell from 22.44% to 12.62% despite 5.49 nats of embedding KL drift and changes to every embedding parameter. That domain was physically and representationally compatible. The update altered the mapping extensively without learning a useful one.
- **Parameter capacity and resistance to updates.** W and its rewired control had identical initial weights and budgets. Under structural-only Lights Out adaptation, both changed exactly 55.6% of embedding parameters on average, and their embedding KL values were close, 1.77 and 1.74 nats. Yet rewired W retained more. Under structural-only remapped adaptation both changed every embedding parameter. This weakens a simple explanation based on one group being generally resistant to updates.
- **Action frequencies.** For W on the remapped line, the combined arm's independent-action baseline rose 0.87 points while actual success dropped 15.99 points. The pooled action mix cannot explain that collapse. D's action-only Lights Out gain of 3.97 points accompanied a 6.50-point improvement in its frequency control. W's combined Lights Out loss of 6.11 points accompanied an 8.18-point drop in its frequency control. The independent controls stop on success or after 12 actions and can continue when actual chains terminate early. Actual minus control also reflects state-conditioned selection and termination, beyond action order.
- **Graph topology and selection.** Rewired W held per-node degree and initial parameter budget fixed and was selected for original competence. Cycles and branching organization can still differ by design, and the selected rewires are not an unselected random-graph population. The test cannot separate lambda derivation from all higher-order topology effects or evolutionary-history differences. The independent-seed replication remains unrun.

## Evidence supporting structural shielding

W retained more original skill than D after incompatible adaptation. It also retained more after Lights Out than after continued same-world adaptation in all three learning-on conditions. These are observations consistent with protection under some comparisons. They do not show the combination of **selective** protection and effective learning of a compatible remap. The same-world advantage also appeared in rewired W.

## Evidence against structural shielding

W did not improve on the compatible remapped line at the endpoint under any enabled learning configuration. Structural-only and combined learning degraded it by **9.82 and 15.99 points**. W showed only **+1.02 points** of incompatible-versus-compatible selectivity with structural-only learning, with its interval crossing zero, and **−2.16** with both updates. The structural-only rewired control showed **+8.95 points** of selectivity. In the 18 baseline-matched pairs, W's selectivity was **7.99 points lower** than rewired W's, with a paired interval of **−10.36 to −5.58 points**. W also retained less original competence than rewired W after structural-only Lights Out adaptation.

## Reproducibility and files

- [Config, task lists, seeds, and source hashes](config.json.gz)
- [Rewired controls and graph characteristics](shuffled-controls.json.gz) and [baseline matching](baseline-matching.json)
- [Candidate-level measurements](candidates.csv) and [paired summaries and confidence intervals](summary.json)
- [Validation](validation.json) and [execution log](experiment.log)
- The checkpoints directory contains fitted weights, exact native RNG state, per-start/goal counts, frozen evaluations, interval traces summarized as diagnostics, and compatibility and action-frequency records.
- [Original retention](plots/01-original-retention.png), [adaptation-world curves](plots/02-adaptation-environment.png), [embedding KL](plots/03-embedding-kl.png), [sensory information](plots/04-sensory-information.png), [successor compatibility](plots/05-successor-compatibility.png), [frequency control](plots/06-frequency-control.png), [action KL](plots/07-action-kl.png), [selective retention](plots/08-selective-retention.png), and [baseline matching](plots/09-baseline-matching.png).

The experiment and report commands are:

    python -m static_agent_training.experimental.wd_structural_shielding <output-directory> --workers 12
    python -m static_agent_training.experimental.wd_structural_shielding_report <output-directory>

The native library compiles with g++. NumPy is required for the independent-action calculation; the report also uses Matplotlib. Restart the same experiment command with the same directory to resume from the last atomic checkpoint of each unfinished trajectory.

## Does This Demonstrate Structural Shielding?

**Classification: Evidence against the hypothesis as operationalized here.** Under these learning rules, W did not demonstrate successful remapping combined with selective protection against incompatible feedback. The degree-preserving, genuinely rewired control preserved more original competence after structural-only Lights Out adaptation. The comparison with D alone would have suggested an advantage, but that comparison mixes baseline competence, topology, and training history. Low puzzle success was not counted as evidence against shielding. The decisive observations were the compatible remap failure and the direction of the matched structural control.
