# Agent Q implementation

[Mathematical specification](AGENT_Q_SPEC.md)

## Train and assess

Run from the project directory. The following commands show the entry points. Choose your own training budget for an actual run.

    ./project agent-q --help
    ./project agent-q --workspace-directory runs/agent-q-my-run --generations 50 --population 32 --episodes 500 --learning both --exploration 0.05
    ./project agent-q --resume runs/agent-q-my-run --generations 100
    ./project agent-q --candidate runs/agent-q-my-run/result.json.gz --learning off
    ./project agent-q --initial-population runs/populations/my-q-candidates.json.gz --population 32

Native CPU is the default. Start with --backend python to inspect the exact 256-byte reference evaluation. The workspace and resumed command use the native CPU. The native path requires g++ and does not call Python during a lifetime.

Training reuses the Agent Z evolution loop, mutation operators, development budgets, and optional structural speciation. Inherited embedding weights and action-conditioned expectation logits mutate by the configured weight noise. Surviving pair IDs carry their rows into offspring. Newly developed pairs receive newly initialized rows. Lifetime learning acts on per-evaluation copies. Final artifacts and checkpoints record inherited candidate parameters.

## Workspace

- In Setup, choose **Agent Q**, select the preset, region count, learning mode, exploration probability, and optional saved Q candidates.
- In Sessions, watch generation summaries and logs. Stop retains the last complete checkpoint. Resume continues from it. A completed run can be extended by increasing the generation target.
- Use **Save candidates** to export a Q population. Choose that set in Setup to seed another run, or in Assessments to run frozen number-line evaluations.
- Open a Q champion in Playground. Run a frozen attempt or turn on lifetime learning and select empirical, structural, or both. Learned parameters persist only in the playground lifetime until reset. Playback shows sampled actions, expected bytes and actual observations. Brain Scan displays the derived action probabilities for region 0 and all fixed successor links.

Q currently supports the original −20 to +20 number line in training, assessments, and Playground. File-stream previews and alternate worlds are outside the current Q adapter. The explorer receives a default 5% uniform action mixture. Terminal anchors end their chain without a final emission.

## Artifacts and verification

Champion artifacts use crt-agent-q-function-first-v1. Candidate populations, session metadata, and checkpoints have distinct Agent Q schema identifiers. A Q model stores lambda pairs, fixed successors, embedding weights, per-anchor per-action expectation logits, region count, and exploration. It stores no independent action-policy weights. Only Q populations with saved development seeds can initialize new evolutionary sessions.

Run the focused tests with:

    .venv/bin/python -m unittest discover -s tests -p test_agent_q.py

The tests check exact compatibility on a hand-computable example, exploration normalization, sampled versus observed targets, same-round aggregation, all learning modes, deterministic Python/native lifetime parity within numerical tolerance, and session lifecycle and artifact flows. These tests do not make claims about navigation performance.
