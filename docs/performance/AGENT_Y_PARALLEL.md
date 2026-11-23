# Agent Y parallel lifetime evaluation

Agent Y Python evaluation now uses the configured workers to evaluate independent
agents in spawned processes. At most one outstanding evaluation per worker is
submitted. Original indices, task lists and derived seeds are retained. Results
are collected in population order; complete champion traces are reconstructed by
one deterministic champion replay. Lifetime inheritance returns stack-safe Species
packets. Existing Agent X evaluation paths are unchanged.

CLI sessions write a small progress.json and completed-agent log lines during a
generation. The workspace detail page displays that evaluation progress. Existing
running trainers retain their imported code; changes apply on launch/resume.

A read-only copied initial checkpoint from
runs/ui-20261122-142351-session-869d63 supplied the first four agents. Each was
measured on the first 20 training and 40 assessment tasks from its generation's
actual task list. Timings include worker startup/transport, not checkpoint saving,
reproduction or the extra champion replay. Serial used the same reduced-detail
worker evaluation path. Measurements were one pass under concurrent system load.

| Workers | Seconds |
| --- | ---: |
| 1 | 6.351 |
| 2 | 3.880 |
| 4 | 1.995 |

Four workers reduced evaluation time by 68.6% (3.18× throughput) for this slice.
Every result field matched exactly. This is not a full-generation speedup claim.
Workers use additional RAM and each inherits the process address-space limit;
that limit is not an aggregate memory cap. Raw timings: AGENT_Y_PARALLEL_SAMPLE.json.

The full suite passed 339 tests. The explicit parallel regression compares result
records, complete champion details and learned inherited Species with serial
execution and checks per-agent progress. Agent Y workspace tests exercise session
launch, saved champion inspection and resume.
