# Streaming development provenance and candidate compression

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](../ARCHITECTURE.md), reviewed 2026-11-22.

Implemented 2026-11-22. Training, reproduction, candidate ranking, checkpoint
frequency and saved schemas are unchanged. Candidate exports still retain the
historical best performers, rather than only the latest generation.

## Changes

Species already hold immutable development provenance as JSON text. Checkpoint,
new candidate and champion-trace writes now reuse that text instead of building
a decoded tree and encoding every node again. An explicit frozen `EncodedJSON`
wrapper validates the entire document with the stack-safe reader before it can
be streamed. Validation is cached per immutable Species; replacing an agent
starts a fresh cache. No decoded provenance tree is retained. Normal `to_data()`
still returns ordinary Python data; its optional encoded mode is only for
streaming `serialization.dump()`, not general JSON APIs.

The overwritten `best-population.json.gz` now uses gzip level 1, like checkpoints.
Other archive filenames retain level 9. Temporary-file writes, gzip close before
fsync, atomic replacement and directory fsync are unchanged.

Reusing the original JSON whitespace changes encoded bytes, but preserves decoded
contents. Exact compressed-byte equality is not the compatibility criterion.

## Measurement

An isolated write benchmark used a previously copied real 32-agent checkpoint
from the offspring benchmark (10,785,311 compressed bytes). It did not write to
the live run. The baseline disabled encoded provenance while keeping the existing
streamed population writer and checkpoint gzip level 1. Validation is included
in the cold optimized timing. The warm timing reuses the same immutable agents.

| Operation | Seconds | Saved bytes |
| --- | ---: | ---: |
| Previous checkpoint writer | 9.517 | 15,412,782 |
| Streamed provenance, first save | 1.858 | 15,745,767 |
| Streamed provenance, cached validation | 1.360 | 15,745,767 |
| Ten-agent candidate payload, gzip 9 | 3.356 | 3,031,603 |
| Same candidate payload, gzip 1 | 2.851 | 4,437,915 |

The first checkpoint save took 80.5% less time with a 2.2% size increase.
Candidate compression alone took 15.0% less time with a 46.4% size increase.
The candidate measurement deliberately used ordinary decoded data in both cases
to isolate compression from provenance streaming. All three checkpoint decoded
canonical SHA-256 hashes match; candidate decompressed byte hashes match.
See [raw measurements](SAFE_SAVE_SAMPLE_20261122.json).

A second pass measured 9.523 s baseline, 1.916 s cold, and 1.377 s warm checkpoint
saves; candidate compression measured 3.371 s versus 2.861 s. All equality checks
and file sizes matched the first pass.

These are two sequential passes under concurrent system load, on a historical
snapshot rather than the larger E3 snapshot from the preceding generation
benchmark. They are not a measured end-to-end generation speedup or a peak-memory
benchmark. Existing processes keep their imported code until restarted; the live
trainer was not restarted for this change.

## Verification

The full suite passed 327 tests after the production changes. Two additional
regressions were then added and passed in the 19-test serialization/training
subset. Coverage includes deeply nested terms, malformed-fragment rejection,
fresh validation after agent replacement, encoding/sync/replace failures leaving
the previous checkpoint intact, and matching old/new generation records,
candidate members and champion traces through the same save/resume boundary.
Existing uninterrupted-versus-resumed training checks also pass.

No checkpoint deduplication, asynchronous saving, reduced checkpoint frequency,
or evaluator changes were introduced.
