# Faster checkpoint saves

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](../ARCHITECTURE.md), reviewed 2026-11-22.

Measured 2026-11-22 on the local i7-14700KF. This updates the earlier
[training stage profile](TRAINING_OPTIMIZATION_REPORT.md) with a focused save
benchmark; it is not an end-to-end training speedup measurement.

## Implemented changes

- Checkpoint gzip compression uses level 1 instead of level 9.
- Stack-safe JSON encoding batches tokens into approximately 64 KiB writes.
  It still streams agents individually through the existing population writer.

The JSON schema, decompressed content, numerical precision, save frequency,
file/directory syncing and atomic replacement are preserved. Historical
checkpoint compression remains readable. An already-running training process
loads these changes only when it next starts or resumes.

## Measurement

Source: `runs/ui-20261121-131148-session-b43354/checkpoint.json.gz`.
The benchmark read the saved population and config once, then performed three
writes per variant to temporary directories using the normal atomic writer.
The baseline restored the previous token-at-a-time JSON writes and gzip level 9.
The comparison includes file/directory sync and excludes reading the source and
validating the outputs. Population conversion was excluded equally from all
variants. The synthetic checkpoint contains the complete real saved population,
config, schema version and generation; historical training records are omitted.

| Save variant | Median seconds | Compressed bytes |
| --- | ---: | ---: |
| Previous token writes, gzip 9 | 1.5002 | 3,555,923 |
| Buffered token writes, gzip 9 | 1.4615 | 3,555,923 |
| Buffered token writes, gzip 1 | 0.8444 | 4,008,379 |

This is 43.7% less save time, with 12.7% larger compressed checkpoints, on this
sample. Most of the measured gain comes from compression level. All nine outputs
had identical decompressed SHA-256:
`91559226a547e1f3f058c0f07456dd321aaee26839643fbd69f75bf11009a7f2`.

A separate 8 MiB decompressed prefix from the newer
`runs/ui-20261122-113151-E3-065d76/checkpoint.json.gz` took 0.1565 s at level 9,
0.0688 s at level 6 and 0.0225 s at level 1 for compression alone. Its level-1
output was 56.8% larger. Compression ratios depend on population contents; the
13% increase above should not be assumed for every run. This prefix experiment
excluded JSON encoding, agent conversion and disk sync.

## Next targets

Profile a current generation before selecting the next optimization. Earlier
measurements identified serial reproduction/development and redundant genome
conversion as candidates. Parallel offspring development must retain seeded
streams, parent selection, child ordering and resumed results. Cached genome
representations must be immutable and invalidated by mutation.
