# Workspace polling and memory

Observed 2026-11-22 while Agent Y was training. The separate viewer process
reported 1,667,912 KiB resident memory and a lifetime-average 77.9% CPU in ps.
The machine had about 12 GiB available and no swap use. This implicated viewer
work rather than system memory pressure; browser CPU was not measured.

The frontend polls every three seconds. Each population-list request previously
reloaded and validated every saved population, including multiple 26–28 MiB
compressed archives, to return only 2,825 bytes of display metadata. One request
took 18.673 seconds while training continued.

Population listings now cache only small summary rows, keyed by absolute filename,
device, inode, modification/change timestamps and size, with at most 256 entries.
Concurrent cold requests share a lock. Every poll still scans current filenames,
so new/deleted populations are reflected; replacements invalidate the row.
Returned rows are independent copies. Actual import/evaluation still reads and
validates the complete saved population. A cold cache still pays the initial read;
no raw genomes or histories are held by the summary cache.

Sensory champion dropdown discovery now derives apparatus metadata from saved run
configuration instead of loading and constructing all champion models. Standalone
snapshot discovery retains its existing model-loading path. The full candidate
cache was reduced from 16 to two entries and evicts before loading another model.
Two very large candidates can still consume considerable memory; the limit is
an entry count, not a byte budget.

After viewer-only restart, with the same trainer PID continuing:

| Request | Seconds |
| --- | ---: |
| Run list | 0.551 |
| First population request measured after restart | 7.018 |
| Next population request | 0.00114 |
| Next population request | 0.00099 |
| Historical E3 champion options | 0.00169 |

The first measured request may overlap another browser request filling the cache.
These are observations under concurrent training, not isolated benchmarks. Viewer
RSS was about 71 MiB after warm-up. A subsequent 15-second process CPU-time
sample measured 0.0113 CPU cores (about 1.1% of one core), with RSS changing
from 69.1 to 68.1 MiB. This short sample is not directly comparable to the old
process lifetime-average CPU figure. Tests cover summary-cache invalidation after
replacement/addition/deletion, caller isolation, concurrent cold access, and
champion dropdown discovery without full-model loads. Candidate selection and
playground tests also pass. Training algorithms and artifacts were unchanged.
