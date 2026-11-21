# Historical fixed retinal apparatus

This document describes the historical `retina` packed-pixel drawing interface,
retained explicitly for snapshot compatibility. New Agent X sessions use
[Receptor signals](RECEPTOR_SIGNALS.md), not this camera interface.

Historical retinal agents use a 16×16 binary retina and four spatial tiles. The apparatus belongs to the agent: width, height, partition scheme and
partition count are saved at birth, inherited and preserved on resume, import,
assessment and environment changes. One to width×height partitions are allowed;
256 partitions at the default resolution means one pixel per partition. Spatial
tiles use deterministic rectangular bisection. They do not track puzzle rows or
change with the environment. A shared presence feature is separate from pixels.

Every environment renders into this same receptor surface. The number line draws
an egocentric view with one world unit per pixel column, ground, the centered
agent, nearby walls and a target marker only when the target is inside the view.
It supplies no absolute position, remote target coordinate or offscreen direction
hint. Translating an otherwise identical scene away from walls produces identical
input. This makes the task partially observable. Puzzle environments draw their
physical lights into the same fixed image. They do not pass semantic cell values
directly to the retinal agent.

The playground's **Agent’s view** shows exactly the observed pixels. The world
overview and numeric task controls are external diagnostics. The sensory activity
overlay additionally compares observations with predictions. Predictions remain
complete retinal frames, with separate pending feedback for each active chain.

Each retinal frame is exactly width×height bits, packed MSB first: 32 bytes for
16×16. Direct frames of another size are rejected. A stream transport can split
opaque bytes into fixed windows and zero-pad the final short window; this does
not change resolution or partition geometry. Files are not automatically decoded
as images or audio. A source adapter must render a suitable scene or encoding
into the fixed receptors. Currently pixel counts must be multiples of 16.

Historical `legacy-bytes` snapshots also retain their original interface.
To explicitly select the historical retinal interface:

```sh
./project train --model senses --sensory-interface retina \
  --retina-width 16 --retina-height 16 --sensory-partitions 4 \
  --structure-preset exploratory --backend native
```

Python and native paths use the same rendered frames and spatial indices.
Structural evolution can change repertoire terms and mappings while keeping the
apparatus fixed. The legacy numeric-goal bias audit rejects retinal runs because
removing a coordinate channel is not a valid visible-target ablation. Comparisons
against full-state categorical agents also differ in available information.

These historical drawing and packing contracts do not establish learning success
or outcome tests for the new `signals` default.
