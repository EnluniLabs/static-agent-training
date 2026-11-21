# Universal receptor input

New Agent X sessions default to the immutable `signals` interface described in
[Receptor signals](RECEPTOR_SIGNALS.md): a 16×16 layout of 256 binary receptors,
four fixed tiles, and one shared generic dual-rail byte transducer for all sources.
The layout indexes receptors; it is not a camera. Historical `retina` packed-pixel
snapshots and `legacy-bytes` controls remain supported with their saved interfaces.

The sensory boundary accepts opaque bytes. It does not know whether they came
from text, pixels, audio, a file, or a sensor. A transport adapter supplies bytes;
the same fixed receptor conversion, partition geometry, learned embeddings and
lambda-pair repertoire process every source. There are no switchboard, object,
word, action-target or modality labels in this input path.

## Conversion and partitions

`SignalSenses.frame` (and historical `FixedSenses.frame`) accepts `bytes`, `bytearray`, `memoryview`, or `None` and returns
the saved fixed-width receptor pattern. The existing dual-rail conversion and its
serialization are unchanged. Each input bit activates its zero or one rail.
`None` is missing input; an empty byte payload is present but empty. Direct
oversized observations retain the historical position-folding behavior.

`SignalSenses.partition_indices(partitions)` selects fixed tiled receptor indices.
Historical `FixedSenses.regions(partitions)` provides contiguous balanced bounds. `active_features(frame, region, partitions)` keeps the original feature
coordinates and the shared presence feature. Regions carry positions, not meanings.
Different regions may select the same pair; all activations remain separate.
The model and 3D mapping labels use this common geometry.

For an unbounded or long stream, `ReceptorStream` reads fixed input windows.
The default is `receptors // 16` bytes, the non-folding capacity of the saved
apparatus. Every source byte is delivered in order across windows; the final
short window is retained. Larger window sizes are rejected rather than silently
folding them. Files are read in bounded requests, and arbitrary transport chunk
sizes produce the same windows. Splitting a network packet differently therefore
does not change the receptor sequence. Empty transport chunks are ignored; an
empty source produces no observations. EOF is distinct from an empty observation.
The reader uses bounded additional buffering; retaining an entire run's traces
still uses memory proportional to the trace length.

A concrete default of 256 receptors and four regions gives 16 input bytes per
window and 64 receptors per tile. Receptor count, partition count and window
size are interface settings, not decisions made from data semantics. Smaller
windows leave the unused receptor positions inactive. Geometry is fixed for an
agent; source changes do not allocate new embedding rows or prediction heads.

## Incoming data advances independently of actions

`run_stream` is observation-driven. For each input window it:

1. Tests pending expectations against the complete new receptor pattern.
2. Independently embeds all configured regions.
3. Advances their continuations for the configured internal rounds.
4. Resolves any action batch through the saved agreement or voting policy.
5. Moves to the next input window, including when chains only predicted or ended.

Predictions concern the complete **next receptor window**, not the whole file.
Duplicate predictions retain independent identities and sequential updates.
The default is one synchronous internal round per input window. An explicit
rounds-per-frame setting can allow more internal work before the next observation;
action resolution ends that window's internal processing. The stream never waits
for an action before admitting the next observation. EOF and explicit limits
expire pending expectations; they do not fabricate feedback or reward.

An optional `action_sink` receives the resolved `Action` or arbitration `NoOp`.
The universal input layer assigns those signals no physical meaning. A robot,
game or other consumer must define its own motor interface. With no sink, outputs
are inspected without affecting the external world.

`StreamEnvironment` also supplies an **action-paced** compatibility adapter for
existing `run_episode` consumers. Its next observation is requested on an action.
Use `run_stream` for continuously arriving input and the workspace preview.

## Use

```python
from static_agent_training.experimental.stream_input import run_stream

# agent is a BitModel with saved SignalSenses and its fixed partition count.
with open('input.bin', 'rb') as source:
    result = run_stream(agent, source, frame_limit=100, learn=False)
```

The source can also be a bytes-like value or an iterable of bytes-like chunks,
including a lazy source. Binary readers remain caller-owned. Text needs an explicit
encoding such as UTF-8; images, audio and numeric arrays need an explicit byte
layout. No file-type detection, decompression or learned preprocessing is hidden
inside the receptor. A one-dimensional byte layout does not automatically preserve
camera geometry or make two modality encodings equivalent.

```sh
./project stream --input input.bin --receptors 256 --partitions 4 --max-frames 16
./project stream --input input.bin --snapshot saved-model.json.gz --output inspection.json
```

Reports refuse to overwrite existing files. The snapshot's geometry and action
policy are retained. Execution is frozen by default; optional CLI learning affects
only the in-memory copy. The default frame budget is explicit and bounded; reaching
it is reported as a limit without reading ahead to guess EOF. The arbitrary-source
runner uses Python. Existing native evaluators still accelerate their registered
worlds and retain their numerical parity tests.

In **Animal playground**, select a fixed-senses snapshot and open the universal
input panel. Local file bytes pass through a frozen copy of that snapshot or its
current playground lifetime. The preview shows receptor regions, state activation,
predictions, action resolution and the 3D mappings. File names and content labels
are not inputs to the agent. Previewing does not change the saved agent or the
playground's learned state.

This establishes a common input mechanism and a way to inspect it. It does not
establish understanding of arbitrary data, spatial vision, temporal attention,
retention, or cross-modal transfer. Those require controlled experiments.
