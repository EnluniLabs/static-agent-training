# Receptor signals

New Agent X sessions default to `signals`, implemented by `SignalSenses`. This is
an immutable binary receptor apparatus, not a drawing image or camera. Width,
height, partition scheme and partition count belong to the agent and persist
across inheritance, resume, import, assessment and environment changes.

## Layout and capacity

| Quantity | Default meaning |
| --- | --- |
| Resolution | 16×16 receptor indexing layout |
| Bits per receptor | 1 binary activation |
| Receptor count / total capacity | 256 receptors / 256 bits |
| Source-byte capacity without folding | 16 bytes, because each source bit needs two rails |
| Partitions | Four fixed tiles; configurable from 1 through 256 |

Tiles use deterministic rectangular bisection of the indexing layout. At 256
partitions each tile contains one receptor. Partitions select subsets of the same
bank; they do not increase capacity, change the transducer or follow world objects.
The shared presence feature is separate from the 256 receptor bits.

## One conversion for every source

`SignalSenses` applies the same generic dual-rail bytes transducer as historical
`FixedSenses`. Each source bit activates either its zero rail or its one rail.
For receptor count `R`, source bit position `k` maps to receptor
`2 * (k % (R // 2)) + bit`, with bits read MSB first. Direct arbitrary-length
byte observations fold positions beyond capacity and combine activations by OR,
just as the legacy generic conversion does. Folding can activate both rails of a
pair and lose information; it does not enlarge the bank.

Streams instead window at `R // 16` source bytes (16 by default), without folding.
A partial final window leaves unfilled slots inactive: it is not padded with
source zero bytes, which would activate zero rails. Missing input (`None`) is
distinct from empty but present bytes via presence. Empty transport chunks produce
no observation. See [Universal receptor input](UNIVERSAL_INPUT.md).

## World adapters

The new Number Line signals source supplies 16 local numeric sensor readings at
relative offsets −8 through 7. These offsets are independent of receptor layout,
width, height and tiles. Each reading is one byte:

| Local reading | Value |
| --- | ---: |
| Outside the world bounds | 0 |
| Normal location | 64 |
| Self | 128 |
| In-view goal, when `goal_conditioned` | 255 |

The adapter supplies no glyph drawing, absolute coordinates or offscreen goal
hints. Other worlds supply their existing `bytes(board_bits(...))`. All these
source bytes pass through the **same** dual-rail conversion; none is inserted as
a receptor activation or rendered as a picture.

Analog sensor sampling and digitization belong to the source adapter. Binary
receptors approximate digitized signals; this is not an analog voltage engine.
Text, audio and other sources likewise require explicit byte encodings.

## Configuration and compatibility

```sh
./project train --model senses --sensory-interface signals \
  --receptor-width 16 --receptor-height 16 --sensory-partitions 4 \
  --structure-preset exploratory --backend native
```

`--receptor-width` and `--receptor-height` alias the internal `retina_width` and
`retina_height` fields. Those historical field names do not imply camera input.
The Exploratory structural settings, learning rules and action arbitration are
unchanged by this interface correction.

Historical `retina` snapshots retain their packed-pixel drawing interface; see
[Fixed retina](FIXED_RETINA.md). Historical `legacy-bytes` snapshots retain their
generic conversion and contiguous partitions, including explicit 144-receptor,
three-partition puzzle controls. Neither is silently converted on load.

This documents an interface contract. Learning success, navigation, transfer and
outcome tests for the new default have not yet been established.
