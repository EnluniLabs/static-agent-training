# Exploratory structural settings

Current system context and model boundaries: [architecture reference](ARCHITECTURE.md), reviewed 2026-11-22.

New Agent X sessions default to the immutable `signals` interface described in
[Receptor signals](RECEPTOR_SIGNALS.md): a 16×16 layout of 256 binary receptors,
four fixed tiles, and one shared generic dual-rail byte transducer for all sources.
The layout indexes receptors; it is not a camera. Historical `retina` packed-pixel
snapshots and `legacy-bytes` controls remain supported with their saved interfaces.

New **Agent X** sessions in the workspace now start with the **Exploratory**
structure profile. The categorical and single-chain fixed-senses forms retain
**Baseline**. Both profiles remain selectable, and individual fields can be
edited under **Custom**. This changes structural search and resource allowances;
it adds no reward for complexity and no artificial graph connections.

| Setting | Baseline | Exploratory |
| --- | ---: | ---: |
| Starting pairs per freshly initialized animal | 4 | 16 |
| Initial reduction budget per pair | 64 | 256 |
| Maximum nodes per inherited term | 31 | 127 |
| Maximum inherited-term depth | 6 | 10 |
| Explicit I/K/S mixture probability | 0.50 | 0.15 |
| Maximum reduction-expression nodes | 256 | 2,048 |
| Maximum development transitions | 1,000 | 2,000 |
| Subtree replacement probability | 0.15 | 0.35 |
| Whole-term replacement probability | 0.10 | 0.20 |
| Starting-pair insertion probability | 0.05 | 0.25 |
| Starting-pair deletion probability | 0.05 | 0.02 |
| Development budget of a newly inserted pair | 64 | 256 |

Mutation draws are independent; probabilities do not sum to one. Weight mutation,
learning rules, rates, receptor conversion, action arbitration and evolutionary
fitness are unchanged. Syntax-node limits are upper bounds, not target term sizes.
The recursive term generator can still produce small terms. Reducing the explicit
combinator mixture does not prohibit terms equivalent to I, K or S.

The larger allowance exposed recursion failures in the old AST utilities. Counting,
depth/closure checks, canonical encoding, freezing, redex enumeration and beta
reduction now use explicit traversal stacks. Canonical bytes and seeded reduction
order remain compatible; baseline replay is tested against the original behavior.
There is no global recursion-limit change. JSON storage and resource limits are
still finite; the preset is a tested starting configuration rather than a promise
that arbitrarily large custom settings are practical.

## Configuration and persistence

```sh
./project train --model senses --sensory-interface signals \
  --receptor-width 16 --receptor-height 16 --sensory-partitions 4 \
  --structure-preset exploratory --backend native
```

For the historical `legacy-bytes` row-aligned 3×3 puzzle control:

```sh
./project train --model senses --environment lights-out --sensory-interface legacy-bytes --receptors 144 \
  --sensory-partitions 3 --structure-preset exploratory --backend native
```

Flat flags override a preset, such as `--initial-pairs 24`,
`--seed-term-max-nodes 255`, or `--development-max-nodes 4096`. The command help
lists all controls. Comparisons apply the same structural settings to both arms.
CLI defaults remain the historical configuration unless a preset or override is
explicitly supplied. The workspace sends resolved field values for review.

Saved configs/checkpoints record the actual nested mutation and development values,
not a preset label that could silently change after a software update. Resume
retains those values and rejects structural overrides. Starting a new profile does
not patch or restart existing trainers, delete terms, or modify saved candidates.

Initial pair count and initial budget apply to **fresh initialization**. Imported
parents retain their existing starting terms, budgets and developed repertoires;
the new mutation probabilities and inserted-pair budget govern later offspring.
Changing imported structural bounds must still accommodate their inherited terms.
To obtain the complete new starting distribution, choose **Start fresh**.

## Inspecting the result

Session details now offer **Track → Structural growth**: mean developed states,
mean states with at least two distinct incoming edges, and mean inherited-term
syntax nodes. Population metrics weight each animal equally. The latter averages
its starting terms first. A convergence state is a topology statistic, not evidence
of useful cooperation between independently activated chains. Champion summaries
also record development stop reasons and conserved budget accounting. Missing
historical metrics are displayed as unavailable. Self-successor links are drawn
as loops in the 3D view.

A 16-seed fresh-development probe produced these means:

| Measure | Baseline | Exploratory |
| --- | ---: | ---: |
| Developed states | 13.50 | 50.50 |
| Convergence states | 1.81 | 7.56 |
| Starting-term syntax nodes | 7.86 | 15.97 |

This is a bundled structural intervention with a larger total reduction allowance,
not a compute-matched ablation or a task-performance result. More structure is not
itself useful behavior. A bounded native 3×3 training smoke run confirms evaluation,
checkpointing and deterministic resume; its tiny task sample does not demonstrate
solving ability. Further comparisons need matched task distributions and separately
reported computation, repertoire use and held-out performance.

The structural observations above are historical evidence about structure and
execution, not learning results or outcome tests for the new signals default.
