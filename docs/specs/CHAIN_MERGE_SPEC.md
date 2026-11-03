# Chain intersections, equivalence, and mergers

Status: analysis specification supplementing the three supplied papers.

Implemented execution policy (updated by user decision): exact ordered-pair
mergers pool remaining reduction budgets, including arrivals at different times.
Historical junctions forward budget to the surviving execution's current position;
budget-exhausted executions may resume. Shared single terms do not pool budgets.
Cycles continue under the remaining budget and a transition cap. This policy
supersedes the independent-budget and stop-on-cycle defaults below. The engine
currently implements exact pair detection and pooling; equivalence certification,
structural similarity, and the full analysis registry remain future work.

## Purpose

Detect when independently developed chains reach the same mathematical term, when differently written terms have a certified common normal form, and when chains actually share a continuation. Also expose approximate structural resemblance for analysis without silently identifying different mathematical objects.

The engine develops `q = (A, B)` by attempting `C = NF(B A)`, then visiting `(B, C)`. A common term is an intersection. A common ordered pair is a merger of continuation state. These require separate records: reaching `J` from different preceding terms does not by itself establish a shared future.

## Relations and allowed consequences

| Relation | Required evidence | Consequence |
| --- | --- | --- |
| Exact term intersection | Identical canonical ASTs | Share immutable term storage; record both occurrences |
| Certified normal-form intersection | Both terms normalize successfully to identical canonical ASTs | Record beta-equivalence with witnesses; optionally use in a separately configured quotient graph |
| Approximate resemblance | Structural similarity exceeds configured threshold | Analysis annotation only |
| Exact pair merger | Both components of ordered pairs are respectively identical canonical ASTs | Share mathematical state identity and certified successor facts |
| Certified equivalent pair | Each corresponding component has a certified common normal form | Record equivalent continuation states; operational collapsing is an explicit ablation |
| Repeated pair within one chain | Exact ordered-pair identity already visited in that chain | Record a cycle and stop development |

Alpha-equivalence is exact identity under De Bruijn representation. Beta-equivalence is broader. Eta-equivalence is excluded from the baseline. Failure to obtain a normal form is UNKNOWN, never a certificate of inequality or divergence.

Structural identity is independent of variable names. Bound variable occurrences are identified by their relationship to their enclosing binders, not by spelling. For example, `λx.x` and `λy.y` are identical structures, as are `λx.λy.x` and `λa.λb.a`. However, `λx.λy.x` and `λa.λb.b` are different structures: the occurrence refers to a different binder. Erasing variable labels without preserving those binding relationships is incorrect.

## Canonical identity

Use immutable `Var(index)`, `Abs(body)`, and `App(function, argument)` nodes. Serialize with a versioned, unambiguous binary encoding: distinct node tags, unsigned variable indices with explicit boundaries, and recursively encoded children. Do not include source names, chain IDs, RNG state, or provenance.

Convert any named input syntax to De Bruijn indices before comparison or hashing. Index zero denotes the nearest enclosing binder, index one the next enclosing binder, and so on. Thus `λx.x` becomes `Abs(Var(0))`, while the two nested examples above become `Abs(Abs(Var(1)))` and `Abs(Abs(Var(0)))`. Source variable names may be retained for display only. The baseline accepts closed terms; reject unbound variables rather than assigning them name-based identity. Generated display names must never affect IDs, similarity scores, or merger decisions.

Define:

```
term_id(M) = SHA256("crt-term-v1" || canonical_bytes(M))
pair_id(A, B) = SHA256("crt-pair-v1" || term_id(A) || term_id(B))
```

Verify canonical bytes on hash matches before identifying objects; a detected collision must raise an error. Pair order matters. Keep raw AST identity separate from normal-form identity: do not quietly normalize inherited seed terms when calculating their raw IDs. Generated successful reductions already produce normal forms; inherited seeds need not be normal.

## Online intersection registry

Maintain a registry within each developed genome:

```
TermRecord: term_id, canonical_ast, node_count, normal_form_evidence?
Occurrence: chain_id, pair_index, component(left|right), term_id
PairRecord: pair_id, left_id, right_id, occurrences, certified_successor?
Attempt: chain_id, source_pair_id, seed, budget_before, steps,
         status, result_term_id?, budget_after, trace_reference
```

Occurrences retain component role even though lambda terms themselves have no intrinsic state/transformation type. Reappearance in overlapping consecutive pairs is routine: report distinct-chain intersections separately from same-chain recurrence. Deduplicate an intersection by term and participating chain IDs; retain all occurrences for tracing.

On registering a pair:

1. Intern both terms and append their occurrences.
2. Emit `TERM_INTERSECTION` when a term gains an occurrence in a new chain.
3. If the pair previously occurred in this chain, emit `PAIR_CYCLE` and stop this chain.
4. Otherwise, if it occurred in another chain, emit `PAIR_MERGER` and retain both provenances.
5. Attempt the next reduction under this chain's own resource policy. Record success or failure independently of the registry's mathematical knowledge.

The registry may retain a successful successor as a mathematical fact even when another execution of that same application exhausts its budget. Do not overwrite success with failure, or turn a failed execution into success because another chain supplied a witness. Every successful attempt for an exact pair must agree on the resulting normal form; disagreement is an evaluator error.

Default: merged chains continue their own seeded reduction attempts until their own stopping condition. Sharing storage does not pool budgets or eliminate charged computation. A later memoization treatment must specify cost accounting and reduction-path semantics separately. Indexes used for analysis across genomes must never transfer mapping weights, execution budgets, or learned state between animals.

## Bounded equivalence certification

Exact identity finds common generated normal forms immediately. To detect nonidentical seed expressions that denote the same normal form, add an optional analysis pass:

1. Attempt full beta-normalization of each distinct term using a separately seeded stochastic reducer.
2. Default analysis limits: 256 contractions per term, 256 AST nodes, one attempt; record these as provisional configuration values.
3. Store status, actual contractions, reducer version, RNG seed, and reduction witness.
4. On NORMAL, index the resulting canonical normal-form ID. Two successful certificates with the same ID establish equivalence.
5. On BUDGET or SIZE, record UNKNOWN. A later successful retry can add evidence without erasing the earlier failed attempt.

Normal-form classes are auxiliary indexes. The primary evolutionary experiment retains exact pair identities and raw inherited expressions, preserving the specified resource-bounded execution. A quotient-graph experiment may collapse certified equivalent pairs, but must report changed execution costs and an explicit mapping-weight reconciliation policy. No automatic averaging or transfer of physical logits is permitted in the baseline.

## Approximate structural resemblance

If “similar” means resemblance rather than proven equality, compute a separate score. For each term, collect a multiset of canonical rooted subtree IDs, including De Bruijn indices. Let `c_M(h)` be the number of occurrences of subtree hash `h` in `M`. Define:

```
similarity(M, N) = sum_h min(c_M(h), c_N(h))
                   / sum_h max(c_M(h), c_N(h))
```

This is a reproducible structural statistic, not a semantic equivalence test. Provisional reporting threshold: 0.8. Report AST sizes and exact/equivalence status alongside the score. Optionally compare certified normal forms instead of raw terms, but label which view was used. Embedded subtree hashes describe syntax in context; they do not assert that an open subtree is an independently meaningful closed term.

For a pair, report the minimum of the two corresponding term scores. Do not swap components to improve the score. Do not merge successors, physical mappings, budgets, or learning events on this basis. Similarity is not transitive; do not use connected components of threshold matches as equivalence classes.

For the small baseline repertoire, exhaustive distinct-term comparisons are acceptable. Larger experiments can add an inverted subtree index for candidate retrieval, with exact scoring afterward and separately measured retrieval recall.

## Example

Suppose two developed sequences contain:

```
chain 1: ..., (P, J), (J, K), ...
chain 2: ..., (Q, J), (J, L), ...
```

Both intersect at `J`. Their next applications are `J P` and `J Q`, so `K` and `L` may differ. No pair merger has occurred. If `K = L` exactly, both subsequently reach `(J, K)`, which is a certified pair merger. From that pair, any successful normalizations of `K J` agree, although finite-budget executions can still differ in success and cost.

For equivalence beyond syntax, `I = λ.0` and `App(I, I)` are distinct raw terms but share normal form `I`. Their equality is certified only after successful normalization. Two superficially similar abstractions can instead have distinct normal forms; approximate resemblance grants no continuation guarantee.

## Events and measurements

Every event includes genome ID, chain IDs, occurrence indices, raw term/pair IDs, relation kind, evidence references, and configuration version. Similarity events additionally record score and threshold. The trace viewer must distinguish term intersections, pair mergers, cycles, certified equivalence, and approximate matches visually.

Report distinct shared terms, distinct merged pairs, participating chains per merger, first merger depth, cycles, equivalence certification coverage and failures, approximate-match scores, and analysis computation separately from development computation. Physical interpretations remain conjectural even at a mathematically certified merger.

## Acceptance checks

- Alpha-renamed terms have identical De Bruijn IDs; distinct free/bound variable structure does not collide.
- Consistently renaming every bound variable leaves all term IDs, pair IDs, similarity scores, and intersection/merger events unchanged.
- Identical unlabeled tree shapes with different binder references remain distinct, such as `λx.λy.x` versus `λx.λy.y`.
- Pair IDs are stable across processes and distinguish component order.
- Shared `J` with different predecessors produces a term intersection without a premature pair merger.
- Two chains reaching the same ordered pair produce one shared state with both provenances.
- A repeated pair in one chain produces a cycle event.
- Distinct expressions reducing to `I` obtain a common normal-form certificate; unequal normal forms remain separate.
- Exhaustion and size limits produce UNKNOWN equivalence results and never erase successful evidence.
- Successful reductions from a shared pair agree; failed attempts remain individually visible.
- Detection changes neither chain budgets nor inherited/lifetime mapping weights.
- Approximate matches never alter the execution graph, including nontransitive triples of matches.
- Identical configuration and seeds reproduce IDs, attempts, and event ordering.

## Baseline decision

Implement exact term intersections and exact pair mergers as required engine features. Include bounded normal-form equivalence and structural resemblance as analysis tools. Leave budget pooling, equivalence-based operational collapsing, and similarity-driven learning as separately named experiments.
