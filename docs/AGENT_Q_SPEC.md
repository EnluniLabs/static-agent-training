# Agent Q specification

Agent Q uses a fixed graph of coherent lambda pairs and two learned distributions. Environmental success selects inherited candidates during evolution. Lifetime corrections use sensory observations and sampled expectations.

## Structure and sensory embedding

An ordered pair q = (B, A) has successor q′ = (NF(B A), B) when development finds the normal form within its existing budgets. Agent Q shares Agent Z's function-first development and pair validation. The graph is immutable during a lifetime. A pair with no successor ends its chain before emitting an action.

The number-line adapter emits one byte containing signed goal-relative position. In region r of R, the embedding logit for anchor x is

    h(x, s, r) = b[x] + Σ(j mod R = r) w[x,j] s[j]
    Eθ(x | s, r) = softmax_x(h(x, s, r))

Each region samples an initial anchor independently from its embedding. All regions use the same table and repertoire. A chain follows its fixed successor after acting. The actual observation does not activate a new chain.

## Action-conditioned outcomes and derived policy

Every anchor and physical action has eight expectation logits z[q,a,j]. The probability of bit j being one is p[q,a,j] = sigmoid(z[q,a,j]). For a sensory byte s, the model is

    Pφ(s | q,a) = Π_j p[q,a,j]^s[j] (1 − p[q,a,j])^(1 − s[j])
    K_q,r(a) = Σ_(s = 0 … 255) Pφ(s | q,a) Eθ(q′ | s,r)
    π(a | q,r) = K_q,r(a) / Σ_b K_q,r(b)
    π_ε(a | q,r) = (1 − ε) π(a | q,r) + ε / |A|

The prior is uniform over the adapter's nine physical action signals. The default exploration probability ε is 0.05. If every compatibility underflows to zero, use a uniform distribution. Embedding normalization subtracts the maximum logit before evaluating the log normalizer. Bernoulli zero and one probabilities use stable sigmoids evaluated at opposite logits.

The Python implementation explicitly sums all 256 bytes. The native implementation marginalizes the unused bits analytically. The embedding in region r is independent of those bits, so summing their Bernoulli factors gives one. It enumerates every assignment of the remaining bits. This is an exact rearrangement up to floating-point rounding.

## Round execution

1. List the active regions whose anchors have successors. Rotate the sorted list by the round index modulo its length. This exposes each eligible region to first access without learned priority.
2. For each region in that order, derive its action distribution from the current parameters. Sample the physical action, then sample an expectation byte from its eight independent Bernoulli bits.
3. Execute that action and capture its immediate post-action observation. Associate the observation only with the chain that emitted the action. Reaching the goal or exhausting the 12-action limit ends the task before any later chain samples an action.
4. Calculate the two losses from the round's unchanged parameter tables. Record the originating anchor, region, action, sampled expectation, observation, successor, and losses.
5. Sum each parameter's contributions in sorted order using compensated arithmetic, then apply the updates. Each successfully executed chain advances to its fixed successor.

The reference evaluator uses Python's seeded RNG. Native execution consumes the same seeded MT19937 stream, with the same order and count of draws on matching trajectories.

## Empirical and structural corrections

For the originating anchor q, executed action a, sampled expectation s, actual observation o and successor q′:

    L_emp = −log Pφ(o | q,a)
    Δz[q,a,j] = η_emp (o[j] − p[q,a,j])

The empirical update touches only the chosen anchor–action expectation row. It uses the observation, even if the sampled expectation was wrong.

    L_struct = −log Eθ(q′ | s,r)
    Δh(x) = η_struct (1[x=q′] − Eθ(x | s,r))

Apply Δh(x) to the embedding bias and, multiplied by s[j], to the weights for that region's input bits. The structural target is the actual fixed successor. The input is the exact sampled pre-action expectation, including empirically incorrect samples.

The enabled learning rates are 0.05. Modes are both, empirical, structural, and off. Off disables both parameter updates and retains the same action and observation loop.

Updates to φ and θ change future K values and therefore the derived action policy. This is the sole coupling between learning and action selection. There are no learned action logits, policy gradients, goal-reward terms, or chain backpropagation. Compatibility is an experimental criterion. The model does not establish correct explanations, convergence, or structural shielding.
