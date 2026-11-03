# Repertoire Libraries: Building the Inherited Animal Before Universal Explanation

**Research idea — November 2, 2026.** Conceptual proposal and experimental roadmap, not an established result. Related: *The Coherence Engine*, *The Universal Explainer*, *Universal Explanation Before Cumulative Innovation*, `docs/THREE_CORE_EXPERIMENTS.md`, and `docs/COHERENT_VS_RANDOM_TEST_SPEC.md`.

## 1. The original bottleneck

A universal explainer cannot usefully be introduced into an otherwise empty cognitive architecture. Before an animal can independently construct and criticize novel explanations, it needs enough inherited functional structure to perceive, act, explore, predict, learn, and persist in an environment. The cumulative-innovation paper describes a transition from dependence on genetically supplied coherent structures (`C_G`) to increasing resources for constructing novel structure (`R_C`), with `R = R_G + R_C`. The difficulty was practical: how could we develop the large repertoire of useful inherited structures in feasible time? Brute-force evolution of an entire competent animal at once would create a vast, entangled search problem. Nor do we yet know a complete hand-authored taxonomy of the structures an animal requires. In this sense we must understand/build *animal* before attempting *universal explanation*.

**Proposed answer:** evolve and train smaller, useful repertoires independently under varied environments and selection pressures, preserve them as transferable packages, and gift a collection of these repertoires to new animals. Build up the inherited foundation cumulatively instead of re-evolving every prerequisite together.

## 2. Why the existing architecture may permit this

A coherent structure is a fixed, internally constrained computational relationship; a representational theory associates its states and transformations with observations, actions, and expectations. Its logical constraints are shared across independently generated structures because they arise from the same formal rules. Their physical interpretations need not be shared.

**Critical clarification: coherent structures are never merged or spliced together.** This is true even for an animal developed from scratch. An animal has a repertoire (a collection of independent coherent structures) and learns which structures to traverse, when, and how to interpret them. Multiple independently evolved packages should simply enlarge the collection available to that animal. We do **not** propose graph merging, artificial inter-structure edges, a new structural-composition operation, or requiring a single coherent structure that combines all skills.

The animal's ordinary representational learning is the proposed mechanism for adapting inherited structures to its current environment and for making independently acquired repertoires useful together in behavior. A new task may be solved by selecting and reinterpreting existing structures, without constructing or merging any new coherent structure. In the papers' terminology this can be **learning by analogy**: `(C,T) -> (C,T')`, with `C` fixed and `T` revised. Novel behavior does not by itself constitute structural innovation.

Shared coherence rules **do not mathematically guarantee** that any arbitrary pair of packages will be compatible or that a useful cross-package interpretation exists. The specific hypothesis is that independently developed coherent repertoires can coexist without structural modification and that an animal can efficiently discover useful representations, selections, and behavioral sequences across them. This is an empirical claim to test, not a proof of automatic interoperability.

## 3. Developmental strategy: a library of gifted repertoires

1. Create bounded evolutionary/ecological settings that exercise a relatively tractable capacity: exploration, spatial navigation, prediction, interaction/manipulation, obstacle avoidance, environmental contingencies, etc. These are candidate domains, not an asserted exhaustive list of innate faculties.
2. Evolve populations within each setting, with lifetime learning continuously active. Some settings may have explicit goals; others should allow voluntary exploration without goals.
3. Preserve successful candidate **repertoire packages**: immutable coherent structures and provenance, plus optionally learned representational parameters and the interface assumptions under which they were acquired. Keep structural inheritance distinct from transferring acquired mappings.
4. Build new animals gifted several packages. The structures remain independent. The animal learns representations and selection policies in its new environment; it may reuse, revise, ignore, or behaviorally coordinate their applications.
5. Evaluate retention of the constituent capacities, adaptation to new contexts, and genuinely new composite tasks. Preserve useful packages/variants for later animals. Incrementally increase complexity without restarting evolution of all prerequisites.
6. Eventually, if a sufficiently capable animal can construct genuinely new coherent structures **during its lifetime**, test the transition from inherited repertoire reuse to structural innovation. This is a future capability, not one demonstrated by the current fixed-repertoire implementation.

The practical development process itself becomes cumulative: human-run experiments preserve and gift useful cognitive foundations **before** the animals demonstrate autonomous cumulative innovation.

## 4. Why exploration may generate generally useful packages

A goal-free animal can explore an environment, choose whether and when to emit expectations, observe failures, and revise representations. It may learn spatial, causal, or action-related regularities without knowing which future goal will matter. Later, an unexpected goal (e.g. reaching food) can be introduced. The animal may solve it faster because it has already developed useful interpretations or conjectures during self-directed exploration. Do not impose prediction at fixed intervals or secretly train on the future food objective.

This motivates multiple evolutionary regimes:
- **Neutral/no task selection** as a control: no reward for reaching a future goal.
- **Exploration-based selection:** novelty, informative coverage, diverse interactions, and/or discovery of environmental contingencies, guarding against aimless wandering.
- **Predictive competence or improvement:** voluntary, testable expectations and their learning progress, guarding against never predicting or only making trivial, easy predictions.
- **Explicit-goal evolution** as a comparison, not the default assumption for all animals.

Separate *evolutionary selection*, *lifetime exploratory learning*, and *later unforeseen goal evaluation*. Selection for exploration or predictive learning is a proxy and does not establish general intelligence. The current architecture's expectation updates are not yet equivalent to explicitly inventing arbitrary problems; test whether useful self-generated conjecture-and-criticism behavior actually emerges.

## 5. What exactly is transferred?

Three conceptually distinct cases must be separated:

- **Structural package only:** evolved coherent repertoire, initialized with fresh learned mappings. Tests whether the evolved structure itself offers a reusable learning advantage.
- **Structure plus acquired interpretation:** the same repertoire plus learned observation/output/expectation parameters, when signal/action interfaces are compatible. Tests direct transfer of acquired knowledge and may require adaptation or re-representation.
- **Multiple independent packages:** several repertoires present concurrently, with either fresh or inherited representations. Tests selection, representational flexibility, interference, and the emergence of behavior requiring capabilities from more than one package.

A package manifest should record generator/lineage, immutable nodes and successor links, normalization/developmental constraints, environment and selection history, learned mapping snapshots (if included), signal and action encoding, expected interface compatibility, performance evidence, and version. This is a proposal for future packaging infrastructure, not a claim that a package format exists today.

Do not confuse shared formal coherence with compatible signal encodings. If one animal's observation/action interface differs, transferred mappings may be meaningless; a generalized bitwise signal interface is a separate planned research direction. Even with a common interface, learned logits can interfere through selection and credit assignment. Re-representation is a hypothesis about how the animal can resolve this, not an automatic guarantee.

## 6. Core experiments

### A. Independent repertoire reuse
Evolve repertoire A for exploration/navigation and B for object interaction or another distinct capability. Compare fresh animals given A only, B only, A+B, and appropriate capacity-matched control repertoires. Test each constituent task and a **new** task that requires both kinds of competence. No structural merging or added links. Lifetime learning remains active. Measure sample efficiency, retention, repertoire utilization, voluntary expectation rates, and interference. A larger collection alone is not evidence of compositional benefit.

### B. Transfer mechanism ablation
Compare (i) fresh structure and fresh mappings, (ii) evolved structure with fresh mappings, (iii) evolved structure with inherited mappings, (iv) independently evolved A+B with fresh mappings, and (v) A+B with inherited mappings, where meaningful. Match node/edge counts, lifetime experience, compute budgets, and evolutionary effort. Distinguish structural bias from transferred interpretation.

### C. Goal-free exploration, then surprise food
Evolve or initialize exploration-capable animals without any information about the later food-finding objective. Allow uninterrupted exploratory lifetimes and voluntary prediction. Then introduce food at a novel location or a novel environment and measure adaptation. Compare no exploration, noninformative exposure, different exploratory-selection regimes, coherent versus random-node repertoires, and transferred exploration packages. Avoid leakage of the later goal into selection, representations, or environments.

### D. Cumulative package library
Repeat the process over multiple generations of *experimental development*: discover a useful package, preserve it, gift it to later animals alongside other packages, and test whether each addition creates measurable new competence without erasing prior abilities. Track marginal gains, negative transfer, package count, compute cost, and adaptation time. Compare against monolithic evolution given an equivalent total budget. This tests **engineering scalability**, not yet autonomous cultural accumulation.

### E. Eventually, animal-to-animal cumulative innovation
Once lifetime construction of novel coherent structures exists, investigate whether animal A can create a new structure, whether B can reconstruct/use/criticize it through an interface, and whether C can extend what B learned. This is the stronger cross-individual cumulative innovation described in the paper. Mere copying of logits or structures is not proof that recipients understand or can criticize explanations.

## 7. Architectural invariants and risks

- Coherent structures are **independent and immutable during normal lifetime traversal**; they are not combined. Lambda reduction is used in development/evolution, not each traversal step.
- The animal chooses whether and when to emit an expectation; no forced prediction schedule.
- Lifetime learning stays on in normal operation. Cloned/frozen diagnostic evaluations may be used for controlled scientific measurement, never mistaken for the deployed animal's normal mode.
- Goals may be explicit, implicit through environmental cues, or absent.
- Evolved repertoire topology and generative distribution matter; compare coherent and random-chain controls, including topology-matched variants. If only labels differ and graph/learning dynamics are isomorphic, no coherence-specific advantage follows.
- Current lifetime learning adjusts mappings and local compatibility signals on fixed structures. It does **not yet** construct novel coherent structures during a lifetime; distinguish analogy, transfer, and true innovation.
- Independent packages may compete for selection probability, suffer destructive mapping interference, or be inaccessible through a shared interface. More structure can be worse.
- Training and testing must use separate environments/seeds and account for total evolutionary compute; avoid interpreting memorized skills or greater model capacity as general transfer.
- Current number-line environment may be too simple to test exploration, compositional skills, or novel goal generalization. Richer environments and interface design will be necessary.

## 8. The central hypothesis and significance

**Hypothesis:** A useful inherited animal can be assembled from independently evolved coherent repertoires because their formal constraints are universal and their environment-specific meanings can be learned or revised by the animal, without merging structures. This modular *development process* may make the otherwise intractable prerequisite of building an animal feasible.

This bridges the papers: *The Coherence Engine* supplies a constrained generator of structure; *The Universal Explainer* distinguishes coherent structure from representational theory and eventual novel structure construction; *Universal Explanation Before Cumulative Innovation* motivates the substantial inherited foundation and the later transition to individually generated and socially accumulated explanations.

There are **two different kinds of accumulation**:
1. **Researcher-mediated accumulation of inherited packages**, potentially achievable with the present fixed-repertoire architecture after suitable implementation and testing.
2. **Autonomous cumulative innovation by animals**, requiring lifetime generation of genuinely novel coherent structures and a way for other animals to reconstruct, criticize, and extend them. This is a longer-term goal.

**Decisive near-term question:** Can a new animal, gifted independently evolved but unmerged repertoires, learn useful representations that retain old capabilities and solve a previously unseen composite problem more efficiently than fair controls?

**Long-term question:** Can this approach build enough inherited competence to support an animal that independently generates, tests, and later communicates genuinely novel coherent structure?

No success is claimed yet. The value of this idea is that it converts an open-ended brute-force evolutionary prerequisite into a sequence of smaller, measurable, reusable experiments.
