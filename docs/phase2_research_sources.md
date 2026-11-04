# Phase 2 primary-source traceability

Checked 2026-11-04. Maturity: **Diagnostic** source audit, **Experimental** transfer hypotheses.
This is not mathematical proof checking, model qualification or empirical alignment evidence.

## Competing interpretability hypotheses

[Nick Cammarata, *Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) is an
informal working hypothesis. It argues for investigating native mechanisms, including signed
poles, readers, context and recursive generalization. This motivates better discovery tools,
not acceptance of a semantic label as a verified mechanism. Its preliminary interpretability
estimates are not GEPA measurements or universal properties of neurons.

[Leo Gao, *Neural networks are not already clean*](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean)
(2026-10-17) distinguishes a coherent pattern in examples from reproducing the neuron's actual
function and downstream effects. It motivates held-out magnitude prediction, weak-activation
tests and activation substitution scored by downstream KL. Small individual effects require
controls and grouped tests. This is a proposed standard of evidence, not proof that every native
neuron is uninterpretable or that decomposition necessarily solves the problem.

Both sources were read directly. Their disagreement is retained as testable alternatives.
The project will report prediction and intervention evidence rather than a cleanliness score.
Neither essay establishes deceptive intent from one activation or authorizes optimizer reward.

## Mathematical collection snapshot

Primary repository pinned to
[`openai/math@4eaee87548250b2682affda7d05d947c05554bae`](https://github.com/openai/math/commit/4eaee87548250b2682affda7d05d947c05554bae)
(2026-10-30). The [repository statement](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/README.md)
describes model-produced manuscripts at mixed verification stages as raw research data.
The inspected READMEs cite OpenAI Math Release preprints. Numbered editions are not supplied for
the seven artifacts below; dated directories plus this SHA identify the audited versions.
Available formalization artifacts do not establish completed independent review or replication.
This audit inspected manuscript statements and formalization scopes; it did not compile Lean,
check proof terms or reproduce algorithms. No peer-review acceptance was established.

### Random 3-SAT threshold

[Computing the Random 3-SAT Threshold, 2026-09-27](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Computing-the-Random-3-SAT-Threshold-September-27-2026/README.md)
claims deterministic approximation of the limiting threshold within `2^-r`. The model uses
proper three-distinct-variable clauses, fair signs and sampling with replacement. The
[introduction](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Computing-the-Random-3-SAT-Threshold-September-27-2026/build/sections/introduction.tex)
does not supply a practical efficiency bound, numerical threshold evaluation or individual-formula
solver. [Family 235 scope](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/docs/235.md)
and [SATComputability configuration](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/ComparatorChallenges/SATComputability.json)
document the selected formalization. Transfer: retain upper/lower evidence separately and do not
equate computability with efficient task solving or general alignment decidability.

### Counting perfect matchings

[A Fully Polynomial Randomized Approximation Scheme for Perfect Matchings in General Graphs,
2026-09-23](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/A-Fully-Polynomial-Randomized-Approximation-Scheme-for-Perfect-Matchings-in-General-Graphs-September-23-2026/README.md)
claims relative-error `epsilon` approximate counting with failure probability at most `delta`
for finite simple undirected graphs, exact zero on zero counts, and polynomial worst-case bit
time in input size, `1/epsilon` and `log(1/delta)`.
[Family 113](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/docs/113.md)
and [MatchingFPRAS configuration](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/ComparatorChallenges/MatchingFPRAS.json)
describe the formalized scope. Transfer: store estimator identity, sampling assumptions, error,
confidence and limits; an estimate is not an exact count or a verified outcome certificate.

### Existential theory of the reals

[Existential–universal real sentences in the counting hierarchy, 2026-10-04](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Existential-universal-real-sentences-in-the-counting-hierarchy-October-4-2026/README.md)
claims fixed-level counting-hierarchy decision for the stated real sentences with explicit
Boolean formulas and arithmetic-circuit polynomials over signed binary integer constants.
The [statement and encoding](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Existential-universal-real-sentences-in-the-counting-hierarchy-October-4-2026/build/sections/01-introduction.tex)
places the existential fragment at level 26. No family-141 formalization entry was found at this
snapshot. Transfer: use compact typed symbolic obligations; compact representation alone does
not imply efficient decision, numerical robustness or a practical general solver.

### Turing-machine time and space

[Simulating One-Tape Time in Two-Fifths-Power Space, 2026-09-25](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Simulating-One-Tape-Time-in-Two-Fifths-Power-Space-September-25-2026/README.md)
claims `O(T^(2/5) polylog(T+2))` writable bits for capped simulation of the stated fixed
deterministic one-writable-tape/head machine, with specified initial-symbol access and read-only
heads. Simulation time is unrestricted.
[Family 137](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/docs/137.md)
and [OneTapeSpace configuration](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/ComparatorChallenges/OneTapeSpace.json)
cover selected capped/unknown-halting statements, not all multitape consequences. Transfer:
measure retained state versus reconstruction cost/fidelity while preserving original audit
evidence. No theorem about transformer inference follows.

### Subset Sum

[Subset Sum in Time O(2^0.49n), 2026-10-04](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Subset-Sum-in-Time-2-power-0-49n-October-4-2026/README.md)
claims a uniform randomized classical decision algorithm, correctness at least `2/3` per input
and the stated time bound on every execution under fixed polynomial input-bit bounds. Repeated
positive integers are allowed. The [model assumptions](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Subset-Sum-in-Time-2-power-0-49n-October-4-2026/build/sections/introduction.tex)
specify word-RAM operations and word length. No family-138 formalization entry was found.
Transfer: faster search remains search; a faster exponential algorithm is not a polynomial
solution, an infeasibility certificate or measured practical improvement here.

### Maximum-cardinality matching

[Almost-Linear-Time Maximum-Cardinality Matching in General Graphs, 2026-09-24](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/Almost-Linear-Time-Maximum-Cardinality-Matching-in-Sparse-General-Graphs-September-24-2026/README.md)
claims exact maximum matching with probability at least `2/3` in `(n+m)^(1+o(1))` instructions
on every path, for the stated simple undirected unweighted adjacency-list/logarithmic-word model.
The directory retains “Sparse” although the manuscript title omits it. No family-120 formalization
entry was found. Transfer: approximate/randomized construction can yield an exactly checkable
witness. Checking a valid matching does not prove maximum cardinality without an optimality check.

### Shortest common superstring

[A Polynomial-Time 2-Approximation for Shortest Common Superstring, 2026-09-24](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/A-Polynomial-Time-2-Approximation-for-Shortest-Common-Superstring-September-24-2026/README.md)
claims a deterministic construction at most twice optimal symbol length for finite families of
explicit strings, polynomial in the full encoding including symbol labels.
[Family 128](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/docs/128.md)
and [Superstring configuration](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/ComparatorChallenges/Superstring.json)
describe available formalization. The guarantee is for the specified algorithm, not classical
maximum-overlap Greedy. Transfer: exact/certified-approximate/statistical/heuristic/unknown are
distinct guarantees; no semantic prompt-compression guarantee follows.

### October ETH manuscript and review status

[The Exponential Time Hypothesis is False](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/README.md)
is dated October 17, revised October 25 and announced October 26. It claims fixed-width CNF exact
counting in subexponential time and polynomial space. The
[family 378 Lean scope](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/lean/docs/378.md)
documents a weaker still-subexponential bound than the manuscript's displayed general bound.
The [history](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/history.md)
announces intended examination by an independent group, not a completed verdict. This establishes
neither `P = NP`, polynomial-time general SAT solving nor alignment decidability.

The formalization catalogue is not treated as exhaustive: SATComputability and MatchingFPRAS
have direct challenge artifacts and scope documents despite omissions from the inspected YAML
catalogue. Proof availability, compilation, statement adequacy, independent verification,
replication and peer review remain distinct evidence categories.
