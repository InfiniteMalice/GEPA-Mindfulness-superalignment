# PR 5: verified knowledge and milestone preparation

Specification: master prompt sections 3, 8, 9, 17–23. Base PR 806 merge
bd275f54d989ae275f5058a050c8c730fa0d5738, tree 1d4c57da0723e5905a741cca42015c8d6f010aa2.
The operator-approved unavailable repo-quality-gate fallback continues: specification, design,
implementation, tests, final independent review and evidence. No beads commands or edits.

Design choice: extend existing EvidenceState/EvidenceClaim, RewardRequest, evaluator contracts,
ResearchCreditClaim/audit and host-owned review. A new reward engine or model-certified state
would compete with established authority; importing the robotics stack would add unrelated
runtime/dependency assumptions. Use bounded, task-local condition manifests and full observed
state transitions instead. These are experiment schemas, not a second canonical case registry.

Source check: KnowGPO arXiv:2610.22331v1 sections 3.2–3.3 and Algorithm 1; official repository
ZJU-REAL/KnowGPO revision 5b4f8bb5a782a09f4e9a0bcc630557c5deb7715d README and Apache-2.0
license (implementation advertised as coming soon). Paper license is arXiv non-exclusive.
PRM4RL arXiv:2610.22056v1 sections 4.1–4.2, Algorithm 1, NeurIPS 2025 proceedings DOI
10.52202/085713-3370; paper CC BY 4.0. Official deng-ai-lab/PRM4RL revision
f9727ed0d62aa7106b1dd8d76a52b5a6d6775079 README inspected. No code license identified in
that tree, and no upstream code is copied. Checked 2026-10-27. Method artifacts will retain
short mathematical/source notes and their exact local digests, not claim publisher-byte hashes.

Interfaces: KnowledgeCondition describes stable task/version/proposition/verifier/kind/dependency
identity. LearningStep binds an episode/event, exact RewardRequest, before/after EvidenceStates,
separate optional belief states, and continuing/terminated/truncated status. LearningContext
pins authority/run/task/version/model/harness/producer/evaluator/dataset and explicit gamma.
prepare_knowledge_progress accepts bounded tuples, snapshots all input, validates full episode
continuity, exact condition coverage, source eligibility and observable evidence membership,
then builds a complete envelope before a single mutation-safe host review. No callback runs
on invalid/oversized/held-out/self-reviewed input. Review rejection supplies no credit claims.

Knowledge counts only newly supported conditions never previously supported in that episode;
revocation and reacquisition are separate diagnostics, preventing a repeated support toggle
from minting new acquisition. Belief correction counts are distinct from evidence acquisition.
Groups use prior verified support/contradiction conditions, independent of wording or belief;
answer and terminal steps have no process comparison. Population normalization uses explicit
epsilon. No blended episode objective or optimizer is attached. This departs from monotonic
OR accumulation so contradictions can be preserved and corrected; it is an explicit adaptation.

Milestones form a checked DAG and only dependency-closed supported milestones count. Phi is the
fraction of completed milestones, fixed by the task manifest. F = gamma * Phi(next) - Phi(before)
retains signed arithmetic and a zero potential at genuine episode termination. Truncation yields
no shaping contribution: it is not treated as successful termination. The host must independently
confirm fixed bounded Markov potential, matching discount, terminal handling and evidence meaning.
There is no success bonus here, no policy-invariance claim for partial observations or unverified
assumptions, and no hidden-thought penalty. Negative shaping is an observable potential diagnostic,
not epistemic-process punishment.

Claims reuse the existing knowledge/progress kinds and same host-normalized semantic event ID.
Raw metrics and overlap audits stay separate; no sum or automatic resolution. Declared
admissibility remains unresolved even after evidence review. Existing independent admission and
durable event reservations remain necessary and unchanged. Outputs stay DEVELOPMENT and cannot
be relabeled into training. The package cannot authenticate principal identity or evidence truth.

Implementation sequence: CPU tests first for grouping/state changes, no duplicate acquisition,
correction, DAG/dependencies, telescoping/terminal/truncation, and evidence/authority/holdout
boundaries; bounded records and adapter; existing bibliography/matrix/guide/package data;
focused regressions, whole-suite attempt, wheel source recovery and one independent final review.

Pre-flight: condition manifests feed state validation, potential and group identity; one digest
pins all three. Evidence transition outputs feed ResearchCreditClaim and existing overlap audit;
the existing ledger requires later host-resolved admissibility and layer accounting. No new
reward pipeline or eligibility mechanism is introduced. Baseline: 1048 local source files matched
the merged Git tree; local git/network remain unavailable, so use isolated hash-verified source
snapshots and connector-backed branch/PR updates. PyTorch absence blocks full local collection.

Review focus: alias/replay and support toggles; continuity across interleaved episodes; contradictory
or stale sources; unresolved moral admission; terminal shortcuts; source-kind collisions; model or
reviewer mutation; full-envelope size before review; numerical normalization; reference integrity.

Task evidence: new adapter tests failed collection before the module existed, then passed.
The existing bibliography regression correctly rejected PRM4RL's 2025 publication year under
its all-2026 assumption. Extend its exact source-metadata exception with 2025, the known 2026
arXiv ID, NeurIPS venue and proceedings DOI; do not misdate the paper or weaken general checks.
Nineteen CPU contracts cover output-before-review budgets and existing event-reservation replay.

Final independent review: no Critical findings, four Important findings and no separate minors.
The single fix pass first ran five new regressions: all failed on the reviewed code. Identity
serialization exposed evaluator dictionaries and callback closures could change retained catalog
identity; return fresh identity dictionaries. Source-kind validation inspected only observable
references; inspect the entire recorded trajectory roster. Empty/oversized action IDs on zero-gain
steps bypassed claim validation; validate them before grouping. Epsilon was absent from retained
configuration; include it in the source and digest. These are fixes to the declared boundaries,
not changes to reward authority. No second reviewer or second implementation agent is used.

Ruling: principal authentication, evidence truth/relevance/freshness, semantic aliases and real
episode/action/termination identity remain the existing host's responsibility — labels alone
cannot authenticate the environment — an untrusted host cannot establish verified evidence.
Ruling: constitutional admissibility and runtime authorization remain unresolved/external —
evidence acceptance is not permission — no automatic optimizer or runtime admission is supplied.
Ruling: training effectiveness, protected behavior and Markov/discount assumptions remain
unqualified — no live experiment was authorized/configured — this stage establishes CPU software
contracts, not empirical improvement or policy invariance.
Ruling: primary definitions/licenses were retrieved by the implementer, not re-retrieved by the
independent reviewer — retained notes/digests and adaptation disclosures were reviewed — these
artifacts identify local checked notes and cannot be mistaken for publisher raw-byte evidence.
Ruling: full-suite/tensor acceptance requires CI because local PyTorch is unavailable — do not
install or fabricate a backend — local test success is explicitly limited to the selected suite.

Final fix evidence: five reviewer regressions failed on the reviewed implementation and passed
after the single fix pass. The final selected run passed 739 tests with 23 optional real-tensor
skips (existing SST, LoCD and DeLoop). The bare repository command was rerun and still stops
collecting tests/test_dapo_hybrid_trainer.py with ModuleNotFoundError: torch, after six skips.
No failing test was weakened or hidden. Six changed Python files pass Black, Python 3.10 syntax
and the 100-column limit. All 1214 checked Markdown targets, including every changed guide/reader,
resolve. Both retained method-note digests/byte counts match the matrix. A real setuptools wheel
bundles all expected resources byte-for-byte; isolated import/default-off adapter and resource
smokes pass outside source imports. The build frontend and local Ruff/mypy executables remain
unavailable; CI supplies the repository's full tool/runtime checks. No dependency was installed.

Only 20 intended files change, with no protected manifest, Constitution, trusted verifier,
training eligibility, runtime authorization or canonical reward-pipeline edits. Source snapshots
were verified against 1048 baseline files. Exactly these changes will be blob-hash checked
before advancing an isolated connector-backed branch. Open a draft PR for independent CI and
operator review; no merge, deployment, teacher/model call or empirical training occurred.

PR 807 review follow-up (2026-10-28): both CodeRabbit findings were verified against current
head f920fa0e25273ae2d168f42c27aa3dd3e1db124c. With five milestones, three before and four
after, gamma 0.75 gives an exact zero but float multiplication yielded 1.1102230246251565e-16
and a spurious progress claim. The new regression failed before the fix. Use exact rational
milestone counts and the represented gamma, converting only row values to floats. A neighboring
represented gamma retains a valid 8.881784197001253e-17 contribution; no tolerance is introduced.
The accounting guide linked from the installed knowledge/progress guide was absent in the actual
wheel. Its artifact assertion failed before the fix; add it to existing package data and the
existing wheel-resource contract. No prior entries or authority rules change.

Follow-up validation: 740 focused tests passed, with 23 existing real-tensor skips. Full local
collection remains blocked by missing PyTorch in tests/test_dapo_hybrid_trainer.py after six
skips. Three changed Python files pass Black, Python 3.10 syntax and 100-column checks. The
rebuilt wheel satisfies every existing and added resource assertion byte-for-byte; the accounting
guide is independently readable through importlib.resources outside the checkout. The maintainer
now authorizes ready PRs, CI/review monitoring, fixes until clean, then merge and continuation;
no failing check is bypassed and no empirical improvement is inferred from these tests.

PR 807 outside-diff review follow-up (2026-10-29): verified that gamma 5e-324 with one of two
milestones complete produces nonzero exact shaping but serializes to float zero. The regression
failed because no progress proposal was emitted. Retain exact shaping per step for the emission
decision, preserve the knowledge zero check, and keep float result/claim serialization unchanged.
The regression includes a following zero-progress episode so the decision cannot reuse another
step's shaping. Exact zero and truncation still emit no progress proposal. The focused adapter
suite passes all 26 tests after the fix. No reward authority or empirical claim changes.

Fresh follow-up validation: 792 focused tests passed with eight existing skips; the wheel-resource
test passed separately after restoring the missing build frontend. Both changed Python files pass
Black, Python 3.10 syntax and 100-column checks. Full local collection still stops at
tests/test_dapo_hybrid_trainer.py because PyTorch is missing (six preceding skips). A fresh
independent read-only review found no issues and independently passed the 26 adapter tests.

Further review verified a downstream contract limitation: the float-only claim score can round
nonzero shaping to zero, and an independently admitted zero score could still reserve its event.
The original underflow-proposal expectation is superseded: retain `shaping_exact` as a rational
string and mark underflowed rows `progress_diagnostic_only`; emit no reservable progress claim
until the credit contract can represent the amount. Exact per-step nonzero detection remains,
with no tolerance. Updated boundary regressions failed before the fix; they cover exact retention,
a following zero-progress episode, unaffected knowledge gain, and the smallest representable
positive progress score. No shared credit contract, ledger, authority or protected policy changes.

Final underflow-contract validation: 795 focused tests passed with eight existing skips, including
the wheel-resource contract. All 28 adapter tests pass. Changed Python files pass Black, Python
3.10 syntax and 100-column checks. Full local collection again stops at the missing PyTorch
import in tests/test_dapo_hybrid_trainer.py after six skips; fresh CI remains the full-suite gate.
