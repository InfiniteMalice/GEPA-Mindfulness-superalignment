# PR 1 continuation: reviewed credit allocation and durable consumption

Goal: persist host-reviewed allocation decisions once per semantic event without creating a
reward provider, optimizing diagnostic scores or replacing runtime authorization.

Design: add ResearchCreditUse as a versioned DEVELOPMENT proposal and ResearchCreditLedger as
a host-owned SQLite accounting journal. The ledger reserve method requires a real RewardRequest,
the original ResearchCreditClaim records, one selected claim per event, explicit not_counted
declarations for provider/reward_integrity/causal_reward layers, and a fresh independent host
callback over the complete snapshot. Reuse the causal adapter's mutation-safe callback helper,
request digest, eligibility checker and bounded JSON validation. No scalar reward is returned.

Claims must be bound to the exact target request through retained source_metadata_json and all
selected evidence must fall inside RewardRequest.observable_references. The host authenticates
producer/evaluator independence and semantic event identity. Unselected claims remain raw evidence.
Admitted does not imply deployed, true or authorized to act. Both diagnostic and allocation
reports remain DEVELOPMENT and cannot be replayed as live optimizer authority.

SQLite BEGIN IMMEDIATE plus a unique (run, task, event) key inside a pinned authority domain
prevents replay after restart and serializes competing consumers. Store a canonical immutable
review envelope and receipt. A reservation is conservative: if later training fails, the event
remains reserved. No automatic refund or serialized receipt reuse can issue additional credit.
The host must explicitly inspect recovery; this PR performs no optimizer or dataset mutation.

Alternative considered: adding scalars directly to RewardPipeline. Deferred because it would
conflate foundational accounting with method-specific objective semantics and overlap policy.
The ledger is accounting beneath the existing authority, not a parallel governance policy.

Execution: write failing CPU tests for missing module, exact serialization, default-off behavior,
request binding, private/teacher-only credit, host callback rejection/mutation/exception,
baseline-layer overlap, duplicate allocation, restart/replay, concurrent writers, rollback on
transaction failure, source restrictions and receipt immutability. Implement and run the focused
suite; perform independent whole-change review, fix verified findings, document evidence and open
a distinct draft PR. Also satisfy the prior review's docstring warning without weakening tests.

Constraints: existing schema versions, canonical cases, Constitution, protected suites, reward
scales and runtime grants remain unchanged. Use the previously approved repo-quality-gate
fallback. Source snapshot and local tool limitations from PR #802 remain applicable.

Evidence: initial test import failed because the new module did not exist. The implemented ledger
then passed its 15-test lane plus the existing 23 claim tests. Independent whole-change review
found no important defects; it requested current scope documentation and committed corruption/
multi-event coverage. Those probes are now committed tests, along with an unchanged RewardPipeline
check and foreign-identity rejection. The combined lane passes 42 tests. The old docstring warning
is addressed without changing test assertions. The Constitution, manifests, datasets, reward
providers, runtime authority and prior schema versions are unchanged.

Final checks: all four touched Python files pass Black 24.10.0's sequential formatting API,
100-column checks, Python 3.10 AST parsing and function-docstring presence checks. Executable
ASTs of the previous claim module and tests match their merged versions after removing only
docstrings. One isort known-first-party entry records the shared test fixture import; it adds
no dependency. PR #802's complete CI is now successful on Python 3.10 and 3.12. That result is
upstream baseline evidence, not verification of this continuation's new journal.

Ruling: reservation is permanent conservative accounting, not proof of successful optimizer
consumption. This prevents automatic retry double credit; the cost is operator recovery if a
later training operation fails. No automatic recovery or refund authority is introduced.

Ruling: retain the diagnostic source-kind warning for compatibility and reject contradictory
capture kinds at reservation intake. Cross-batch alias/capture semantics still require host
verification. The cost of an incorrect host equivalence decision remains false duplicate credit
or over-conservative rejection; no automated semantic equivalence claim is made.
