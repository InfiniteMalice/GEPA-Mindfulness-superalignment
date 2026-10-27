# DeLoop + PEO experiment: design and implementation plan

Goal: add a disabled-by-default repeated-action audit, independently reviewed recovery-pair proposals and a separate DPO numerical objective. The master specification sections 3, 7, 18–23 govern this bounded PR 4 stage. Base: `f3080ff7aea24e610eec2e814eebf24d5f2f7555`, merged PR 805.

Source: Chen et al., arXiv:2610.19376v1, sections 3.1, 4.1–4.2 and Appendix G.1, checked 2026-10-27. The primary text states code/data will be released; no authoritative implementation was identified or copied. Paper license CC BY 4.0. The retained, explicitly scoped web-reader equation extract is `docs/recommendations/source_artifacts/deloop-2610.19376v1-equations.txt`; its digest is recorded in the integration matrix. Raw publisher bytes are not available through the shell proxy.

Approaches: a blanket duplicate blocker cannot know whether a new observation will arrive and risks suppressing required verification. Generic repetition penalties have the same problem. Select a trace-bound diagnostic ledger plus reviewed pair preparation; leave runtime blocking and GRPO penalties for separately validated experiments. No action is executed by this module.

Architecture: `training/deloop.py` reuses RewardRequest, ActionOPD SemanticAction, immutable EventEnvelope and the existing complete PEO sequence validator. Explicit step metadata carries environment version and exemptions with observable evidence. A pre-action prediction binds the public-state/action/context fingerprint. Completed action/observation signatures detect exact structural repeats; new observations, state changes and necessary/human-requested repeats remain distinct. Audit records are DEVELOPMENT-only and report case-17 diagnostic subtypes only for clarification actions; no canonical classification or manifest is changed.

The miner considers the first later structurally non-looping, different action in the same verified trace, preserving the rejected step's actual public prefix. Independent host review must bind the exact proposal digest, use an existing evaluator identity distinct from the generating model, confirm unproductive repetition and original-state applicability, and provide existing local/relational verification results. Missing, conflicting, unsafe, unsupported or premature-ending recovery evidence yields no pair; a failed first recovery is not replaced by a more favorable future action. Outputs remain DEVELOPMENT proposals, not optimizer-admitted data or runtime permission. DPO loss is a separate lazy-PyTorch kernel with detached reference log probabilities and stable softplus, not a new reward authority or trainer.

Implementation sequence, inline under the approved missing-repo-quality-gate fallback:

1. Write CPU tests first for default-off behavior, typed snapshots, exact action/observation repeats, changed observations/state, explicit recheck/verification exemptions, answered clarifications, complete causal binding, bounds, raw-versus-verified separation and unchanged canonical taxonomy. Implement the audit in the new module, reusing existing contracts.
2. Write review/pair tests for original-prefix preservation, first-recovery selection, self-review rejection, digest/action/evidence mismatches, unknown judgments, premature termination, required-verification bypass and no silent training eligibility. Implement the miner and separate DPO kernel; real tensor checks skip explicitly if unavailable.
3. Extend the existing REF-DELOOP bibliography, reciprocal recommendations and strict inventory; document source adaptations, experimental maturity, serialization, source digest, scope limits and matched ablation protocols. Run focused regressions and attempt full collection. Use one independent final review before an isolated draft PR.

Review focus: omission/reordering of executed steps must not change which recovery is first; equality of evidence IDs cannot prove unchanged observations; exemptions must remain source-bound rather than self-reported escape hatches; privileged later state must never become the rejected step's prompt; review records must not authorize rewards or actions.

Pre-flight: the miner consumes the same validated audit/source snapshots rather than accepting a caller-written ledger; the DPO kernel receives already-scored action log probabilities rather than hidden reasoning. Existing reward-integrity flat-pair rules are closed and unrelated; do not extend them or invent a new case registry.

Ruling: use a hash-verified isolated source snapshot and GitHub connector commits because this environment has no usable git checkout or shell network proxy. Baseline hashes match 1,043 available files. PyTorch remains unavailable locally; tensor execution requires CI. This limits local numerical evidence and does not justify hardware or effectiveness claims. Canonical trainer attachment, live blockers, GRPO penalties and behavioral ablations remain unexecuted.

Implementation evidence: the missing module and missing overlay flag failed before implementation. Focused negative tests exposed action-class mismatch, contradictory loop findings, unrelated outcome-support evidence, malformed review result types, and optional-runtime import before coefficient validation; each was corrected and passed locally. Existing PEO payload contracts required immutable tuple-reference normalization and unique verification-record IDs in authored fixtures; no existing validator was weakened. The overlay and bibliography inventories were extended exactly, preserving prior entries and strict validation.

Independent final review: original-prefix preservation, first-recovery selection, complete executed-step coverage, conservative exemptions and DEVELOPMENT boundaries were inspected. One P2 evidence-integrity defect was reproduced: an evidence ID could be reused across steps with a different source kind and overwrite the earlier reference. The new regression failed by reaching the reviewer before the fix; source capture now rejects conflicting typed identities before review. No further blocking defect was reported. The DPO tensor tests remain unexecuted locally and require CI; the review did not qualify training effectiveness.

Ruling: the existing experimental overlay registry is extended with one default-off diagnostic declaration; no canonical case or robustness-stripe definition changes. The declaration does not execute the adapter, authorize an action, admit a dataset or attach an optimizer. This preserves established overlay ownership at the cost of requiring explicit host invocation alongside the declaration.

Final evidence: 500 focused tests passed; 23 real-tensor tests skipped (three new DPO tests and 20 existing SST/LoCD tests). The DPO kernel also rejects a nonfinite intermediate margin rather than allowing softplus saturation to hide it; its numerical regression requires CI. Full local collection stopped at `tests/test_dapo_hybrid_trainer.py` because PyTorch is absent, after six skips. Six touched Python files pass Black, Python 3.10 syntax and the 100-column limit. The 875 relative Markdown links resolve, the source-extract digest/byte count match the matrix, and the guide is included in package data. Exactly 20 intended files changed; the canonical case/stripe manifests, Constitution, verification policy and reward eligibility are unchanged. Only the existing experimental overlay declaration catalog is extended. No external training, API spending, live action, deployment or merge occurred.

Review follow-up (2026-10-27): both supplied findings were verified against PR 806 head
`ed64be515706cf528264d241c52f982935c99018`. The approved quality-gate fallback remains in
use. Reserve the full candidate plus the maximum valid serialized LoopPairReview under the
existing unique-reference/binding and fixed-subject contracts before invoking the host reviewer.
Check accepted and rejected output routes including prior records; retain final envelope
validation and fail closed on insufficient space. This may reject a candidate whose smaller
actual review would fit. The sizing shape has no evidence or admission authority. Add only the
guide's missing matrix and DeLoop extract to existing package data, and extend the existing
wheel-resource contract.

Regression evidence: both new capacity tests first failed because over-budget entries reached
the callback (including the second entry after a completed review). Both now pass. A real
setuptools wheel first failed the missing-resource assertion; the corrected wheel bundles both
records byte-for-byte and an isolated importlib.resources reader can open them without the
checkout. No build isolation or network access was required. The optional CodeRabbit CLI is
not installed. Full local collection still stops at missing PyTorch in
`tests/test_dapo_hybrid_trainer.py`; no full-suite success is claimed.

Follow-up validation: 502 focused tests passed, with 23 optional real-tensor skips. The three
changed Python files pass Black, Python 3.10 syntax and the 100-column limit. The standalone
wheel build and isolated installed-resource smoke check passed; the existing wheel pytest
requires the unavailable build frontend, so its assertions were additionally checked against
the real wheel directly. PR 806's pre-fix CI run 40526338937 succeeded; fresh CI is required
for this follow-up. Changes are limited to the miner, its regressions, package data, the existing
wheel-resource assertions, the guide and this evidence record.
