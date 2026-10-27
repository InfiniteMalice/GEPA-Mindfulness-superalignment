# PR 2: source-checked ActionOPD and localized distillation adapters

Goal: implement bounded optional numerical/data adapters while preserving teacher, student and
verification separation. No tool execution, API spending or default training changes are included.

Primary definitions checked on 2026-10-27: ActionOPD arXiv:2610.19820v1, sections 2/A.2–A.4,
official thu-nics/ActionOPD revision d447bdf2ad3158ae7c315cb96ce0a180f94459b1 (Apache-2.0);
LoCD arXiv:2610.21427v1 section 4, official zz-haooo/LoCD revision
33d950954f14c0f4e087274ae9365c7f2d2cbd80 (MIT). No upstream code is copied. Paper licenses
are CC BY 4.0. These are adaptations, not reproductions of benchmark training runs.

ActionOPD: typed semantic actions use canonical JSON; teacher samples retain model/version,
configuration and exact public-prefix identity. Invalid student actions never match. Repeated
teacher samples retain multiplicity. Empty teacher sets supply no labels. Implement the published
frequency-bonus reward and shared-batch whitening math. Exact canonical matching is a deliberately
bounded replacement for upstream LLM semantic matching. Independent outcome records remain separate
and cannot be inferred from agreement. Output is DEVELOPMENT and grants no optimizer/action authority.

LoCD: prepare one original-prefix/local-continuation instance per verified correction region using
CorrectionProposal and CoevolutionStore.correction_source_events. Keep trusted teacher context
separate from student inputs; mask all original prefix tokens. Preserve original trajectory and
source records. Implement a lazy-PyTorch forward-KL segment loss with detached teacher targets.
This is the forward-KL numerical path only; no full self-distillation trainer, divergence clipping,
same-model teacher requirement, source tokenizer or global loss merger is supplied.

Sequence: write failing deterministic CPU tests; implement action scoring/whitening and local
prefix/mask preparation plus loss; run real imports and numerical tests; extend the existing strict
reference registry/reader and their exact identity tests with the two verified sources; update
applicability/maturity honestly; independent whole-change review; publish one reviewable draft PR.

Review focus: different student states, teacher provenance mutation, unverified reward promotion,
unauthorized attractive shortcuts, private context leakage, whole-trajectory imitation, wrong loss
mask, teacher gradient flow, no-supervision groups, and changed canonical manifests. The operator's
approved quality-gate fallback continues. Hardware effectiveness and model training remain untested.

Evidence: initial tests failed for the missing adapters. Subsequent red tests exposed missing
source-registry entries, insufficient public-state fingerprinting and late aggregate-input bounds;
all were corrected. Primary Appendix A.3 checking led to explicit nominal teacher-count handling
for partial retention and no supervision when all teacher samples are invalid.

Pinned pytest 8.4.2 (bfae4224fd554d3d7f2c277a4cc092b6ec6af3ae) and Pluggy 1.6.0
(fd08ab5f811a9b2fa9124ae8cbbd393221151e2c) sources enabled local compatibility testing outside
the repository. The omitted original historical design link fixture was restored unchanged for
existing registry tests; it was not reviewed or edited as a prior plan. Bibliography identity
oracles were extended exactly for the two verified sources, preserving all existing assertions.

Independent review confirmed the limited-scope numerical definitions and found a masked-prefix
float32 overflow/backward risk by inspection. The loss now selects active positions before
softmax and preserves float64; a real-PyTorch regression is present but skipped locally. A negative
source-token regression failed before validation was added. The new guide is included in wheel
package data. The focused suite passes 365 tests with four explicit PyTorch skips; no fake tensor
backend was used. Full CI and actual model experiments remain separate evidence.

Ruling: all prepared records remain DEVELOPMENT, and numerical helpers have no automatic trainer
attachment. This keeps independent admission outside teacher agreement; the cost is that operator
integration work is still required before these adapters can train a model.

Ruling: exact canonical matching and exact-length token regions are bounded adaptations; there
is no claim of arbitrary semantic equivalence, automatic mechanistic localization or full upstream
reproduction. The cost is narrower coverage until independently validated adapters are added.
