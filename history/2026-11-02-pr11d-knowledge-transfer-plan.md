# Knowledge Transfer Probing implementation plan

Goal: bounded, default-off text-proxy protocol and saved-public-answer evaluation, separating
source access, target recall and application. No native multimodal training or empirical claim.
Authority: user master16.3/22/23 and standing ready-PR/review/merge instructions. Apply inline
superpowers workflow and approved repo-quality-gate fallback; no bd/.beads. One fresh final
whole-branch reviewer, no implementer delegation. Isolated snapshot copied after verifying all
21 prior-stage hashes; no local Git metadata. Main1d909436e86bd8a216f693cbe968e817dac592b8,
tree89cd95488e02990f038176796c704cb27af402a4 (merged PR815).

Source/spec: Li, Hong, Yang, Tang, arXiv2610.21459v1,2026-10-15, arXiv non-exclusive license.
Primary3.1–3.3/4.1–4.4 and AppendixA.1–A.3 checked. Author repository
kevinlllee/Knowledge-Transfer-Probing at218a274154dbadc2f4e679ebaf3741706dc1d5c0:
README and textual scoring script read. No explicit code/data license established; copy neither.
Primary studies image-acquired facts versus text QA. Our text-only protocol is an original
adaptation; no image generation, visual judge, learned model or knowledge-possession proof.

Approaches: native multimodal reproduction needs compatible backbones/operator budgets.
Callbacks for training would add unconfigured side effects. Choose independently supplied
fictional units, predeclared model/checkpoint/exposure conditions, public-only question projection
and deterministic saved-final-answer scoring. Reuse DiagnosticRecord/SystemConfig/CostEntry,
unchanged training guard, and existing paired estimator/ResamplingPolicy rather than new authorities.
Existing representation repair metrics are not learned-knowledge transfer measurements.

Interfaces: evaluation/knowledge_transfer_records.py defines versioned DEVELOPMENT records:
TransferUnit(unit_id,entity_id,attribute,cohort,value,source_question,target_question,
application_question,application_answer,evidence_refs), cohort alignment/test.
TransferCondition(condition_id,system,base_checkpoint_digest,checkpoint_digest,
acquisition_manifest_digest,acquired_unit_ids,alignment_manifest_digest,alignment_entities,
source_representation,target_representation,evidence_refs). All identities/exposures declared,
not authenticated; evaluator packets never become training data.
TransferObservation(condition_id,unit_id,repeat_id,channel,system,status,final_answer,
evidence_refs,costs=()), channel source/target/application, status answered/abstained/error.
Config TransferProbeConfig(enabled=False,dependencies_known=False,
resampling_policy=ResamplingPolicy(seed=0,resamples=1000,minimum_clusters=2)).

evaluation/knowledge_transfer.py: public_transfer_prompt(unit,channel,enabled=False)->str|None
returns only the authored question, never truth, cohort, IDs, hashes or references added by us.
evaluate_knowledge_transfer(units,conditions,observations,*,repeat_ids,baseline_condition_id,
evaluator_version,config=...)->dict|None validates all rosters before scoring. No callbacks/IO.
Shared declared base and representation pair; baseline checkpoint equals base with no exposures.
Acquired unit IDs may include test entities by design through separately admitted source facts;
target-QA alignment for test entities is forbidden, including entity/cohort aliases. Duplicated
entity/attribute units, mismatched model/harness systems, unknown/duplicate cells fail closed.

Expected source/target answer is unit.value; application truth is separately host-authored.
Normalize NFC/casefold, collapsed whitespace and terminal ASCII periods/U+3002 ideographic
full stops; preserve
internal punctuation/numbers. This stricter policy is explicit; never parse or score hidden CoT.
Reject literal value-token sequences in source/target questions. This guard does not prove
semantic isolation; application prompts may legitimately contain answer options and need host
audit. No generator can certify its own source or isolation. Public final-answer extraction,
source/checkpoint authenticity, admission and unseen aliases remain host responsibilities.

Metrics: every planned condition×unit×repeat×channel cell, explicit missing/abstained/error/wrong,
known-correct attainment over planned cells and conditional answered accuracy. Retain raw public
answers/sources/costs and per-attribute distributions. Source/target answered-pair contingency
and attainment gap never certify acquisition. Separate model/harness versions and raw exposures.
Pair nonbaseline versus baseline within cohort/channel using estimate_paired, entity clusters,
repeat averaging and complete expected roster; missing observations leave incomplete pairs.
Intervals default unavailable because dependencies_known=False; conditional entity bootstrap
requires explicit audited dependence declarations and enough clusters. No training-seed inference
or post-selection guarantee. Preserve negative effects. No scalar composite or promotion.

Bounds:64units,4conditions,8repeats,1536planned cells,16refs/8costs per record,512-byte questions,
256-byte values,4096-byte final answers,2MiB complete input and4MiB output. At most2000bootstrap
resamples; cost values at most1e12, aggregate only identical unit/basis/currency. These are original
CPU controls. No new dependency. All reports DEVELOPMENT/optimizer-ineligible/authority=false.

Task1: tests first for isolated public prompts, literal leakage/known-guard limitations, mixed
acquired source facts versus held-out QA alignment, matched systems/base, full missing rosters,
wrong/abstained/error behavior, normalization, negative paired effects and cluster/repeat semantics,
cost separation, strict serde and opt-in short circuit. Observe missing-module RED; implement two
modules; run GREEN existing eligibility/statistics/documentation contracts.
Task2: disabled config, source note/digest, protocol, existing REC-019 matrix/reference/registry/
reader traceability and installed-wheel resources; baseline119 checks already pass. No protected
manifest, Constitution, case registry, reward/verifier/admission policy or fixed Studies A–F edits.
Task3: syntax/Black/100columns/import checks, wheel and isolated default-off smoke, one independent
whole-branch review; verified RED/GREEN corrections; ready PR, babysit CI/reviews, merge clean head.

Review focus: truth/caption leakage versus explicitly limited literal checks; entity alias/cohort
overlap; malformed tuple restoration; missing-data denominators/paired inference; declared exposure
or self-reported answers mistaken for verified knowledge or permission. No actual model is trained.
Pre-flight: projection consumes the same exact unit records as evaluation; conditions bind each
observation to SystemConfig; CostEntry/paired estimator contracts reused unchanged.
Ruling: source-access success is evidence of answering a probe, not proof that a model acquired
or possesses knowledge. Misreading it would invalidate learning claims, so retain unverified scope.
Ruling: source training facts and sealed QA packets are distinct. This module never admits either
for training; any real acquisition must use existing host admission. A declaration is no authority.

Evidence: baseline119 checks passed. Missing-module RED preceded implementation; first14 transfer
controls passed. Initial spelling-alias fixture hit an existing whitespace guard; corrected to
case-only, then disabled the new guard temporarily to observe DID NOT RAISE RED and restored it
for GREEN. New alias guard prevents ambiguous NFC/case/space identities; semantic aliases remain
host audit work. A partial-cost regression reproduced RED (known subtotal mislabeled total);
unknown entries now leave total unavailable with a separate known_subtotal. Final145 related
checks passed in13.54s, including existing eligibility/statistics and source/doc registries.

Independent whole-branch review ran94 checks and20 malformed-array probes. No Critical/Important
findings; one Minor terminology mismatch: U+3002 was called fullwidth. Corrected docs to explicitly
name ASCII U+002E and IDEOGRAPHIC FULL STOP U+3002; U+FF0E remains significant. Direct behavior
check confirms the documented distinction. No scoring behavior changed or minor left deferred.
Black/Python3.10/100columns and new-file AST import-use checks pass. Rebuilt wheel40resources and
3runtimefiles byte-equal source; isolated config/defaultoff/registry/resource smoke passes without
NumPy. Wheel SHA256259827dd5b8338f1aa1cc1d08e022a424b979fcd82c78efe80ffc3d6de3bacfb.

Final: Ruling: semantic truth/isolation/application validity, unknown aliases, checkpoint/exposure
authenticity, source admission and public-final extraction remain independent host duties. The
cost of a failed duty is invalid research evidence, never a new authorization or reward claim.
Final: Ruling: knowledge possession/causal transfer/native multimodal effects remain Unsupported;
conditional entity statistics cannot establish independence or training-seed/post-selection
validity. Provided cost receipts cannot establish complete/authentic billing. Misreading these
limits would invalidate scientific or resource claims; no model executes here.
Final: Ruling: reviewer used retained source notes, not independent publisher/upstream retrieval;
implementer verified primary and pinned upstream definitions before code. No code/data license
established, nothing copied. Source mismatch would invalidate fidelity, so claim adaptation only.
Final: Ruling: local full Torch/Ruff/Python matrix and network-isolated build dependency install
were unavailable. Source-equal no-isolation wheel verification is separate evidence; final remote
CI must pass before merge. Ignoring this distinction would leave compatibility unverified.
