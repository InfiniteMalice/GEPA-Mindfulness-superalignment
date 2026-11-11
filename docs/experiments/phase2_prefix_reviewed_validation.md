# Reviewed publication qualification gate

Maturity: **Experimental** software; **Diagnostic**, `DEVELOPMENT` evidence. No optimizer input,
parameter update, promotion authority or trained-model alignment claim. This supersedes the
software gate of the preceding prefix snapshot without changing or removing its evidence.

## Review and one fix pass

A fresh read-only reviewer found one Important issue: nested original comparison commitments
could not roundtrip when v1 references were compared with the new v2 factory. The review was
static: PyTorch initialization required a writable temporary directory, so its attempted runtime
reproduction was blocked. The reviewer did not verify new probe measurements or final reports.
The root executor independently reproduced the precise failure against the original gzip probe,
then fixed version-aware source binding. All six historical commitments now roundtrip exactly,
but new qualified execution rejects their missing prefix metadata before model execution.

Additional remote Minor findings were fixed: specific errors for short nonempty v2 prefix arrays,
generation/locked geometry and controls mismatch; accurate default parameter-budget documentation;
faithful-report denominator documentation; adjacent manifest snapshot identities. Actual legacy
and geometry RED outputs and the later GREEN suite are retained. Source/version hashes, actual
measurements and software checks are independently checked by deterministic tools; the reviewer's
language does not confer verification authority.

The reviewer accepted these explicit limits: hashes do not authenticate source truth, custody,
semantic independence or actual researcher independence; retrospective discovery may use shared
BOS/future context; causal simulation rejects future features; standard large pretrained tracing
and scalable external evidence storage remain unsupported; timestamp/cost identity is distinct
from numerical reproduction; no trained alignment or semantic validity is claimed.

## Final software gate

- **7,338 passed**, 11 skipped, 18 warnings, 162 subtests, 585.42 seconds, offline CPU.
- **120 affected tests passed**, 96.46 seconds, including original nested v1 records.
- Repository Ruff and Black pass; Black leaves 866 files unchanged. Both protected CI mypy commands
  passed; supplemental six-module checks pass with the repository's `MYPYPATH=src`.
- Original stdout, compressed JUnit, RED/GREEN and static-check logs are retained. Missing CUDA
  hardware and Mojo account for the recorded skips; those backends are not represented as tested.
- Canonical 17-case manifest, coevolution/learning-surface and reward-provenance owners remain
  byte-identical to the frozen baseline. The Constitution retains exactly 21 numbered sections.

## Retained reruns

Four `phase2_prefix_reviewed_prN_probe.json.gz` files retain the reviewed source hashes, exact model
identities/seeds, inputs, measurements and actual timestamps/costs. They are repetitions of the
same synthetic protocols, not independent alignment samples. All declared source hashes match
current source. The preceding prefix snapshot remains available under its original filenames.

The discovery metrics for 24 hypotheses, simulation's 684 per-position output rows, comparison's
1,104 per-position output rows, and stage-7 baseline behavior structure match the preceding
prefix-qualified snapshot exactly. Prediction criteria, null controls and falsified hypotheses
remain as recorded in `phase2_prefix_results.json`: discovery 23 falsified/one numeric criterion
met; comparison 14 prediction criteria met/eight falsified and 19 causal prediction rows met,
41 falsified, six nonspecific. Matching rows establish numerical replay under this environment,
not semantic explanation, causal admission, statistical independence or alignment improvement.

Historical stage manifests retain their original bytes; `phase2_manifest_snapshots.json` binds
them to original full source revisions and initial published mirrors. The original-history bundle
and source pins remain available. The reviewed manifest separately hashes current source, tests,
reports and measurements. Host custody, independent research verification and governed promotion
remain separate boundaries. The ordinary inference path has no prerequisite on these tools.

Remote current-head checks and PR timelines determine merge readiness. This software report does
not substitute for a protected trained-model capability comparison; those experiments remain
`NOT_RUN`, and general alignment improvement remains `NOT_ESTABLISHED`.
