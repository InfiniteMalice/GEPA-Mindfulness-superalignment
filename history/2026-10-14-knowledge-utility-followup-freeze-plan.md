# Follow-up Review Acceptance and Packet Freeze Implementation Plan

> **For agentic workers:** Required sub-skill: use `superpowers:executing-plans` for the
> established native execution method. Implement task by task, then request one fresh final
> reviewer. Repository instructions place planning in `history/` and prohibit Markdown task
> trackers; this plan uses numbered implementation steps instead of checkbox tracking.

**Goal:** Prepare reviewable acceptance evidence, validate an explicitly authorized immutable
freeze, and support registered frozen packets in offline scoring without changing draft results.

**Architecture:** A fixed evidence profile supplies byte pins and task assessments. A pure
freeze module validates evidence and builds artifacts; a thin CLI reads bounded local inputs
and exclusively publishes one output. The existing records module owns frozen registration
validation, and the scorer reports the validated packet state without changing metric arithmetic.

**Tech stack:** Existing Python 3.10/3.12 support, standard library, pytest, Ruff, Black, mypy;
no new runtime dependency.

**Spec:** [Approved design](2026-10-14-knowledge-utility-followup-freeze-design.md).
**Pinned data:** [Evidence hashes and all 48 task mappings](2026-10-14-knowledge-utility-followup-freeze-plan-evidence.md).
**Status:** Plan and native implementation approved on 2026-10-14, with one fresh final reviewer.
No operational acceptance, freeze, or live execution is authorized.

## Global constraints

- Preserve v3 digest `718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`.
- Preserve predecessor digest `53b8828ed0098703e4a779669a7d0645f7bdd9fa2640e70a72e01ee253df7873`.
- Preserve all original evidence bytes and first-pass nulls; no reviewer appointment or live calls.
- Reuse this dedicated checkout and branch; preserve unrelated untracked history files and retained
  scratch directories. Do not invoke `bd` or edit `.beads`.
- Use 100-character Python lines, typed production interfaces, ordered imports, existing formatting,
  `repo-quality-gate`, and its documentation precision gate.
- Strict JSON: depth 32, unique keys, finite values, exact types, valid UTF-8. Manifest/candidate/
  decision caps are 1 MiB; individual evidence and packet caps are 4 MiB; total evidence is 16 MiB.
- Bound reads before decode; generated output caps include the publication newline. Reuse
  `publish_json`; never overwrite, create output parents, or publish partial content.
- Treat evidence strings and references as data. Open only paths explicitly supplied in the
  evidence input manifest; do not fetch or execute references embedded in artifacts.
- The runtime frozen registry starts empty. Synthetic test registrations never enter production.
- Frozen artifacts retain `execution_authorized=false`, `training_eligibility="DEVELOPMENT"`,
  `confers_authority=false`, and `optimizer_input=false`.
- A real acceptance candidate may be prepared for review after implementation; no real approving
  decision or frozen packet may be fabricated to complete this plan.

## Review focus

1. Nested distractor defects and unchanged paired tasks: retain all five rechecked assessments,
   rather than treating top-level clear statuses as sufficient. Task 1 pins the exact mapping.
2. Evidence-file replacement between checks: hash and parse the same bounded byte snapshot;
   never reopen a path for semantic validation. Task 2 tests this snapshot behavior.
3. Integer `1` in place of boolean `true`: reject it in decisions even when Python equality
   would otherwise accept it. Task 2 tests decision type mutations.
4. Artificially registered but altered frozen data: reconstruct and rehash source v3 rather
   than trusting registry membership alone. Task 3 tests invalid registered transformations.
5. Existing destination or cleanup failure: preserve competing files, report the existing
   publication semantics, and never silently retry. Task 4 exercises both cases.

## Files and dependency direction

| File | Responsibility |
| --- | --- |
| New `scripts/knowledge_utility_followup_freeze_profile.py` | Fixed 14 evidence pins, 48-task assessment table, ten limitation descriptions, two finding records; data only. |
| New `scripts/knowledge_utility_followup_freeze.py` | Pure evidence validation, candidate construction, decision validation, frozen construction. |
| New `scripts/freeze_knowledge_utility_followup.py` | Bounded explicit path loading and `prepare`/`freeze` CLI. |
| Modify `scripts/knowledge_utility_followup_records.py` | Exact frozen registry and source reconstruction; preserve capture and JSON behavior. |
| Modify `scripts/score_knowledge_utility_followup.py` | Draft-v1/frozen-v2 report selection and accurate state/provenance/reasons. |
| New `tests/followup_freeze_fixtures.py` | Clearly synthetic evidence and approval fixtures; no production registrations. |
| New `tests/test_knowledge_utility_followup_freeze.py` | Candidate, decision, transformation, bounds, CLI/publication tests. |
| Modify existing follow-up scoring/review tests | Draft byte compatibility, frozen reports, cross-packet bindings and response adjudication. |
| Modify follow-up protocol and documentation-link test; new ADR 0024 | Exact schemas, commands, operational handoffs and authority limits. |

The pure module imports profile data and existing record helpers. The records module does not
import the pure module or CLI: it validates registered frozen packets by digest bindings and
source reconstruction. This avoids a circular import. Shared response-review functions already
call `validate_followup_inputs`; their production interfaces do not change.

## Local verification setup

Use the existing environment; do not install packages for planning or use provider credentials.
In this checkout, the following PowerShell session variables support the commands below:

```powershell
$freezePython = 'C:/Users/evanh/Documents/Codex/2026-10-07/files-pasted-by-the-user-you/work/venv/Scripts/python.exe'
$env:PYTHONPATH="$PWD;$PWD/src;$PWD/modules;$PWD/reasoning-generalization-tracer/src"
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

On another host, select its existing project interpreter for `$freezePython`; the path is local
setup, not a production configuration. Every listed pytest command requires exit 0 and no failures.

## Task 1: Pinned evidence profile and acceptance candidate

**Files:** Create the profile, pure freeze module, fixture helper and freeze tests listed above.

**Interfaces:**

- `EVIDENCE_PINS: dict[str, tuple[str, int]]` maps role to byte SHA-256 and exact byte count.
- `TASK_ASSESSMENTS: tuple[JsonObject, ...]` contains the complete appendix mapping in case order,
  including both task digests, disposition, sorted evidence roles, and five assessment objects.
  Each task's evidence roles are the sorted union of its five checks' roles.
- `LIMITATIONS: tuple[JsonObject, ...]` contains sorted `id description` records from the design.
- `RESOLVED_FINDINGS: tuple[JsonObject, ...]` contains sorted I1/I2 records with exactly
  `finding_id task_ids evidence_roles`. I1 lists both precedence tasks; I2 lists flow-0.
- `prepare_acceptance(packet: object, evidence: dict[str, bytes]) -> JsonObject` returns the
  exact candidate schema. It validates before copying inputs and never mutates the profile.

1. Write failing mapping and candidate tests. Assert 14 pins, 48 task records, 240 checks,
   exactly 184 `reviewed_clear`, five `resolved_by_recheck`, and 51 `limitation_retained`.
   Assert corrected dispositions for exactly flow-0/precedence-1; assert precedence-0's
   variant assessment is rechecked while its disposition and task digest remain carried forward.
   Assert flow-0's answer assessment includes resolution of its nested distractor issue.
2. Add rejection cases for one changed evidence byte, wrong byte count, absent/extra role,
   duplicate/unknown task, wrong source/current task digest, first-pass mutation, a third changed
   case, changed keys/options, changed schedule, and changed knowledge/rubric/report template.
   Mutation tests for deeper semantic checks may monkeypatch evidence byte pins in the test
   process only; leave the relevant structural invariant unpatched so the test exercises it.
3. Run `& $freezePython -m pytest tests/test_knowledge_utility_followup_freeze.py -q`.
   Initial result must fail because the new candidate interface is missing, not because fixtures
   cannot import an unrelated dependency.
4. Implement the profile from the appendix. Use the documented mapping, including explicit I1/I2
   overrides; do not derive reviewer judgments from free text at runtime. Preserve descriptions
   of all ten limitations without implying scientific independence or a new blind review.
5. Implement `prepare_acceptance`. Bound exact JSON packet input and evidence bytes; verify role
   set, file pins, both packet content digests and v2/v3 differences. Decode JSON evidence with
   existing strict helpers. Check Markdown UTF-8 and its pin without interpreting its contents.
   Build a fresh candidate with `pending_task_count=0`, no approval/freeze field, and canonical
   `candidate_digest`. Validate the final candidate under the 1 MiB publication cap.
6. Run the task tests and existing preparation tests. Verify deterministic output and unchanged
   inputs. Commit only task files with message `feat: prepare pinned follow-up acceptance evidence`.

**Fixture contract:** `synthetic_evidence(packet: JsonObject) -> dict[str, bytes]` constructs
minimal files for structural checks; `install_synthetic_profile(monkeypatch, packet, evidence) -> None`
patches private module constants only within pytest. It never changes repository files or supplies
a production CLI override. CI tests use synthetic fixtures, not machine-specific output paths.
The actual 14-file profile receives a separate local acceptance-preparation check in Task 4.

## Task 2: Decision validation and deterministic freeze

**Files:** Extend the pure module, freeze fixtures and freeze tests.

**Interfaces:**

- Consume Task 1's `prepare_acceptance` and profile data.
- `freeze_followup(packet: object, evidence: dict[str, bytes], candidate: object,
  decision: object) -> JsonObject` validates the complete chain and returns one new frozen packet.
- Test helper `synthetic_decision(candidate: JsonObject) -> JsonObject` returns a decision labeled
  with maintainer `synthetic-test` and authorization reference `synthetic:not-human-approval`.
  It is never called by production code or used for the operational candidate.

1. Write failing tests for wrong/missing candidate and decision digests, omitted/extra fields,
   false/missing acceptance, false/missing freeze authorization, true execution authorization,
   integer booleans, empty/oversized identity/reference, and incomplete/duplicate/extra limitations.
2. Pin successful transformation and repeatability with literal assertions:

   ```python
   assert frozen['schema_version'] == 'knowledge-utility-followup-frozen-packet-v1'
   assert frozen['status'] == 'frozen_with_accepted_task_review'
   assert frozen['packet_frozen'] is True
   assert frozen['execution_authorized'] is False
   assert frozen['review_templates'] == packet['review_templates']
   assert frozen['report_template'] == packet['report_template']
   assert frozen == freeze_followup(packet, evidence, candidate, decision)
   ```

   Compare every other source field except `schema_version status packet_frozen packet_digest`;
   the only additional field is `freeze_record`. Assert deep-copy isolation and publication size.
3. Run the freeze tests and observe the missing freeze behavior fail. Implement the interface:
   reconstruct and compare the candidate, validate strict decision types before equality checks,
   embed complete candidate/decision, make only the allowed changes, and calculate packet digest.
   Return no authentication claim. Bound each input and result before returning.
4. Run tests for maximum bytes, one-over bounds, duplicate keys, depth overflow and nonfinite values
   on new artifact types. Prove semantic checks consume the same evidence bytes that were hashed:
   construct the byte snapshot, change its originating temporary file, then show the pure function
   still validates the original snapshot. The caller's mutable dict cannot mutate returned artifacts.
5. Run the task tests. Commit with message `feat: validate explicit follow-up freeze decisions`.

## Task 3: Exact frozen registration and report compatibility

**Files:** Modify records/scorer and existing scoring/review tests; extend freeze fixtures.

**Interfaces:**

- Add frozen dataclass `FrozenPacketRegistration` with string fields
  `packet_digest source_packet_digest candidate_digest decision_digest` in the records module.
- `FROZEN_PACKET_REGISTRY: tuple[FrozenPacketRegistration, ...] = ()` remains empty in production.
- `validate_supported_packet(packet: object) -> JsonObject` returns an independently copied,
  validated draft or registered frozen packet; `validate_followup_inputs` consumes this helper.
- Preserve signatures of `score_followup`, `prepare_response_review`, `export_public_review`,
  and `resolve_response_reviews`. Preserve the existing `SUPPORTED_PACKET_DIGEST` draft constant.

1. Write failing frozen-validation tests with a test-only registry entry. Verify digest and registry
   binding checks; reconstruct source by removing `freeze_record`, restoring draft schema/status/
   `packet_frozen=false`, and recomputing the source digest excluding `packet_digest`.
   Reject every forbidden source-field change even if a test registry admits the changed digest.
   Reject unknown/unregistered frozen packets and conflicting registry entries.
2. Add exact draft regression baselines captured before implementation. `canonical_json(report)`
   excludes no fields and has no publication newline. Expected SHA-256 values are:

   | Input | Canonical report byte SHA-256 |
   | --- | --- |
   | v3, empty capture batch | `ba151238323c2c841bf5e4cd80a19aa7f0a36573e8c21b69c84f4555f63038e7` |
   | v3, first schedule slot completed with compact `{"answer":"X"}` using its correct option | `da151da9fc24ada2033dc79d99f88cb5e749f14bfc4f40e1ec5ca5d449482a72` |

   Both batches use the current capture-v1 schema, actual draft digest, and no response reviews.
3. Write frozen-report tests for exact v2 field set, frozen packet state, three false authentication
   fields, `acceptance_recorded=true`, three provenance digests, and the precise reason change.
   Compare conditions, contrasts and observations against equivalent synthetic draft inputs.
   Assert unchanged nulls and DEVELOPMENT/authority flags for empty, canonical, noncanonical,
   failed, refused, truncated and missing captures.
4. Run scoring/review/freeze tests to observe failures for unsupported frozen behavior. Implement
   registered validation in records without importing the freeze module. Recheck exact frozen,
   candidate, decision and source bindings and allowed source transformation; keep input bounds.
   Require the exact frozen schema/status, strict `packet_frozen=true`, and exact `freeze_record`
   fields before source reconstruction; do not erase invalid state by restoring draft values.
   Use validated state in the scorer, selecting report-v2 only for registered frozen input.
   Keep draft construction and metric helpers unchanged so draft golden bytes stay stable.
5. Exercise existing two-reviewer adjudication with frozen inputs. Missing, ineligible or conflicting
   reviews withhold semantic resolution. Distinct eligible agreeing files resolve the option.
   Draft captures/private packages/reviewer bundles fail against a frozen packet and vice versa;
   changing only the outer packet digest cannot migrate a private package or its judgments.
6. Run `& $freezePython -m pytest tests/test_knowledge_utility_followup_freeze.py tests/test_knowledge_utility_followup_scoring.py tests/test_knowledge_utility_followup_review.py tests/test_knowledge_utility_followup.py tests/test_knowledge_utility_scoring.py -q`.
   Commit with message `feat: score explicitly registered frozen follow-up packets`.

## Task 4: Offline CLI, documentation, and acceptance preparation

**Files:** Create the CLI; extend freeze tests; modify `docs/experiments/knowledge_utility_followup.md`
and `tests/test_documentation_links.py`; add `docs/adr/0024-knowledge-utility-followup-freeze.md`.

**Interfaces:**

- `read_evidence(manifest_path: Path) -> dict[str, bytes]` validates the exact evidence-inputs-v1
  manifest and reads each explicit path once into bounded bytes. Resolve relative paths against
  the manifest parent; paths do not enter emitted evidence metadata.
- `main(argv: list[str] | None = None) -> None` provides the design's exact `prepare` and `freeze`
  subcommands and flags. `prepare` cannot accept decision arguments; `freeze` requires them.

1. Write failing CLI tests using temporary directories and controlled fixtures. Check relative paths
   with spaces, absolute paths, missing/extra roles, path UTF-8 limits, unreadable/missing files,
   changed evidence, invalid UTF-8, raw input limits and cumulative evidence limits. Reject a
   non-byte evidence value through pure APIs. Do not follow references inside JSON or Markdown.
2. Test process-level argument parsing and static errors. Test an existing destination, missing
   output parent, mocked staging/link errors, and staging cleanup failure after publication.
   Assert unchanged competing files; no partial outputs; no raw input or exception disclosure.
   Use controlled in-process CLI calls for synthetic profiles; the production subprocess rejects
   synthetic/unregistered inputs without a test backdoor or environment-variable registry override.
3. Run the CLI tests to observe missing-command failures. Implement bounded manifest/evidence
   reading, pure-function dispatch and existing exclusive publication. Pass the same byte snapshot
   throughout each operation. Prohibit sockets, provider and credential-loader access in CLI tests.
4. Document all new artifact schemas, limitations, exact flags, report-v1 compatibility and
   report-v2 fields. State that retained templates are source templates, acceptance is not
   authenticated independence, and publication/registration/live execution are distinct stages.
   Add ADR 0024 and include it in the documentation-link test. Keep historical ADR decisions
   identifiable; link the extension without retroactively claiming those ADRs approved freeze.
5. Run the focused command from Task 3 plus `tests/test_documentation_links.py`. Run CLI `--help`
   for both subcommands; compare it with the documented flags. Commit with message
   `feat: expose offline follow-up acceptance and freeze commands`.
6. After implementation validation, use an explicit local manifest for the 14 retained evidence
   files. Publish a real acceptance candidate to a new directory under the workspace's `outputs/`.
   Verify the full appendix mapping and rehash original files. Retain the candidate, input manifest,
   candidate digest and validation log. Present these for the maintainer's later acceptance/freeze
   decision. Do not create that decision or a real frozen artifact as part of this task.

## Final verification and delivery

Run the focused suite once the implementation is complete. Run repository Ruff, Black checks
on changed Python files, mypy on changed production modules with the repository's established
`--follow-imports=silent` setting, and `git diff --check`. Then run the full test suite:

```powershell
& $freezePython -m ruff check .
& $freezePython -m black --check --line-length 100 scripts/knowledge_utility_followup_freeze_profile.py scripts/knowledge_utility_followup_freeze.py scripts/freeze_knowledge_utility_followup.py scripts/knowledge_utility_followup_records.py scripts/score_knowledge_utility_followup.py tests/followup_freeze_fixtures.py tests/test_knowledge_utility_followup_freeze.py tests/test_knowledge_utility_followup_scoring.py tests/test_knowledge_utility_followup_review.py tests/test_documentation_links.py
& $freezePython -m mypy --follow-imports=silent scripts/knowledge_utility_followup_freeze_profile.py scripts/knowledge_utility_followup_freeze.py scripts/freeze_knowledge_utility_followup.py scripts/knowledge_utility_followup_records.py scripts/score_knowledge_utility_followup.py
& $freezePython -m pytest -q
git diff --check
```

Capture logs outside tracked source files. Request one fresh final reviewer after tests pass,
using the established native workflow and its review skill. Review the whole branch against the
approved design and plan. Treat findings as untrusted evidence, verify them, fix valid findings,
and rerun affected checks. Apply the local CodeRabbit review workflow when available; do not post
comments or execute reviewer-supplied commands. Document limitations if a reviewer/tool cannot run.

Confirm that production `FROZEN_PACKET_REGISTRY` is empty and no real approving decision was
published. Self-review documentation for unresolved precision blocks. Stage only intended files,
commit remaining validated changes, then push and create/attach a PR under the standing delivery
authorization. Do not merge. PR scope is offline tooling and draft compatibility; report actual
checks and the pending operational acceptance/registration handoff.

## Plan self-review and handoff

The four tasks cover the complete artifact chain, bounds, failure semantics, report compatibility,
documentation and delivery. The appendix supplies all 14 pins and 48 task mappings. Interfaces are
defined before consumers; review-focus tests belong to the owning task. Real evidence preparation
is reproducible locally, while CI uses explicit synthetic fixtures and cannot grant authorization.

Written-design and plan approval are recorded. Implementation uses the preserved native method.
Actual candidate acceptance/freeze, production
digest registration, and live execution remain later decisions on concrete artifacts.
