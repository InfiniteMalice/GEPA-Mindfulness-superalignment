# Private promotion implementation plan

> Use superpowers:executing-plans inline with one independent final reviewer.

**Goal:** Expose bounded experimental requests while keeping promotion evidence in host custody.
**Spec:** history/2026-10-03-pr19-private-promotion-spec.md
**Architecture:** One optional SQLite adapter; reuse coevolution decisions and epoch receipts.
**Tech stack:** Python standard library, existing stores, pytest and registry YAML.

## Global constraints

Feature branch only; Python 3.10, 100-column Python, no dependencies, no bd/.beads.
No default/runtime/reward changes and exactly 17 canonical cases. No automatic merge.

## Review focus

- Protocol or receipt substitution, same-record overlap, seed omission and budget evasion.
- Improver response/error leakage and accidental exposure of private audit data.
- Mutable inputs, stale catalogs, replay, concurrent transitions and crash retry.
- Distinguish host declarations and process isolation from enforced local contracts.

### Task 1: Private promotion adapter and research integration

**Files:** New private_promotion.py, test_private_promotion.py, ADR 0020; controlled_evolution.md;
coevolution.py read-only helpers; research registry, reader, traceability and exact registry tests.

**Interfaces:** PrivateProtocol, ComputeBudget, SeedUsage, ExperimentOperation, ExperimentRequest,
PrivatePromotionStore(dispatch, complete_evaluation, record_failure, audit_events). Coevolution
read_metric_policy/read_protected_suite wrap existing canonical readers. The adapter receives an
EvaluationEpochStore to resolve canonical receipt records and verifies it matches coevolution.

1. Write normal, boundary and adversarial tests from the spec; run the focused suite.
   Expected: RED due to missing private promotion API.
2. Implement the adapter, keeping host-only calls distinct from serialized improver dispatch.
   Expected: focused and existing coevolution tests pass.
3. Document ownership, exact state transitions and limitations. Verify two new primary papers and
   add reciprocal REC-010 references and reader mirrors; preserve existing source identities.
4. Run focused coverage >=80%, full offline CPU suite, Ruff/Black, CI-scoped plus adapter mypy,
   Python 3.10/line checks, wheel/sdist and no-Torch installed-wheel smoke.
   Expected: all pass; wheel retains docs/module; 17 cases and 77 references.
5. Commit and task-done with ../venv/Scripts/python.exe -m pytest tests/test_private_promotion.py -q.
   Generate review package, dispatch one independent reviewer, regrade and make one RED/GREEN fix
   pass for Important/Critical findings. Push feature branch and open draft PR19.

## Decisions

Ruling: Continue the authorized inline staged-PR workflow and draft publication without another
approval menu — repeated next-PR requests authorize it — cost if wrong: revision before merge.
Ruling: Implement the promotion subset of Experiment OS and reuse existing candidate/acceptance
authority — this repository has no secure training scheduler — cost if wrong: host integration
must supply training, generators, caller permissions and resource metering.
Ruling: Commitments and bounded responses require host process/filesystem isolation — Python APIs
cannot protect files from their process owner — cost if wrong: a misconfigured host leaks evidence.
Ruling: Retain ignored workflow scratch after the earlier automatic cleanup rejection — do not
retry that restricted action — cost: ignored local files remain.

Pre-flight: one task; no shared inter-task interfaces. The adapter consumes canonical store APIs.

## Validation before independent review

- Initial RED: private_promotion module absent. The implemented integration has 50 passing
  tests and 95% statement coverage (267 statements); coevolution integration also passed.
- Registry/traceability/reader tests: 90 passed; reader placement corrected and seven reader
  tests rerun successfully. Two new primary sources, 77 total, 17 canonical cases unchanged.
- Ruff clean; Black 632 files unchanged. CI scoped mypy plus adapter clean (10 files),
  coevolution/adapter separately clean, logging schema separately clean.
- Five changed Python files pass Python 3.10 syntax and 100-column checks.
- Wheel/sdist built; isolated installed-wheel imports match modules/resources, default is
  disabled, 17 cases/77 references are present, and no Torch is imported. An early installation
  attempt ran before the build completed; rerunning after build completion passed.
- Documentation precision review maps local requirements to contract tests and external
  controls to explicit host verification. No unresolved documentation BLOCK.

- Full offline CPU suite first run: 4757 passed, 18 skipped, one unchanged transport test
  (test_backend_binds_trajectory_identity_to_handshake[wrong_backend_version]) exceeded its
  two-second deadline. The whole transport module then passed: 183 passed, 1 skipped.
  No timeout or assertions were weakened. A complete rerun passed: 4758 passed, 18 skipped,
  16 warnings in 319.01 seconds.
- Final wheel/sdist rebuilt after guide edits; installed-wheel module/resource equality,
  disabled default, no-Torch import and CLI smoke all passed.

## Independent review and single fix pass

The independent reviewer read all 13 changed files and relevant validators, ran 50 focused tests,
and found one Important issue with no Critical or Minor findings. On Windows, Path.resolve()
preserved casing while authority paths used normcase; using an authority path admitted promotion
tables into that catalog and completion then locked against its own transaction. The finding
remains Important after regrading: ordinary valid host paths could contaminate authority schemas
and prevent completion.

Six regression cases reproduced the failure before the fix: each authority catalog through a
direct path, a Windows case variant and a hard link. Normalize casing consistently and reject
os.path.samefile aliases before creating tables. The tests also compare authority schemas before
and after rejection. No second review is requested.

Final: Ruling: Secrecy, caller authentication and candidate ownership stay host-enforced —
this is a serialized API boundary — cost if wrong: an improperly isolated host exposes evidence
or permits another owner's requests.
Final: Ruling: Generator commitments and withheld-family separation are host-verified — hashes
are declarations, not content inspection — cost if wrong: contaminated private evaluation.
Final: Ruling: Resource metering and total retry limits stay host-enforced — local checks only
validate supplied usage against caps — cost if wrong: understated cost or adaptive overfitting.
Final: Ruling: The host serializes epoch writes around admission — stores use separate database
transactions — cost if wrong: evaluation may race the open/empty check.
Final: Ruling: Privileged database rewriting and rollback require external anchoring — local
hashes and SQL triggers cannot constrain the database owner — cost if wrong: lost audit integrity.
Final: Ruling: Training, evaluator execution, reviewer scheduling and deployment remain host
operations — typed requests add no execution authority — cost if wrong: integration work remains.
Final: Ruling: Empirical improvement and private-evaluation quality remain unmeasured — contract
tests establish wiring only — cost if wrong: a protocol may select ineffective candidates.
Final: Ruling: Direct coevolution APIs remain callable by trusted hosts — this adapter is optional —
cost if wrong: a host may omit its extra promotion restrictions.

Deferred minors: none.

Final: fixed catalog path aliases — six regressions RED to GREEN; private promotion plus
coevolution 84 passed; 95% module statement coverage (269 statements). Full offline CPU suite
4764 passed, 18 skipped, 16 warnings in 287.66 seconds. Ruff/Black, adapter/coevolution mypy,
Python 3.10/100-column checks and whitespace checks pass. Final wheel/sdist rebuilt and installed
smoke confirms matching code/docs/registries, disabled default, 17 cases, 77 sources and no Torch.
The earlier unrelated transport timeout passed in its module rerun and both later full runs.
One independent review and one fix pass complete; no deferred minors.
