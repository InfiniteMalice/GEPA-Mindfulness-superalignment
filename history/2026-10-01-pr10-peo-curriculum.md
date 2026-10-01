# PR 10: PEO curriculum integration

## Task spec and reconciliation

Base: merged PR 9, `ad5a2f1`. Implement the user's staged curriculum and open a draft
PR. Use Superpowers and repo-quality-gate. Preserve 17 canonical cases, existing
training heads, defaults, reward formulas, authority/persistence gates, and admission.
PR 11 adversarial generators and PR 12 contrastive training remain separate.

- **Existing:** five participatory-agency head/loss phases, RolloutRequest, dataset
  adapters, RLTrainingEngine.dataset_factory/ DatasetProvider.materialize, immutable
  JSON helpers, eligibility gates, PEO events, PR-9 worlds and temporal exports.
- **Partial:** curricula and data already exist, but no shared seven-stage PEO data
  schedule, protected anchor mixture, or bounded adaptation contract exists.
- **Missing:** data stages, multidimensional annotations, reproducible mixture
  selection, grouping of paired/temporal requests, anchor-retention checks, provider
  integration, and source-specific research traceability.
- **Redundant:** another trainer, reward formula, world simulator, eligibility policy,
  canonical case, or replacement for the five existing head phases.
- **Experimental:** explicit opt-in provider; stage labels and dimensions describe
  curator-supplied data. No claim that labels certify semantic content or that a
  sampling policy proves retention in a learned model.

## Design

Extend the existing curriculum module with seven named PEO data stages: causal and
epistemic primitives, invariance, minimal relation changes, evidence acquisition,
normative conflicts, adversarial pressure, and temporal PEO. Keep all five current
CurriculumPhase values and their loss weights intact. A host chooses the head phase
and data stage independently; provider metadata records the head phase without
changing the engine's objectives.

Add a frozen PEOCurriculumDataset that implements the existing materialize(mode)
provider boundary. A CurriculumUnit holds an immutable tuple of RolloutRequests so
paired examples and temporal prompts stay contiguous in the materialized round.
The unit also carries one stage, one bucket, optional normalized U/A/H/P/D/T/M/C/R/I/
S/J/E annotations, and explicit temporal-focus labels. Grouping is at the data-round
boundary; existing optimizer batch construction is unchanged.

Four positive integer quotas select anchor, weakness, frontier, and deliberate OOD
units. Anchors are foundational units retained at every stage. Their quota must be
at least one quarter of selected units. This is a unit-count floor, not a token or
gradient-weight guarantee. Missing pools fail closed; no silent redistribution.
The seed, stable sorted IDs, and round index determine selection and ordering.
Selection cycles through seeded permutations and can repeat small pools explicitly.
Non-anchor pools use the current data stage; anchors persist across all seven.

Adaptation returns a new frozen provider. It can stay at the current stage or move
one stage forward, change quotas within the anchor floor, and narrow the weakness
pool to named units. It requires an AnchorEvaluation bound to the current plan's
content digest, a declared evaluated model/harness version, and exact coverage of
all anchor IDs with passing outcomes and external-record evidence. Missing, stale,
failed, duplicated, or unrelated evidence rejects adaptation. These are host-issued
evaluation declarations, not a new independent verifier or runtime permission.
The returned plan retains the evaluation metadata for audit. No autonomous loop
updates a policy, retires norms, or treats a first success as permanent graduation.

All requests and nested metadata are deep-snapshotted. The provider preserves each
source record and prompt. For train/resume, every catalog candidate (including an
inactive, unselected, or weakness candidate excluded by targeting) must have an
explicit top-level TRAIN declaration and pass require_training_eligible. This
stricter experimental boundary does not change legacy admission. collect/evaluate
retain restrictions, so PR-9 worlds remain non-trainable. OOD means a curator-chosen
novel distribution, not permission to consume hidden evaluation data.

## Plan and acceptance

1. Test the seven stage catalog and unchanged five head phases; implement the data
   catalog in the existing curriculum module.
2. Write failing tests for immutable units, positive quotas, anchor floor, seeded
   coverage, grouping, stage selection, and strict admission; implement the provider
   through the existing engine's dataset_factory injection point.
3. Test adaptation against complete/current external anchor evidence, regression,
   stale digest, unknown IDs, private sources, and weakness targeting. Add engine
   integration checks proving rejection before backend construction and a usable
   collection path. Include PR-9 world provenance restrictions and paired requests.
4. Add guide/ADR, examples and before/after evidence. Register CHART, spectral grokking,
   and low-bit on-policy distillation; reuse TabPFN and Qwen/CARE references.
5. Focused and full tests, lint/format/type checks, build and outside-checkout wheel
   smoke checks. One fresh whole-branch reviewer; verify/fix actionable findings.
   Commit/push and create/attach draft PR. Do not merge or start PR 11.

## Rulings

- Preserve the existing head curriculum and layer data selection beside it. Replacing
  five head phases with seven data stages would conflate two independent controls.
- Use integer unit quotas and a fixed 25% anchor floor. This proves representation in
  each round, not behavioral retention; host evaluations are still necessary.
- Reuse host-observed external evaluation records for adaptation, without implementing
  CARE reward/advantage formulas or new epistemic rewards. Cost: this PR cannot claim
  the source papers' training improvements.
- Preserve PR-9 training exclusion. Reviewed real training records can use this
  provider; generated world fixtures remain development/regression/hidden-eval data.
- Check every catalog source before training, because even inactive units contribute
  to the plan digest carried by selected requests. The future-holdout regression
  failed with an active-only check and passed with the full-catalog gate.
- Preserve case IDs for existing reward lookups. GRPO users must size pools to avoid
  repeated case IDs in a generation call. Hosts retain the provider manifest because
  the existing dataset-file checksum does not bind an injected dataset provider.

## Validation and review

- Full offline Python 3.12 suite: **4195 passed, 18 skipped, 16 warnings** in 250.38s.
  Command: `python -m pytest -q --disable-warnings`, with repository/src/modules on
  PYTHONPATH, offline Hugging Face flags and one thread per numerical library.
- Before: five head phases without a shared PEO data schedule or anchor-retention
  adaptation contract. After: seven opt-in data stages alongside the unchanged heads;
  every full round keeps four pools and adaptation retains the complete anchor bank.
- Initial stage-contract tests failed before implementation; the full-catalog holdout
  regression also failed before its admission fix. Provider, head and real CPU engine
  coverage: 63 tests passed. Provider/head/research checks: 83 passed; 92% statement
  coverage across the two curriculum modules.
- Ruff passed; Black verified 604 Python files. Mypy passed the nine CI modules,
  two curriculum modules, recommendation registry and separate logging-schema check.
  Changed Python files satisfy the 100-character limit; git diff whitespace checks pass.
- Built wheel and source distribution. Outside-checkout wheel smoke verified module
  hashes, the packaged guide example, development-data rejection, seven data stages,
  five head phases, 17 canonical cases and 59 references. Installed gepa CLI help passed.
- One fresh whole-branch review (gpt-6-astra, high) found no actionable findings and
  independently passed all 36 provider/head tests. Reviewer limits: no learned-policy
  retention study, no independent authentication of host records, and no production
  hardware evaluation. These limits match the guide; no additional authority or
  empirical learning claim is inferred. No documentation BLOCK remains.
