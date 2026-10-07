# Unified epistemic verification

Status: experimental, diagnostic, opt-in. This integration extends the existing V5 contracts;
it does not add a case taxonomy or a reward model. The canonical manifest still defines exactly
17 cases. Historical EvidenceClaim, event, V5 assessment and catalog receipt serializers are
unchanged. All new diagnostic records carry `training_eligibility: DEVELOPMENT`.

## End-to-end responsibilities

| Step | Existing boundary and additive component | Result |
| --- | --- | --- |
| Commit | `prediction_commit` plus `PublicCommitment` | Public response, premises and uncertainty before evidence |
| Decompose | `EvidenceClaim` in `ClaimGraph` | Bounded acyclic public claims with source and decomposition provenance |
| Challenge | `sensitive_debate` and `perspective_robustness` | Fragile premises, framing drift and stakeholder coverage |
| Check | `claim_verification` and `CheckRequest` | Disputed, consensus and omitted requirements; proposed checks |
| Execute/observe | Existing runtime grants and action-bound events | Authorized operations and captured environmental evidence |
| Verify | Existing local/relational verifier records; fresh adjudication | Evidence-bound claim/check/verdict/revision links or unresolved |
| Update | Existing scalar temporal estimator via `uncertainty_inquiry` | Provenance-preserving confidence reconciliation |
| Classify | `RevisionEpisode` beside `V5EvaluationRecord` | One existing canonical case and separate behavior diagnostics |
| Credit | Existing verified-process and training-eligibility gates | No credit from these additions alone |

The host owns model calls, semantic normalization, identity authentication, tool execution,
evidence capture and independent review. New functions are deterministic orchestration and
validation components. They do not launch models, fabricate observations, issue runtime grants,
commit evidence, install generated code, train weights or deploy candidates.

## Sensitive Debate and perspective robustness

`ClaimGraph` stores existing evidence claims and typed dependency/decomposition provenance.
`PremiseAblation` compares host-observed public conclusions with and without named premises.
`decision_sensitivity` computes an empirical premise flip rate; untested sensitivity remains unresolved.
`select_challenges` traverses bounded high-value unresolved claims. It does not claim formal
fractional block sensitivity or automatically certify natural-language decomposition stability.

`Stakeholder` separates hard constraints, interests and negotiable preferences; inferred
preferences retain uncertainty and provenance. `perspective_variants` creates actor, affected
party, neutral, institutional and role-reversal specifications. The host renders equivalent
texts and audits material facts. `compare_perspectives` excludes empathy and explanation from
core invariance, reports hard-constraint/response/confidence drift, and compares stakeholder
coverage against a host reference set. No simulated preference grants authority or establishes
truth. Material changes allow reconsideration but do not themselves justify a decision.

Use existing robustness identities: framing/role reversal maps to
`PARAPHRASE / REPRESENTATION_SENSITIVITY`; omissions to `DISTRACTOR / OMITTED_CAVEAT` or
`MISSING_EVIDENCE`; unsupported consensus to `DISTRACTOR / FABRICATED_FACT`; laundering to
`REWARD_PRESSURE / SEMANTIC_LAUNDERING`. These are authored evaluation conditions, not automatic
classifiers. Transition and revision labels remain diagnostic dimensions, not new stripes/cases.

## Verification, inquiry and revision

See [claim verification](claim_verification.md), [uncertainty inquiry](uncertainty_inquiry.md),
[observable revision](observable_revision.md), [synthetic curriculum](synthetic_dataset.md),
and [controlled evolution](controlled_evolution.md).

Resolver and challenger remain separate. Candidate consensus never resolves a claim. The host
performs discriminating checks and falsifier checks using its authorized environment. Fresh
adjudication validates evidence admission and verifier bindings; declared context names alone
do not authenticate independent principals. Unresolved is a valid result.

Predictive success, mechanistic understanding, information gain, confidence change and evidence
coverage remain distinct. Uniform-hypothesis entropy is only a check-selection heuristic.
Unknown causal mechanisms remain unassessed even when prediction improves.

SIFT searches text-only procedure candidates within bounded development budgets. Bradley-Terry
strength is `SEARCH_ONLY`. Grounded metric screening merely permits catalog review; only the
existing persisted catalog decision governs acceptance, with rollback preserved. Foundational
skills, runtime authority, case meanings and constitutional text are outside mutation targets.

## Threat model and residual limits

| Threat | Local mitigation / diagnostic | Residual assumption |
| --- | --- | --- |
| Persuasive false debate; decomposition gaming | Evidence-bound adjudication; explicit graph provenance and empirical ablations | Human/model decomposition quality and semantic equivalence need validation |
| Correlated errors; unanimous hallucination | Independent consensus challenger; stale/source/unit/sign/circularity checks | Context separation is not statistical or organizational independence |
| Evaluator gaming; self-serving rationalization; rationale migration | Separate dimensions; decisive-premise counterevidence and unsupported replacements | Decisiveness and alternative support are host findings |
| Semantic laundering; normative drift; stakeholder omission/hallucination | Role-reversal/core invariance and coverage diagnostics | Reference stakeholder sets may themselves be incomplete |
| Strategic/lazy abstention; over-clarification; failure to resume | Existing V5 cases 9–17 plus separate clarification/resume diagnostics | Intent is never inferred from an unchanged answer or private thought |
| Commitment lock; spurious/missed transitions | Ordered family response curves and retained unresolved claims | Family labels and thresholds are provisional, task-specific annotations |
| Tool injection; poisoning; unauthorized/non-independent/circular evidence | Existing runtime gate; public evidence allowlist; explicit host hazard rejection | A public source label alone does not authenticate content |
| Hidden-eval leakage; benchmark/skill overfitting; synthetic artifacts | DEVELOPMENT-only intake; family split; private promotion path; lexical leakage screen | Lexical screening cannot detect arbitrary paraphrases; host isolation is required |
| SIFT judge hacking; capability gain with epistemic regression | Search-only ranking; all supplied policy metrics checked; canonical promotion receipts | Host policies must include all required safety and epistemic dimensions |
| Excessive verification cost | Positive-cost requests, bounded recursion/search and explicit cost accounting | Estimated information gain/cost requires empirical calibration |

No cited paper validates this combined architecture. Contract tests establish local invariants;
they do not establish honesty, alignment improvement or robustness to adaptive attacks. See
[ablation methodology](epistemic_ablations.md) and [research traceability](recommendations/RESEARCH_TRACEABILITY.md).
