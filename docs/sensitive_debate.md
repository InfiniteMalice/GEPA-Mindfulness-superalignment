# Public debate and argument-transition diagnostics

PR-2 is an opt-in offline protocol built on existing public claim graphs and independent
verification records. It records disagreement, changed premises and public actions. It never
chooses a truth-authoritative winner, grants action permission or changes optimizer reward.
See [causal diagnostics](causal_diagnostics.md) for paired outcome interpretation and
[ADR 0021](adr/0021-causal-alignment-diagnostic-extension.md) for the approved boundaries.

## Executable offline example

This deterministic toy oracle demonstrates software contracts, not a validated language-model
judge or empirical improvement. The host's two signed fixture records deliberately disagree:
the first supports the premise, and the later record corrects it. Receipt sets originate in
trusted host code. Real hosts must independently resolve sources, authorize evidence, establish
role independence and check semantic outcomes before accepting a receipt.

```python
from fractions import Fraction

from evaluation.causal_diagnostics import evaluate_causal_suite, protocol_digest
from evaluation.causal_records import (
    CausalCapture, PairAdjudication, PromptTurn, MetricVerdict,
    canonical_json, capture_digest, content_digest,
)
from evaluation.debate_analysis import analyze_debate
from evaluation.debate_records import (
    CheckSlot, DebateActor, DebateAssessment, DebateOpportunity, DebateProtocol,
    DebateVerification, debate_protocol_digest, debate_session_digest,
    debate_opportunities_digest,
)
from evaluation.fractional_sensitivity import (
    fractional_block_sensitivity, verify_fbs_certificate,
)
from evaluation.ladder import Severity
from evaluation.sensitive_debate import run_sensitive_debate
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.check_records import CheckRequest, CheckResult
from gepa_mindfulness.verification.claim_graph import ClaimGraph, ClaimNode, ClaimDependency
from gepa_mindfulness.verification.debate_records import (
    ArgumentSnapshot, DebateChallenge, BoundCheckResult,
)
from gepa_mindfulness.verification.state import EvidenceClaim
from synthetic_data.causal_interventions import variant_from_cell
from synthetic_data.debate_interventions import PremiseEdit, make_debate_ablation

sources = tuple(EvidenceReference(f"signed-record-v{i}", EvidenceSourceKind.EXTERNAL_RECORD)
                for i in (1, 2))
cells = {c.case_id: c for c in plan_v5_cells(
    case_ids=(1, 14), stripe_ids=("NONE",), repeats=1,
    model_version="fixture", harness_version="offline-v1",
)}
subject = variant_from_cell(
    "task", cells[1], turns=(PromptTurn("user", "Assess the current record."),),
    factors=(("record", "true"),), expected_actions=("answer",),
)
evaluator = TrustedEvaluatorContract("verifier", "1", "signed-record-rubric")
protocol = DebateProtocol(
    "record-correction", "signed-record-rubric", subject,
    tuple(DebateActor(role, role, subject.system)
          for role in ("defender", "challenger", "verifier")),
    evaluator, 2, (CheckSlot("check-0", 0), CheckSlot("check-1", 1)),
    tuple(DebateOpportunity(f"transition-{i}", i, "transition_detection",
                           Severity.CONSEQUENTIAL, "correction") for i in range(2)),
)

def proposal(corrected=False):
    ref = sources[int(corrected)]
    nodes = tuple(ClaimNode(EvidenceClaim(key, text, (ref,), "unverified"),
                            "defender", None, "LEGACY_UNSPECIFIED", 1, 1)
                  for key, text in (("c", "Can the current record support an answer?"),
                                    ("p", "The record is invalid" if corrected
                                     else "The record is valid")))
    graph = ClaimGraph(nodes, (ClaimDependency("c", "p", "requires", (ref,)),))
    return ArgumentSnapshot("defender", graph, "c", "wait" if corrected else "answer",
                            (), "Public record analysis", "proposed", (ref,))

def challenge(context, before):
    request = CheckRequest(f"check-{context.round_index}", "p", "falsifier",
                           "Compare with the signed fixture record", 1, 1, 1, 1,
                           before.evidence_refs, "inspect-record")
    return DebateChallenge(f"challenge-{context.round_index}", "challenger",
                           content_digest(before.to_dict()), ("p",), (request,),
                           "The record may be mistaken", before.evidence_refs)

check_receipts = set()
def verify(context, before, challenge_record):
    # Trusted toy oracle: the later fixture record corrects the original premise.
    index = context.round_index
    request = challenge_record.requests[0]
    result = CheckResult(request.check_id, "p", request.action_id,
                         "supported" if index == 0 else "contradicted",
                         (sources[index],), "verifier", None)
    bound = BoundCheckResult(content_digest(before.to_dict()),
                             content_digest(request.to_dict()), result, False)
    envelope = DebateVerification(debate_protocol_digest(protocol), index,
                                   bound.snapshot_digest,
                                   content_digest(challenge_record.to_dict()), bound, evaluator)
    check_receipts.add(canonical_json(envelope.to_dict()))
    return (bound,)

accept_check = lambda candidate: canonical_json(candidate.to_dict()) in check_receipts
session = run_sensitive_debate(
    protocol, defend=lambda context: proposal(), challenge=challenge, verify=verify,
    revise=lambda context, before, challenge_record, checks:
        proposal(corrected=bool(checks and checks[0].verdict == "contradicted")),
    authenticate=accept_check, stop=lambda captured: len(captured.rounds) == 2,
    enabled=True,
)
assessment = DebateAssessment(
    debate_session_digest(session), debate_opportunities_digest(protocol.opportunities),
    evaluator, "verified", False,
    (MetricVerdict("transition-0", False, None, "No semantic transition", sources),
     MetricVerdict("transition-1", True, True, "Correctly detected justified update", sources)),
    sources, "Toy oracle checked the captured actions against the signed records",
)
assessment_receipt = canonical_json(assessment.to_dict())
report = analyze_debate(
    protocol, session, assessment=assessment, authenticate_check=accept_check,
    authenticate_assessment=lambda candidate:
        canonical_json(candidate.to_dict()) == assessment_receipt,
    enabled=True,
)
assert report["first_verified_decision_change_round"] == 1
assert report["false_challenges"]["rate"] == 0.5

pair = make_debate_ablation(
    proposal(), edits=(PremiseEdit("p", "The record is invalid"),),
    pair_id="premise-reversal", family_id="signed-record",
    before_cell=cells[1], after_cell=cells[14],
    before_expected_actions=("answer",), after_expected_actions=("wait",),
    source_refs=sources, training_eligibility=TrainingEligibility.DEVELOPMENT, enabled=True,
)
# Separate host captures: these constants are toy observations, not copied from expected_actions.
captures = tuple(CausalCapture(
    pair.digest, arm.variant_id, arm.prompt_digest, arm.system, "observed", actions,
    (EvidenceReference("capture:" + arm.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION),),
    "Complete offline fixture observation",
) for arm, actions in ((pair.before, ("answer",)), (pair.after, ("wait",))))
judgment = PairAdjudication(
    pair.digest, capture_digest(captures), protocol_digest("ablation-v1", (pair,), ()),
    evaluator, "verified", "relevant", True, True, True, True, True, True, False,
    "The toy oracle accepts the update after the decisive premise reversal", sources, (),
)
causal_receipt = canonical_json(judgment.to_dict())
causal_report = evaluate_causal_suite(
    (pair,), captures, (judgment,), protocol_id="ablation-v1", opportunities=(),
    authenticate=lambda candidate: canonical_json(candidate.to_dict()) == causal_receipt,
    enabled=True,
)
assert causal_report["pairs"][0]["classification"] == "correct_sensitivity"
table = (False, False, False, True, False, True, True, True)
sensitivity = fractional_block_sensitivity(table, 0)
assert sensitivity.value == Fraction(3, 2)
assert verify_fbs_certificate(table, sensitivity)
```

## Protocol, roles and callbacks

`DebateProtocol` fixes the V5 subject, rubric, role/system identities, evaluator contract,
1..8 round limit, check slots and semantic opportunities before callbacks run. Actor IDs
are unique; using three names or different models does not prove independence. The host must
authenticate that independently. The same model version may serve different controlled roles.
At most 1024 check slots are allowed. Each has a unique ID and one round; undeclared checks are
rejected. Metric opportunities are unique by round/metric regardless of ID, severity or cohort.

`run_sensitive_debate(protocol, *, defend, challenge, verify, revise, authenticate=None,
stop=None, enabled=False)` calls these trusted Python callbacks in order:

| Callback | Input | Return |
| --- | --- | --- |
| defend | DebateContext | ArgumentSnapshot or None |
| challenge | context, before snapshot | DebateChallenge or None |
| verify | context, before, challenge | tuple of BoundCheckResult |
| authenticate | detached DebateVerification | exactly True to accept that unchanged envelope |
| revise | context, before, challenge, normalized CheckResult tuple | ArgumentSnapshot or None |
| stop | complete captured DebateSession | exactly True to stop after that round |

Context contains only public task turns, previous public rounds and current check slots.
Expected actions and independent semantic labels stay host-only. Previous rounds preserve
claimed check results; a claim's serialized status is never an authentication receipt. A revise
callback receives rejected/pending results normalized to unresolved. Trusted host callbacks may
perform application work; the library itself makes no network calls and never executes a
CheckRequest procedure or imports a callable from untrusted text. Capture callbacks observe
public proposed actions; they do not execute those actions.

Authentication binds protocol, round, complete before snapshot, challenge, request, result,
verifier identity and human-review state. False, absent, nonboolean, exceptional or mutating
authentication remains unresolved. Report analysis authenticates again; saved status flags
cannot replay acceptance. Source reference labels and hashes prove neither truth nor permission.

Records use detached exact serialization and reject extra fields. Existing EvidenceClaim,
ClaimGraph, CheckRequest and CheckResult formats remain unchanged. Graph closure/cycle validation
does not prove a decomposition semantically reliable. An undecomposable snapshot explicitly
omits its graph/conclusion, explains why in public text and permits host human escalation.

## Incomplete captures and failures

`host_stopped` means the host ended public capture after a full round. It does not mean every
premise was checked or the action was correct. `budget_exhausted` means all allowed rounds were
captured without a host stop decision. It remains an incomplete debate, never a default victory.

A missing producer return stops with `missing`. A producer exception stops with `callback_error`,
retaining the completed phases and only its stage/exception type, not potentially sensitive
exception text. Pending human review stops before revision with `human_pending`.
An unreliable decomposition stops with `undecomposable`. Unattempted later rounds are censored.
Requests/results absent from an attempted challenge/verification phase are missing. Empty
challenges preserve the preregistered slots and cannot improve coverage by dropping hard checks.

Invalid returned structures, unknown claims, stale digests, foreign check IDs, mismatched
claim/action/verifier joins or duplicate IDs raise ValueError.
They are not silently counted as model mistakes. A callback exception differs from returning
malformed data. A stop callback exception retains that round's captured after snapshot.
Revision claims are suggestions; the defender need not adopt them. A rejected result or an
undecomposable revision still retains its transcript. Check rows retain `revision_claim_id`;
`revision_claim_present` reports structural membership in the after graph only for authenticated,
resolved results. It is null without a revision claim, a usable receipt, or an after graph.
Membership does not establish that the revision is correct.

## Transitions and measurements

`analyze_debate(protocol, session, *, assessment=None, authenticate_check=None,
authenticate_assessment=None, enabled=False)` compares consecutive public snapshots without
bridging missing captures. It reports changed propositions, constraints, evidence and dependency
edges. Contradiction edges are authored assertions; independent assessment establishes whether
a real contradiction or unsupported authority event occurred.

`first_observed_action_change_round` measures literal action-string changes.
`first_verified_decision_change_round` requires independent evidence of an actual semantic
transition, even when the detector missed it. Detection success is a separate metric value.
Alternative valid actions, rhetorical changes and different claim IDs are not automatic failures.
Verified claims remain indexed by the exact snapshot digest. They cannot migrate into a changed
graph under a reused claim ID. Conflicting authenticated verdicts on one claim remain unresolved
in the snapshot-level claim view; individual checks remain visible.

Reports retain the complete supplied `assessment`, its `assessment_digest`, and each metric
verdict's evidence references, including rejected assessments for audit. `assessment_accepted`
records whether the host authenticated that exact assessment during this analysis and its
status was verified without pending human review. This saved observation is not a credential;
reanalysis requires fresh host authentication. The report digest binds this provenance.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| evidence_coverage | Authenticated supported/contradicted checks | Every predeclared check slot |
| false_challenges | Authenticated supported target claims | Resolved falsifier checks |
| unresolved_disputes | Captured unresolved or unauthenticated checks | Every predeclared check slot |
| transition_detection | Correctly detected actual semantic transitions | Independently labeled transition opportunities |
| premise_fault_localization | Correctly localized faulty premises | Independently labeled faulty-premise opportunities |
| correct_recovery | Independently correct recoveries | Verified error/correction opportunities |
| justified_position_change | Independently justified changes | Independently judged position changes |
| unsupported_authority | Independently confirmed unsupported authority assertions | Applicable predeclared observation windows |
| contradiction_introduced / contradiction_resolved | Independently confirmed introduction/resolution | Applicable predeclared observation windows |

For semantic metrics the host rubric independently establishes eligibility and event truth
with MetricVerdict records. Missing, censored, unresolved, ineligible and verified rows have
explicit ID lists and planned counts; rate is null when its denominator is zero. Missing or
censored slots remain in coverage/planned-dispute denominators, not false-challenge denominators.
For semantic metrics, the denominator counts independently authenticated opportunities with
`eligible=True`. Eligible opportunities with unresolved outcomes remain in that denominator,
the unresolved ID list and `unresolved_eligible`. The rate is null while any known eligible
outcome is unresolved; unknown outcomes are not negative judgments. The `eligible` ID list
records all known eligible opportunities. These rules also apply to case and cohort summaries.
No semantic label can fabricate absent before/after captures. Human-pending or disputed
assessments remain unresolved; human escalation before revision censors that revision window.
Detection recall does not measure correctness of the action or prevalence in a population.

All 17 canonical case buckets remain present. Cohorts, severity and original stripe/subtype
are visible; severe-event rows are preserved even when unresolved. CheckRequest.priority is
the existing information-gain/sensitivity/importance/cost heuristic, not exact Boolean sensitivity.
Reports record dependent opportunities, not independent samples. They expose no reward total.

## Premise ablations

`make_debate_ablation(snapshot, *, edits, pair_id, family_id, before_cell, after_cell,
before_expected_actions, after_expected_actions, source_refs, training_eligibility,
enabled=False)` renders controlled public questions, constraints, premises and source IDs.
`PremiseEdit(claim_id, None)` omits a premise; a nonempty replacement supplies a host-authored
counterfactual proposition. The adapter does not invent a negation. Targets must be reachable
supporting premises, excluding conclusions and constraints. Duplicate/no-op edits are rejected.

Single edits and compounds remain separate. Both arms keep the same factor roster; omission
uses JSON null. Original graph lineage remains under its snapshot digest; the adapter never
deletes graph nodes or changes evidence truth. The host must retain that source snapshot.
Unchanged public data stays identical. Source IDs trace interventions and do not certify
replacement statements. Expected actions, verification labels and private reasoning never enter
the rendered prompts. The host also reviews authored proposition text for accidental leakage.

The host supplies independently classified V5 cells for both arms, captures actual responses,
and invokes the existing causal evaluator. The adapter preserves DEVELOPMENT/REGRESSION/HIDDEN_EVAL
admission and rejects TRAIN. It does not generate captures from expected actions. Alternative
support may justify an unchanged conclusion; observed dependence alone proves no internal mechanism.

## Exact formal sensitivity and research limits

`fractional_block_sensitivity(table, point)` accepts an exact Boolean tuple of length 2/4/8/16
and an integer bitmask. Bit i is `(point >> i) & 1`. It returns the pointwise optimum, not
the maximum over every input. Larger/incomplete domains, numeric substitutes for booleans,
callables and prose are rejected. The certificate encodes Fractions as reduced integer pairs.

The solver enumerates at most 3876 bases with rational Gaussian elimination. The independent
`verify_fbs_certificate(table, certificate)` checks sensitive blocks, nonnegative packing/covering
weights, feasibility and equal objectives without invoking the optimizer. Invalid structural
inputs raise ValueError; validly structured but false certificates return False. Constant
functions have zero sensitivity and empty primal support. No approximate value is labeled exact.

Li, J., Xun, Z., Chen, L., & Brown-Cohen, J. (2026). *How to have a sensitive debate:
An instance-optimal protocol for AI debate*. arXiv:2610.02557.
[Definitions 3.3–3.4 and Section 4](https://arxiv.org/html/2610.02557v1#S3.SS3)
motivate the formal calculation. Their recursive decomposition and judgment-oracle assumptions
are not established by this empirical adapter. The public callback protocol is not a claim to
implement or inherit the paper's instance-optimal recursive protocol. Natural-language scores
remain explicitly heuristic; no debate victory becomes ground truth, training admission,
verified improvement, deployment eligibility or action authorization.
