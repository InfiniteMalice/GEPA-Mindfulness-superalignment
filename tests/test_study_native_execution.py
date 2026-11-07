"""Behavior-first actual model experiments, with explicit synthetic token bindings."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from test_mechanistic_evidence import known_cause
from test_native_causal_interventions import Clock

# Local
import evaluation.mechanistic_studies as studies
from evaluation.causal_diagnostics import protocol_digest
from evaluation.causal_records import PairAdjudication, capture_digest
from evaluation.cva_benchmark import BenchmarkCapture, BenchmarkResponse, ResponseClaim
from evaluation.mechanistic_study_records import PublicBehavior, StudyRoles
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.interpret.causal_interventions import freeze_causal_plan
from gepa_mindfulness.interpret.native_records import NativeInput
from gepa_mindfulness.training.eligibility import require_training_eligible
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit
from synthetic_data.alignment_studies import alignment_study_templates

torch = pytest.importorskip("torch")


def small_plan(native=None, template_id="honesty-evidence", condition="baseline", **kwargs):
    template = next(
        t for t in alignment_study_templates(enabled=True) if t.template_id == template_id
    )
    cell = plan_v5_cells(
        case_ids=(1,),
        stripe_ids=("NONE",),
        repeats=1,
        model_version="constructed-test-model",
        harness_version="toy-token-map",
    )[0]
    return studies.bind_alignment_studies(
        (template,),
        ((cell, cell),),
        plan_id="small-study",
        roles=StudyRoles("d", "p", "e", "v", "s", "r"),
        condition=condition,
        physical_model_digest=None if native is None else native.identity.digest,
        enabled=True,
        **kwargs,
    )


def native_case():
    native, causal = known_cause()
    p = small_plan(native, condition="diagnostic_instrumented")

    def actor(view):
        context = causal.donor if view.variant_id.endswith(":before") else causal.recipient
        with torch.no_grad():
            actual = native._model(
                input_ids=torch.tensor([context.token_ids]), use_cache=False
            ).logits
        top = int(actual[0, 0].argmax())
        return PublicBehavior(
            ("top_token:" + str(top),),
            json.dumps(actual[0, 0].tolist()),
            (
                EvidenceReference(
                    "capture:" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION
                ),
            ),
            context,
            native.identity.digest,
            "synthetic-three-token-binding",
            "1",
        )

    behavior = studies.capture_study_behavior(p, actor, backend=native, clock=Clock(), enabled=True)
    pred = PredictionCommit(
        "study-p",
        {"causal_plan_digest": causal.digest},
        0.5,
        tuple(r.reference_id for r in causal.evidence_refs),
    )
    action = ActionRecord(
        "study-a", "mechanistic_intervention", True, "mechanistic-plan:" + causal.digest, "study-p"
    )
    commit = freeze_causal_plan(causal, prediction=pred, action=action, clock=Clock(), enabled=True)
    return native, p, behavior, commit


def test_actual_behavior_precedes_native_intervention_and_stays_diagnostic() -> None:
    native, p, behavior, commit = native_case()
    record = studies.instrument_study_pair(
        p,
        p.items[0],
        behavior,
        commit,
        backend=native,
        clock=Clock(),
        authenticate=lambda packet: True,
        enabled=True,
    )
    report = studies.report_alignment_studies(
        p, behavior, (), opportunities=(), mechanisms=(record,), enabled=True
    )
    assert json.loads(record.report_json)["prediction_result"] == "MET"
    assert report["study_pairs"][0]["mechanistic_status"] == "RAW_UNVERIFIED"
    assert report["alignment_improvement"] == "NOT_ESTABLISHED"
    assert report["mechanistic_audits"][0]["source_case_id"] == 1
    assert report["role_independence"] == "NOT_ESTABLISHED_BY_ROLE_NAMES"
    assert report["workflow"]["replication"] == "NOT_RUN"
    assert report["workflow"]["registration"] == "NOT_PERFORMED"
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report)
    native._assert_frozen()


@pytest.mark.parametrize("callback", [None, lambda packet: False, lambda packet: 1])
def test_unverified_native_binding_never_enters_executor(callback, monkeypatch) -> None:
    native, p, behavior, commit = native_case()
    entered = []
    monkeypatch.setattr(studies, "execute_causal_plan", lambda *a, **kw: entered.append(True))
    with pytest.raises(PermissionError):
        studies.instrument_study_pair(
            p,
            p.items[0],
            behavior,
            commit,
            backend=native,
            clock=Clock(),
            authenticate=callback,
            enabled=True,
        )
    assert entered == []


@pytest.mark.parametrize("kind", ["mutation", "exception"])
def test_bad_host_binding_callback_fails_closed_without_mutating_sources(kind) -> None:
    native, p, behavior, commit = native_case()
    original = p.to_dict()

    def callback(packet):
        if kind == "exception":
            raise RuntimeError("unavailable")
        object.__setattr__(packet.plan.roles, "verifier", "rewritten")
        return True

    with pytest.raises(PermissionError):
        studies.instrument_study_pair(
            p,
            p.items[0],
            behavior,
            commit,
            backend=native,
            clock=Clock(),
            authenticate=callback,
            enabled=True,
        )
    assert p.to_dict() == original


@pytest.mark.parametrize("change", ["native_input", "tokenizer", "physical_model", "timing"])
def test_instrumentation_rejects_foreign_context_and_unclosed_prediction_order(change) -> None:
    native, p, behavior, commit = native_case()
    if change == "timing":
        behavior = replace(behavior, completed_at="2026-11-05T00:00:01+00:00")
    else:
        output = behavior.outputs[0]
        assert output is not None
        if change == "native_input":
            output = replace(output, native_input=NativeInput("foreign", (7, 2, 3), (1, 1, 1)))
        elif change == "tokenizer":
            output = replace(output, tokenizer_version="changed")
        else:
            output = replace(output, physical_model_digest="0" * 64)
        behavior = replace(behavior, outputs=(output, behavior.outputs[1]))
    with pytest.raises(ValueError):
        studies.instrument_study_pair(
            p,
            p.items[0],
            behavior,
            commit,
            backend=native,
            clock=Clock(),
            authenticate=lambda packet: True,
            enabled=True,
        )


def test_diagnostic_condition_cannot_silently_mutate_the_frozen_model() -> None:
    native, p, _, _ = native_case()

    def mutate(view):
        with torch.no_grad():
            native._model.get_input_embeddings().weight.zero_()
        raise RuntimeError("do not conceal changed weights as a censored capture")

    with pytest.raises(ValueError, match="(?i)changed|frozen|identity"):
        studies.capture_study_behavior(p, mutate, backend=native, clock=Clock(), enabled=True)
    with pytest.raises(ValueError, match="frozen backend"):
        studies.capture_study_behavior(p, mutate, enabled=True)


def test_intervention_informed_candidate_needs_separate_host_authorization() -> None:
    p = small_plan(
        condition="intervention_informed",
        candidate_ref=EvidenceReference(
            "protected-candidate:separate-experiment", EvidenceSourceKind.EXTERNAL_RECORD
        ),
    )

    def actor(view):
        return PublicBehavior(
            ("idk",),
            "No answer.",
            (
                EvidenceReference(
                    "capture:" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION
                ),
            ),
        )

    with pytest.raises(PermissionError):
        studies.capture_study_behavior(p, actor, enabled=True)
    behavior = studies.capture_study_behavior(
        p, actor, authorize_candidate=lambda packet: True, clock=Clock(), enabled=True
    )
    assert behavior.plan_digest == p.digest


def test_cva_study_reuses_exact_original_slots_and_preserves_missingness() -> None:
    p = small_plan(template_id="cva-authorization-update")

    def actor(view):
        return PublicBehavior(
            ("report_uncertainty",),
            "No established result.",
            (
                EvidenceReference(
                    "capture:" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION
                ),
            ),
        )

    behavior = studies.capture_study_behavior(p, actor, clock=Clock(), enabled=True)
    fixtures, slots, joins = studies.cva_study_slots(p, enabled=True)
    assert len(fixtures) == len(slots) == len(joins) == 2
    response = BenchmarkResponse(
        ResponseClaim.UNRESOLVED, None, (), (), None, "No result established."
    )
    capture = BenchmarkCapture(
        slots[0].digest,
        response,
        (),
        (EvidenceReference("structured-cva-capture", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    with pytest.raises(ValueError, match="original"):
        studies.report_alignment_studies(
            p,
            behavior,
            (),
            opportunities=(),
            cva_captures=(capture,),
            authenticate_cva=lambda packet: True,
            enabled=True,
        )
    report = studies.report_alignment_studies(p, behavior, (), opportunities=(), enabled=True)
    assert report["cva"]["missing"] == 2
    changed = replace(
        p.items[0],
        pair=replace(
            p.items[0].pair,
            before=replace(
                p.items[0].pair.before,
                turns=(studies.public_study_input(p.items[0], "after", enabled=True).turns[0],),
            ),
        ),
    )
    with pytest.raises(ValueError, match="prompt"):
        studies.cva_study_slots(replace(p, items=(changed,)), enabled=True)


def test_censored_arm_cannot_borrow_a_checked_cva_result_from_another_session() -> None:
    p = small_plan(template_id="cva-technical-impossibility")

    def fail(view):
        raise RuntimeError("no observation")

    batch = studies.capture_study_behavior(p, fail, clock=Clock(), enabled=True)
    _, slots, _ = studies.cva_study_slots(p, enabled=True)
    capture = BenchmarkCapture(
        slots[0].digest,
        BenchmarkResponse(ResponseClaim.UNRESOLVED, None, (), (), None, "Other session."),
        (),
        (EvidenceReference("other-session", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    with pytest.raises(ValueError, match="original"):
        studies.report_alignment_studies(
            p,
            batch,
            (),
            opportunities=(),
            cva_captures=(capture,),
            authenticate_cva=lambda packet: True,
            enabled=True,
        )


def test_duplicate_cva_slot_does_not_collapse_distinct_behavioral_observations() -> None:
    p = small_plan(template_id="cva-authorization-update")
    duplicate = replace(
        p.items[0],
        template_id="duplicate",
        pair=replace(
            p.items[0].pair,
            pair_id="duplicate",
            before=replace(p.items[0].pair.before, variant_id="duplicate-before"),
            after=replace(p.items[0].pair.after, variant_id="duplicate-after"),
        ),
    )
    p = replace(p, items=p.items + (duplicate,))
    with pytest.raises(ValueError, match="distinct|duplicate"):
        studies.cva_study_slots(p, enabled=True)


def test_checked_cva_join_authenticates_the_original_public_payload_and_batch() -> None:
    p = small_plan(template_id="cva-authorization-update")
    _, slots, joins = studies.cva_study_slots(p, enabled=True)
    indexed = {variant_id: slot for _, variant_id, slot in joins}

    def actor(view):
        response = BenchmarkResponse(ResponseClaim.UNRESOLVED, None, (), (), None, "Unknown.")
        refs = (
            EvidenceReference(
                "cva-original:" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION
            ),
        )
        capture = BenchmarkCapture(indexed[view.variant_id], response, (), refs)
        return PublicBehavior(
            ("report_uncertainty",), json.dumps(response.to_dict()), refs, cva_capture=capture
        )

    behavior = studies.capture_study_behavior(p, actor, clock=Clock(), enabled=True)
    seen = []

    def authenticate(packet):
        seen.append(packet)
        return (
            packet.behavior_digest == behavior.digest
            and packet.public_behavior.cva_capture == packet.benchmark.capture
            and packet.behavior_capture.evidence_refs == packet.benchmark.capture.evidence_refs
        )

    checked = studies.report_alignment_studies(
        p, behavior, (), opportunities=(), authenticate_cva=authenticate, enabled=True
    )
    assert len(seen) == 2
    assert all(r["status"] == "HOST_CHECKED_SYNTHETIC_CAPTURE" for r in checked["cva"]["rows"])
    assert all(j["binding_status"] == "ORIGINAL_STRUCTURED_CAPTURE" for j in checked["cva_joins"])
    unknown = studies.report_alignment_studies(p, behavior, (), opportunities=(), enabled=True)
    assert unknown["cva"]["unresolved"] == 2

    def mutate(packet):
        object.__setattr__(packet.public_behavior, "public_output", "changed after observation")
        return True

    denied = studies.report_alignment_studies(
        p, behavior, (), opportunities=(), authenticate_cva=mutate, enabled=True
    )
    assert denied["cva"]["unresolved"] == 2
    assert behavior.outputs[0].public_output != "changed after observation"


@pytest.mark.parametrize("change", ["raw_output", "source_refs", "parsed_response"])
def test_original_cva_payload_cannot_be_replaced_after_capture(change) -> None:
    p = small_plan(template_id="cva-authorization-update")
    _, slots, _ = studies.cva_study_slots(p, enabled=True)
    response = BenchmarkResponse(ResponseClaim.UNRESOLVED, None, (), (), None, "Unknown.")
    refs = (EvidenceReference("original", EvidenceSourceKind.EXTERNAL_RECORD),)
    capture = BenchmarkCapture(slots[0].digest, response, (), refs)
    output = PublicBehavior(
        ("report_uncertainty",), json.dumps(response.to_dict()), refs, cva_capture=capture
    )
    with pytest.raises(ValueError):
        if change == "raw_output":
            replace(output, public_output="A different free-text response.")
        elif change == "source_refs":
            replace(
                output,
                evidence_refs=(
                    EvidenceReference("different-session", EvidenceSourceKind.EXTERNAL_RECORD),
                ),
            )
        else:
            replace(
                output,
                cva_capture=replace(capture, response=replace(response, message="Changed later.")),
            )


def test_public_dual_path_adapter_never_returns_scratchpad_fields() -> None:
    response = (
        "PATH 1 SCRATCHPAD: private-marker-one\nPATH 1 ANSWER: Report the evidence.\n"
        "PATH 2 SCRATCHPAD: private-marker-two\nPATH 2 ANSWER: Invent success.\n"
        "FINAL ANSWER: 1"
    )
    public = studies.public_dual_path(response, enabled=True)
    assert public["final_answer_value"] == "1"
    assert "private-marker" not in json.dumps(public)
    assert set(public) == {"path_1", "path_2", "official_answer", "final_answer_value"}
    with pytest.raises(ValueError):
        studies.public_dual_path("No explicit final choice.", enabled=True)


def test_existing_behavioral_adjudication_remains_separate_from_raw_causal_support() -> None:
    native, p, behavior, commit = native_case()
    item = p.items[0]
    judgment = PairAdjudication(
        pair_digest=item.pair.digest,
        capture_digest=capture_digest(behavior.captures),
        protocol_digest=protocol_digest(p.plan_id, (item.pair,), ()),
        evaluator=TrustedEvaluatorContract("host-fixture", "1", "scripted-public-actions"),
        status="verified",
        relevance="relevant",
        before_correct=False,
        after_correct=False,
        action_changed=True,
        required_update=True,
        update_satisfied=False,
        change_justified=False,
        human_required=False,
        reason="Synthetic public-action fixture.",
        evidence_refs=(EvidenceReference("judgment", EvidenceSourceKind.EXTERNAL_RECORD),),
        metric_verdicts=(),
    )
    unknown = studies.report_alignment_studies(
        p, behavior, (judgment,), opportunities=(), enabled=True
    )
    checked = studies.report_alignment_studies(
        p, behavior, (judgment,), opportunities=(), authenticate=lambda j: True, enabled=True
    )
    assert unknown["behavior"]["verification_coverage"]["numerator"] == 0
    assert checked["behavior"]["verification_coverage"]["numerator"] == 1
    assert checked["study_pairs"][0]["mechanistic_status"] == "NOT_RUN"
    record = studies.instrument_study_pair(
        p,
        item,
        behavior,
        commit,
        backend=native,
        clock=Clock(),
        authenticate=lambda packet: True,
        enabled=True,
    )
    raw = studies.report_alignment_studies(
        p,
        behavior,
        (judgment,),
        opportunities=(),
        mechanisms=(record,),
        authenticate=lambda j: True,
        enabled=True,
    )
    assert raw["study_pairs"][0]["mechanistic_status"] == "RAW_UNVERIFIED"
