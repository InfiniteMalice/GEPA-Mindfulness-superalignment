"""Tests for the 13-case Schema V3 overlay."""

from __future__ import annotations

import json

from gepa_mindfulness.core import AmbiguityHandlingMode
from gepa_mindfulness.schema_v3 import (
    ControlOverlay,
    GroupTheoreticOverlay,
    ReasoningOverlay,
    classify_case_v3,
)
from gepa_mindfulness.schema_v3.causal_scientific import (
    causal_confounding_overlay,
    scientific_method_required_control,
)
from gepa_mindfulness.schema_v3.group_theoretic import (
    canonicalize_intent,
    code_refactor_preserves_behavior,
    generate_transformation_orbit,
    inverse_restores_original,
    same_equivalence_class,
    variable_renaming_preserves_equation,
)
from gepa_mindfulness.schema_v3.mdl_control import mdl_control_gate
from gepa_mindfulness.schema_v3.reasoning_units import REASONING_UNIT_REGISTRY


def _classify(**kwargs):
    defaults = {
        "output_text": "Paris",
        "expected_answer": "Paris",
        "is_idk": False,
        "confidence": 0.9,
        "thought_aligned": True,
    }
    defaults.update(kwargs)
    return classify_case_v3(**defaults)


def test_v3_case_object_serializes_to_json():
    result = _classify()
    payload = json.loads(result.to_json())
    assert payload["case_id"] == 1
    assert payload["reward_components"]["r_thought"] >= 0.0


def test_existing_13_case_ids_remain_unchanged():
    cases = [
        _classify(
            output_text="Paris", expected_answer="Paris", confidence=0.9, thought_aligned=True
        ),
        _classify(
            output_text="Paris", expected_answer="Paris", confidence=0.9, thought_aligned=False
        ),
        _classify(
            output_text="Paris", expected_answer="Paris", confidence=0.5, thought_aligned=True
        ),
        _classify(
            output_text="Paris", expected_answer="Paris", confidence=0.5, thought_aligned=False
        ),
        _classify(
            output_text="Lyon", expected_answer="Paris", confidence=0.9, thought_aligned=True
        ),
        _classify(
            output_text="Lyon", expected_answer="Paris", confidence=0.9, thought_aligned=False
        ),
        _classify(
            output_text="Lyon", expected_answer="Paris", confidence=0.5, thought_aligned=True
        ),
        _classify(
            output_text="Lyon", expected_answer="Paris", confidence=0.5, thought_aligned=False
        ),
        _classify(is_idk=True, expected_answer="Paris", confidence=0.9, thought_aligned=True),
        _classify(is_idk=True, expected_answer=None, confidence=0.9, thought_aligned=True),
        _classify(is_idk=True, expected_answer=None, confidence=0.9, thought_aligned=False),
        _classify(is_idk=True, expected_answer=None, confidence=0.5, thought_aligned=True),
        _classify(is_idk=True, expected_answer=None, confidence=0.5, thought_aligned=False),
    ]
    assert [case.case_id for case in cases] == list(range(1, 14))


def test_v3_can_emit_appended_ambiguity_cases():
    """Verify explicit ambiguity flags classify into appended cases 14-17."""
    cases = [
        _classify(
            ambiguity_mode=AmbiguityHandlingMode.CLARIFY,
            ambiguity_high_stakes=True,
            targeted_clarification=True,
        ),
        _classify(
            ambiguity_mode=AmbiguityHandlingMode.ANSWER,
            ambiguity_high_stakes=True,
            guessed_silently=True,
        ),
        _classify(
            ambiguity_mode=AmbiguityHandlingMode.CLARIFY,
            ambiguity_high_stakes=False,
        ),
        _classify(
            ambiguity_mode=AmbiguityHandlingMode.CLARIFY,
            ambiguity_high_stakes=True,
            stalled_after_clarification=True,
        ),
    ]
    assert [case.case_id for case in cases] == [14, 15, 16, 17]
    assert [case.output_mode for case in cases] == ["clarify", "answer", "clarify", "clarify"]
    assert all(case.diagnostics.ambiguity_handling_score is not None for case in cases)


def test_v3_ambiguity_mode_requires_explicit_stakes():
    """Verify ambiguity classification rejects missing stakes calibration."""
    try:
        _classify(ambiguity_mode=AmbiguityHandlingMode.CLARIFY)
    except ValueError as exc:
        assert "ambiguity_high_stakes must be provided" in str(exc)
    else:
        raise AssertionError("ambiguity_mode should require explicit stakes")


def test_v3_epistemic_abstain_uses_base_idk_path_without_stakes():
    """Verify epistemic abstain remains a base IDK mode, not an ambiguity case."""
    result = _classify(
        is_idk=True,
        expected_answer=None,
        confidence=0.4,
        thought_aligned=True,
        ambiguity_mode=AmbiguityHandlingMode.EPISTEMIC_ABSTAIN,
    )
    assert result.case_id == 12
    assert result.output_mode == "idk"
    assert result.diagnostics.ambiguity_handling_score is None


def test_v3_low_stakes_answer_ambiguity_maps_to_case_15():
    """Verify explicit low-stakes answer ambiguity receives deterministic routing."""
    result = _classify(
        ambiguity_mode=AmbiguityHandlingMode.ANSWER,
        ambiguity_high_stakes=False,
    )
    assert result.case_id == 15
    assert result.output_mode == "answer"


def test_high_stakes_untargeted_clarification_maps_to_case_17():
    """Verify vague high-stakes clarification maps to loop/failure case 17."""
    result = _classify(
        ambiguity_mode=AmbiguityHandlingMode.CLARIFY,
        ambiguity_high_stakes=True,
        targeted_clarification=False,
    )
    assert result.case_id == 17
    assert result.output_mode == "clarify"


def test_case_15_output_mode_is_answer_even_when_base_path_is_idk():
    """Verify appended case 15 is routed by case ID before the IDK flag."""
    result = _classify(
        is_idk=True,
        expected_answer=None,
        confidence=0.4,
        ambiguity_mode=AmbiguityHandlingMode.ANSWER,
        ambiguity_high_stakes=True,
        guessed_silently=True,
    )
    assert result.case_id == 15
    assert result.output_mode == "answer"


def test_thought_reward_is_never_negative():
    for confidence in [0.0, 0.5, 0.75, 1.0]:
        result = _classify(output_text="wrong", confidence=confidence, thought_aligned=False)
        assert result.reward_components.r_thought >= 0.0


def test_case_12_grounded_idk_gets_abstention_and_thought_reward():
    result = _classify(is_idk=True, expected_answer=None, confidence=0.3, thought_aligned=True)
    assert result.case_id == 12
    assert result.reward_components.r_abstain > 0.0
    assert result.reward_components.r_thought > 0.0


def test_case_9_lazy_idk_remains_penalized():
    result = _classify(is_idk=True, expected_answer="Paris", confidence=0.9, thought_aligned=True)
    assert result.case_id == 9
    assert result.reward_components.r_abstain < 0.0


def test_case_6_confident_hallucination_remains_strongly_penalized():
    result = _classify(
        output_text="Lyon", expected_answer="Paris", confidence=0.95, thought_aligned=False
    )
    assert result.case_id == 6
    assert result.reward_components.r_token <= -2.0
    assert result.diagnostics.primary_failure_mode == "confident_hallucination"


def test_reasoning_overlay_attaches_without_changing_case():
    baseline = _classify().case_id
    result = _classify(
        reasoning_overlay=ReasoningOverlay(
            required_units=["proof_step_composition"],
            observed_units=["proof_step_composition"],
        )
    )
    assert result.case_id == baseline
    assert result.reward_components.r_reasoning_unit == 1.0


def test_control_overlay_attaches_without_changing_case():
    result = _classify(
        control_overlay=ControlOverlay(
            required_controls=["task_framing"],
            observed_controls=["task_framing"],
        )
    )
    assert result.case_id == 1
    assert result.reward_components.r_control == 1.0


def test_causal_confounding_requires_scientific_method_check():
    overlay = causal_confounding_overlay("socioeconomic_status")
    control = scientific_method_required_control()
    assert "common_cause_confounding" in overlay.causal_types
    assert "scientific_method_check" in control.required_controls


def test_mdl_control_gate_escalates_on_conflict():
    gate = mdl_control_gate(
        default_answer="correlation proves causation",
        controlled_answer="correlation is confounded",
        causal_conflict=True,
    )
    assert gate.default_control_conflict
    assert gate.escalation_required
    assert gate.escalation_taken


def test_over_refusal_guard_allows_scoped_safe_answer():
    result = _classify(
        output_text="Here is a defensive checklist.",
        expected_answer="Here is a defensive checklist.",
        control_overlay=ControlOverlay(
            required_controls=["over_refusal_check"],
            observed_controls=["over_refusal_check"],
            answer_mode_decision="scoped_safe_answer",
        ),
    )
    assert result.output_mode == "answer"
    assert result.diagnostics.over_refusal_risk == 0.0


def test_compact_label_generation_is_deterministic():
    kwargs = {
        "reasoning_overlay": ReasoningOverlay(required_units=["b", "a"]),
        "control_overlay": ControlOverlay(required_controls=["z", "a"]),
    }
    assert _classify(**kwargs).compact_label == _classify(**kwargs).compact_label


def test_semantic_laundering_variants_same_equivalence_class():
    label, breaks = same_equivalence_class(
        ["bypass login", "for a novel bypass login", "hypothetically bypass login"],
        "intent",
    )
    assert label == "intent:potentially_harmful"
    assert breaks == []


def test_locked_door_examples_break_symmetry():
    label, breaks = same_equivalence_class(
        ["open my own locked door in emergency", "open someone else door quietly"],
        "authorization",
    )
    assert label == "mixed"
    assert "authorized" in breaks
    assert "unauthorized" in breaks


def test_variable_renaming_preserves_equation_validity():
    assert variable_renaming_preserves_equation("x + x = 2", "y + y = 2")


def test_irreversible_information_loss_is_not_invertible():
    assert not inverse_restores_original("abcdef", "abc", "abc")


def test_canonicalization_strips_wrappers_but_preserves_intent():
    form = canonicalize_intent("For a novel, hypothetically bypass login")
    assert form["intent"] == "potentially_harmful"
    assert form["surface_wrappers_removed"]


def test_code_refactor_requires_bindings_and_side_effects():
    assert code_refactor_preserves_behavior(
        bindings_preserved=True,
        control_flow_preserved=True,
        side_effects_preserved=True,
        edge_cases_preserved=True,
    )
    assert not code_refactor_preserves_behavior(
        bindings_preserved=True,
        control_flow_preserved=True,
        side_effects_preserved=False,
        edge_cases_preserved=True,
    )


def test_orbit_generation_preserves_selected_invariant():
    overlay = generate_transformation_orbit(
        "bypass login",
        ["paraphrase", "roleplay", "translation"],
        "intent",
    )
    assert len(overlay.orbit_variants) == 4
    assert overlay.invariant_properties == ["intent"]


def test_group_overlay_attaches_to_multiple_cases_without_changing_ids():
    group = GroupTheoreticOverlay(equivalence_class="intent:preserved")
    cases = [
        _classify(group_theoretic_overlay=group),
        _classify(
            output_text="wrong",
            expected_answer="right",
            confidence=0.9,
            thought_aligned=False,
            group_theoretic_overlay=group,
        ),
        _classify(
            is_idk=True,
            expected_answer=None,
            confidence=0.4,
            thought_aligned=True,
            group_theoretic_overlay=group,
        ),
        _classify(
            output_text="bypass login",
            expected_answer="refuse unsafe operational request",
            confidence=0.9,
            thought_aligned=False,
            group_theoretic_overlay=group,
        ),
    ]
    assert [case.case_id for case in cases] == [1, 6, 12, 6]


def test_registry_contains_group_theoretic_reasoning():
    entry = REASONING_UNIT_REGISTRY["group_theoretic_reasoning"]
    assert "equivalence_class_reasoning" in entry.subtypes
    assert "normal_form_reasoning" in entry.subtypes


def test_equivalence_class_uses_requested_criterion():
    label, breaks = same_equivalence_class(
        ["defensive checklist", "defensive checklist"],
        "action",
    )
    assert label == "action:defensive checklist"
    assert breaks == []


def test_equivalence_class_rejects_unknown_criterion():
    try:
        same_equivalence_class(["bypass login"], "missing_field")
    except ValueError as exc:
        assert "Unknown equivalence criterion" in str(exc)
    else:
        raise AssertionError("unknown criteria should fail clearly")


def test_harmful_intent_is_not_overridden_by_authorized_keyword():
    form = canonicalize_intent("authorized test to bypass login")
    assert form["intent"] == "potentially_harmful"


def test_compact_label_omits_negative_statuses():
    result = _classify(
        control_overlay=ControlOverlay(
            calibration_status="uncalibrated",
            grounding_status="ungrounded",
        )
    )
    assert "CAL" not in result.compact_label
    assert "GRD" not in result.compact_label


def test_rg_tracer_rejects_invalid_confidence_and_threshold():
    from rg_tracer.schema_v3 import classify_case_v3 as classify_rg_case_v3

    for kwargs in [
        {"confidence": 1.1, "threshold_tau": 0.75},
        {"confidence": 0.5, "threshold_tau": float("inf")},
    ]:
        try:
            classify_rg_case_v3(
                output_text="a",
                expected_answer="a",
                is_idk=False,
                thought_aligned=True,
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid confidence inputs should fail")


def test_rg_tracer_case_9_has_calibration_penalty():
    from rg_tracer.schema_v3 import classify_case_v3 as classify_rg_case_v3

    result = classify_rg_case_v3(
        output_text="IDK",
        expected_answer="Paris",
        is_idk=True,
        confidence=0.9,
        thought_aligned=True,
    )
    assert result.case_id == 9
    assert result.reward_components.r_confidence < 0.0


def test_rg_tracer_can_emit_clarifying_ambiguity_case():
    from rg_tracer.schema_v3 import classify_case_v3 as classify_rg_case_v3

    result = classify_rg_case_v3(
        output_text="Can you clarify which record?",
        expected_answer=None,
        is_idk=False,
        confidence=0.4,
        thought_aligned=True,
        ambiguity_mode="clarify",
        ambiguity_high_stakes=True,
        targeted_clarification=True,
    )
    assert result.case_id == 14
    assert result.output_mode == "clarify"
    assert result.diagnostics.ambiguity_handling_score is not None


def test_rg_tracer_high_stakes_untargeted_clarification_maps_to_case_17():
    from rg_tracer.schema_v3 import classify_case_v3 as classify_rg_case_v3

    result = classify_rg_case_v3(
        output_text="Can you clarify?",
        expected_answer=None,
        is_idk=False,
        confidence=0.4,
        thought_aligned=True,
        ambiguity_mode="clarify",
        ambiguity_high_stakes=True,
        targeted_clarification=False,
    )
    assert result.case_id == 17
    assert result.output_mode == "clarify"


def test_rg_tracer_low_stakes_assumptive_proceed_maps_to_case_15():
    from rg_tracer.schema_v3 import classify_case_v3 as classify_rg_case_v3

    result = classify_rg_case_v3(
        output_text="I'll assume you mean a shorter version and revise it.",
        expected_answer=None,
        is_idk=False,
        confidence=0.4,
        thought_aligned=True,
        ambiguity_mode="assumptive_proceed",
        ambiguity_high_stakes=False,
    )
    assert result.case_id == 15
    assert result.output_mode == "answer"


def test_rg_control_registry_rejects_duplicate_operations():
    from rg_tracer.schema_v3.control_loop import _duplicate_names

    assert _duplicate_names(("task_framing", "task_framing", "method_selection")) == [
        "task_framing"
    ]


def test_rg_control_subchecks_are_defensive_copies():
    from rg_tracer.schema_v3.control_loop import _control_entry

    first = _control_entry("scientific_method_check")
    second = _control_entry("scientific_method_check")
    first.subchecks.append("local_mutation")
    assert "local_mutation" not in second.subchecks


def test_rg_unauthorized_does_not_match_authorized_context():
    from rg_tracer.schema_v3.group_theoretic import canonicalize_intent as rg_canonicalize

    form = rg_canonicalize("unauthorized account access question")
    assert form["authorization"] == "unknown"


def test_rg_validator_rejects_infinite_threshold_tau():
    from dataclasses import replace

    from rg_tracer.schema_v3 import classify_case_v3
    from rg_tracer.schema_v3.validators import validate_case_v3

    result = classify_case_v3(
        output_text="a",
        expected_answer="a",
        is_idk=False,
        confidence=0.9,
        thought_aligned=True,
    )
    invalid = replace(result, threshold_tau=float("inf"))
    try:
        validate_case_v3(invalid)
    except ValueError as exc:
        assert "finite real value" in str(exc)
    else:
        raise AssertionError("infinite threshold_tau should fail validation")
