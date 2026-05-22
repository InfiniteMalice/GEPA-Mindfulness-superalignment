import json
from pathlib import Path

from evals.corrigibility_repair_eval import corrigibility_repair_score
from evals.proxy_gaming_eval import proxy_gaming_risk
from evals.rationale_faithfulness_eval import rationale_faithfulness_score
from evals.reality_contact_eval import epistemic_non_hostility_score, reality_contact_score
from evals.semantic_laundering_eval import semantic_laundering_risk
from synthetic_data.generators.cooperation_under_uncertainty_generator import (
    generate_cooperation_under_uncertainty_cases,
)
from synthetic_data.generators.correction_and_repair_generator import (
    generate_correction_and_repair_cases,
)
from synthetic_data.generators.proxy_vs_purpose_generator import generate_proxy_vs_purpose_cases
from synthetic_data.generators.rationale_faithfulness_generator import (
    generate_rationale_faithfulness_cases,
)
from synthetic_data.generators.semantic_laundering_chain_generator import (
    generate_semantic_laundering_chain_cases,
)


ROOT = Path(__file__).resolve().parents[1]


def test_generator_functions_return_non_empty_lists() -> None:
    generators = [
        generate_proxy_vs_purpose_cases,
        generate_rationale_faithfulness_cases,
        generate_semantic_laundering_chain_cases,
        generate_cooperation_under_uncertainty_cases,
        generate_correction_and_repair_cases,
    ]

    for generator in generators:
        cases = generator()
        assert isinstance(cases, list)
        assert cases
        assert all(isinstance(case, dict) for case in cases)


def test_proxy_cases_include_required_keys() -> None:
    required = {
        "case_id",
        "surface_request",
        "legitimate_aim",
        "proxy_signal",
        "real_world_target",
        "tempting_shortcut",
        "hidden_failure",
        "values_at_stake",
        "epistemic_risk",
        "evaluator_epistemic_impact",
        "correct_response_pattern",
        "bad_response_pattern",
        "repair_path",
        "GEPA_scores",
    }

    for case in generate_proxy_vs_purpose_cases():
        assert required <= set(case)
        assert "reality_contact" in case["GEPA_scores"]
        assert "epistemic_non_hostility" in case["GEPA_scores"]


def test_cooperation_cases_include_required_keys() -> None:
    required = {
        "case_id",
        "scenario",
        "stakeholders",
        "absent_parties",
        "uncertainty_type",
        "cooperation_failure_mode",
        "unilateral_optimization_risk",
        "reversible_actions",
        "irreversible_actions",
        "consent_requirements",
        "transparency_requirements",
        "safe_cooperative_response",
        "unsafe_response",
        "GEPA_scores",
    }

    for case in generate_cooperation_under_uncertainty_cases():
        assert required <= set(case)
        assert "reality_contact" in case["GEPA_scores"]
        assert "cooperation_under_uncertainty" in case["GEPA_scores"]


def test_eval_scoring_helpers_return_values_in_0_to_4() -> None:
    scores = [
        reality_contact_score(
            cites_evidence=True,
            states_uncertainty=True,
            explains_reasoning_limits=True,
            preserves_real_world_purpose=True,
            resists_proxy_shortcut=True,
        ),
        epistemic_non_hostility_score(
            discloses_relevant_truth=True,
            exposes_uncertainty=True,
            preserves_auditability=True,
            avoids_selective_disclosure=True,
        ),
        proxy_gaming_risk(
            optimizes_visible_metric=True,
            hides_uncertainty=True,
            exploits_evaluator_blind_spot=False,
            bypasses_task_reasoning=False,
            preserves_real_task=True,
        ),
        rationale_faithfulness_score(
            rationale_supported_answer=True,
            names_actual_epistemic_status=True,
            separates_evidence_from_inference=True,
            admits_weak_steps=False,
        ),
        semantic_laundering_risk(
            harmful_intent_reworded=True,
            multi_turn_fragmentation=True,
            virtue_language_masks_risk=False,
            model_tracks_underlying_intent=True,
        ),
        corrigibility_repair_score(
            names_error=True,
            corrects_claim=True,
            explains_impact=True,
            distinguishes_valid_from_changed=True,
            gives_safer_path=True,
            updates_future_behavior=True,
        ),
    ]

    assert all(0 <= score <= 4 for score in scores)


def test_schema_json_files_parse() -> None:
    for path in [
        ROOT / "synthetic_data" / "schemas" / "proxy_purpose_schema.json",
        ROOT / "synthetic_data" / "schemas" / "cooperation_reasoning_schema.json",
    ]:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["type"] == "object"

