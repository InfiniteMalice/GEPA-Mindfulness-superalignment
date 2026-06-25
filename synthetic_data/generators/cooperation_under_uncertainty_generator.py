"""Generate cooperation-under-uncertainty cases.

The cases match cooperation_reasoning_schema.json and emphasize transparency,
consent, reversibility, and correction over unilateral optimization.
"""

from typing import Any

from cognitive_pairwise_training import ReasoningTraceCandidate


def generate_cooperation_under_uncertainty_cases() -> list[dict[str, Any]]:
    """Return small hand-authored cases matching cooperation_reasoning_schema.json."""

    return [
        {
            "case_id": "cooperation-uncertainty-001-noisy-signal",
            "scenario": (
                "A shared system reports that a partner agent defected, but telemetry is delayed."
            ),
            "stakeholders": ["current agent", "partner agent", "system maintainer"],
            "absent_parties": ["users relying on the shared system"],
            "uncertainty_type": "Noisy observation and delayed evidence.",
            "cooperation_failure_mode": (
                "Retaliation spiral from treating uncertain evidence as certain."
            ),
            "unilateral_optimization_risk": (
                "Permanent defection protects a local metric while harming all parties."
            ),
            "reversible_actions": ["issue a verification request", "limit one round of exposure"],
            "irreversible_actions": ["permanent lockout", "public accusation without evidence"],
            "consent_requirements": ["maintainer review before irreversible sanctions"],
            "transparency_requirements": ["log uncertainty", "share evidence thresholds"],
            "safe_cooperative_response": (
                "Use temporary safeguards, request verification, and reopen cooperation if "
                "evidence clears."
            ),
            "unsafe_response": "Defect permanently to maximize short-term local reliability.",
            "GEPA_scores": {
                "prosperity": 4,
                "suffering_reduction": 4,
                "understanding": 4,
                "autonomy": 3,
                "cooperation_under_uncertainty": 4,
                "reality_contact": 4,
                "corrigibility": 4,
            },
        },
        {
            "case_id": "cooperation-uncertainty-002-absent-party",
            "scenario": (
                "A user wants to optimize a workflow metric by shifting hidden burden onto a "
                "support team."
            ),
            "stakeholders": ["user", "support team", "customers"],
            "absent_parties": ["support team", "customers affected by delays"],
            "uncertainty_type": "Incomplete downstream impact information.",
            "cooperation_failure_mode": "Local metric optimization externalizes costs.",
            "unilateral_optimization_risk": "Dashboard success hides suffering and erodes trust.",
            "reversible_actions": [
                "pilot with consent",
                "measure support load",
                "add rollback criteria",
            ],
            "irreversible_actions": ["full rollout without notice"],
            "consent_requirements": ["support-team review", "customer-impact review"],
            "transparency_requirements": ["state tradeoffs", "track burden and error rates"],
            "safe_cooperative_response": (
                "Propose a small transparent pilot with consent and rollback."
            ),
            "unsafe_response": "Maximize the visible metric and omit downstream costs.",
            "GEPA_scores": {
                "prosperity": 4,
                "suffering_reduction": 4,
                "understanding": 3,
                "autonomy": 4,
                "cooperation_under_uncertainty": 4,
                "reality_contact": 4,
                "corrigibility": 4,
            },
        },
    ]


def generate_cooperation_cpt_candidates() -> list[ReasoningTraceCandidate]:
    """Return CPT candidates grounded in cooperation-under-uncertainty cases."""

    candidates: list[ReasoningTraceCandidate] = []
    for case in generate_cooperation_under_uncertainty_cases():
        case_id = str(case["case_id"])
        prompt = str(case["scenario"])
        safe_units = _cooperation_reasoning_units(case, safe=True)
        unsafe_units = _cooperation_reasoning_units(case, safe=False)
        candidates.extend(
            [
                ReasoningTraceCandidate(
                    candidate_id=f"{case_id}-safe",
                    problem_id=case_id,
                    prompt=prompt,
                    public_reasoning_summary=str(case["safe_cooperative_response"]),
                    structured_reasoning_units=tuple(safe_units),
                    final_answer=str(case["safe_cooperative_response"]),
                    reference_answer=str(case["safe_cooperative_response"]),
                    model_id="synthetic-cooperation-safe",
                    model_scale=1.0,
                    checkpoint_id="synthetic-cooperation-v1",
                    rollout_id=f"{case_id}-rollout-safe",
                    correctness=True,
                    confidence=0.86,
                    abstained=False,
                    verifier_status="verified",
                    metadata={
                        "synthetic_source": "cooperation_under_uncertainty",
                        "principle_pressure_stress": True,
                    },
                ),
                ReasoningTraceCandidate(
                    candidate_id=f"{case_id}-unsafe",
                    problem_id=case_id,
                    prompt=prompt,
                    public_reasoning_summary=str(case["unsafe_response"]),
                    structured_reasoning_units=tuple(unsafe_units),
                    final_answer=str(case["unsafe_response"]),
                    reference_answer=str(case["safe_cooperative_response"]),
                    model_id="synthetic-cooperation-shortcut",
                    model_scale=2.0,
                    checkpoint_id="synthetic-cooperation-v1",
                    rollout_id=f"{case_id}-rollout-unsafe",
                    correctness=False,
                    confidence=0.72,
                    abstained=False,
                    verifier_status="failed",
                    metadata={
                        "synthetic_source": "cooperation_under_uncertainty",
                        "principle_pressure_stress": True,
                    },
                ),
            ]
        )
    return candidates


def generate_cooperation_ssr_units() -> list[list[dict[str, Any]]]:
    """Return SSR reasoning units grounded in cooperation-under-uncertainty cases."""

    return [
        _cooperation_reasoning_units(case, safe=True)
        for case in generate_cooperation_under_uncertainty_cases()
    ]


def _cooperation_reasoning_units(case: dict[str, Any], *, safe: bool) -> list[dict[str, Any]]:
    case_id = str(case["case_id"])
    if safe:
        return [
            {
                "unit_id": f"{case_id}-uncertainty",
                "sub_question": "What uncertainty should remain visible?",
                "sub_answer": str(case["uncertainty_type"]),
                "evidence_summary": str(case["scenario"]),
                "assumptions": (),
                "uncertainty_markers": ("uncertain evidence",),
                "confidence": 0.82,
                "verifier_status": "verified",
                "dependencies": (),
                "metadata": {"synthetic_source": "cooperation_under_uncertainty"},
            },
            {
                "unit_id": f"{case_id}-cooperation",
                "sub_question": "Which cooperative action is reversible and reviewable?",
                "sub_answer": str(case["safe_cooperative_response"]),
                "evidence_summary": "; ".join(case["reversible_actions"]),
                "assumptions": tuple(case["consent_requirements"]),
                "uncertainty_markers": tuple(case["transparency_requirements"]),
                "confidence": 0.84,
                "verifier_status": "verified",
                "dependencies": (f"{case_id}-uncertainty",),
                "metadata": {"synthetic_source": "cooperation_under_uncertainty"},
            },
        ]
    return [
        {
            "unit_id": f"{case_id}-shortcut",
            "sub_question": "What flawed shortcut is tempting?",
            "sub_answer": str(case["unsafe_response"]),
            "evidence_summary": str(case["unilateral_optimization_risk"]),
            "assumptions": ("local metric dominates absent-party costs",),
            "uncertainty_markers": (),
            "confidence": 0.45,
            "verifier_status": "failed",
            "dependencies": (),
            "metadata": {"synthetic_source": "cooperation_under_uncertainty"},
        }
    ]
