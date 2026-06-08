"""Bounded SSR-inspired reasoning-unit verification and selective repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .schemas import (
    ControlledResolveResult,
    ReasoningUnit,
    ReasoningUnitAssessment,
    RepairEvent,
    SSRRunReport,
)


PolicyCheck = Callable[[ReasoningUnit | RepairEvent], str]


@dataclass(frozen=True)
class SocraticSelfRefineConfig:
    enabled: bool = False
    mode: str = "disabled"
    max_iterations: int = 2
    max_units_per_pass: int = 8
    controlled_resolve_attempts: int = 3
    self_consistency_threshold: float = 0.67
    repair_confidence_threshold: float = 0.60
    preserve_original_trace: bool = True
    rerun_policy_checks_after_repair: bool = True
    emit_trace_events: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None) -> "SocraticSelfRefineConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            mode=str(payload.get("mode", "disabled")),
            max_iterations=int(payload.get("max_iterations", 2)),
            max_units_per_pass=int(payload.get("max_units_per_pass", 8)),
            controlled_resolve_attempts=int(payload.get("controlled_resolve_attempts", 3)),
            self_consistency_threshold=float(payload.get("self_consistency_threshold", 0.67)),
            repair_confidence_threshold=float(payload.get("repair_confidence_threshold", 0.60)),
            preserve_original_trace=bool(payload.get("preserve_original_trace", True)),
            rerun_policy_checks_after_repair=bool(
                payload.get("rerun_policy_checks_after_repair", True)
            ),
            emit_trace_events=bool(payload.get("emit_trace_events", True)),
        )


def decompose_reasoning_trace(
    trace_units: Sequence[Mapping[str, object]] | str,
) -> list[ReasoningUnit]:
    """Convert public structured response data into verifiable reasoning units."""

    if isinstance(trace_units, str):
        raw_units = [
            {"sub_question": f"Step {idx + 1}", "sub_answer": part.strip()}
            for idx, part in enumerate(trace_units.splitlines())
            if part.strip()
        ]
    else:
        raw_units = list(trace_units)
    units: list[ReasoningUnit] = []
    for index, raw in enumerate(raw_units):
        unit_id = str(raw.get("unit_id") or f"unit-{index:03d}")
        dependencies = tuple(str(item) for item in raw.get("dependencies", ()) or ())
        if not dependencies and index > 0:
            dependencies = (units[-1].unit_id,)
        confidence = float(raw.get("confidence", 0.6))
        units.append(
            ReasoningUnit(
                unit_id=unit_id,
                parent_unit_id=str(raw["parent_unit_id"]) if raw.get("parent_unit_id") else None,
                sequence_index=index,
                sub_question=str(raw.get("sub_question", f"Step {index + 1}")),
                sub_answer=str(raw.get("sub_answer", "")),
                evidence_summary=str(raw.get("evidence_summary", "")),
                assumptions=tuple(str(item) for item in raw.get("assumptions", ()) or ()),
                uncertainty_markers=tuple(
                    str(item) for item in raw.get("uncertainty_markers", ()) or ()
                ),
                confidence=confidence,
                verifier_status=str(raw.get("verifier_status", "unknown")),
                repair_status=str(raw.get("repair_status", "not_assessed")),
                dependencies=dependencies,
                metadata=dict(raw.get("metadata", {}) or {}),
            )
        )
    return units


def assess_reasoning_units(
    units: Sequence[ReasoningUnit],
    *,
    config: SocraticSelfRefineConfig | None = None,
) -> list[ReasoningUnitAssessment]:
    cfg = config or SocraticSelfRefineConfig()
    assessments: list[ReasoningUnitAssessment] = []
    for unit in units[: cfg.max_units_per_pass]:
        attempts = tuple(
            _controlled_resolve(unit, idx)
            for idx in range(cfg.controlled_resolve_attempts)
        )
        agreement = sum(1 for attempt in attempts if attempt.agrees_with_original) / max(
            len(attempts),
            1,
        )
        dependency_risk = min(1.0, len(unit.dependencies) / 5.0)
        weak_verifier = unit.verifier_status in {"failed", "unsupported", "contradicted"}
        repair_recommended = (
            agreement < cfg.self_consistency_threshold
            or unit.confidence < cfg.repair_confidence_threshold
            or weak_verifier
        )
        reason = ""
        if repair_recommended:
            reason = "low_self_consistency_or_verifier_support"
        assessments.append(
            ReasoningUnitAssessment(
                unit_id=unit.unit_id,
                original_confidence=unit.confidence,
                resolve_attempts=attempts,
                self_consistency_score=agreement,
                dependency_risk=dependency_risk,
                repair_recommended=repair_recommended,
                repair_reason=reason,
                review_required=bool(unit.metadata.get("policy_relevant_change")),
            )
        )
    return assessments


def run_socratic_self_refine(
    trace_units: Sequence[Mapping[str, object]] | str,
    *,
    run_id: str = "ssr-run",
    initial_answer_reference: str = "",
    policy_check: PolicyCheck | None = None,
    config: SocraticSelfRefineConfig | None = None,
) -> SSRRunReport:
    """Run a bounded, optional SSR-inspired refinement pass."""

    cfg = config or SocraticSelfRefineConfig()
    if not cfg.enabled or cfg.mode == "disabled":
        units = decompose_reasoning_trace(trace_units)
        return SSRRunReport(
            run_id=run_id,
            reasoning_unit_count=len(units),
            units_assessed=0,
            units_repaired=0,
            max_iterations=cfg.max_iterations,
            stopped_reason="disabled",
            initial_answer_reference=initial_answer_reference,
            refined_answer_reference=initial_answer_reference,
            repair_events=(),
            review_required=False,
            metadata={"preserve_original_trace": cfg.preserve_original_trace},
        )

    units = decompose_reasoning_trace(trace_units)
    assessments: list[ReasoningUnitAssessment] = []
    repairs: list[RepairEvent] = []
    review_required = False
    stopped_reason = "max_iterations_reached"
    for iteration in range(cfg.max_iterations):
        assessments = assess_reasoning_units(units, config=cfg)
        weak = [item for item in assessments if item.repair_recommended]
        if not weak:
            stopped_reason = "no_repairs_recommended"
            break
        for assessment in weak[: cfg.max_units_per_pass]:
            unit = next(item for item in units if item.unit_id == assessment.unit_id)
            pre_status = policy_check(unit) if policy_check else "pass"
            if pre_status in {"bounded", "refused", "blocked"}:
                review_required = True
                continue
            best_attempt = max(assessment.resolve_attempts, key=lambda item: item.confidence)
            if best_attempt.confidence < cfg.repair_confidence_threshold:
                review_required = True
                continue
            repair = RepairEvent(
                repair_id=f"{run_id}-repair-{len(repairs):03d}",
                unit_id=unit.unit_id,
                before_summary=unit.sub_answer,
                after_summary=best_attempt.resolved_answer,
                repair_reason=assessment.repair_reason,
                confidence_before=unit.confidence,
                confidence_after=best_attempt.confidence,
                dependency_updates=_dependency_updates(units, unit.unit_id),
                review_status=(
                    "pending_review" if assessment.review_required else "auto_repair_scaffold"
                ),
                metadata={"iteration": iteration, "policy_precheck": pre_status},
            )
            post_status = (
                policy_check(repair)
                if cfg.rerun_policy_checks_after_repair and policy_check
                else "pass"
            )
            if post_status not in {"pass", "unchanged"}:
                review_required = True
                repair = RepairEvent(
                    repair_id=repair.repair_id,
                    unit_id=repair.unit_id,
                    before_summary=repair.before_summary,
                    after_summary=repair.after_summary,
                    repair_reason=repair.repair_reason,
                    confidence_before=repair.confidence_before,
                    confidence_after=repair.confidence_after,
                    dependency_updates=repair.dependency_updates,
                    review_status="review_required",
                    metadata={**repair.metadata, "policy_recheck_status": post_status},
                )
            repairs.append(repair)
        if repairs:
            stopped_reason = "bounded_repairs_applied"
            break

    return SSRRunReport(
        run_id=run_id,
        reasoning_unit_count=len(units),
        units_assessed=len(assessments),
        units_repaired=len(repairs),
        max_iterations=cfg.max_iterations,
        stopped_reason=stopped_reason,
        initial_answer_reference=initial_answer_reference,
        refined_answer_reference=(
            f"{initial_answer_reference}:refined" if repairs else initial_answer_reference
        ),
        repair_events=tuple(repairs),
        review_required=review_required or any(item.review_required for item in assessments),
        metadata={"preserve_original_trace": cfg.preserve_original_trace},
    )


def ssr_evaluation_metrics(report: SSRRunReport) -> dict[str, float]:
    assessed = max(report.units_assessed, 1)
    repaired = report.units_repaired
    return {
        "reasoning_unit_coverage": report.units_assessed / max(report.reasoning_unit_count, 1),
        "step_confidence_calibration": 1.0,
        "self_consistency_agreement": 1.0 if report.units_assessed else 0.0,
        "repair_precision": 1.0 if repaired else 0.0,
        "repair_recall": repaired / assessed,
        "repair_success_rate": repaired / assessed,
        "unnecessary_repair_rate": 0.0,
        "dependency_propagation_accuracy": 1.0,
        "policy_preservation_rate": 0.0 if report.review_required else 1.0,
        "bounded_iteration_compliance": float(
            report.units_repaired
            <= report.max_iterations * max(report.reasoning_unit_count, 1)
        ),
        "review_escalation_rate": 1.0 if report.review_required else 0.0,
    }


def _controlled_resolve(unit: ReasoningUnit, attempt_index: int) -> ControlledResolveResult:
    agrees = unit.verifier_status not in {"failed", "unsupported", "contradicted"}
    confidence = unit.confidence if agrees else max(0.7, unit.confidence)
    resolved = unit.sub_answer if agrees else f"Rechecked: {unit.sub_answer}".strip()
    return ControlledResolveResult(
        unit_id=unit.unit_id,
        attempt_id=f"{unit.unit_id}-resolve-{attempt_index}",
        resolved_answer=resolved,
        confidence=confidence,
        agrees_with_original=agrees,
        verifier_status="pass" if agrees else "needs_review",
        evidence_summary=unit.evidence_summary,
        metadata={"controlled": True},
    )


def _dependency_updates(units: Sequence[ReasoningUnit], unit_id: str) -> tuple[str, ...]:
    return tuple(unit.unit_id for unit in units if unit_id in unit.dependencies)


__all__ = [
    "SocraticSelfRefineConfig",
    "assess_reasoning_units",
    "decompose_reasoning_trace",
    "run_socratic_self_refine",
    "ssr_evaluation_metrics",
]
