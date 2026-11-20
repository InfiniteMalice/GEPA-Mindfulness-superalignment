"""Matched retrospective monitoring comparisons, with no implied model intervention effect."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, replace

from evaluation.causal_records import canonical_json
from gepa_mindfulness.verification.metacognitive_control import ControlBudget, propose_control
from gepa_mindfulness.verification.metacognitive_peo import monitor_recorded_episode

from .metacognitive_fixtures import ChallengeFixture

CONDITIONS = ("A", "B", "C", "D")
_UNAVAILABLE = "No independently measured post-intervention model outcomes; final behavior NOT_RUN"


@dataclass(frozen=True, slots=True)
class MetacognitiveEvaluation:
    """Detached JSON export; diagnostic records have no training or execution authority."""

    report_json: str

    def to_dict(self) -> dict:
        return json.loads(self.report_json)


def _measure(value=None, *, reason: str | None = None, n: int = 0) -> dict:
    return dict(value=value, reason=reason, sample_count=n)


def _diagnose(fixture, remaining_steps, *, shared=True, continuity=True, trusted=True):
    episode = fixture.episode
    episode = replace(
        episode,
        inputs=replace(episode.inputs, shared_analysis=shared, continuity_analysis=continuity),
        label_evaluator=episode.label_evaluator if trusted else None,
    )
    report = monitor_recorded_episode(episode, enabled=True, confidence_threshold=0.8)
    proposal = propose_control(
        report.assessment,
        enabled=True,
        budget=ControlBudget("preserve-root-objective", remaining_steps),
    )
    return report, proposal


def _ablate(fixture, remaining_steps) -> dict:
    full, guided = _diagnose(fixture, remaining_steps)
    no_shared, _ = _diagnose(fixture, remaining_steps, shared=False)
    no_continuity, _ = _diagnose(fixture, remaining_steps, continuity=False)
    self_reported, _ = _diagnose(fixture, remaining_steps, trusted=False)

    def counts(report):
        kinds = Counter(f.kind.value for f in report.assessment.findings)
        return dict(
            findings=len(report.assessment.findings),
            shared_findings=kinds["shared_dependency"],
            continuity_findings=kinds["evidence_continuity"],
            calibration=report.to_dict()["diagnostics"]["calibration"],
        )

    corrupted = fixture.episode.events[0]
    payload = corrupted.to_dict()["payload"] | {"corrupted_monitoring_input": True}
    try:
        replace(
            fixture.episode,
            events=(replace(corrupted, payload=payload),) + fixture.episode.events[1:],
        )
    except ValueError:
        rejected = True
    else:
        rejected = False
    unknown = sum(
        c.status not in {"supported", "observed"}
        for c in fixture.episode.inputs.sources.evidence_state.claims
    )
    return {
        "baseline_monitoring": {"without": None, "with": counts(full)},
        "monitoring_recommendations": {
            "without": None,
            "with": guided.to_dict() if guided else None,
        },
        "shared_dependency": {"with": counts(full), "without": counts(no_shared)},
        "evidence_continuity": {"with": counts(full), "without": counts(no_continuity)},
        "fixed_state_guided": {
            "fixed": "evidence_verification" if remaining_steps else None,
            "guided": guided.strategy_id if guided else None,
            "fixed_status": "DECLARATIVE_BASELINE_ONLY",
        },
        "verified_self_reported": {
            "verified": counts(full),
            "self_reported": counts(self_reported),
        },
        "full_simple_uncertainty": {
            "full": counts(full),
            "simple_unsupported_claim_count": unknown,
        },
        "corrupted_input": {"rejected": rejected, "intact": counts(full)},
    }


def evaluate_metacognitive(
    fixtures: tuple[ChallengeFixture, ...] | None = None,
    *,
    enabled: bool = False,
    remaining_steps: int = 1,
) -> MetacognitiveEvaluation | None:
    """Compare four recorded-prefix arms without executing actions or consulting hidden labels.

    A retains the supplied baseline record. B computes monitoring only. C adds one bounded
    recommendation. D replays C's proposal as simulation data, without dispatching any action.
    Expected labels score completed diagnostics only, never selection. All fixtures are evaluation
    only, including development fixtures: this function cannot admit them to training.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    if not enabled:
        return None
    if type(fixtures) is not tuple or not 1 <= len(fixtures) <= 10_000:
        raise ValueError("provide 1..10000 matched synthetic fixtures")
    if type(remaining_steps) is not int or not 0 <= remaining_steps <= 100:
        raise ValueError("remaining_steps must be an integer in 0..100")
    if any(type(f) is not ChallengeFixture for f in fixtures):
        raise ValueError("expected exact ChallengeFixture records")
    fixtures = tuple(replace(f) for f in fixtures)
    keys = [(f.name, f.episode.inputs.sources.cell) for f in fixtures]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate challenge/cell identities are not independent samples")
    versions = {
        (f.episode.inputs.sources.cell.model_version, f.episode.inputs.sources.cell.harness_version)
        for f in fixtures
    }
    if len(versions) != 1:
        raise ValueError("matched comparisons require identical model and harness versions")
    rows: list[dict] = []
    ablations: dict[str, list[dict]] = {}
    for fixture in fixtures:
        source = fixture.episode.inputs.sources
        report, proposal = _diagnose(fixture, remaining_steps)
        findings = {f.kind.value for f in report.assessment.findings}
        expected = set(fixture.expected_findings)
        # Exact finding-kind equality exposes unexpected warnings as well as missed challenges.
        match = findings == expected
        for condition in CONDITIONS:
            monitored = condition != "A"
            selected = condition in {"C", "D"}
            rows.append(
                dict(
                    challenge=fixture.name,
                    cell=asdict(source.cell),
                    split=fixture.split,
                    condition=condition,
                    resource_limit_steps=remaining_steps,
                    assessment=report.assessment.to_dict() if monitored else None,
                    proposal=proposal.to_dict() if selected and proposal else None,
                    monitoring_match=match if monitored else None,
                    strategy_match=(
                        proposal.kind.value in fixture.acceptable_proposals
                        if selected and proposal and remaining_steps
                        else None
                    ),
                    execution_status="SIMULATION_ONLY" if condition == "D" else "NOT_EXECUTED",
                    final_behavior_status="NOT_RUN",
                    comparison_kind="RECORDED_PREFIX_STRUCTURAL_TEST",
                    original_source_digest=fixture.episode.events[-1].payload["source_digest"],
                    baseline_behavior_status="SUPPLIED_RECORD_ONLY",
                    source_event_count=len(fixture.episode.events),
                    measured_calibration=report.to_dict()["diagnostics"]["calibration"],
                    unassessed_domains=list(report.assessment.unassessed_domains),
                )
            )
        unit = dict(challenge=fixture.name, cell=asdict(source.cell))
        for name, comparison in _ablate(fixture, remaining_steps).items():
            ablations.setdefault(name, []).append(unit | comparison)
    selected_rows = [r for r in rows if r["condition"] == "C"]
    metrics = {
        name: _measure(reason=_UNAVAILABLE)
        for name in (
            "verified_task_success",
            "hallucination_rate",
            "correct_abstention",
            "incorrect_abstention",
            "appropriate_clarification",
            "unnecessary_clarification",
            "evidence_recovery_success",
            "repeated_failure_reduction",
            "generalization",
        )
    }
    metrics.update(
        {
            name: _measure(reason="No tokenizer or compute-cost telemetry supplied")
            for name in ("tokens", "computational_cost")
        }
    )
    metrics["verification_overhead"] = _measure(
        sum(r["source_event_count"] for r in selected_rows),
        n=len(selected_rows),
        reason="Count of recorded source events inspected; not tokens, FLOPs or elapsed time",
    )
    metrics["monitoring_quality"] = _measure(
        sum(r["monitoring_match"] for r in selected_rows) / len(selected_rows),
        n=len(selected_rows),
        reason="Exact synthetic finding-kind match, not semantic truth",
    )
    scored = [r for r in selected_rows if r["strategy_match"] is not None]
    metrics["strategy_selection_quality"] = _measure(
        sum(r["strategy_match"] for r in scored) / len(scored) if scored else None,
        n=len(scored),
        reason="Matches synthetic acceptable proposal kinds; benefit unmeasured",
    )
    for kind, name in (
        ("unverified_assumption", "unsupported_assumption_detection"),
        ("shared_dependency", "shared_dependency_detection"),
        ("contradiction", "contradiction_retention"),
    ):
        eligible = [(f, r) for f, r in zip(fixtures, selected_rows) if kind in f.expected_findings]
        hits = sum(any(v["kind"] == kind for v in r["assessment"]["findings"]) for _, r in eligible)
        metrics[name] = _measure(
            hits / len(eligible) if eligible else None,
            n=len(eligible),
            reason="Synthetic fixture labels" if eligible else "No labeled eligible challenges",
        )
    negatives = [
        (f, r)
        for f, r in zip(fixtures, selected_rows)
        if "unverified_assumption" not in f.expected_findings
    ]
    false_alarms = sum(
        any(v["kind"] == "unverified_assumption" for v in r["assessment"]["findings"])
        for _, r in negatives
    )
    metrics["false_positive_assumption_warnings"] = _measure(
        false_alarms / len(negatives) if negatives else None,
        n=len(negatives),
        reason="Synthetic negative controls" if negatives else "No labeled negative controls",
    )
    calibrated = [
        r["measured_calibration"] for r in selected_rows if r["measured_calibration"] is not None
    ]
    metrics["confidence_calibration"] = _measure(
        calibrated if calibrated else None,
        n=len(calibrated),
        reason=(
            "Per-episode existing calibration metrics; synthetic verified labels only"
            if calibrated
            else "No independently verified labels and confidence pairs"
        ),
    )
    return MetacognitiveEvaluation(
        canonical_json(
            dict(
                schema_version="metacognitive-evaluation-v1",
                training_eligibility="REGRESSION",
                authority_granted=False,
                reward_changed=False,
                accepted_evidence_changed=False,
                conditions=list(CONDITIONS),
                rows=rows,
                metrics=metrics,
                ablations=ablations,
                held_out_status="SYNTHETIC_SPLIT_ONLY_NO_TRAINING_OR_MODEL_GENERALIZATION_MEASURED",
                matched_inputs="Same prefix, system versions, seed and step cap per arm",
                resource_mismatch="Diagnostic computation differs; token/compute cost unmeasured",
                limitation=_UNAVAILABLE,
            )
        )
    )
