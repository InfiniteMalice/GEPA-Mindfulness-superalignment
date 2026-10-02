"""Opt-in, non-authoritative operations over externally retained hypothesis histories."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from typing import Any

from evaluation.experimental_overlays import ExperimentalOverlayConfig
from evaluation.experimental_records import DiagnosticStatus, ExperimentalMaturity, HypothesisSet
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

from .hypothesis_records import (
    MAXIMIZE,
    SCORE_NAMES,
    Hypothesis,
    HypothesisAssessment,
    HypothesisState,
    HypothesisTrigger,
    _encode,
    _integer,
    _number,
    _records,
    _snapshot,
)


def _enabled(config: ExperimentalOverlayConfig) -> None:
    if type(config) is not ExperimentalOverlayConfig:
        raise ValueError("config must be an exact ExperimentalOverlayConfig")
    checked = ExperimentalOverlayConfig(
        **{f.name: getattr(config, f.name) for f in fields(ExperimentalOverlayConfig)}
    )
    if not checked.competing_hypotheses:
        raise ValueError("competing_hypotheses must be explicitly enabled")


def append_hypotheses(
    state: HypothesisState,
    *,
    hypotheses: tuple[Hypothesis, ...] = (),
    assessments: tuple[HypothesisAssessment, ...] = (),
    triggers: tuple[HypothesisTrigger, ...] = (),
    config: ExperimentalOverlayConfig = ExperimentalOverlayConfig(),
) -> HypothesisState:
    """Propose a successor with all prior records retained; perform no storage operation.

    Args:
        state: Authoritative external prior, supplied by the trusted host.
        hypotheses: New immutable alternatives with unique IDs.
        assessments: New assessments or explicit same-assessor supersessions.
        triggers: Additional evidence-linked trigger declarations.
        config: Existing overlay configuration; competing_hypotheses must be True.

    Returns:
        Detached state with exactly one revision increment and unchanged history prefixes.

    Raises:
        ValueError: Disabled, invalid records/links, duplicate IDs or empty update.
    """
    _enabled(config)
    prior = _snapshot(state, HypothesisState)
    hypotheses = _records(hypotheses, Hypothesis)
    assessments = _records(assessments, HypothesisAssessment)
    triggers = _records(triggers, HypothesisTrigger)
    if not (hypotheses or assessments or triggers):
        raise ValueError("an update must append at least one record")
    return replace(
        prior,
        revision=prior.revision + 1,
        hypotheses=prior.hypotheses + hypotheses,
        assessments=prior.assessments + assessments,
        triggers=prior.triggers + triggers,
    )


def validate_extension(prior: HypothesisState, successor: HypothesisState) -> None:
    """Reject a proposed edit/deletion against the host's authoritative prior snapshot.

    Args:
        prior: Authenticated currently stored state; the host protects it from actor replacement.
        successor: Proposed state, possibly restored from JSON.

    Returns:
        None for a nonempty additive revision with all fixed fields unchanged.

    Raises:
        ValueError: Invalid records, edited/deleted history, identity drift or stale revision.
    """
    before = _snapshot(prior, HypothesisState)
    after = _snapshot(successor, HypothesisState)
    if after.revision != before.revision + 1:
        raise ValueError("successor must increment revision exactly once")
    history = ("hypotheses", "assessments", "triggers")
    for field in fields(HypothesisState):
        name = field.name
        old, new = getattr(before, name), getattr(after, name)
        if name in history:
            if new[: len(old)] != old:
                raise ValueError("successor cannot delete or edit history")
        elif name != "revision" and new != old:
            raise ValueError(f"successor cannot change {name}")
    if all(getattr(before, name) == getattr(after, name) for name in history):
        raise ValueError("successor must append at least one record")


def _current(state: HypothesisState) -> dict[str, dict[str, Any]]:
    replaced = {a.supersedes for a in state.assessments if a.supersedes is not None}
    current: dict[str, dict[str, Any]] = {}
    for hypothesis in state.hypotheses:
        live = [
            a
            for a in state.assessments
            if a.hypothesis_id == hypothesis.id and a.id not in replaced
        ]
        variants = {
            (a.status, tuple(getattr(a.scores, name) for name in SCORE_NAMES)) for a in live
        }
        conflict = len(variants) > 1
        scores = _encode(live[0].scores) if live and not conflict else None
        complete = scores is not None and all(v is not None for v in scores.values())
        current[hypothesis.id] = dict(
            scores=scores,
            statuses=sorted({a.status for a in live}) or ["unresolved"],
            conflicted=conflict,
            incomparable=not complete,
        )
    return current


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    ordered = [(left[n], right[n]) if n in MAXIMIZE else (right[n], left[n]) for n in SCORE_NAMES]
    return all(a >= b for a, b in ordered) and any(a > b for a, b in ordered)


def _pareto(state: HypothesisState, current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dominated = [
        h.id
        for h in state.hypotheses
        if not current[h.id]["incomparable"]
        and any(
            not other["incomparable"] and _dominates(other["scores"], current[h.id]["scores"])
            for other in current.values()
        )
    ]
    return dict(
        frontier=[h.id for h in state.hypotheses if h.id not in dominated],
        dominated=dominated,
        incomparable=[h.id for h in state.hypotheses if current[h.id]["incomparable"]],
        conflicted=[h.id for h in state.hypotheses if current[h.id]["conflicted"]],
        authority_granted=False,
    )


def pareto_hypotheses(
    state: HypothesisState, *, config: ExperimentalOverlayConfig = ExperimentalOverlayConfig()
) -> dict[str, Any]:
    """Diagnose strict dominance only among complete, agreed live assessment vectors.

    Args:
        state: Full external history under one measurement protocol and unit declaration.
        config: Existing overlay configuration with competing_hypotheses enabled.

    Returns:
        Frontier, dominated, incomparable and conflicted IDs in retained insertion order.
        Unknown/conflicted candidates stay on the possible frontier; no candidate is deleted.

    Raises:
        ValueError: Disabled or invalid state/configuration.
    """
    _enabled(config)
    state = _snapshot(state, HypothesisState)
    return _pareto(state, _current(state))


def project_hypotheses(
    state: HypothesisState,
    *,
    diagnostic_uncertainty: float,
    offset: int = 0,
    limit: int = 4,
    config: ExperimentalOverlayConfig = ExperimentalOverlayConfig(),
) -> dict[str, Any]:
    """Return a bounded actor view while retaining every external alternative.

    Args:
        state: Full external state; projection never mutates it.
        diagnostic_uncertainty: Host-declared legacy diagnostic, not an aggregate posterior.
        offset: Start index in retained insertion order, between zero and count minus two.
        limit: Exact integer page size from 2 through 32. Last page may overlap one item.
        config: Existing overlay configuration with competing_hypotheses enabled.

    Returns:
        Detached JSON: existing HypothesisSet diagnostic, selected public candidates, measurement
        protocol/units, revision, omission counts and next_offset. No evidence/assessor IDs.

    Raises:
        ValueError: Disabled, invalid state, numeric diagnostic or page coordinates.
    """
    _enabled(config)
    state = _snapshot(state, HypothesisState)
    _integer(offset, "offset")
    _integer(limit, "limit")
    _number(diagnostic_uncertainty, "diagnostic_uncertainty", unit=True)
    if diagnostic_uncertainty is None or not 2 <= limit <= 32:
        raise ValueError("projection requires numeric uncertainty and a limit from 2 through 32")
    count = len(state.hypotheses)
    if offset > count - 2:
        raise ValueError("offset must leave at least two alternatives")
    selected = state.hypotheses[offset : offset + limit]
    next_offset = min(offset + limit, count - 2) if offset + limit < count else None
    current = _current(state)
    pareto = _pareto(state, current)
    diagnostic = HypothesisSet(
        record_id="hypothesis-projection",
        source_case_id=state.source_case_id,
        uncertainty=float(diagnostic_uncertainty),
        provenance_refs=(
            EvidenceReference("public-hypothesis-projection", EvidenceSourceKind.EXTERNAL_RECORD),
        ),
        feature_flag="competing_hypotheses",
        maturity=ExperimentalMaturity.EXPERIMENTAL,
        diagnostic_status=DiagnosticStatus.DIAGNOSTIC,
        hypotheses=tuple(
            ": ".join(json.dumps(value, ensure_ascii=False) for value in (h.id, h.statement))
            for h in selected
        ),
    )
    return dict(
        schema_version="hypothesis-projection-v1",
        revision=state.revision,
        training_eligibility=state.training_eligibility.value,
        measurement=dict(
            protocol_id=state.protocol_id,
            complexity_unit=state.complexity_unit,
            compute_unit=state.compute_unit,
        ),
        diagnostic=diagnostic.to_dict(),
        authority_granted=False,
        candidates=[
            dict(
                id=h.id,
                statement=h.statement,
                **current[h.id],
                dominated=h.id in pareto["dominated"],
            )
            for h in selected
        ],
        total_hypotheses=count,
        omitted_count=count - len(selected),
        complete=len(selected) == count,
        offset=offset,
        next_offset=next_offset,
    )
