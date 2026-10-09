"""Opt-in capture of simulated public preferences, with no authority promotion."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.debate_records import _record

from .perspective_protocol import (
    PerspectiveCandidate,
    PerspectiveCapture,
    PerspectivePlan,
    perspective_digest,
)


@dataclass(frozen=True)
class PerspectiveGenerationContext:
    """Allowlisted public semantics, without experimental IDs or admission metadata."""

    source_text: str
    facts: tuple[str, ...]
    constraints: tuple[str, ...]
    roles: tuple[dict[str, Any], ...]
    stakeholders: tuple[dict[str, Any], ...]
    slots: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON values without diagnostic-record envelope fields."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_plan(cls, plan: PerspectivePlan) -> PerspectiveGenerationContext:
        """Keep public actor names meaningful while replacing experiment slot identities."""
        return cls(
            plan.source.source_text,
            tuple(c.proposition for c in plan.source.facts),
            tuple(c.proposition for c in plan.source.constraints),
            tuple(
                dict(
                    actor_id=r.actor_id,
                    role_id=r.role_id,
                    rights=r.rights,
                    duties=r.duties,
                    authority=r.authority,
                )
                for r in plan.source.roles
            ),
            tuple(
                dict(
                    stakeholder_id=s.stakeholder_id,
                    stakeholder_role=s.stakeholder_role,
                    impact=s.impact,
                    inclusion=s.inclusion,
                    interests=s.interests,
                    hard_constraints=s.hard_constraints,
                    preferences=s.preferences,
                    preference_uncertainty=s.preference_uncertainty,
                )
                for s in plan.stakeholders
            ),
            tuple(
                dict(
                    slot_id=f"slot-{i}",
                    perspective=s.perspective.perspective,
                    stakeholder_ids=s.perspective.stakeholder_ids,
                )
                for i, s in enumerate(plan.slots)
            ),
        )


def capture_perspectives(
    plan: PerspectivePlan,
    *,
    candidates: tuple[PerspectiveCandidate, ...] | None = None,
    generate: (
        Callable[[PerspectiveGenerationContext], tuple[PerspectiveCandidate, ...]] | None
    ) = None,
    enabled: bool = False,
) -> PerspectiveCapture:
    """Capture one bounded attempt; the trusted host callback is not sandboxed or timed out."""
    if enabled is not True:
        raise ValueError("perspective capture requires enabled=True")
    plan = _record(plan, PerspectivePlan)
    digest = perspective_digest(plan)
    if candidates is not None and generate is not None:
        raise ValueError("choose either candidates or a generation callback")
    if generate is not None:
        if not callable(generate):
            raise ValueError("generate must be callable")
        try:
            candidates = generate(PerspectiveGenerationContext.from_plan(plan))
        except Exception as error:
            return PerspectiveCapture(digest, (), "callback_error", type(error).__name__)
        if candidates is None:
            raise ValueError("callback must return an exact candidate tuple")
        public_capture = PerspectiveCapture(digest, candidates, "observed", "callback result")
        aliases = {f"slot-{i}": s.slot_id for i, s in enumerate(plan.slots)}
        if not {c.slot_id for c in public_capture.candidates} <= set(aliases):
            raise ValueError("candidate references an unplanned public slot")
        candidates = tuple(
            replace(c, slot_id=aliases[c.slot_id]) for c in public_capture.candidates
        )
    elif candidates is None:
        return PerspectiveCapture(digest, (), "censored", "no generation source supplied")
    captured = PerspectiveCapture(digest, candidates, "observed", "public candidates captured")
    if not {c.slot_id for c in captured.candidates} <= {s.slot_id for s in plan.slots}:
        raise ValueError("candidate references an unplanned slot")
    return captured
