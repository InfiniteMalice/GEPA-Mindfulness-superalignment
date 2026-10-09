"""Opt-in capture of simulated public preferences, with no authority promotion."""

# Standard library
from __future__ import annotations

from collections.abc import Callable

# Third-party
# Local
from gepa_mindfulness.verification.debate_records import _record

from .perspective_protocol import (
    PerspectiveCandidate,
    PerspectiveCapture,
    PerspectivePlan,
    perspective_digest,
)


def capture_perspectives(
    plan: PerspectivePlan,
    *,
    candidates: tuple[PerspectiveCandidate, ...] | None = None,
    generate: Callable[[PerspectivePlan], tuple[PerspectiveCandidate, ...]] | None = None,
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
            candidates = generate(PerspectivePlan.from_dict(plan.to_dict()))
        except Exception as error:
            return PerspectiveCapture(digest, (), "callback_error", type(error).__name__)
        if candidates is None:
            raise ValueError("callback must return an exact candidate tuple")
    elif candidates is None:
        return PerspectiveCapture(digest, (), "censored", "no generation source supplied")
    captured = PerspectiveCapture(digest, candidates, "observed", "public candidates captured")
    if not {c.slot_id for c in captured.candidates} <= {s.slot_id for s in plan.slots}:
        raise ValueError("candidate references an unplanned slot")
    return captured
