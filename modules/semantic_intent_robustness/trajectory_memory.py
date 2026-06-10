"""Cross-session trajectory summaries without raw KV persistence."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrajectorySummary:
    """Bounded cross-session trajectory summary."""

    summary_id: str
    principal_scope: str
    created_at: str
    expires_at: str
    capability_domains: tuple[str, ...]
    redacted_fragment_summaries: tuple[str, ...]
    contextual_risk: float
    closure_risk: float
    trajectory_reasons: tuple[str, ...]
    provenance_references: tuple[str, ...]
    review_status: str
    raw_kv_persisted: bool = False


def quarantine_if_untrusted(summary: TrajectorySummary) -> TrajectorySummary:
    """Quarantine summaries that attempt to alter authority or policy."""

    joined = " ".join(summary.redacted_fragment_summaries).lower()
    if any(marker in joined for marker in ("ignore policy", "override authority", "new goal")):
        return replace(summary, review_status="quarantined")
    return summary


__all__ = ["TrajectorySummary", "quarantine_if_untrusted"]
