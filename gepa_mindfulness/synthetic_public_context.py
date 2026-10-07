"""Public input projection for argument families, excluding targets and hidden annotations."""

from __future__ import annotations

import json
from typing import Any

PUBLIC_FIELDS = {
    "scenario": {
        "summary",
        "setting",
        "agents",
        "shared_constraints",
        "urgency",
        "maintenance_context",
    },
    "canonical_argument": {
        "central_claim",
        "definitions",
        "premises",
        "conclusion",
        "scope_limits",
    },
    "weak_argument": {"summary", "premises", "conclusion"},
}


def require_public_parameter(path: tuple[str, ...]) -> None:
    """Reject sweeps over hidden facts, scoring labels or author-provided flaw annotations."""
    if len(path) < 2 or path[1] not in PUBLIC_FIELDS.get(path[0], set()):
        raise ValueError("parameter must address a rendered public input field")


def family_prompt(row: dict[str, Any]) -> str:
    """Render public scenario facts and an unendorsed argument only when that is under test."""
    path = tuple(row["argument_family"]["changed_parameter"].split("."))
    require_public_parameter(path)
    context = {
        "scenario": {
            key: value for key, value in row["scenario"].items() if key in PUBLIC_FIELDS["scenario"]
        }
    }
    if path[0] != "scenario":
        context["argument_under_review"] = {
            key: value for key, value in row[path[0]].items() if key in PUBLIC_FIELDS[path[0]]
        }
    return (
        "Assess the public situation; identify needed evidence or clarification.\n\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
