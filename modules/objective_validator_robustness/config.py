"""Configuration for optional objective-validator robustness behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


@dataclass(frozen=True)
class ObjectiveValidatorRobustnessConfig:
    enabled: bool = False
    detect_validator_capture: bool = True
    detect_proxy_breakdown: bool = True
    detect_novelty: bool = True
    infer_plausible_objectives: bool = True
    preserve_optionality_under_uncertainty: bool = True
    prefer_reversible_actions: bool = True
    escalate_catastrophic_downside: bool = True
    raise_objective_validation_interrupts: bool = True
    emit_trace_events: bool = True
    inference_mode: str = "heuristic"

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "ObjectiveValidatorRobustnessConfig":
        data = dict(payload or {})
        section = data.get("objective_validator_robustness")
        if isinstance(section, Mapping):
            data = dict(section)
        return cls(
            enabled=bool(data.get("enabled", False)),
            detect_validator_capture=bool(data.get("detect_validator_capture", True)),
            detect_proxy_breakdown=bool(data.get("detect_proxy_breakdown", True)),
            detect_novelty=bool(data.get("detect_novelty", True)),
            infer_plausible_objectives=bool(data.get("infer_plausible_objectives", True)),
            preserve_optionality_under_uncertainty=bool(
                data.get("preserve_optionality_under_uncertainty", True)
            ),
            prefer_reversible_actions=bool(data.get("prefer_reversible_actions", True)),
            escalate_catastrophic_downside=bool(
                data.get("escalate_catastrophic_downside", True)
            ),
            raise_objective_validation_interrupts=bool(
                data.get("raise_objective_validation_interrupts", True)
            ),
            emit_trace_events=bool(data.get("emit_trace_events", True)),
            inference_mode=str(data.get("inference_mode", "heuristic")),
        )


def load_objective_validator_robustness_config(
    path: str | Path,
) -> ObjectiveValidatorRobustnessConfig:
    """Load objective robustness configuration from a YAML file."""

    target = Path(path)
    raw = target.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw) or {}
    else:
        payload = _parse_simple_yaml(raw)
    if not isinstance(payload, Mapping):
        raise TypeError("objective validator robustness config must load to a mapping")
    return ObjectiveValidatorRobustnessConfig.from_mapping(payload)


def _parse_simple_yaml(raw: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            current_section = key
            data.setdefault(key, {})
            continue
        target = data[current_section] if current_section else data
        lowered = value.lower()
        if lowered in {"true", "false"}:
            target[key] = lowered == "true"
        else:
            target[key] = value.strip("'\"")
    return data
