"""Configuration helpers for Mindful Trace GEPA."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback parser
    yaml = None  # type: ignore

DEFAULT_DSPY_CONFIG_PATH = Path("policies/dspy.yml")


@dataclass
class DSPyConfig:
    enabled: bool = False
    allow_optimizations: bool = False
    forbidden_phrases: tuple[str, ...] = tuple()
    max_variants_per_module: int = 1

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "DSPyConfig":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            allow_optimizations=bool(payload.get("allow_optimizations", False)),
            forbidden_phrases=tuple(payload.get("forbidden_phrases", []) or []),
            max_variants_per_module=int(payload.get("max_variants_per_module", 1)),
        )


def load_dspy_config(path: Path | None = None) -> DSPyConfig:
    target = path or DEFAULT_DSPY_CONFIG_PATH
    if not target.exists():
        return DSPyConfig()
    with target.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    if yaml is not None:
        data = yaml.safe_load(raw) or {}
    else:
        data = _parse_simple_yaml(raw)
    return DSPyConfig.from_mapping(data)


def _parse_simple_yaml(raw: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_list: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and current_list is not None:
            value = stripped[1:].strip().strip("'\"")
            data.setdefault(current_list, []).append(value)
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            current_list = key
            data.setdefault(key, [])
            continue
        current_list = None
        lowered = value.lower()
        if lowered in {"true", "false"}:
            data[key] = lowered == "true"
        else:
            try:
                if "." in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value.strip("'\"")
    return data


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


__all__ = ["DSPyConfig", "load_dspy_config", "dump_json", "DEFAULT_DSPY_CONFIG_PATH"]
