"""Offline trace viewer builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

VIEWER_DIR = Path(__file__).resolve().parent


def load_static_asset(name: str) -> str:
    path = VIEWER_DIR / name
    candidate = path.with_name(f"{path.name}.new")
    if candidate.exists():
        path = candidate
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def build_viewer_html(
    trace_events: List[Dict[str, Any]],
    token_events: List[Dict[str, Any]],
    output_path: Path,
    deception: Dict[str, Any] | None = None,
    manifest: Dict[str, Any] | None = None,
    settings: Dict[str, Any] | None = None,
    scoring: Dict[str, Any] | None = None,
    *,
    dual_path: Dict[str, Any] | None = None,
    paired: Dict[str, Any] | None = None,
) -> Path:
    dual_path_payload = dual_path if dual_path is not None else paired
    html = load_static_asset("viewer.html")
    script = load_static_asset("viewer.js")
    css = load_static_asset("viewer.css")

    payload = {
        "trace": trace_events,
        "tokens": token_events,
        "deception": deception or {},
        "paired": dual_path_payload or {},
        "dualPath": dual_path_payload or {},
        "manifest": manifest or {},
        "settings": settings or {},
        "scoring": scoring or {},
    }
    data_blob = json.dumps(payload)

    bundle = html.replace("/*__VIEWER_CSS__*/", css).replace("/*__VIEWER_JS__*/", script)
    bundle = bundle.replace("/*__GEPA_DATA__*/", data_blob)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(bundle)
    return output_path


__all__ = ["build_viewer_html", "load_static_asset", "VIEWER_DIR"]
