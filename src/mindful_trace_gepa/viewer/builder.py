"""Offline trace viewer builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mindful_trace_gepa.logging_schema import normalize_trace_event
from mindful_trace_gepa.path_utils import atomic_write_text

VIEWER_DIR = Path(__file__).resolve().parent


def load_static_asset(name: str) -> str:
    path = VIEWER_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _safe_json_payload(payload: Dict[str, Any]) -> str:
    data_blob = json.dumps(payload, ensure_ascii=False)
    return (
        data_blob.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


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

    normalized_trace = [normalize_trace_event(event) for event in trace_events]
    payload = {
        "trace": normalized_trace,
        "tokens": token_events,
        "deception": deception or {},
        "paired": dual_path_payload or {},
        "dualPath": dual_path_payload or {},
        "manifest": manifest or {},
        "settings": settings or {},
        "scoring": scoring or {},
    }
    data_blob = _safe_json_payload(payload)

    bundle = html.replace("/*__VIEWER_CSS__*/", css).replace("/*__VIEWER_JS__*/", script)
    bundle = bundle.replace("/*__GEPA_DATA__*/", data_blob)
    atomic_write_text(output_path, bundle, encoding="utf-8")
    return output_path


__all__ = ["build_viewer_html", "load_static_asset", "VIEWER_DIR"]
