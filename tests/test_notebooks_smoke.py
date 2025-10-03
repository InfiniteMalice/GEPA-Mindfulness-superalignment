from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = [
    Path("notebooks/ft_phi3_mini_unsloth_gepa.ipynb"),
    Path("notebooks/ft_llama3_8b_unsloth_gepa.ipynb"),
]


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_final_cell_contains_cli(notebook_path: Path) -> None:
    """Ensure the notebook ends with the GEPA scoring/viewer commands.

    The regression suite does not execute the notebooks; instead we verify that
    the static artefacts include the documented quickstart commands so users can
    reproduce the scoring flow offline.
    """

    payload = json.loads(notebook_path.read_text())
    cells = payload["cells"]
    assert cells, f"Notebook {notebook_path} contains no cells"
    final_cell = cells[-1]
    assert final_cell["cell_type"] == "code"
    lines = "\n".join(final_cell["source"]).strip().splitlines()
    assert lines[0].startswith("!gepa score --trace runs/trace.jsonl"), lines
    assert any("gepa view" in line for line in lines), lines


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_mentions_abstention_threshold(notebook_path: Path) -> None:
    payload = json.loads(notebook_path.read_text())
    sources = []
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            sources.extend(cell["source"])
    joined = "\n".join(sources)
    assert "ABSTENTION_THRESHOLD" in joined
