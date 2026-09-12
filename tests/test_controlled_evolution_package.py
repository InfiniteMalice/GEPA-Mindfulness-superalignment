"""Wheel-level contracts for controlled-evolution documentation resources."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RESOURCES = {
    "docs/__init__.py",
    "docs/controlled_evolution.md",
    "docs/experimental_v5_overlays.md",
    "docs/recommendations/RESEARCH_TRACEABILITY.md",
    "docs/recommendations/UNIFIED_RECOMMENDATIONS.md",
    "docs/recommendations/references.yaml",
    "docs/recommendations/registry.yaml",
    "evaluation/cases/experimental_overlays.yaml",
    "evaluation/experimental_overlays.py",
    "evaluation/experimental_records.py",
}


def test_wheel_bundles_readable_controlled_evolution_resources(tmp_path: Path) -> None:
    wheel_directory = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_directory),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert EXPECTED_RESOURCES <= names
        metadata_paths = tuple(name for name in names if name.endswith(".dist-info/METADATA"))
        assert metadata_paths == ("gepa_mindfulness-0.1.0.dist-info/METADATA",)
        metadata = archive.read(metadata_paths[0]).decode("utf-8")
        metadata_lines = metadata.splitlines()
        assert "Name: gepa-mindfulness" in metadata_lines
        assert "Version: 0.1.0" in metadata_lines

    smoke = """
import importlib.resources
import sys

sys.path.insert(0, sys.argv[1])
import docs

assert docs.__file__.startswith(sys.argv[1])
root = importlib.resources.files("docs")
recommendations = importlib.resources.files("docs.recommendations")
assert root.joinpath("controlled_evolution.md").read_text(encoding="utf-8").startswith(
    "# Controlled Learning and Offline Evolution"
)
assert root.joinpath("experimental_v5_overlays.md").read_text(encoding="utf-8").startswith(
    "# Experimental V5 Overlays"
)
from evaluation.experimental_overlays import ExperimentalOverlayConfig, enabled_overlays
from evaluation.experimental_records import ExperimentalMaturity
assert enabled_overlays(ExperimentalOverlayConfig()) == ()
assert ExperimentalMaturity.EXPERIMENTAL.value == "experimental"
assert importlib.resources.files("evaluation.cases").joinpath(
    "experimental_overlays.yaml"
).is_file()
assert recommendations.joinpath("UNIFIED_RECOMMENDATIONS.md").read_text(
    encoding="utf-8"
).startswith("# Unified V5 Recommendations")
assert recommendations.joinpath("RESEARCH_TRACEABILITY.md").read_text(
    encoding="utf-8"
).startswith("# V5 Research Traceability")
assert recommendations.joinpath("registry.yaml").read_text(encoding="utf-8").startswith(
    "registry_version: 17case-v5"
)
assert recommendations.joinpath("references.yaml").read_text(encoding="utf-8").startswith(
    "registry_version: 17case-v5"
)
print("wheel documentation resources are readable")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", smoke, str(wheel)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "wheel documentation resources are readable"
