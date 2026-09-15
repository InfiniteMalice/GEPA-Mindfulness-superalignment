"""Keep the invariant inventory linked to executable regression suites."""

from pathlib import Path

import yaml


def test_invariants_reference_real_tests_and_disclose_scoped_enforcement() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((root / "research/invariants.yaml").read_text(encoding="utf-8"))
    assert registry["version"] == "17case-v5"
    for invariant in registry["invariants"].values():
        assert invariant["rule"]
        assert invariant["enforcement"] in {"machine", "machine_scoped", "host_integration"}
        assert invariant["tests"]
        for path in invariant["tests"]:
            source = (root / path).read_text(encoding="utf-8")
            assert "def test_" in source
        if invariant["enforcement"] != "machine":
            assert invariant["scope"]
