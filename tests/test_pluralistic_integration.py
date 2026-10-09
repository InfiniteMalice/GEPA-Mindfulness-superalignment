"""Execute the packaged public workflow offline, without promoting simulated preferences."""

# Standard library
import re
from pathlib import Path

# Third-party
import pytest

# Local
from gepa_mindfulness.training.eligibility import require_training_eligible


def test_offline_pluralistic_guide_executes(monkeypatch):
    """The guide composes all three PRs and retains one explicitly missing condition."""

    def blocked(*args, **kwargs):
        pytest.fail("pluralistic guide attempted network access")

    monkeypatch.setattr("socket.socket.connect", blocked)
    path = Path(__file__).resolve().parents[1] / "docs" / "pluralistic_robustness.md"
    code = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)[0]
    namespace = {}
    exec(compile(code, str(path), "exec"), namespace)
    report = namespace["report"]
    assert len(namespace["families"]) == 9 and len(namespace["curriculum"]) == 21
    assert namespace["role_pair"].claimed_equivalence
    assert not namespace["authority_pair"].claimed_equivalence
    assert report["conditions"]["plain_synthetic"]["missing_runs"] == ["plain_synthetic"]
    combined = report["run_rows"][-1]
    assert combined["combined_protocol_complete"]
    assert combined["pluralistic_report"]["verified_claims"]["risk"]["status"] == "supported"
    assert (
        combined["pluralistic_report"]["protocol"]["plan"]["source"]["facts"][0]["status"]
        == "unverified"
    )
    assert report["training_effect_established"] is False and report["optimizer_input"] is False
    assert report["paired"][0]["deltas"]["perspective_robustness"] is None
    with pytest.raises(ValueError):
        require_training_eligible(report)
