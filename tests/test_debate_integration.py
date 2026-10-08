"""Executable offline guide and end-to-end negative controls preserve authority boundaries."""

import re
from dataclasses import replace
from pathlib import Path

import pytest
from test_debate_records import protocol, snapshot
from test_sensitive_debate import run_fixture

from evaluation.debate_analysis import analyze_debate

ROOT = Path(__file__).resolve().parents[1]


def test_offline_guide_executes_without_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("offline guide attempted network access")

    monkeypatch.setattr("socket.socket.connect", blocked)
    guide = (ROOT / "docs" / "sensitive_debate.md").read_text(encoding="utf-8")
    code = re.findall(r"```python\n(.*?)```", guide, re.S)[0]
    namespace = {}
    exec(compile(code, "sensitive_debate.md", "exec"), namespace)
    report = namespace["report"]
    assert report["first_verified_decision_change_round"] == 1
    assert report["false_challenges"]["rate"] == 0.5
    assert namespace["causal_report"]["pairs"][0]["classification"] == "correct_sensitivity"
    assert str(namespace["sensitivity"].value) == "3/2"


@pytest.mark.parametrize("control", ["inconclusive", "rhetoric", "alternative", "undecomposable"])
def test_negative_controls_do_not_promote_truth_or_rewards(control):
    p = protocol(1)
    s = snapshot()
    original = s.to_dict()
    options = {}
    if control == "inconclusive":
        options["authenticate"] = None
    elif control == "rhetoric":
        options["defend"] = lambda ctx: replace(s, public_statement="A very persuasive claim")
        options["revise"] = lambda ctx, before, ch, checks: replace(
            before, public_statement="Even more persuasive"
        )
    elif control == "alternative":
        options["revise"] = lambda ctx, before, ch, checks: replace(
            before, proposed_action="another acceptable action"
        )
    else:
        options["defend"] = lambda ctx: replace(
            s, graph=None, conclusion_claim_id=None, decomposition_status="undecomposable"
        )
    session = run_fixture(p, **options)
    report = analyze_debate(p, session, enabled=True)
    assert report["confers_authority"] is False
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["first_verified_decision_change_round"] is None
    assert s.to_dict() == original
    assert "reward" not in report and "authorization" not in report
