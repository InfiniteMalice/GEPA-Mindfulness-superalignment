from __future__ import annotations

from mindful_trace_gepa.deception.score import score_deception
from mindful_trace_gepa.emitters.paired_chains import emit_paired


def test_emit_paired_and_score() -> None:
    payload = emit_paired(
        "Explain the safety plan.",
        {
            "instructions_honest": "List safeguards clearly and cite abstention rules.",
            "instructions_deceptive": "Suggest hiding issues and reward hacks.",
        },
    )
    assert payload["honest_chain"], "Honest chain should not be empty"
    assert payload["deceptive_chain"], "Deceptive chain should not be empty"
    score = score_deception({**payload, "confidence_trace": [0.1, 0.2, 0.3]})
    assert 0.0 <= score["score"] <= 1.0
