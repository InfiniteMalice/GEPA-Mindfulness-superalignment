from factuality_certification import EvidenceItem, FactualityCertificationConfig, certify_answer
from factuality_certification.certification import positive_only_reward_features


def test_supported_claim():
    res = certify_answer(
        "What happened?",
        "The paper reports 82% accuracy.",
        [
            EvidenceItem(
                id="e1",
                text="The paper reports 82% accuracy.",
                quality_score=1.0,
                retrieval_score=1.0,
            )
        ],
    )
    assert res.recommended_action == "answer"


def test_unsupported_claim():
    res = certify_answer("Q", "Mars has 3 moons.", [])
    assert res.overall_label in {"uncertified", "partial"}


def test_contradicted_claim():
    res = certify_answer(
        "Q",
        "The trial was not successful.",
        [EvidenceItem(id="e1", text="The trial was successful.")],
    )
    assert any(s.support_label == "contradicted" for s in res.claim_support)


def test_partial_answer():
    res = certify_answer(
        "Q",
        "The paper reports 82% accuracy and was released today.",
        [
            EvidenceItem(
                id="e1",
                text="The paper reports 82% accuracy.",
                quality_score=1.0,
                retrieval_score=1.0,
            )
        ],
    )
    assert res.recommended_action in {"answer_partially", "answer_with_qualifications", "answer"}


def test_overrefusal_detected():
    res = certify_answer(
        "Q",
        "I cannot answer this question.",
        [
            EvidenceItem(
                id="e1",
                text="Answer: Paris is the capital of France.",
                quality_score=1.0,
                retrieval_score=1.0,
            )
        ],
    )
    assert res.logs["overrefusal_detected"] is True


def test_proper_abstention():
    res = certify_answer("Q", "Insufficient evidence to answer.", [])
    assert res.recommended_action == "abstain"


def test_current_source_requirement():
    res = certify_answer("Q", "The latest law changed today.", [])
    assert res.logs["current_source_claim_ids"]


def test_no_thought_penalty():
    res = certify_answer("Q", "Insufficient evidence to answer.", [])
    feats = positive_only_reward_features(res)
    assert all(v >= 0 for v in feats.values())


def test_config_modes():
    for mode in ["off", "shadow", "advisory", "gated", "training"]:
        cfg = FactualityCertificationConfig(mode=mode, enabled=True)
        res = certify_answer("Q", "A factual claim.", [], config=cfg)
        assert res.mode == mode
