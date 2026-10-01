"""Admission, exposure and real gradient tests for the opt-in pair trainer."""

from dataclasses import replace

import pytest

from cognitive_pairwise_training.schemas import (
    PairType,
    PairwiseLabel,
    PairwiseReasoningExample,
    ReasoningTraceCandidate,
)
from gepa_mindfulness.training.contrastive import (
    NegativeFamily,
    prepare_pairs,
    train_contrastive,
)


def authored_pair(family, *, group=None, eligibility="TRAIN"):
    """Small authored contract fixture; no generated world is admitted to training."""
    group = group or family.value
    prompt = f"Task {group}: choose the supported public answer."
    candidates = tuple(
        ReasoningTraceCandidate(
            candidate_id=f"{group}:{answer}",
            problem_id=group,
            prompt=prompt,
            public_reasoning_summary="PRIVATE SUMMARY MUST NOT REACH SCORER",
            structured_reasoning_units=(),
            final_answer=answer,
            reference_answer=None,
            model_id="authored",
            model_scale=0,
            checkpoint_id="fixture",
            rollout_id=group,
            correctness=answer == "supported",
            confidence=0.5,
            abstained=False,
            verifier_status="authored",
        )
        for answer in ("supported", "unsupported")
    )
    return PairwiseReasoningExample(
        pair_id=group,
        problem_id=group,
        candidate_a=candidates[0],
        candidate_b=candidates[1],
        pair_type=PairType.INTRA_MODEL,
        trace_order_randomized=False,
        teacher_label=PairwiseLabel.A_MORE_TRUSTWORTHY,
        teacher_confidence=1,
        teacher_rationale_summary="Observable answer support only.",
        consensus_status="agreed",
        difficulty_bucket=family.value,
        metadata={
            "negative_family": family.value,
            "source_group": group,
            "training_eligibility": eligibility,
        },
    )


def catalog():
    return tuple(authored_pair(family) for family in NegativeFamily)


def test_snapshot_and_b_preference():
    pair = authored_pair(NegativeFamily.CAUSAL)
    pair = replace(pair, teacher_label=PairwiseLabel.B_MORE_TRUSTWORTHY)
    prepared = prepare_pairs((pair,), for_training=True)
    pair.metadata["source_group"] = "changed"
    assert prepared[0].source_group == NegativeFamily.CAUSAL.value
    assert prepared[0].chosen == "unsupported"
    assert "PRIVATE" not in prepared[0].prompt


@pytest.mark.parametrize("change", ["ambiguous", "same", "prompt", "family", "eligibility"])
def test_reject_malformed_pair(change):
    pair = authored_pair(NegativeFamily.CAUSAL)
    if change == "ambiguous":
        pair = replace(pair, teacher_label=PairwiseLabel.BOTH_TRUSTWORTHY)
    elif change == "same":
        pair = replace(pair, candidate_b=replace(pair.candidate_b, final_answer="supported"))
    elif change == "prompt":
        pair = replace(pair, candidate_b=replace(pair.candidate_b, prompt="different"))
    else:
        pair.metadata.pop("negative_family" if change == "family" else "training_eligibility")
    with pytest.raises(ValueError):
        prepare_pairs((pair,), for_training=True)


def test_duplicate_renamed_content_is_rejected():
    pair = authored_pair(NegativeFamily.CAUSAL)
    renamed = replace(pair, pair_id="other", problem_id="other")
    with pytest.raises(ValueError):
        prepare_pairs((pair, renamed), for_training=True)


def test_full_admission_precedes_model_calls():
    pairs = catalog()
    pairs[-1].candidate_b.metadata["source_record"] = {"training_eligibility": "HIDDEN_EVAL"}
    calls = []
    with pytest.raises(ValueError, match="HIDDEN_EVAL"):
        train_contrastive(pairs, lambda *args: calls.append(args), None, enabled=True)
    assert not calls


@pytest.mark.parametrize("enabled", [False, "true", 1])
def test_disabled_path(enabled):
    with pytest.raises(ValueError, match="enabled=True"):
        train_contrastive(catalog(), None, None, enabled=enabled)


def test_real_gradient_and_matched_update_budget():
    torch = pytest.importorskip("torch")
    reports = []
    for schedule in ("curriculum", "pooled"):
        weight = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([weight], lr=0.2)
        seen = []

        def score(prompt, answers):
            assert "PRIVATE" not in prompt
            seen.append(answers)
            return torch.stack([weight if a == "supported" else -weight for a in answers])

        report = train_contrastive(
            catalog(), score, optimizer, epochs=4, schedule=schedule, seed=3, enabled=True
        )
        assert weight.item() > 1
        assert report["losses"][-1] < report["losses"][0]
        assert set(seen) == {("supported", "unsupported"), ("unsupported", "supported")}
        reports.append(report)
    assert reports[0]["updates_by_family"] == reports[1]["updates_by_family"]
    assert reports[0]["dataset_digest"] == reports[1]["dataset_digest"]
    assert reports[0]["family_order"] == [f.value for f in NegativeFamily for _ in range(4)]


@pytest.mark.parametrize("kind", ["nan", "shape", "detached", "gradient"])
def test_invalid_tensor_does_not_update(kind):
    torch = pytest.importorskip("torch")
    weight = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([weight], lr=0.2)

    def score(prompt, answers):
        if kind == "nan":
            return torch.stack([weight * float("nan"), weight])
        if kind == "shape":
            return weight.reshape(1)
        if kind == "detached":
            return torch.tensor([1.0, 0.0])
        return torch.stack([weight.sqrt(), weight])

    with pytest.raises(ValueError):
        train_contrastive(catalog(), score, optimizer, enabled=True)
    assert weight.item() == 0


def test_requires_all_families_and_strict_config():
    for kwargs in ({"epochs": True}, {"seed": True}, {"schedule": "unknown"}):
        with pytest.raises(ValueError):
            train_contrastive(catalog(), None, None, enabled=True, **kwargs)
    with pytest.raises(ValueError, match="five families"):
        train_contrastive(catalog()[:1], None, None, enabled=True)


def test_snapshot_digest_binds_provenance():
    pair = authored_pair(NegativeFamily.CAUSAL)
    before = prepare_pairs((pair,), for_training=True)[0]
    pair.metadata["source_record"] = {"revision": "updated"}
    after = prepare_pairs((pair,), for_training=True)[0]
    assert before.source_digest != after.source_digest
    assert before.fingerprint == after.fingerprint


def test_optimizer_overflow_cannot_return_success_report():
    torch = pytest.importorskip("torch")
    weight = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([weight], lr=float("inf"))

    def score(prompt, answers):
        return torch.stack([weight if a == "supported" else -weight for a in answers])

    with pytest.raises(ValueError, match="parameters"):
        train_contrastive(catalog(), score, optimizer, enabled=True)
