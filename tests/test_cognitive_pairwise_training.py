from __future__ import annotations

import json

from cognitive_pairwise_training import (
    CognitivePairwiseTrainingConfig,
    PairType,
    PairwiseLabel,
    ReasoningTraceCandidate,
    build_pairwise_examples,
    compute_pairwise_label_loss,
    export_pairwise_jsonl,
)
from gepa_mindfulness.training.configs import TrainingConfig
from mindful_trace_gepa.logging_schema import StructuredEventType, make_event_envelope


def _candidate(**overrides) -> ReasoningTraceCandidate:
    payload = {
        "candidate_id": "cand-a",
        "problem_id": "p1",
        "prompt": "Answer the abstract puzzle.",
        "public_reasoning_summary": "Supported concise reasoning.",
        "structured_reasoning_units": ({"unit_id": "u1"},),
        "final_answer": "A",
        "reference_answer": "A",
        "model_id": "small",
        "model_scale": 1.0,
        "checkpoint_id": "ckpt",
        "rollout_id": "rollout",
        "correctness": True,
        "confidence": 0.8,
        "abstained": False,
        "verifier_status": "verified",
        "metadata": {},
    }
    payload.update(overrides)
    return ReasoningTraceCandidate(**payload)


def _candidates() -> list[ReasoningTraceCandidate]:
    return [
        _candidate(
            candidate_id="s-good",
            model_id="small",
            model_scale=1,
            correctness=True,
            verifier_status="verified",
        ),
        _candidate(
            candidate_id="s-bad",
            model_id="small",
            model_scale=1,
            correctness=False,
            verifier_status="failed",
            final_answer="B",
        ),
        _candidate(
            candidate_id="l-bad",
            model_id="large",
            model_scale=10,
            correctness=False,
            verifier_status="failed",
            final_answer="C",
        ),
        _candidate(
            candidate_id="l-same",
            model_id="large",
            model_scale=10,
            correctness=True,
            verifier_status="failed",
            final_answer="A",
        ),
        _candidate(
            candidate_id="abstain",
            model_id="small",
            model_scale=1,
            correctness=False,
            abstained=True,
            verifier_status="verified",
            final_answer="I don't know",
        ),
    ]


def test_intra_model_pairs_are_generated() -> None:
    pairs = build_pairwise_examples(_candidates())
    assert any(pair.pair_type == PairType.INTRA_MODEL for pair in pairs)


def test_inter_model_pairs_are_generated() -> None:
    pairs = build_pairwise_examples(_candidates())
    assert any(pair.pair_type == PairType.INTER_MODEL for pair in pairs)


def test_smaller_correct_versus_larger_incorrect_pairs_are_generated() -> None:
    pairs = build_pairwise_examples(_candidates())
    assert any(pair.pair_type == PairType.SMALL_CORRECT_LARGE_INCORRECT for pair in pairs)


def test_order_randomization_is_deterministic_under_fixed_seed() -> None:
    cfg = CognitivePairwiseTrainingConfig(seed=123, randomize_pair_order=True)
    first = [pair.to_dict() for pair in build_pairwise_examples(_candidates(), config=cfg)]
    second = [pair.to_dict() for pair in build_pairwise_examples(_candidates(), config=cfg)]
    assert first == second
    assert any(pair["trace_order_randomized"] for pair in first)


def test_pair_labels_do_not_depend_only_on_final_answer_correctness() -> None:
    pairs = build_pairwise_examples(
        _candidates(),
        config=CognitivePairwiseTrainingConfig(randomize_pair_order=False),
    )
    same_answer_pair = next(
        pair for pair in pairs if pair.pair_type == PairType.ANSWER_PROCESS_MISMATCH
    )
    assert same_answer_pair.candidate_a.final_answer == same_answer_pair.candidate_b.final_answer
    assert same_answer_pair.teacher_label in {
        PairwiseLabel.A_MORE_TRUSTWORTHY,
        PairwiseLabel.B_MORE_TRUSTWORTHY,
    }


def test_same_answer_different_reasoning_quality_pairs_are_supported() -> None:
    assert any(
        pair.pair_type == PairType.SAME_ANSWER_DIFFERENT_REASONING_QUALITY
        for pair in build_pairwise_examples(_candidates())
    )


def test_abstention_versus_guess_pairs_are_supported() -> None:
    assert any(
        pair.pair_type == PairType.ABSTENTION_VERSUS_GUESS
        for pair in build_pairwise_examples(_candidates())
    )


def test_teacher_consensus_filtering_works() -> None:
    close = [
        _candidate(candidate_id="a", confidence=0.5, correctness=None, verifier_status="unknown"),
        _candidate(
            candidate_id="b",
            confidence=0.5,
            correctness=None,
            verifier_status="unknown",
            model_id="other",
        ),
    ]
    assert (
        build_pairwise_examples(
            close,
            config=CognitivePairwiseTrainingConfig(consensus_filtering=True),
        )
        == []
    )


def test_diagnostics_are_computed_and_serialized(tmp_path) -> None:
    pairs = build_pairwise_examples(_candidates())
    metrics = compute_pairwise_label_loss(pairs)
    assert metrics.position_bias_diagnostic >= 0.0
    assert metrics.length_bias_diagnostic >= 0.0
    assert metrics.final_answer_shortcut_diagnostic >= 0.0
    assert metrics.model_identity_shortcut_diagnostic >= 0.0
    out = export_pairwise_jsonl(pairs, tmp_path / "pairs.jsonl")
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "teacher_label" in row


def test_cpt_is_disabled_by_default_and_trainer_config_preserves_defaults() -> None:
    config = TrainingConfig.from_mapping({})
    assert config.cognitive_pairwise_training.enabled is False


def test_cpt_trace_events_join_to_rollouts_by_stable_id() -> None:
    pair = build_pairwise_examples(_candidates())[0]
    event = make_event_envelope(
        StructuredEventType.CPT_PAIRWISE_EXAMPLE,
        {"pair_id": pair.pair_id},
        rollout_id=pair.candidate_a.rollout_id,
    ).to_dict()
    assert event["rollout_id"] == pair.candidate_a.rollout_id
