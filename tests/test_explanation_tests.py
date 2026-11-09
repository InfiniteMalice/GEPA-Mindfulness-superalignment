"""Locked prediction protocol tests; synthetic amplitudes are software fixtures, not neurons."""

# Standard library
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timezone

# Third-party
import pytest

# Local
from gepa_mindfulness.interpret import explanation_tests as explanation_module
from gepa_mindfulness.interpret.explanation_tests import (
    ActivationPredictor,
    ExampleRef,
    ExplanationHypothesis,
    ExplanationKind,
    SplitManifest,
    TokenFeature,
    evaluate_explanation,
    fit_ridge_predictor,
    freeze_explanation,
    generate_context_hypotheses,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput
from gepa_mindfulness.training.eligibility import require_training_eligible

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class Clock:
    def now(self):
        return datetime(2026, 11, 5, tzinfo=timezone.utc)


@pytest.fixture
def data():
    torch.manual_seed(23)
    source = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_embd=8,
            n_inner=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    native = FrozenNativeModel(
        source, model_id="software-fixture", revision="seed-23", enabled=True
    )

    def batch(ids, tokens):
        actual = native.capture(
            native.hook(layer=0, kind=HookKind.MLP_POST),
            torch.tensor(tokens),
            component_indices=(0,),
            example_ids=ids,
            token_positions=(0, 1, 2),
            enabled=True,
        )
        # An explicitly synthetic target function for exact prediction/falsification tests.
        values = tuple(
            replace(m, activation=(1.0 if m.context_tokens[m.token_position] == 3 else -1.0,))
            for m in actual.measurements
        )
        return replace(actual, measurements=values)

    generation = batch(("g1", "g2"), [[0, 3, 2], [1, 2, 3]])
    development = batch(("d1",), [[2, 1, 3]])
    locked = batch(("l1", "l2", "l3"), [[3, 1, 1], [4, 2, 2], [5, 3, 3]])
    manifest = SplitManifest(
        tuple(ExampleRef.from_input(i, distribution="id") for i in generation.inputs),
        tuple(ExampleRef.from_input(i, distribution="id") for i in development.inputs),
        tuple(
            ExampleRef.from_input(i, distribution=group)
            for i, group in zip(locked.inputs, ("positive", "near_miss", "ood"))
        ),
    )
    return generation, development, locked, manifest


def hypothesis(generation, *, weights=(2.0,), bias=-1.0):
    predictor = ActivationPredictor(
        (TokenFeature("token_three", "token_is", 0, 3),), weights, bias, "rule", ()
    )
    return ExplanationHypothesis(
        "token-three-hypothesis",
        "Synthetic equality rule; semantics remain unspecified.",
        ExplanationKind.RELATED_CONTEXT,
        ("unrelated mixture", "unknown mechanism"),
        ("Replacing token three with token one should reverse the signed prediction.",),
        predictor,
        tuple(i.example_id for i in generation.inputs),
    )


def frozen(data, **kwargs):
    generation, _, _, manifest = data
    return freeze_explanation(
        hypothesis(generation, **kwargs),
        manifest,
        generation,
        clock=Clock(),
        max_mae=0.01,
        activation_threshold=0.5,
        enabled=True,
    )


def test_closed_predictor_reads_only_permitted_tokens_positions_and_relations() -> None:
    context = NativeInput("input", (3, 1, 3), (1, 1, 1))
    predictor = ActivationPredictor(
        (
            TokenFeature("preceding_three", "token_is", -1, 3),
            TokenFeature("neighbors_match", "tokens_equal", -1, None, 1),
        ),
        (2.0, 3.0),
        -1.0,
        "rule",
        (),
    )
    assert predictor.predict(context, 1) == 4.0
    assert predictor.predict(context, 0) == -1.0
    with pytest.raises(ValueError):
        predictor.predict({"actual_target_activation": 4.0}, 1)


def test_split_manifest_rejects_id_and_exact_context_leakage(data) -> None:
    generation, _, locked, manifest = data
    with pytest.raises(ValueError, match="split"):
        replace(manifest, locked=(manifest.generation[0],))
    same_context = replace(generation.inputs[0], example_id="different-id")
    with pytest.raises(ValueError, match="context"):
        replace(manifest, locked=(ExampleRef.from_input(same_context, distribution="ood"),))
    with pytest.raises(ValueError, match="generation"):
        fit_ridge_predictor(
            manifest,
            locked,
            features=(TokenFeature("t", "token_is", 0, 3),),
            ridge=1e-6,
            enabled=True,
        )


def _legacy_sum(values: Iterable[float], start: float = 0) -> float:
    """Reproduce the sequential float accumulation used before Python 3.12."""
    total = start
    for value in values:
        total += value
    return total


@pytest.mark.parametrize("legacy_accumulation", (False, True))
def test_rule_freezes_before_locked_prediction_and_can_be_falsified(
    data, monkeypatch, legacy_accumulation
) -> None:
    """Reject the wrong predictor under both old and current float accumulation."""
    if legacy_accumulation:
        monkeypatch.setattr(explanation_module, "sum", _legacy_sum, raising=False)
    _, _, locked, _ = data
    commit = frozen(data)
    restored = type(commit).from_dict(commit.to_dict())
    assert restored == commit
    report = evaluate_explanation(commit, locked, enabled=True)
    assert report["metrics"]["mae"] == 0
    assert report["metrics"]["false_positives"] == 0
    assert report["metrics"]["false_negatives"] == 0
    assert report["prediction_result"] == "PREREGISTERED_CRITERION_MET"
    assert report["evidence_status"] == "PREDICTIVE_HYPOTHESIS"
    assert report["measurement_authentication"] == "NOT_PERFORMED"
    assert len(report["rows"]) == 9
    wrong = evaluate_explanation(frozen(data, weights=(-2.0,), bias=1.0), locked, enabled=True)
    assert wrong["prediction_result"] == "PREDICTIONS_FALSIFIED"
    assert wrong["metrics"]["mae"] == pytest.approx(2.0, abs=1e-12, rel=0.0)
    assert wrong["metrics"]["sign_errors"] == 9
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report)


def test_bounded_learned_predictor_is_fitted_on_generation_only(data) -> None:
    generation, _, locked, manifest = data
    predictor = fit_ridge_predictor(
        manifest,
        generation,
        features=(TokenFeature("t", "token_is", 0, 3),),
        ridge=1e-6,
        enabled=True,
    )
    assert set(predictor.fit_example_ids) == {"g1", "g2"}
    assert predictor.kind == "ridge"
    assert abs(predictor.predict(locked.inputs[0], 0) - 1.0) < 1e-4
    h = replace(hypothesis(generation), predictor=predictor)
    commit = freeze_explanation(
        h, manifest, generation, clock=Clock(), max_mae=0.01, activation_threshold=0.5, enabled=True
    )
    assert evaluate_explanation(commit, locked, enabled=True)["metrics"]["mae"] < 1e-4


def test_locked_identity_model_hook_and_predictor_binding_are_enforced(data) -> None:
    generation, _, locked, manifest = data
    commit = frozen(data)
    with pytest.raises(ValueError, match="locked"):
        evaluate_explanation(commit, generation, enabled=True)
    other_input = replace(locked.inputs[0], token_ids=(2, 1, 1))
    changed_measurements = tuple(
        (
            replace(m, input_digest=other_input.digest, context_tokens=other_input.token_ids)
            if m.example_id == other_input.example_id
            else m
        )
        for m in locked.measurements
    )
    changed = replace(
        locked, inputs=(other_input,) + locked.inputs[1:], measurements=changed_measurements
    )
    with pytest.raises(ValueError, match="locked"):
        evaluate_explanation(commit, changed, enabled=True)
    forged = replace(hypothesis(generation).predictor, fit_example_ids=("l1",), kind="ridge")
    with pytest.raises(ValueError, match="generation"):
        freeze_explanation(
            replace(hypothesis(generation), predictor=forged),
            manifest,
            generation,
            clock=Clock(),
            max_mae=0.01,
            activation_threshold=0.5,
            enabled=True,
        )


def test_generation_retains_competitors_and_uses_no_locked_outcomes(data) -> None:
    generation, _, _, manifest = data
    candidates = generate_context_hypotheses(generation, manifest, max_hypotheses=4, enabled=True)
    assert 2 <= len(candidates) <= 4
    assert any(h.mechanism_kind is ExplanationKind.UNKNOWN for h in candidates)
    assert all(h.competitors and h.falsification_predictions for h in candidates)
    assert all(set(h.generation_example_ids) == {"g1", "g2"} for h in candidates)


def test_explanation_paths_are_disabled_by_default(data) -> None:
    with pytest.raises(ValueError, match="enabled=True"):
        evaluate_explanation(None, None)
    with pytest.raises(ValueError, match="enabled=True"):
        fit_ridge_predictor(None, None, features=(), ridge=0.0)


def test_near_misses_are_new_public_inputs_with_predictions_not_fabricated_measurements(
    data,
) -> None:
    from gepa_mindfulness.interpret.explanation_tests import generate_near_misses

    generation, _, _, manifest = data
    report = generate_near_misses(
        hypothesis(generation), generation, manifest, replacement_tokens=(1, 2, 3), enabled=True
    )
    assert report["candidates"]
    assert any(row["predicted_change"] != 0 for row in report["candidates"])
    for row in report["candidates"]:
        assert "actual_activation" not in row
        assert row["measurement_status"] == "NOT_MEASURED"
        assert row["input"]["example_id"].startswith("near-miss:")
    assert report["optimizer_input"] is False


def test_null_prediction_cannot_receive_negative_pole_sign_credit(data) -> None:
    report = evaluate_explanation(frozen(data, weights=(0.0,), bias=0.0), data[2], enabled=True)
    assert report["metrics"]["sign_errors"] == 9


def test_overflowing_metrics_fail_clearly_instead_of_serializing_infinity(data) -> None:
    locked = data[2]
    enormous = replace(
        locked, measurements=tuple(replace(m, activation=(1e308,)) for m in locked.measurements)
    )
    with pytest.raises(ValueError, match="numeric metric"):
        evaluate_explanation(frozen(data), enormous, enabled=True)


def test_locked_positions_cannot_be_selected_after_frozen_predictions(data) -> None:
    commit = replace(frozen(data, weights=(0.0,), bias=1.0), max_mae=1.0)
    locked = data[2]
    assert (
        evaluate_explanation(commit, locked, enabled=True)["prediction_result"]
        == "PREDICTIONS_FALSIFIED"
    )
    favorable = replace(
        locked,
        token_positions=(0,),
        measurements=tuple(m for m in locked.measurements if m.token_position == 0),
    )
    with pytest.raises(ValueError, match="positions"):
        evaluate_explanation(commit, favorable, enabled=True)


def test_every_locked_example_requires_measurement_coverage(data) -> None:
    generation, _, locked, manifest = data
    masked = replace(locked.inputs[0], attention_mask=(0, 0, 1))
    new_inputs = (masked,) + locked.inputs[1:]
    partial = replace(
        locked,
        inputs=new_inputs,
        token_positions=(0, 1),
        measurements=tuple(
            m
            for m in locked.measurements
            if m.example_id != masked.example_id and m.token_position < 2
        ),
    )
    manifest = replace(
        manifest,
        locked=tuple(
            ExampleRef.from_input(i, distribution=r.distribution)
            for i, r in zip(new_inputs, manifest.locked)
        ),
    )
    generation = replace(
        generation,
        token_positions=(0, 1),
        measurements=tuple(m for m in generation.measurements if m.token_position < 2),
    )
    with pytest.raises(ValueError, match="masked"):
        freeze_explanation(
            hypothesis(generation),
            manifest,
            generation,
            clock=Clock(),
            max_mae=0.01,
            activation_threshold=0.5,
            enabled=True,
        )
    with pytest.raises(ValueError, match="coverage"):
        explanation_module._batch_for_split(partial, manifest.locked, "locked")


def test_shared_causal_prefix_is_rejected_before_heldout_commit(data) -> None:
    generation, _, locked, manifest = data
    changed = replace(
        locked.inputs[0],
        token_ids=(generation.inputs[0].token_ids[0],) + locked.inputs[0].token_ids[1:],
    )
    refs = (ExampleRef.from_input(changed, distribution=manifest.locked[0].distribution),)
    splits = replace(manifest, locked=refs + manifest.locked[1:])
    with pytest.raises(ValueError, match="causal.*prefix|prefix.*overlap"):
        freeze_explanation(
            hypothesis(generation),
            splits,
            generation,
            clock=Clock(),
            max_mae=0.0,
            activation_threshold=0.5,
            enabled=True,
        )


def test_masked_token_jitter_cannot_hide_an_identical_visible_context() -> None:
    gen = NativeInput("g", (1, 2, 3), (1, 0, 1))
    dev = NativeInput("d", (7, 6, 5), (1, 1, 1))
    locked = NativeInput("l", (1, 4, 3), (1, 0, 1))
    with pytest.raises(ValueError, match="context"):
        SplitManifest(
            (ExampleRef.from_input(gen, distribution="id"),),
            (ExampleRef.from_input(dev, distribution="id"),),
            (ExampleRef.from_input(locked, distribution="id"),),
        )


def test_reference_v2_and_legacy_v1_roundtrip_without_promoting_old_evidence(data) -> None:
    generation, _, _, manifest = data
    ref = manifest.generation[0]
    assert ExampleRef.from_dict(ref.to_dict()) == ref
    legacy = ref.to_dict()
    legacy.pop("causal_prefix_digests")
    legacy["schema_version"] = "native-example-reference-v1"
    restored = ExampleRef.from_dict(legacy)
    assert restored.to_dict() == legacy
    assert restored.causal_prefix_digests == ()
    from gepa_mindfulness.interpret.explanation_tests import require_causal_prefix_isolation

    old = replace(manifest, generation=(restored,) + manifest.generation[1:])
    with pytest.raises(ValueError, match="rebuild legacy"):
        require_causal_prefix_isolation(old, (0,))
    malformed = ref.to_dict()
    malformed["causal_prefix_digests"] = []
    with pytest.raises(ValueError, match="metadata"):
        ExampleRef.from_dict(malformed)


def test_prefix_qualification_uses_query_position_and_visible_tokens() -> None:
    from gepa_mindfulness.interpret.explanation_tests import require_causal_prefix_isolation

    def ref(name, tokens, mask=(1, 1, 1)):
        return ExampleRef.from_input(NativeInput(name, tokens, mask), distribution="id")

    g, d, locked_ref = ref("g", (1, 2, 3)), ref("d", (4, 5, 6)), ref("l", (1, 7, 8))
    splits = SplitManifest((g,), (d,), (locked_ref,))
    with pytest.raises(ValueError, match="prefix overlap"):
        require_causal_prefix_isolation(splits, (0,))
    require_causal_prefix_isolation(splits, (1, 2))
    assert g.causal_prefix_digests[0] != g.causal_prefix_digests[1]
    jitter = ref("j", (1, 9, 3), (1, 0, 1))
    masked = ref("m", (1, 2, 3), (1, 0, 1))
    assert jitter.causal_prefix_digests == masked.causal_prefix_digests
    assert jitter.input_digest != masked.input_digest
    assert jitter.context_digest == masked.context_digest
    with pytest.raises(ValueError, match="masked"):
        require_causal_prefix_isolation(SplitManifest((masked,), (d,), (locked_ref,)), (1,))
    with pytest.raises(ValueError, match="prefix overlap"):
        require_causal_prefix_isolation(SplitManifest((g,), (locked_ref,), (d,)), (0,))


def test_forged_generation_prefix_metadata_does_not_match_actual_inputs(data) -> None:
    generation, _, _, manifest = data
    altered = replace(manifest.generation[0], causal_prefix_digests=("a" * 64,) * 3)
    forged = replace(manifest, generation=(altered,) + manifest.generation[1:])
    with pytest.raises(ValueError, match="generation context"):
        freeze_explanation(
            hypothesis(generation),
            forged,
            generation,
            clock=Clock(),
            max_mae=0.1,
            activation_threshold=0.5,
            enabled=True,
        )


def test_directly_constructed_commit_cannot_bypass_prefix_guard(data) -> None:
    from gepa_mindfulness.interpret.explanation_tests import ExplanationCommit

    _, _, locked, manifest = data
    commit = frozen(data)
    leak = replace(
        manifest.locked[0], causal_prefix_digests=manifest.generation[0].causal_prefix_digests
    )
    forged = replace(commit, splits=replace(manifest, locked=(leak,) + manifest.locked[1:]))
    restored = ExplanationCommit.from_dict(forged.to_dict())
    with pytest.raises(ValueError, match="prefix overlap"):
        evaluate_explanation(restored, locked, enabled=True)


def test_present_short_prefix_metadata_reports_geometry_instead_of_legacy() -> None:
    from gepa_mindfulness.interpret.explanation_tests import require_causal_prefix_isolation

    groups = tuple(
        (ExampleRef.from_input(NativeInput(name, tokens, (1,) * len(tokens)), distribution="id"),)
        for name, tokens in [("g", (1, 2, 3)), ("d", (4,)), ("l", (5, 6, 7))]
    )
    manifest = SplitManifest(*groups)
    with pytest.raises(ValueError, match="outside.*geometry"):
        require_causal_prefix_isolation(manifest, (2,))
