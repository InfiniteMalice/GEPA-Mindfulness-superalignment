"""Matched native/decomposition comparisons use real forwards and preserve uncertainty."""

# Standard library
import gzip
import json
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest
from test_mechanistic_evidence import known_cause
from test_native_causal_interventions import Clock

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.interpret.comparative import (
    encode_decode,
    evaluate_comparison,
    native_dictionary,
    prepare_comparison,
)
from gepa_mindfulness.interpret.comparative_records import (
    ComparisonArm,
    ComparisonBudget,
    ComparisonProtocol,
)
from gepa_mindfulness.interpret.explanation_tests import ExampleRef, SplitManifest, TokenFeature
from gepa_mindfulness.interpret.native_causal_records import EffectPrediction, LogitContrast
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput
from gepa_mindfulness.training.eligibility import require_training_eligible

torch = pytest.importorskip("torch")


def trial():
    native, _ = known_cause()
    hook = native.hook(layer=0, kind=HookKind.MLP_RESIDUAL)
    dictionary = native_dictionary(native, hook, enabled=True)
    relu = replace(
        dictionary, dictionary_id="constructed-relu", nonlinearity="relu", origin="constructed"
    )
    groups = {
        "generation": (
            NativeInput("g1", (0, 2, 3), (1, 1, 1)),
            NativeInput("g2", (1, 1, 2), (1, 1, 1)),
            NativeInput("g3", (2, 4, 1), (1, 1, 1)),
        ),
        "development": (NativeInput("d1", (3, 3, 4), (1, 1, 1)),),
        "locked": (
            NativeInput("l1", (4, 3, 4), (1, 1, 1)),
            NativeInput("l2", (5, 2, 5), (1, 1, 1)),
        ),
    }
    refs = {
        k: tuple(
            ExampleRef.from_input(i, distribution="ood" if i.example_id == "l2" else "id")
            for i in v
        )
        for k, v in groups.items()
    }
    manifest = SplitManifest(refs["generation"], refs["development"], refs["locked"])
    features = (TokenFeature("one", "token_is", 0, 1), TokenFeature("four", "token_is", 0, 4))
    prediction = EffectPrediction(-1, 1e-8, 20.0, 1e-8)
    context_ref = EvidenceReference(
        "fixture:writer-reader-context", EvidenceSourceKind.EXTERNAL_RECORD
    )
    arms = (
        ComparisonArm(
            "native",
            "native",
            dictionary,
            (2,),
            features,
            prediction,
            "Current-token hypothesis for the constructed component.",
            0.01,
        ),
        ComparisonArm(
            "context",
            "native_context",
            dictionary,
            (2,),
            (
                TokenFeature("previous-one", "token_is", -1, 1),
                TokenFeature("same-token", "tokens_equal", -1, other_offset=0),
            ),
            prediction,
            "A competing contextual hypothesis.",
            0.01,
            context_refs=(context_ref,),
        ),
        ComparisonArm(
            "sae",
            "sparse_autoencoder",
            relu,
            (2,),
            features,
            prediction,
            "Constructed ReLU dictionary control, not a learned SAE.",
            0.01,
        ),
        ComparisonArm(
            "other",
            "other_decomposition",
            None,
            (2,),
            features,
            prediction,
            "No other dictionary supplied.",
            0.01,
            unavailable_reason="No supported fixed dictionary supplied.",
        ),
    )
    protocol = ComparisonProtocol(
        "comparison-test",
        native.identity.digest,
        hook,
        manifest,
        arms,
        ComparisonBudget(1, 2, 9, 1000000),
        (0, 1, 2),
        (0, 1),
        LogitContrast(0, 3, 4),
        79,
        1e-4,
    )
    return native, protocol, groups


def test_dictionary_reconstruction_is_independent_of_explanation_prediction() -> None:
    native, protocol, _ = trial()
    dictionary = protocol.arms[0].dictionary
    values = torch.tensor([[[-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    z, reconstructed = encode_decode(dictionary, values)
    assert torch.equal(values, reconstructed)
    z_relu, reconstructed_relu = encode_decode(protocol.arms[2].dictionary, values)
    assert reconstructed_relu[0, 0, 0] == 0.0
    assert float(((values - reconstructed_relu) ** 2).sum()) == 1.0
    assert z_relu[0, 0, 0] == 0.0 and z[0, 0, 0] == -1.0
    native._assert_frozen()


def test_actual_comparison_retains_unavailable_methods_and_real_distribution_evidence() -> None:
    native, protocol, groups = trial()
    rng = torch.get_rng_state().clone()
    commit = prepare_comparison(native, protocol, groups["generation"], clock=Clock(), enabled=True)
    report = evaluate_comparison(native, commit, groups["locked"], enabled=True)
    assert report["method_superiority"] == "NOT_ESTABLISHED"
    assert report["evidence_status"] == "PREDICTIVE_HYPOTHESIS"
    assert report["arms"][-1]["status"] == "UNAVAILABLE"
    assert len(report["arms"]) == 4
    for arm in report["arms"][:-1]:
        assert arm["status"] == "MEASURED"
        assert arm["conditions"]["identity"]["summary"]["mean_kl"] == 0.0
        assert arm["conditions"]["explanation"]["rows"]
        assert arm["reconstruction"]["mse"] >= 0.0
        assert arm["prediction"]["feature_mae"] >= 0.0
        assert set(arm["by_distribution"]) == {"id", "ood"}
        assert arm["causal"]["evidence_status"] == "UNRESOLVED"
        assert arm["coverage"]["negative_native_values"] > 0
    assert torch.equal(rng, torch.get_rng_state())
    native._assert_frozen()
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report)
    assert type(commit).from_dict(commit.to_dict()) == commit


def test_model_budget_and_future_token_mismatches_fail_closed() -> None:
    native, protocol, groups = trial()
    with pytest.raises(ValueError, match="model"):
        replace(protocol, model_digest="0" * 64)
    with pytest.raises(ValueError, match="budget"):
        replace(protocol, budget=replace(protocol.budget, selected_features=2))
    with pytest.raises(ValueError, match="future|prefix"):
        replace(
            protocol.arms[0],
            public_features=(
                TokenFeature("future", "token_is", 1, 2),
                protocol.arms[0].public_features[1],
            ),
        )
    with pytest.raises(ValueError, match="enabled=True"):
        evaluate_comparison(None, None, None)
    with pytest.raises(ValueError, match="budget"):
        prepare_comparison(
            native,
            replace(protocol, budget=replace(protocol.budget, max_forwards_per_arm=1)),
            groups["generation"],
            clock=Clock(),
            enabled=True,
        )


def test_modified_fits_and_locked_contexts_cannot_be_selected_after_measurement() -> None:
    native, protocol, groups = trial()
    commit = prepare_comparison(native, protocol, groups["generation"], clock=Clock(), enabled=True)
    first = commit.fits[0]
    wrong_predictor = replace(first.predictors[0], bias=first.predictors[0].bias + 1.0)
    forged = replace(
        commit, fits=(replace(first, predictors=(wrong_predictor,)),) + commit.fits[1:]
    )
    with pytest.raises(ValueError, match="fit|generation"):
        evaluate_comparison(native, forged, groups["locked"], enabled=True)
    with pytest.raises(ValueError, match="locked|split"):
        evaluate_comparison(native, commit, groups["generation"], enabled=True)


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_random_architecture_results_recompute_kl_nll_and_preserve_source(family) -> None:
    import math

    from test_native_causal_interventions import trial as random_trial

    source, native, _, _ = random_trial(family=family)
    _, old, groups = trial()
    hook = native.hook(layer=0, kind=HookKind.MLP_RESIDUAL)
    dictionary = native_dictionary(native, hook, enabled=True)
    arms = tuple(
        replace(
            arm,
            dictionary=(
                None
                if arm.dictionary is None
                else (
                    replace(dictionary, nonlinearity="relu", origin="constructed")
                    if arm.method == "sparse_autoencoder"
                    else dictionary
                )
            ),
            context_refs=(
                (EvidenceReference("random-context:" + family, EvidenceSourceKind.EXTERNAL_RECORD),)
                if arm.method == "native_context"
                else ()
            ),
        )
        for arm in old.arms
    )
    protocol = replace(old, model_digest=native.identity.digest, hook=hook, arms=arms)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    commit = prepare_comparison(native, protocol, groups["generation"], clock=Clock(), enabled=True)
    result = evaluate_comparison(native, commit, groups["locked"], enabled=True)
    originals = {(r["example_id"], r["position"]): r["logits"] for r in result["original_logits"]}

    def logsumexp(values):
        largest = max(values)
        return largest + math.log(sum(math.exp(v - largest) for v in values))

    for arm in result["arms"]:
        if arm["status"] != "MEASURED":
            continue
        measured = 0
        for condition in arm["conditions"].values():
            if condition["status"] != "MEASURED":
                continue
            measured += 1
            for row in condition["rows"]:
                original = originals[row["example_id"], row["position"]]
                changed = row["substituted_logits"]
                a, b = logsumexp(original), logsumexp(changed)
                kl = sum(math.exp(v - a) * ((v - a) - (w - b)) for v, w in zip(original, changed))
                assert row["kl_original_to_substituted"] == pytest.approx(kl, abs=1e-12)
                gold = row["gold_next_token"]
                if gold is not None:
                    assert row["substituted_nll"] == pytest.approx(b - changed[gold], abs=1e-12)
        assert arm["cost"]["forward_calls"] == measured <= protocol.budget.max_forwards_per_arm
    assert all(torch.equal(state[k], v) for k, v in source.state_dict().items())
    assert source.training is True
    native._assert_frozen()


def test_group_shuffling_and_other_linear_dictionary_are_actual_supported_conditions() -> None:
    native, p, groups = trial()
    dictionary = p.arms[0].dictionary
    encoder = list(dictionary.encoder)
    encoder[0], encoder[2] = encoder[2], encoder[0]
    swapped = replace(
        dictionary,
        dictionary_id="swapped-basis",
        origin="constructed",
        encoder=tuple(encoder),
        decoder=tuple(tuple(row) for row in zip(*encoder)),
    )
    other = replace(p.arms[-1], dictionary=swapped, unavailable_reason=None)
    arms = tuple(replace(a, selected_features=(1, 2)) for a in p.arms[:-1] + (other,))
    p = replace(p, arms=arms, budget=replace(p.budget, selected_features=2))
    commit = prepare_comparison(native, p, groups["generation"], clock=Clock(), enabled=True)
    result = evaluate_comparison(native, commit, groups["locked"], enabled=True)
    for arm in result["arms"]:
        assert arm["status"] == "MEASURED"
        assert arm["conditions"]["shuffled_labels"]["status"] == "MEASURED"
        assert arm["cost"]["forward_calls"] == 9
    assert result["arms"][-1]["reconstruction"]["mse"] == 0.0
    assert result["arms"][0]["reconstruction"]["mse"] == 0.0
    assert all(f.label_permutation == (1, 0) for f in commit.fits)


@pytest.mark.parametrize("change", ["means", "random_indices", "generation"])
def test_control_and_generation_tampering_cannot_enter_locked_evaluation(change) -> None:
    native, p, groups = trial()
    commit = prepare_comparison(native, p, groups["generation"], clock=Clock(), enabled=True)
    first = commit.fits[0]
    if change == "means":
        first = replace(first, means=(first.means[0] + 1.0,))
        bad = replace(commit, fits=(first,) + commit.fits[1:])
    elif change == "random_indices":
        choices = [i for i in range(p.hook.width) if i not in first.random_indices and i != 2]
        first = replace(first, random_indices=(choices[0],))
        bad = replace(commit, fits=(first,) + commit.fits[1:])
    else:
        generation = commit.generation
        measurement = generation.measurements[0]
        changed = replace(measurement, activation=(measurement.activation[0] + 1.0,))
        bad = replace(
            commit,
            generation=replace(generation, measurements=(changed,) + generation.measurements[1:]),
        )
    with pytest.raises(ValueError, match="generation|fit|control"):
        evaluate_comparison(native, bad, groups["locked"], enabled=True)


def test_unsupported_maps_hooks_and_prediction_features_fail_explicitly() -> None:
    native, p, groups = trial()
    dictionary = p.arms[0].dictionary
    with pytest.raises(ValueError, match="geometry"):
        replace(dictionary, decoder=dictionary.decoder[:-1])
    with pytest.raises(ValueError, match="nonlinearity"):
        replace(dictionary, nonlinearity="opaque-pickled-module")
    with pytest.raises(ValueError, match="identity"):
        replace(p.arms[0], dictionary=replace(dictionary, nonlinearity="relu"))
    with pytest.raises(ValueError, match="head vectors"):
        replace(p, hook=native.hook(layer=0, kind=HookKind.ATTENTION_HEAD))
    with pytest.raises(ValueError):
        replace(p, activation_positions=(True,))
    with pytest.raises(ValueError, match="clock"):

        class ClockWithoutTimezone:
            def now(self):
                from datetime import datetime

                return datetime(2026, 11, 5)

        prepare_comparison(
            native, p, groups["generation"], clock=ClockWithoutTimezone(), enabled=True
        )


def test_zero_decoder_cannot_hide_unexplained_reconstruction_behind_zero_kl() -> None:
    native, p, groups = trial()
    sae = p.arms[2]
    bad = replace(
        sae.dictionary, decoder=tuple((0.0,) * p.hook.width for _ in sae.dictionary.decoder)
    )
    arm = replace(sae, dictionary=bad, max_feature_mae=0.0)
    p = replace(p, arms=p.arms[:2] + (arm,) + p.arms[3:])
    commit = prepare_comparison(native, p, groups["generation"], clock=Clock(), enabled=True)
    result = evaluate_comparison(native, commit, groups["locked"], enabled=True)
    measured = result["arms"][2]
    assert measured["conditions"]["explanation"]["summary"]["mean_kl"] == 0.0
    assert measured["reconstruction"]["mse"] > 0.0
    assert measured["prediction"]["feature_mae"] > 0.0
    assert measured["prediction"]["criterion"] == "FALSIFIED"
    assert result["method_superiority"] == "NOT_ESTABLISHED"
    assert measured["causal"]["evidence_status"] == "UNRESOLVED"


def test_generation_scale_normalization_does_not_turn_dictionary_units_into_a_ranking() -> None:
    native, p, groups = trial()
    native_arm = p.arms[0]
    d = native_arm.dictionary
    scaled = replace(
        d,
        dictionary_id="scaled-linear",
        origin="constructed",
        encoder=tuple(tuple(v * 10.0 for v in row) for row in d.encoder),
        decoder=tuple(tuple(v / 10.0 for v in row) for row in d.decoder),
    )
    other = replace(p.arms[-1], dictionary=scaled, unavailable_reason=None)
    p = replace(p, arms=p.arms[:-1] + (other,))
    commit = prepare_comparison(native, p, groups["generation"], clock=Clock(), enabled=True)
    result = evaluate_comparison(native, commit, groups["locked"], enabled=True)
    first, last = result["arms"][0]["prediction"], result["arms"][-1]["prediction"]
    assert last["feature_mae"] == pytest.approx(10.0 * first["feature_mae"], rel=1e-9, abs=1e-9)
    assert last["generation_std_normalized_mae"] == pytest.approx(
        first["generation_std_normalized_mae"], rel=1e-9, abs=1e-9
    )
    assert result["method_superiority"] == "NOT_ESTABLISHED"


def test_all_unavailable_protocol_is_rejected_before_any_capture() -> None:
    _, p, _ = trial()
    unavailable = tuple(
        replace(a, dictionary=None, unavailable_reason="No dictionary available.") for a in p.arms
    )
    with pytest.raises(ValueError, match="available"):
        replace(p, arms=unavailable)


def test_dense_biased_dictionary_residuals_and_signed_effects_are_independently_checkable() -> None:
    native, p, groups = trial()
    d = p.arms[0].dictionary
    dense = replace(
        d,
        dictionary_id="dense-biased-control",
        origin="constructed",
        encoder_bias=tuple(0.1 * (i + 1) for i in range(d.feature_count)),
        decoder=tuple(
            tuple(0.5 if i == j else 0.125 for j in range(d.width)) for i in range(d.feature_count)
        ),
        decoder_bias=tuple(0.03 * (i + 1) for i in range(d.width)),
    )
    arm = replace(p.arms[-1], dictionary=dense, unavailable_reason=None)
    p = replace(p, arms=p.arms[:-1] + (arm,))
    commit = prepare_comparison(native, p, groups["generation"], clock=Clock(), enabled=True)
    result = evaluate_comparison(native, commit, groups["locked"], enabled=True)
    measured = result["arms"][-1]
    predicted = {
        (r["example_id"], r["position"], r["feature_index"]): r["predicted"]
        for r in measured["prediction"]["rows"]
    }
    for b, context in enumerate(groups["locked"]):
        for j, position in enumerate(p.activation_positions):
            actual = result["original_native_values"][b][j]
            features = [
                sum(actual[i] * dense.encoder[i][k] for i in range(dense.width))
                + dense.encoder_bias[k]
                for k in range(dense.feature_count)
            ]
            reconstruction = [
                dense.decoder_bias[i]
                + sum(features[k] * dense.decoder[k][i] for k in range(dense.feature_count))
                for i in range(dense.width)
            ]
            applied = measured["conditions"]["explanation"]["applied_native_values"][b][j]
            for i in range(dense.width):
                residual = actual[i] - reconstruction[i]
                replaced = [
                    (
                        predicted[context.example_id, position, k]
                        if k in arm.selected_features
                        else features[k]
                    )
                    for k in range(dense.feature_count)
                ]
                expected = (
                    residual
                    + dense.decoder_bias[i]
                    + sum(replaced[k] * dense.decoder[k][i] for k in range(dense.feature_count))
                )
                assert applied[i] == pytest.approx(expected, abs=1e-6)
                assert measured["reconstruction"]["decoded_native_values"][b][j][
                    i
                ] == pytest.approx(reconstruction[i], abs=1e-12)
    originals = {(r["example_id"], r["position"]): r["logits"] for r in result["original_logits"]}
    conditions = measured["conditions"]

    def effect(condition, example):
        row = next(
            r
            for r in conditions[condition]["rows"]
            if r["example_id"] == example and r["position"] == p.metric.position
        )
        old = originals[example, p.metric.position]
        new = row["substituted_logits"]
        return (new[p.metric.target_token] - new[p.metric.reference_token]) - (
            old[p.metric.target_token] - old[p.metric.reference_token]
        )

    for row in measured["causal"]["rows"]:
        actual = effect("ablation", row["example_id"])
        control = effect("random_features", row["example_id"])
        assert row["ablation_effect"] == pytest.approx(actual, abs=1e-12)
        assert row["random_feature_effect"] == pytest.approx(control, abs=1e-12)
        pred = arm.causal_prediction
        outcome = "MET"
        if not pred.minimum_effect <= pred.direction * actual <= pred.maximum_effect:
            outcome = "FALSIFIED"
        elif pred.direction * (actual - control) < pred.minimum_control_gap:
            outcome = "NOT_SPECIFIC"
        assert row["prediction_result"] == outcome


def test_report_byte_budget_rejects_wide_vocabulary_before_capture(monkeypatch) -> None:
    import transformers

    from gepa_mindfulness.interpret import comparative
    from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel

    _, p, groups = trial()
    source = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16384,
            n_positions=8,
            n_embd=8,
            n_inner=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    backend = FrozenNativeModel(source, model_id="wide-fixture", revision="1", enabled=True)
    hook = backend.hook(layer=0, kind=HookKind.MLP_RESIDUAL)
    arms = tuple(
        replace(
            a,
            dictionary=(
                None
                if a.dictionary is None
                else replace(
                    a.dictionary, model_digest=backend.identity.digest, hook_digest=hook.digest
                )
            ),
        )
        for a in p.arms
    )
    p = replace(
        p,
        model_digest=backend.identity.digest,
        hook=hook,
        arms=arms,
        manifest=replace(p.manifest, generation=p.manifest.generation[:1]),
        budget=replace(p.budget, max_distribution_values=2000000),
    )

    def forbidden_capture(*args):
        raise AssertionError("oversized report executed a capture")

    monkeypatch.setattr(comparative, "_capture", forbidden_capture)
    with pytest.raises(ValueError, match="report.*byte|serialized.*budget"):
        prepare_comparison(backend, p, groups["generation"][:1], clock=Clock(), enabled=True)


def test_original_nested_comparison_commitments_remain_inert_readable_records() -> None:
    """Historical v1 commitments roundtrip but cannot enter qualified held-out execution."""
    from gepa_mindfulness.interpret.comparative_records import ComparisonCommit

    source = Path(__file__).resolve().parents[1] / (
        "docs/experiments/phase2_pr8_comparison_probe.json.gz"
    )
    original = json.loads(gzip.decompress(source.read_bytes()))
    for result in original["results"]:
        payload = result["report"]["commit"]
        restored = ComparisonCommit.from_dict(payload)
        assert restored.to_dict() == payload
        assert all(not ref.causal_prefix_digests for ref in restored.protocol.manifest.generation)
        with pytest.raises(ValueError, match="rebuild legacy"):
            evaluate_comparison(None, restored, (), enabled=True)
