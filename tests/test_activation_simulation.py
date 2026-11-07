"""Actual downstream KL, exact replacement geometry and matched frozen-model controls."""

# Standard library
import math
from dataclasses import replace

# Third-party
import pytest

# Local
from gepa_mindfulness.interpret.activation_simulation import (
    downstream_kl,
    prepare_simulation_controls,
    replace_selected_activations,
    simulate_activations,
)
from gepa_mindfulness.interpret.activation_simulation_records import ChoiceToken, LogitChoiceProbe
from gepa_mindfulness.interpret.explanation_tests import (
    ActivationPredictor,
    ExampleRef,
    ExplanationHypothesis,
    ExplanationKind,
    SplitManifest,
    TokenFeature,
    freeze_explanation,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput
from gepa_mindfulness.training.eligibility import require_training_eligible

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def trial(family="gpt2", neurons=(0, 3)):
    from datetime import datetime, timezone

    class Clock:
        def now(self):
            return datetime(2026, 11, 5, tzinfo=timezone.utc)

    torch.manual_seed(31)
    if family == "gpt2":
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
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
        )
        kind = HookKind.MLP_POST
    else:
        source = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=16,
                max_position_embeddings=8,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                bos_token_id=1,
                eos_token_id=2,
                attention_dropout=0,
            )
        )
        kind = HookKind.MLP_PRODUCT
    native = FrozenNativeModel(
        source, model_id=f"random-{family}", revision="seed-31", enabled=True
    )
    hook = native.hook(layer=0, kind=kind)
    generation = (NativeInput("g1", (1, 2, 3), (1, 1, 1)), NativeInput("g2", (3, 2, 1), (1, 1, 1)))
    development = (NativeInput("d1", (2, 1, 3), (1, 1, 1)),)
    locked = (NativeInput("l1", (3, 1, 2), (1, 1, 1)), NativeInput("l2", (4, 5, 6), (1, 1, 1)))
    manifest = SplitManifest(
        tuple(ExampleRef.from_input(i, distribution="id") for i in generation),
        tuple(ExampleRef.from_input(i, distribution="id") for i in development),
        tuple(
            ExampleRef.from_input(i, distribution="ood" if i.example_id == "l2" else "id")
            for i in locked
        ),
    )
    batches, commits = [], []
    for neuron in neurons:
        batch = native.capture(
            hook,
            torch.tensor([i.token_ids for i in generation]),
            component_indices=(neuron,),
            example_ids=("g1", "g2"),
            token_positions=(0, 1, 2),
            enabled=True,
        )
        predictor = ActivationPredictor(
            (TokenFeature("token_three", "token_is", 0, 3),),
            (0.0,),
            0.05 * (neuron + 1),
            "rule",
            (),
        )
        hypothesis = ExplanationHypothesis(
            f"constant-{neuron}",
            "Deliberately incorrect constant as a fidelity control.",
            ExplanationKind.UNKNOWN,
            ("context-dependent mechanism",),
            ("Constant substitution will not reproduce the actual activation function.",),
            predictor,
            ("g1", "g2"),
        )
        commits.append(
            freeze_explanation(
                hypothesis,
                manifest,
                batch,
                clock=Clock(),
                max_mae=0.0,
                activation_threshold=0.01,
                enabled=True,
            )
        )
        batches.append(batch)
    controls = prepare_simulation_controls(tuple(commits), tuple(batches), seed=41, enabled=True)
    return source, native, hook, tuple(commits), locked, controls


def test_downstream_kl_is_computed_in_the_declared_direction() -> None:
    original = torch.log(torch.tensor([[[0.5, 0.5]]], dtype=torch.float64))
    substituted = torch.log(torch.tensor([[[0.9, 0.1]]], dtype=torch.float64))
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    result = downstream_kl(original, substituted)
    assert result.shape == (1, 1)
    assert result.item() == pytest.approx(expected, abs=1e-12)
    assert downstream_kl(original, original).item() == pytest.approx(0.0, abs=1e-12)


def test_substitution_rejects_wrong_shape_dtype_device_and_positions() -> None:
    original = torch.zeros((1, 3, 4))
    replacement = torch.ones((1, 2, 2))
    result = replace_selected_activations(
        original, replacement, token_positions=(1, 2), component_indices=(0, 3)
    )
    assert original.sum().item() == 0
    assert result.sum().item() == 4
    for value, positions, components in (
        (replacement[:, :1], (1, 2), (0, 3)),
        (replacement.double(), (1, 2), (0, 3)),
        (torch.ones((1, 2, 2), device="meta"), (1, 2), (0, 3)),
        (replacement, (1, 3), (0, 3)),
        (replacement, (1, 2), (0, 0)),
    ):
        with pytest.raises(ValueError):
            replace_selected_activations(
                original, value, token_positions=positions, component_indices=components
            )


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_actual_simulation_keeps_model_rng_and_matched_controls(family) -> None:
    source, native, hook, commits, locked, controls = trial(family)
    weights = {k: v.detach().clone() for k, v in source.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = simulate_activations(
        native, hook, commits, locked, controls=controls, logit_positions=(0, 1), enabled=True
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert source.training is True
    assert all(torch.equal(weights[k], v) for k, v in source.state_dict().items())
    assert {
        "explanation",
        "identity",
        "ablation",
        "mean",
        "random_predictor",
        "shuffled_labels",
        "magnitude_matched_random_direction",
        "random_neurons",
        "matched_complexity",
        "cross_context",
    } <= set(report["conditions"])
    identity = report["conditions"]["identity"]
    assert identity["summary"]["mean_kl"] == pytest.approx(0.0, abs=1e-12)
    assert identity["summary"]["mean_delta_nll"] == pytest.approx(0.0, abs=1e-12)
    candidate = report["conditions"]["explanation"]
    assert len(candidate["per_position"]) == 4
    assert candidate["summary"]["mean_kl"] > 0
    assert candidate["summary"]["mean_activation_mae"] > 0
    assert all(
        "original_nll" in row and "substituted_nll" in row for row in candidate["per_position"]
    )
    assert report["evidence_status"] == "PREDICTIVE_HYPOTHESIS"
    assert report["source_model_state_digest"] == native.identity.state_digest
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in native._model.modules()) == 0
    )
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report)


def test_unfrozen_or_foreign_locked_inputs_cannot_enter_simulation() -> None:
    _, native, hook, commits, locked, controls = trial()
    with pytest.raises(ValueError, match="locked"):
        simulate_activations(
            native,
            hook,
            commits,
            (NativeInput("foreign", (1, 2, 3), (1, 1, 1)),),
            controls=controls,
            logit_positions=(0,),
            enabled=True,
        )
    with pytest.raises(ValueError, match="enabled=True"):
        simulate_activations(None, None, None, None, controls=None, logit_positions=())


def simulate_trial(parts, **kwargs):
    _, native, hook, commits, locked, controls = parts
    return simulate_activations(
        native,
        hook,
        commits,
        locked,
        controls=controls,
        logit_positions=controls.logit_positions,
        enabled=True,
        **kwargs,
    )


def test_retained_logits_independently_reproduce_kl_shifted_nll_and_calibration() -> None:
    report = simulate_trial(trial())
    originals = {(r["example_id"], r["position"]): r["logits"] for r in report["original_logits"]}
    for condition in report["conditions"].values():
        if condition["status"] != "MEASURED":
            continue
        for row in condition["per_position"]:
            old = torch.tensor(originals[row["example_id"], row["position"]], dtype=torch.float64)
            new = torch.tensor(row["substituted_logits"], dtype=torch.float64)
            p = old.softmax(0)
            manual = (p * (old.log_softmax(0) - new.log_softmax(0))).sum().item()
            assert row["kl_original_to_substituted"] == pytest.approx(manual, abs=1e-12)
            gold = row["gold_next_token"]
            assert row["original_nll"] == pytest.approx(-old.log_softmax(0)[gold].item())
            assert row["substituted_nll"] == pytest.approx(-new.log_softmax(0)[gold].item())
        for name in ("calibration_original", "calibration_substituted"):
            calibration = condition[name]
            assert calibration["count"] == 4
            manual_ece = (
                sum(
                    b["count"] * abs(b["accuracy"] - b["confidence"])
                    for b in calibration["bins"]
                    if b["count"]
                )
                / 4
            )
            assert calibration["ece"] == pytest.approx(manual_ece)


@pytest.mark.parametrize("change", ["mean", "locked_source", "logit_positions", "model"])
def test_control_tampering_and_source_relabelling_fail_closed(change) -> None:
    source, native, hook, commits, locked, controls = trial()
    if change == "mean":
        controls = replace(
            controls,
            components=(replace(controls.components[0], mean=99.0),) + controls.components[1:],
        )
    if change == "locked_source":
        batch = controls.generation_batches[0]
        forged = replace(
            batch, measurements=tuple(replace(m, activation=(99.0,)) for m in batch.measurements)
        )
        with pytest.raises(ValueError, match="generation"):
            prepare_simulation_controls(
                commits,
                (forged,) + controls.generation_batches[1:],
                seed=controls.seed,
                enabled=True,
            )
        return
    if change == "model":
        with torch.no_grad():
            next(native._model.parameters()).add_(1.0)
    with pytest.raises(ValueError):
        simulate_activations(
            native,
            hook,
            commits,
            locked,
            controls=controls,
            logit_positions=(0,) if change == "logit_positions" else (0, 1),
            enabled=True,
        )


def test_single_unit_low_kl_does_not_certify_wrong_constant_as_explained() -> None:
    report = simulate_trial(trial(neurons=(0,)))
    candidate = report["conditions"]["explanation"]
    assert candidate["summary"]["mean_activation_mae"] > 0.01
    assert candidate["summary"]["mean_kl"] < 0.01
    assert report["evidence_status"] == "PREDICTIVE_HYPOTHESIS"
    assert report["conditions"]["shuffled_labels"]["status"] == "UNAVAILABLE"
    assert report["activation_coverage"][0]["count"] == 6


def test_zero_random_direction_still_matches_intervention_magnitude(monkeypatch) -> None:
    import random

    parts = trial()
    monkeypatch.setattr(random.Random, "gauss", lambda *args: 0.0)
    report = simulate_trial(parts)
    for name in ("explanation", "magnitude_matched_random_direction"):
        values = report["conditions"][name]["activation_predictions"]
        squared = sum((r["applied"] - r["actual"]) ** 2 for r in values)
        if name == "explanation":
            expected = squared
        else:
            assert squared == pytest.approx(expected, rel=1e-6, abs=1e-10)


def test_probe_mapping_is_frozen_and_reports_actual_choice_scores() -> None:
    parts = trial()
    _, native, hook, commits, locked, controls = parts
    probe = LogitChoiceProbe(
        "answer", "l1", 0, (ChoiceToken("complete", 1), ChoiceToken("uncertain", 2)), "uncertain"
    )
    controls = prepare_simulation_controls(
        commits, controls.generation_batches, seed=41, probes=(probe,), enabled=True
    )
    report = simulate_activations(
        native, hook, commits, locked, controls=controls, logit_positions=(0, 1), enabled=True
    )
    rows = report["conditions"]["explanation"]["per_position"]
    scores = rows[0]["substituted_logits"]
    expected = "complete" if scores[1] >= scores[2] else "uncertain"
    probe_result = report["conditions"]["explanation"]["behavior_probes"][0]
    assert probe_result["substituted_choice"] == expected
    assert probe_result["substituted_correct"] == (expected == "uncertain")
    assert type(controls).from_dict(controls.to_dict()) == controls


def test_distribution_budget_prevents_execution_and_failed_hook_cleans_up(monkeypatch) -> None:
    parts = trial()
    _, native, hook, commits, locked, controls = parts
    with pytest.raises(ValueError, match="budget"):
        simulate_trial(parts, max_distribution_values=1)
    import gepa_mindfulness.interpret.activation_simulation as module

    def failure(*args, **kwargs):
        raise ValueError("controlled replacement failure")

    monkeypatch.setattr(module, "replace_selected_activations", failure)
    with pytest.raises(ValueError, match="controlled replacement"):
        simulate_activations(
            native, hook, commits, locked, controls=controls, logit_positions=(0, 1), enabled=True
        )
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in native._model.modules()) == 0
    )


@pytest.mark.parametrize("positions", [(False, True), (0.0, 1.0)])
def test_malformed_positions_are_rejected_before_model_execution(positions, monkeypatch) -> None:
    _, native, hook, commits, locked, controls = trial()

    def forbidden(*args, **kwargs):
        raise AssertionError("model executed with malformed output positions")

    monkeypatch.setattr(FrozenNativeModel, "capture", forbidden)
    with pytest.raises(ValueError):
        simulate_activations(
            native,
            hook,
            commits,
            locked,
            controls=controls,
            logit_positions=positions,
            enabled=True,
        )


def test_simulation_cannot_use_a_gold_future_token_to_predict_a_causal_activation() -> None:
    _, native, hook, commits, _, controls = trial()
    predictor = ActivationPredictor(
        (TokenFeature("future-gold", "token_is", 1, 3),), (1.0,), 0.0, "rule", ()
    )
    a, b = NativeInput("a", (1, 2, 3), (1, 1, 1)), NativeInput("b", (1, 3, 3), (1, 1, 1))
    assert predictor.predict(a, 0) != predictor.predict(b, 0)
    observed = native.capture(
        hook,
        torch.tensor([a.token_ids, b.token_ids]),
        component_indices=(0,),
        example_ids=("a", "b"),
        token_positions=(0,),
        enabled=True,
    )
    assert observed.measurements[0].activation == observed.measurements[1].activation
    bad = replace(commits[0], hypothesis=replace(commits[0].hypothesis, predictor=predictor))
    with pytest.raises(ValueError, match="future-token"):
        prepare_simulation_controls(
            (bad,) + commits[1:], controls.generation_batches, seed=41, enabled=True
        )
