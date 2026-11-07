"""Prediction-bound causal experiments; effects do not self-admit scientific evidence."""

# Standard library
from dataclasses import replace
from datetime import datetime, timezone

# Third-party
import pytest

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.interpret.causal_interventions import (
    CausalPlan,
    EffectPrediction,
    InterventionMode,
    InterventionSite,
    LogitContrast,
    execute_causal_plan,
    freeze_causal_plan,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput
from gepa_mindfulness.training.eligibility import require_training_eligible
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
REF = EvidenceReference("fixture:causal-pair", EvidenceSourceKind.EXTERNAL_RECORD)


class Clock:
    def now(self):
        return datetime(2026, 11, 5, tzinfo=timezone.utc)


def trial(mode=InterventionMode.ACTIVATION_PATCH, family="gpt2"):
    torch.manual_seed(71)
    if family == "gpt2":
        source = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=16,
                n_positions=8,
                n_embd=8,
                n_inner=16,
                n_layer=2,
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
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        kind = HookKind.MLP_PRODUCT
    backend = FrozenNativeModel(
        source, model_id=f"causal-software-{family}", revision="seed-71", enabled=True
    )
    site = InterventionSite(backend.hook(layer=0, kind=kind), (0,), (0,))
    mediator = InterventionSite(backend.hook(layer=1, kind=HookKind.MLP_PRE), (0,), (2,))
    plan = CausalPlan(
        "trial-1",
        "This precise intervention is predicted to change the token contrast.",
        backend.identity.digest,
        NativeInput("donor", (1, 2, 3), (1, 1, 1)),
        NativeInput("recipient", (4, 2, 3), (1, 1, 1)),
        site,
        (
            mediator
            if mode
            in (
                InterventionMode.PATH_PATCH,
                InterventionMode.MEDIATION,
                InterventionMode.RESTORATION,
            )
            else None
        ),
        mode,
        LogitContrast(2, 3, 4),
        EffectPrediction(1, 1e-10, 1.0, 1e-12),
        (REF,),
        71,
        ((0.1,),) if mode is InterventionMode.ACTIVATION_REPLACE else None,
    )
    prediction = PredictionCommit(
        "predict-1", {"causal_plan_digest": plan.digest}, 0.5, (REF.reference_id,)
    )
    action = ActionRecord(
        "execute-1",
        "mechanistic_intervention",
        True,
        "mechanistic-plan:" + plan.digest,
        prediction.prediction_commit_id,
    )
    commit = freeze_causal_plan(
        plan, prediction=prediction, action=action, clock=Clock(), enabled=True
    )
    return source, backend, plan, commit


def test_prediction_and_action_bind_exact_plan_before_execution() -> None:
    _, native, plan, commit = trial()
    with pytest.raises(ValueError, match="(?i)commit"):
        execute_causal_plan(native, plan, clock=Clock(), enabled=True)
    wrong = PredictionCommit(
        "predict-1", {"causal_plan_digest": "0" * 64}, 0.5, (REF.reference_id,)
    )
    action = ActionRecord(
        "execute-1",
        "mechanistic_intervention",
        True,
        "mechanistic-plan:" + plan.digest,
        "predict-1",
    )
    with pytest.raises(ValueError, match="prediction"):
        freeze_causal_plan(plan, prediction=wrong, action=action, clock=Clock(), enabled=True)
    assert type(commit).from_dict(commit.to_dict()) == commit


@pytest.mark.parametrize("family", ["gpt2", "llama"])
@pytest.mark.parametrize("mode", list(InterventionMode))
def test_supported_modes_use_actual_forwards_and_preserve_source(mode, family) -> None:
    source, native, plan, commit = trial(mode, family)
    original = {k: v.detach().clone() for k, v in source.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = execute_causal_plan(native, commit, clock=Clock(), enabled=True)
    assert result["plan_digest"] == plan.digest
    assert result["commit_digest"] == commit.digest
    assert result["measurement_authentication"] == "NOT_PERFORMED"
    assert result["evidence_status"] == "UNRESOLVED"
    assert result["conditions"]["identity"]["effect"] == pytest.approx(0.0, abs=1e-12)
    assert result["conditions"]["intervention"]["logits"]
    assert math_recompute_contrast(result["conditions"]["intervention"]["logits"], plan.metric) == (
        pytest.approx(result["conditions"]["intervention"]["score"], abs=1e-12)
    )
    assert result["prediction_result"] in ("MET", "FALSIFIED", "NOT_SPECIFIC")
    assert torch.equal(rng, torch.get_rng_state())
    assert source.training is True
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in native._model.modules()) == 0
    )
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(result)


def math_recompute_contrast(logits, metric):
    return (
        logits[metric.position][metric.target_token]
        - logits[metric.position][metric.reference_token]
    )


def test_mediation_decomposition_is_actual_controlled_effect_arithmetic() -> None:
    _, native, _, commit = trial(InterventionMode.MEDIATION)
    report = execute_causal_plan(native, commit, clock=Clock(), enabled=True)
    values = report["mediation"]
    assert values["mediated_effect"] == pytest.approx(
        values["total_effect"] - values["direct_effect"]
    )
    assert values["causal_assumptions"]
    assert report["conditions"]["mediator_restored"]["logits"]


def test_unknown_backend_or_unbound_hook_cannot_become_causal_evidence() -> None:
    _, native, plan, _ = trial()
    forged = replace(
        plan, source=replace(plan.source, hook=replace(plan.source.hook, module_path="unknown"))
    )
    pred = PredictionCommit("p2", {"causal_plan_digest": forged.digest}, 0.5, (REF.reference_id,))
    action = ActionRecord(
        "a2", "mechanistic_intervention", True, "mechanistic-plan:" + forged.digest, "p2"
    )
    commit = freeze_causal_plan(forged, prediction=pred, action=action, clock=Clock(), enabled=True)
    with pytest.raises(ValueError, match="hook"):
        execute_causal_plan(native, commit, clock=Clock(), enabled=True)
    with pytest.raises(ValueError, match="enabled=True"):
        execute_causal_plan(None, None, clock=None)


def rebind(plan):
    pred = PredictionCommit("p", {"causal_plan_digest": plan.digest}, 0.5, (REF.reference_id,))
    action = ActionRecord(
        "a", "mechanistic_intervention", True, "mechanistic-plan:" + plan.digest, "p"
    )
    return freeze_causal_plan(plan, prediction=pred, action=action, clock=Clock(), enabled=True)


@pytest.mark.parametrize("family", ["gpt2", "llama"])
@pytest.mark.parametrize("kind", [HookKind.ATTENTION_HEAD, HookKind.RESIDUAL_STREAM])
def test_vector_heads_and_full_residual_swaps_reinsert_exact_donor_values(family, kind) -> None:
    _, native, plan, _ = trial(InterventionMode.COMPONENT_SWAP, family)
    hook = native.hook(layer=0, kind=kind)
    components = (0,) if kind is HookKind.ATTENTION_HEAD else tuple(range(hook.width))
    plan = replace(plan, source=InterventionSite(hook, components, (0, 2)))
    c = execute_causal_plan(native, rebind(plan), clock=Clock(), enabled=True)["conditions"]
    assert c["intervention"]["sites"]["source"]["after"] == c["donor"]["sites"]["source"]["before"]
    assert (
        c["reverse_swap"]["sites"]["source"]["after"] == c["recipient"]["sites"]["source"]["before"]
    )
    assert len(c["intervention"]["sites"]["source"]["after"][0]) == len(plan.source.columns)


@pytest.mark.parametrize("mode", [InterventionMode.RESTORATION, InterventionMode.PATH_PATCH])
def test_path_and_restoration_use_the_declared_actual_mediator_values(mode) -> None:
    _, native, _, commit = trial(mode)
    report = execute_causal_plan(native, commit, clock=Clock(), enabled=True)
    c = report["conditions"]
    origin = c["donor"] if mode is InterventionMode.RESTORATION else c["intervention_source_patch"]
    assert c["intervention"]["sites"]["mediator"]["after"] == origin["sites"]["mediator"]["before"]
    reference = (
        c["intervention_source_ablated"] if mode is InterventionMode.RESTORATION else c["recipient"]
    )
    assert report["primary_effect"] == c["intervention"]["score"] - reference["score"]


def test_hook_order_failure_removes_all_hooks_and_preserves_the_frozen_model() -> None:
    _, native, plan, _ = trial(InterventionMode.PATH_PATCH)
    mediator = InterventionSite(native.hook(layer=0, kind=HookKind.ATTENTION_RESIDUAL), (0,), (2,))
    plan = replace(plan, mediator=mediator)
    with pytest.raises(ValueError, match="order"):
        execute_causal_plan(native, rebind(plan), clock=Clock(), enabled=True)
    native._assert_frozen()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in native._model.modules())
