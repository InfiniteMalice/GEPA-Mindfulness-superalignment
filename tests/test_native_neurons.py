"""Native hooks measure actual architecture-specific tensors without changing source models."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from gepa_mindfulness.interpret.native_neurons import (
    FrozenNativeModel,
    measurement_graph,
    sample_signed_regimes,
)
from gepa_mindfulness.interpret.native_records import HookKind, SamplingThresholds
from gepa_mindfulness.training.eligibility import require_training_eligible

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def model(family="gpt2"):
    torch.manual_seed(17)
    if family == "gpt2":
        return transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_embd=16,
                n_inner=32,
                n_layer=2,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            max_position_embeddings=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def backend(source=None):
    return FrozenNativeModel(
        model() if source is None else source,
        model_id="random-test-model",
        revision="seed-17",
        enabled=True,
    )


def capture(native, kind=HookKind.MLP_POST, *, components=(0,), positions=(0, 1, 2)):
    return native.capture(
        native.hook(layer=0, kind=kind),
        torch.tensor([[1, 2, 3]]),
        component_indices=components,
        example_ids=("generation-1",),
        token_positions=positions,
        enabled=True,
    )


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_capture_matches_real_hook_and_preserves_source(family) -> None:
    source = model(family)
    source.train()
    original = {k: v.detach().clone() for k, v in source.state_dict().items()}
    native = backend(source)
    assert source.training is True
    assert all(p.requires_grad for p in source.parameters())
    assert native._model is not source
    assert native._model.training is False
    assert all(not p.requires_grad for p in native._model.parameters())
    kind = HookKind.MLP_POST if family == "gpt2" else HookKind.MLP_PRODUCT
    batch = capture(native, kind, components=(0, 3))
    mlp = (
        native._model.transformer.h[0].mlp
        if family == "gpt2"
        else native._model.model.layers[0].mlp
    )
    down = mlp.c_proj if family == "gpt2" else mlp.down_proj
    actual = {}

    def hook(module, inputs):
        actual["values"] = inputs[0].detach().clone()

    handle = down.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            native._model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        handle.remove()
    for record in batch.measurements:
        assert record.activation == (
            actual["values"][0, record.token_position, record.component_index].item(),
        )
        assert record.context_tokens == (1, 2, 3)
        assert record.model_digest == native.identity.digest
        assert record.hook_digest == batch.hook.digest
    assert batch.tensor_shape == (1, 3, 32)
    assert batch.dtype == "torch.float32" and batch.device == "cpu"
    assert batch.evidence_status == "DISCOVERED_CORRELATION"
    assert batch.hook.normalization and batch.hook.upstream_paths and batch.hook.downstream_paths
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in native._model.modules()) == 0
    )
    assert type(batch).from_dict(batch.to_dict()) == batch
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(batch.to_dict())


def test_gated_postactivation_is_not_silently_conflated_with_product() -> None:
    native = backend(model("llama"))
    gate = capture(native, HookKind.MLP_POST)
    up = capture(native, HookKind.MLP_UP)
    product = capture(native, HookKind.MLP_PRODUCT)
    assert gate.hook.module_path.endswith("act_fn")
    assert product.hook.module_path.endswith("down_proj")
    for g, u, p in zip(gate.measurements, up.measurements, product.measurements):
        expected = torch.tensor(g.activation, dtype=torch.float32) * torch.tensor(u.activation)
        assert p.activation[0] == expected.item()
    assert gate.hook.transformation != product.hook.transformation


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_attention_head_and_residual_coordinates(family) -> None:
    native = backend(model(family))
    head = capture(native, HookKind.ATTENTION_HEAD, components=(1,))
    assert head.hook.heads == 2 and head.hook.head_dim == 8
    assert all(len(r.activation) == 8 and r.component_index == 1 for r in head.measurements)
    residual = capture(native, HookKind.MLP_RESIDUAL, components=(15,))
    assert residual.tensor_shape == (1, 3, 16)
    stream = capture(native, HookKind.RESIDUAL_STREAM)
    assert stream.hook.location == "input"
    assert stream.tensor_shape == (1, 3, 16)


def test_wrong_hooks_positions_and_input_types_fail_explicitly() -> None:
    native = backend()
    hook = native.hook(layer=0, kind=HookKind.MLP_POST)
    with pytest.raises(ValueError, match="unsupported"):
        native.hook(layer=0, kind=HookKind.MLP_PRODUCT)
    with pytest.raises(ValueError, match="layer"):
        native.hook(layer=2, kind=HookKind.MLP_POST)
    with pytest.raises(ValueError, match="hook"):
        native.capture(
            replace(hook, module_path="transformer.h.1.mlp.c_proj"),
            torch.tensor([[1, 2, 3]]),
            component_indices=(0,),
            example_ids=("generation-1",),
            token_positions=(0,),
            enabled=True,
        )
    for ids, positions, components in (
        (torch.tensor([[1.0, 2.0]]), (0,), (0,)),
        (torch.tensor([[1, 2]]), (2,), (0,)),
        (torch.tensor([[1, 2]]), (0,), (32,)),
        (torch.tensor([[1, 2]]), (0, 0), (0,)),
        (torch.tensor([[1, 32]]), (0,), (0,)),
    ):
        with pytest.raises(ValueError):
            native.capture(
                hook,
                ids,
                component_indices=components,
                example_ids=("generation-1",),
                token_positions=positions,
                enabled=True,
            )


def test_forward_failure_removes_hooks_and_mutated_frozen_weights_are_rejected(monkeypatch) -> None:
    native = backend()

    def failure(*args, **kwargs):
        raise RuntimeError("controlled forward failure")

    with monkeypatch.context() as patch:
        patch.setattr(native._model, "forward", failure)
        with pytest.raises(ValueError, match="frozen model"):
            capture(native)
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in native._model.modules()) == 0
    )
    with torch.no_grad():
        next(native._model.parameters()).add_(1)
    with pytest.raises(ValueError, match="frozen model"):
        capture(native)


def test_exception_inside_capture_hook_removes_temporary_hook() -> None:
    source = model()
    with torch.no_grad():
        source.transformer.wte.weight.fill_(float("nan"))
    native = backend(source)
    with pytest.raises(ValueError, match="nonfinite"):
        capture(native)
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in native._model.modules()) == 0
    )


def test_explicit_reader_weights_respect_linear_and_conv1d_layouts() -> None:
    for family in ("gpt2", "llama"):
        source = model(family)
        native = backend(source)
        result = native.component_weights(
            native.hook(layer=0, kind=HookKind.MLP_POST), component_index=3, enabled=True
        )
        mlp = source.transformer.h[0].mlp if family == "gpt2" else source.model.layers[0].mlp
        expected = mlp.c_proj.weight[3, :] if family == "gpt2" else mlp.down_proj.weight[:, 3]
        assert result["downstream"][0]["values"] == expected.detach().tolist()
        assert result["evidence_status"] == "DISCOVERED_CORRELATION"
        if family == "llama":
            assert len(result["upstream"]) == 2


def test_sampler_retains_weak_negative_zero_and_explicit_missing_regimes() -> None:
    native = backend()
    values = (-2.0, -0.7, -0.2, -0.01, 0.0, 0.01, 0.2, 0.7, 2.0)
    actual = native.capture(
        native.hook(layer=0, kind=HookKind.MLP_POST),
        torch.tensor([list(range(1, 10))]),
        component_indices=(0,),
        example_ids=("generation-1",),
        token_positions=tuple(range(9)),
        enabled=True,
    )
    # Explicit synthetic amplitudes for sampling software; preserve real coordinates/digests.
    records = tuple(replace(m, activation=(v,)) for m, v in zip(actual.measurements, values))
    batch = replace(actual, measurements=records)
    sampled = sample_signed_regimes(
        batch,
        discovery_ids=("generation-1",),
        seed=19,
        per_regime=2,
        thresholds=SamplingThresholds(0.05, 0.5, 1.0),
        enabled=True,
    )
    assert sampled["missing_regimes"] == []
    assert set(sampled["regimes"]) == {
        "positive_high",
        "negative_high",
        "positive_medium",
        "negative_medium",
        "positive_weak",
        "negative_weak",
        "near_zero",
        "nonactivating",
    }
    assert all(sampled["regimes"].values())
    assert sampled["seed"] == 19
    with pytest.raises(ValueError, match="discovery"):
        sample_signed_regimes(
            batch,
            discovery_ids=("locked-1",),
            seed=19,
            per_regime=2,
            thresholds=SamplingThresholds(0.05, 0.5, 1.0),
            enabled=True,
        )
    zeros = replace(batch, measurements=tuple(replace(m, activation=(0.0,)) for m in records))
    empty = sample_signed_regimes(
        zeros,
        discovery_ids=("generation-1",),
        seed=19,
        per_regime=2,
        thresholds=SamplingThresholds(0.05, 0.5, 1.0),
        enabled=True,
    )
    assert "negative_high" in empty["missing_regimes"]


def test_native_graph_is_observational_and_vector_reduction_is_never_silent() -> None:
    native = backend()
    graph = measurement_graph(capture(native), prompt="toy", response="", enabled=True)
    assert graph.method == "native_activation_observation"
    assert graph.metadata["evidence_status"] == "DISCOVERED_CORRELATION"
    assert not graph.edges
    assert graph.to_networkx().number_of_nodes() == 3
    with pytest.raises(ValueError, match="vector"):
        measurement_graph(
            capture(native, HookKind.ATTENTION_HEAD), prompt="toy", response="", enabled=True
        )


def test_default_off_unsupported_architecture_and_size_limits() -> None:
    with pytest.raises(ValueError, match="enabled=True"):
        FrozenNativeModel(None, model_id="test", revision="v1")
    with pytest.raises(ValueError, match="architecture"):
        FrozenNativeModel(torch.nn.Linear(2, 2), model_id="test", revision="v1", enabled=True)
    with pytest.raises(ValueError, match="parameter"):
        FrozenNativeModel(model(), model_id="test", revision="v1", max_parameters=1, enabled=True)


@pytest.mark.parametrize("family", ["gpt2", "llama"])
def test_attention_residual_and_padding_measure_only_declared_positions(family) -> None:
    native = backend(model(family))
    batch = native.capture(
        native.hook(layer=1, kind=HookKind.ATTENTION_RESIDUAL),
        torch.tensor([[1, 2, 3], [1, 3, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        component_indices=(0, 2),
        example_ids=("generation-1", "generation-2"),
        token_positions=(1, 2),
        enabled=True,
    )
    assert len(batch.measurements) == 6
    assert not any(
        m.example_id == "generation-2" and m.token_position == 2 for m in batch.measurements
    )
    assert batch.hook.location == "output[0]"
    assert type(batch).from_dict(batch.to_dict()) == batch


def test_typed_native_records_do_not_accept_forged_provenance_or_causal_status() -> None:
    batch = capture(backend())
    with pytest.raises(ValueError, match="self-certify"):
        replace(batch, evidence_status="INTERVENTION_SUPPORTED")
    with pytest.raises(ValueError, match="missing measurement"):
        replace(batch, measurements=batch.measurements[:-1])
    with pytest.raises(ValueError, match="provenance"):
        replace(
            batch,
            measurements=(replace(batch.measurements[0], model_digest="0" * 64),)
            + batch.measurements[1:],
        )
    with pytest.raises(ValueError, match="recorded context"):
        replace(batch.measurements[0], context_tokens=(1, 2, 4))
    with pytest.raises(ValueError, match="dtype"):
        replace(batch, dtype="unknown")


def test_input_lengths_fail_at_declared_architecture_limit_before_forward() -> None:
    native = backend()
    with pytest.raises(ValueError, match="position capacity"):
        native.capture(
            native.hook(layer=0, kind=HookKind.MLP_PRE),
            torch.ones((1, 17), dtype=torch.long),
            component_indices=(0,),
            example_ids=("generation-1",),
            token_positions=(0,),
            enabled=True,
        )


def test_graph_and_regime_sampling_do_not_silently_mix_examples_or_components() -> None:
    native = backend()
    batch = native.capture(
        native.hook(layer=0, kind=HookKind.MLP_POST),
        torch.tensor([[1, 2, 3], [2, 3, 1]]),
        component_indices=(0,),
        example_ids=("one", "two"),
        token_positions=(0,),
        enabled=True,
    )
    with pytest.raises(ValueError, match="single example"):
        measurement_graph(batch, prompt="one prompt", response="", enabled=True)
    with pytest.raises(ValueError, match="single component"):
        sample_signed_regimes(
            capture(native, components=(0, 1)),
            discovery_ids=("generation-1",),
            seed=0,
            per_regime=1,
            thresholds=SamplingThresholds(0.05, 0.5, 1.0),
            enabled=True,
        )


def test_internal_module_substitution_cannot_reuse_the_same_computation_identity() -> None:
    source = model()
    source.transformer.h[0].mlp.act = torch.nn.ReLU()
    with pytest.raises(ValueError, match="internal"):
        backend(source)


def test_attention_implementation_changes_invalidate_frozen_computation() -> None:
    source = model()
    source.config._attn_implementation = "eager"
    native = backend(source)
    native._model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="frozen model"):
        capture(native)
