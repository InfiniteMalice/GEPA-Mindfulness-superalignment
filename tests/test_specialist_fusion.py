"""Analytical CPU fusion checks and existing provenance/admission boundaries."""

# Standard library
import builtins
import json
from dataclasses import replace
from unittest.mock import patch

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import SystemConfig
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.specialist_fusion import (
    collect_input_statistics,
    fuse_linear_layer,
    screen_rollout_text,
)
from gepa_mindfulness.training.specialist_fusion_records import (
    FusionCandidate,
    FusionSource,
    LayerStatistics,
    TFOPDConfig,
)


np = pytest.importorskip("numpy", reason="optional fusion extra is not installed")


def source(index=0, **changes):
    """Declare a host-owned admitted source without certifying factual correctness."""
    data = dict(
        base_checkpoint_digest=content_digest("base"),
        specialist_checkpoint_digest=content_digest(f"teacher:{index}"),
        architecture_id="authored-linear-fixture",
        layer_id="projection.weight",
        system=SystemConfig(f"teacher:{index}", "fixture-harness:v1", content_digest(index)),
        rollout_digest=content_digest(f"rollouts:{index}"),
        source_metadata_json=canonical_json(
            dict(
                training_eligibility="TRAIN",
                runtime_version="fixture-v1",
                selection_policy_version="host-selection-v1",
                sampling_config={},
                evaluator_version="fixture-v1",
                cohort="unverified-software-fixture",
            )
        ),
        evidence_refs=(EvidenceReference("record:fixture", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    data.update(changes)
    return FusionSource(**data)


def statistics(index, gram, token_count=2, **changes):
    """Analytical declared statistics are diagnostic input fixtures, not model evidence."""
    return LayerStatistics(
        source(index, **changes),
        token_count,
        tuple(tuple(float(v) for v in row) for row in gram),
        content_digest(f"inputs:{index}"),
        "float64",
    )


def config(**changes):
    """Small matrices use FP64 unless a test explicitly investigates storage rounding."""
    return TFOPDConfig(
        enabled=True, accumulation_dtype="float64", storage_dtype="float64", **changes
    )


def test_default_off_never_imports_numpy_or_inspects_inputs():
    """Optional numerical work requires an explicit enabled configuration."""
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "numpy":
            raise AssertionError("disabled numerical dependency imported")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", guarded):
        assert collect_input_statistics(None, object()) is None
        assert fuse_linear_layer(object(), object(), object()) is None
        assert screen_rollout_text(object(), finished=False) is None


def test_config_round_trip_and_rejected_extensions():
    """Versioning rejects coercible flags, silent fallbacks and unknown fields."""
    wire = TFOPDConfig().to_dict()
    assert TFOPDConfig.from_dict(json.loads(json.dumps(wire))) == TFOPDConfig()
    for changes in (
        {"enabled": 1},
        {"alpha": True},
        {"alpha": -1},
        {"alpha": float("nan")},
        {"condition_limit": 0},
        {"storage_dtype": "bfloat16"},
        {"accumulation_dtype": "float16"},
    ):
        with pytest.raises(ValueError):
            TFOPDConfig(**changes)
    for key, value in (("schema_version", "future"), ("allow_unverified_promotion", True)):
        altered = dict(wire, **{key: value})
        with pytest.raises(ValueError):
            TFOPDConfig.from_dict(altered)


def test_token_normalization_and_source_capture_are_immutable():
    """Each teacher's moments are averaged over its own actual input token positions."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    first = collect_input_statistics(source(), values, config=config())
    duplicated = collect_input_statistics(source(), np.tile(values, (5, 1)), config=config())
    np.testing.assert_array_equal(first.gram, [[5.0, 7.0], [7.0, 10.0]])
    assert first.gram == duplicated.gram
    assert first.token_count == 2 and duplicated.token_count == 10
    values[:] = 100.0
    assert first.gram == ((5.0, 7.0), (7.0, 10.0))
    assert LayerStatistics.from_dict(first.to_dict()) == first
    with pytest.raises(ValueError):
        require_training_eligible(first.to_dict())


def test_identical_grams_give_weighted_average_not_task_arithmetic():
    """Equal input distributions reduce to the weighted teacher mean."""
    base = np.array([[10.0, 20.0]])
    teachers = (base + [2.0, 4.0], base + [6.0, 8.0])
    grams = (statistics(0, np.eye(2)), statistics(1, np.eye(2)))
    result = fuse_linear_layer(base, teachers, grams, domain_weights=(1.0, 3.0), config=config())
    np.testing.assert_allclose(result.weights, [[15.0, 27.0]], rtol=0, atol=1e-12)
    assert FusionCandidate.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    assert result.to_dict()["optimizer_eligible"] is False
    assert result.to_dict()["authority_granted"] is False
    with pytest.raises(ValueError):
        require_training_eligible(result.to_dict())
    assert result.sources == tuple(item.source for item in grams)


def test_complementary_support_preserves_each_specialist_on_its_inputs():
    """Orthogonal teacher supports combine without the unrelated teacher's update."""
    base = np.zeros((1, 2))
    teachers = (np.array([[2.0, 99.0]]), np.array([[88.0, 3.0]]))
    grams = (statistics(0, [[1, 0], [0, 0]]), statistics(1, [[0, 0], [0, 1]]))
    result = fuse_linear_layer(base, teachers, grams, config=config(alpha=1.0))
    np.testing.assert_array_equal(result.weights, [[2.0, 3.0]])
    diagnostics = json.loads(result.diagnostics_json)
    assert [row["unreproduced_share"] for row in diagnostics["teachers"]] == [0.0, 0.0]


def test_underflowing_rhs_cannot_silently_return_the_baseline():
    """A well-conditioned solve still fails closed if its nonzero RHS loses magnitude."""
    with pytest.raises(ValueError, match="numerical"):
        fuse_linear_layer(
            [[0.0]],
            ([[1e-200]],),
            (statistics(0, [[1e-200]]),),
            config=config(),
        )


def test_underflowing_supported_energy_cannot_claim_zero_reference():
    """An unsupported large delta component must not hide positive supported energy."""
    with pytest.raises(ValueError, match="numerical"):
        fuse_linear_layer(
            np.zeros((1, 2)),
            (np.array([[1e-200, 1.0]]), np.array([[0.0, 1.0]])),
            (statistics(0, [[1, 0], [0, 0]]), statistics(1, [[1, 0], [0, 1]])),
            config=config(alpha=1.0),
        )


def test_oblique_system_uses_delta_formula_and_off_diagonal_shrinkage():
    """The hand-derived nonsymmetric numerator must multiply the inverse on the right."""
    base = np.array([[1.0, -1.0], [2.0, 0.0]])
    teachers = (base + [[1.0, 2.0], [3.0, 4.0]], base + [[5.0, 6.0], [7.0, 8.0]])
    grams = (statistics(0, [[2, 1], [1, 3]]), statistics(1, [[4, -1], [-1, 2]]))
    # At alpha=1, the Gram sum is diag(6, 5); numerator is [[18, 14], [30, 24]].
    result = fuse_linear_layer(base, teachers, grams, config=config(alpha=1.0))
    np.testing.assert_allclose(
        result.weights, base + [[18 / 6, 14 / 5], [30 / 6, 24 / 5]], atol=1e-12
    )
    diagonal = fuse_linear_layer(base, teachers, grams, config=config(alpha=0.0))
    np.testing.assert_allclose(
        diagonal.weights, base + [[22 / 6, 18 / 5], [34 / 6, 28 / 5]], atol=1e-12
    )


@pytest.mark.parametrize(
    "bad",
    [
        [[float("nan"), 0.0]],
        [[True, 1.0]],
        [["1", "2"]],
        [[1.0], [1.0, 2.0]],
        np.zeros((4097, 2)),
        np.zeros((2, 65)),
        np.ones((2, 2), dtype=np.float16),
    ],
)
def test_bad_activations_are_rejected(bad):
    """No coercion, oversized allocation or unsupported precision enters statistics."""
    with pytest.raises(ValueError):
        collect_input_statistics(source(), bad, config=config())


@pytest.mark.parametrize(
    "metadata",
    [
        {"training_eligibility": "HIDDEN_EVAL"},
        {"training_eligibility": "TRAIN", "nested": {"training_eligibility": "REGRESSION"}},
        {"runtime_version": "not-admitted"},
    ],
)
def test_source_restrictions_fail_before_activation_access(metadata):
    """The shared guard preserves nested restrictions instead of trusting an outer label."""
    with pytest.raises(ValueError):
        source(source_metadata_json=canonical_json(metadata))


def test_common_base_architecture_layer_and_unique_teachers_are_required():
    """Declared lineage cannot silently merge incompatible specialists."""
    base = np.zeros((1, 2))
    first = statistics(0, np.eye(2))
    for second in (
        first,
        statistics(1, np.eye(2), base_checkpoint_digest=content_digest("other")),
        statistics(1, np.eye(2), architecture_id="other"),
        statistics(1, np.eye(2), layer_id="other"),
    ):
        with pytest.raises(ValueError):
            fuse_linear_layer(base, (base, base), (first, second), config=config())


@pytest.mark.parametrize("gram", [[[1, 2], [0, 1]], [[1, 0], [0, -1]], [[0, 0], [0, 0]]])
def test_invalid_or_singular_grams_do_not_produce_candidates(gram):
    """No implicit repair, pseudo-inverse or baseline substitution is allowed."""
    base = np.zeros((1, 2))
    with pytest.raises(ValueError):
        fuse_linear_layer(base, (base,), (statistics(0, gram),), config=config(alpha=1.0))


def test_ill_conditioned_system_and_bad_weights_are_rejected():
    """Numerically unsafe sources cannot be hidden through a successful status."""
    base = np.zeros((1, 2))
    grams = (statistics(0, [[1, 0], [0, 1e-14]]),)
    with pytest.raises(ValueError):
        fuse_linear_layer(base, (base,), grams, config=config(alpha=1.0))
    for weights in ((0.0,), (-1.0,), (True,), (float("inf"),), (1.0, 2.0)):
        with pytest.raises(ValueError):
            fuse_linear_layer(
                base, (base,), (statistics(0, np.eye(2)),), domain_weights=weights, config=config()
            )


def test_zero_reference_energy_is_missing_and_precision_loss_is_visible():
    """Undefined teacher diagnostics are None; FP32-rounded updates are measured honestly."""
    base = np.ones((1, 2))
    teachers = (base.copy(), base + 1e-9)
    grams = (statistics(0, np.eye(2)), statistics(1, np.eye(2)))
    result = fuse_linear_layer(
        base,
        teachers,
        grams,
        config=TFOPDConfig(enabled=True, accumulation_dtype="float64", storage_dtype="float32"),
    )
    np.testing.assert_array_equal(result.weights, base)
    diagnostics = json.loads(result.diagnostics_json)
    assert diagnostics["teachers"][0]["unreproduced_share"] is None
    assert diagnostics["quantization_error_norm"] > 0
    assert diagnostics["solve_update_norm"] > 0
    assert diagnostics["stored_update_norm"] == 0


def test_heuristic_loop_screen_cannot_certify_recovery_or_training_admission():
    """Finished, low-entropy text is only a screen; no behavior penalty is generated."""
    loop = screen_rollout_text("repeat" * 10000, finished=True, enabled=True)
    assert loop["compression_flag"] is True
    assert loop["keep_by_text_screen"] is False
    unfinished = screen_rollout_text("A finished-looking answer", finished=False, enabled=True)
    assert unfinished["keep_by_text_screen"] is False
    assert loop["authority_granted"] is False
    with pytest.raises(ValueError):
        require_training_eligible(loop)


def test_tiny_nonzero_teacher_energy_does_not_become_an_undefined_share():
    """Scaling a correct single-teacher merge cannot erase its interference diagnostic."""
    base = np.zeros((1, 2))
    teacher = np.array([[1e-200, 1e-200]])
    result = fuse_linear_layer(base, (teacher,), (statistics(0, np.eye(2)),), config=config())
    row = json.loads(result.diagnostics_json)["teachers"][0]
    assert row["unreproduced_share"] == 0.0
    assert row["reference_energy"] is None
    assert row["reference_energy_status"] == "not_representable"
    assert row["update_norm"] > 0


def test_interference_is_invariant_under_tiny_global_update_scaling():
    """Relative error remains measurable when raw squared energies underflow binary64."""
    base = np.zeros((1, 2))
    teachers = (np.array([[1e-200, 0.0]]), np.array([[3e-200, 0.0]]))
    result = fuse_linear_layer(
        base, teachers, (statistics(0, np.eye(2)), statistics(1, np.eye(2))), config=config()
    )
    rows = json.loads(result.diagnostics_json)["teachers"]
    np.testing.assert_allclose(
        [row["unreproduced_share"] for row in rows], [1.0, 1.0 / 9], rtol=1e-12
    )


def test_rounding_underflow_in_statistics_and_overflow_in_storage_fail_visibly():
    """Unsupported moment/storage ranges never manufacture zero-information success."""
    with pytest.raises(ValueError):
        collect_input_statistics(source(), np.full((2, 2), 1e-30), config=TFOPDConfig(enabled=True))
    base = np.full((1, 2), 1e40)
    with pytest.raises(ValueError):
        fuse_linear_layer(
            base, (base.copy(),), (statistics(0, np.eye(2)),), config=TFOPDConfig(enabled=True)
        )


def test_candidate_numbers_and_source_bindings_are_detached_from_inputs():
    """Editing an array or serialized view cannot mutate the retained proposal."""
    base = np.zeros((1, 2))
    teacher = np.array([[1.0, 2.0]])
    stats = statistics(0, np.eye(2))
    result = fuse_linear_layer(base, (teacher,), (stats,), config=config())
    base[:] = 100.0
    teacher[:] = -100.0
    np.testing.assert_array_equal(result.weights, [[1.0, 2.0]])
    assert result.sources == (stats.source,)
    wire = result.to_dict()
    wire["weights"][0][0] = 999.0
    assert result.weights[0][0] == 1.0
    with pytest.raises(ValueError):
        replace(result, authority_granted=True)


def test_teacher_diagnostics_distinguish_solved_and_stored_candidate_fit():
    """A perfect mathematical fit must not conceal a storage-induced loss of the update."""
    base = np.ones((1, 2))
    teacher = base + 1e-9
    result = fuse_linear_layer(
        base,
        (teacher,),
        (statistics(0, np.eye(2)),),
        config=TFOPDConfig(enabled=True, accumulation_dtype="float64", storage_dtype="float32"),
    )
    row = json.loads(result.diagnostics_json)["teachers"][0]
    assert row["unreproduced_share"] == 0.0
    assert row["stored_unreproduced_share"] == 1.0


def test_declared_float32_statistics_cannot_contain_unrounded_float64_values():
    """Serialized precision must match the actual retained moment values."""
    with pytest.raises(ValueError, match="float32"):
        LayerStatistics(
            source(), 2, ((1.0 + 1e-12, 0.0), (0.0, 1.0)), content_digest("inputs"), "float32"
        )
