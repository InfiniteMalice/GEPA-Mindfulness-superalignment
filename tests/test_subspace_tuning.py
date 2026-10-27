"""Contract and real CPU tensor checks for optional applied-update SST."""

# Standard library
import copy
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

# Local
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.subspace_tuning import SSTConfig, create_sst_adapter


class SSTConfigTests(unittest.TestCase):
    """The adapter cannot activate through malformed or implicit configuration."""

    def test_default_off_does_not_inspect_optimizer_or_targets(self):
        """Default-off construction requires neither tensors nor PyTorch."""
        self.assertIsNone(create_sst_adapter(object(), object(), reference_id="fixture"))

    def test_config_round_trip_is_versioned_and_exact(self):
        """Checkpoint configuration retains every experimental setting."""
        config = SSTConfig(enabled=True, rank_fraction=0.5, refresh_every=2, fallback="freeze")
        self.assertEqual(SSTConfig.from_dict(config.to_dict()), config)
        for key, value in (("schema_version", "future"), ("enabled", 1), ("extra", None)):
            with self.subTest(key=key):
                wire = config.to_dict()
                wire[key] = value
                with self.assertRaises(ValueError):
                    SSTConfig.from_dict(wire)

    def test_invalid_configuration_fails_before_tensor_access(self):
        """Ranks, refresh cadence and resource limits cannot be ambiguous."""
        for kwargs in (
            {"enabled": 1},
            {"rank_fraction": True},
            {"rank_fraction": -0.1},
            {"rank_fraction": 1.1},
            {"rank_fraction": float("nan")},
            {"rank_fraction": 10**1000},
            {"refresh_every": 0},
            {"refresh_every": True},
            {"fallback": "unprotected"},
            {"max_matrix_elements": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SSTConfig(**kwargs)


def setup_pair(optimizer_name="AdamW", rank_fraction=0.5, **config):
    """Use an oblique basis so coordinate-wise optimizer scaling can leak."""
    import torch

    initial = torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    parameter = torch.nn.Parameter(initial.clone())
    control = torch.nn.Parameter(initial.clone())
    optimizer_type = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_type([parameter], lr=0.05, weight_decay=0.1)
    control_optimizer = optimizer_type([control], lr=0.05, weight_decay=0.1)
    adapter = create_sst_adapter(
        optimizer,
        {"weight": parameter},
        reference_id="authored-matrix-v1",
        config=SSTConfig(enabled=True, rank_fraction=rank_fraction, **config),
    )
    return parameter, control, optimizer, control_optimizer, adapter


@unittest.skipUnless(importlib.util.find_spec("torch"), "real PyTorch unavailable")
class SSTTensorTests(unittest.TestCase):
    """Run actual optimizers, including moments, weight decay and checkpoint recovery."""

    def test_invalid_ownership_aliases_and_runtime_modes_are_rejected(self):
        """Protection cannot silently target an unowned, aliased or unsupported tensor."""
        import torch

        parameter = torch.nn.Parameter(torch.eye(2, dtype=torch.float64))
        other = torch.nn.Parameter(torch.eye(2, dtype=torch.float64))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        for targets in ({"foreign": other}, {"a": parameter, "b": parameter}):
            with self.subTest(targets=tuple(targets)):
                with self.assertRaises(ValueError):
                    create_sst_adapter(
                        optimizer, targets, reference_id="fixture", config=SSTConfig(enabled=True)
                    )
        optimizer.param_groups[0]["differentiable"] = True
        with self.assertRaises(ValueError):
            create_sst_adapter(
                optimizer,
                {"weight": parameter},
                reference_id="fixture",
                config=SSTConfig(enabled=True),
            )
        alias = torch.nn.Parameter(parameter.detach())
        aliased_optimizer = torch.optim.SGD([parameter, alias], lr=0.1)
        with self.assertRaises(ValueError):
            create_sst_adapter(
                aliased_optimizer,
                {"weight": parameter},
                reference_id="fixture",
                config=SSTConfig(enabled=True),
            )

    def test_changed_optimizer_groups_and_matrix_budget_fail_before_step(self):
        """Captured ownership and bounded decomposition work remain explicit."""
        import torch

        parameter, _control, optimizer, _ordinary, adapter = setup_pair()
        optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaises(ValueError):
            adapter.step()
        self.assertFalse(optimizer.state)
        with self.assertRaises(ValueError):
            setup_pair(max_matrix_elements=1)

    def test_native_optimizers_project_actual_step_and_retain_full_gradient_state(self):
        """Projection happens after native stateful preconditioning and weight decay."""
        import torch

        for name in ("SGD", "Adam", "AdamW"):
            with self.subTest(optimizer=name):
                parameter, control, optimizer, ordinary, adapter = setup_pair(name)
                original = parameter.detach().clone()
                gradient = torch.tensor([[1.0, 4.0], [2.0, -3.0]], dtype=torch.float64)
                parameter.grad, control.grad = gradient.clone(), gradient.clone()
                ordinary.step()
                report = adapter.step()
                free = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float64)
                expected = original + free @ (control.detach() - original) @ free
                torch.testing.assert_close(parameter, expected, atol=1e-12, rtol=0)
                torch.testing.assert_close(parameter.grad, gradient)
                for key, value in ordinary.state.get(control, {}).items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value)
                self.assertLess(report["targets"]["weight"]["left_leakage"], 1e-12)
                self.assertGreater(report["targets"]["weight"]["raw_update_norm"], 0)
                self.assertFalse(report["authority_granted"])
                with self.assertRaises(ValueError):
                    require_training_eligible(report)
                json.dumps(report, allow_nan=False)

    def test_rank_extremes_match_ordinary_update_or_freeze(self):
        """Rank zero preserves the step; full protected rank suppresses it."""
        import torch

        for rank in (0.0, 1.0):
            parameter, control, _optimizer, ordinary, adapter = setup_pair(rank_fraction=rank)
            original = parameter.detach().clone()
            parameter.grad = torch.ones_like(parameter)
            control.grad = torch.ones_like(control)
            ordinary.step()
            adapter.step()
            torch.testing.assert_close(parameter, control if rank == 0 else original)

    def test_rank_zero_preserves_native_result_despite_cancellation(self):
        """An identity transform must not reconstruct small weights from enormous operands."""
        import torch

        parameter = torch.nn.Parameter(torch.eye(2, dtype=torch.float64) * 1e20)
        control = torch.nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW([parameter], lr=1.0, weight_decay=1.0)
        ordinary = torch.optim.AdamW([control], lr=1.0, weight_decay=1.0)
        adapter = create_sst_adapter(
            optimizer,
            {"weight": parameter},
            reference_id="cancellation-fixture",
            config=SSTConfig(enabled=True, rank_fraction=0),
        )
        parameter.grad, control.grad = torch.ones_like(parameter), torch.ones_like(control)
        ordinary.step()
        adapter.step()
        torch.testing.assert_close(parameter, control, rtol=0, atol=0)

    def test_full_rank_preserves_exact_weights_at_extreme_update_scale(self):
        """A fully protected target remains bitwise unchanged despite amplified SVD roundoff."""
        import torch

        parameter = torch.nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64) * 1e-30
        )
        original = parameter.detach().clone()
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        adapter = create_sst_adapter(
            optimizer,
            {"weight": parameter},
            reference_id="full-protection-fixture",
            config=SSTConfig(enabled=True, rank_fraction=1),
        )
        parameter.grad = torch.full_like(parameter, 1e20)
        adapter.step()
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)

    def test_fixed_reference_and_explicit_refresh_have_distinct_generations(self):
        """A refresh is visible and occurs only after the configured successful step."""
        import torch

        for interval in (None, 1):
            parameter, _control, _optimizer, _ordinary, adapter = setup_pair(refresh_every=interval)
            original = adapter.state_dict()["references"]["weight"].clone()
            for index in range(2):
                parameter.grad = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=torch.float64)
                report = adapter.step()
                self.assertEqual(report["basis_step"], index if interval == 1 else 0)
                self.assertEqual(report["refreshed"], interval == 1)
            if interval is None:
                torch.testing.assert_close(adapter.state_dict()["references"]["weight"], original)

    def test_checkpoint_restores_reference_and_next_step(self):
        """Model, native optimizer and adapter state jointly reproduce the next update."""
        import torch

        parameter, _control, optimizer, _ordinary, adapter = setup_pair(refresh_every=2)
        parameter.grad = torch.ones_like(parameter)
        adapter.step()
        saved = adapter.state_dict()
        native = copy.deepcopy(optimizer.state_dict())
        other, _control, other_optimizer, _ordinary, restored = setup_pair(refresh_every=2)
        with torch.no_grad():
            other.copy_(parameter)
        other_optimizer.load_state_dict(native)
        restored.load_state_dict(saved)
        saved["references"]["weight"].fill_(99)
        gradient = torch.tensor([[2.0, -3.0], [1.0, 4.0]], dtype=torch.float64)
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        self.assertEqual(adapter.step(), restored.step())
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)

    def test_corrupt_checkpoint_does_not_mutate_adapter(self):
        """Unknown schema, mismatched configuration and malformed anchors fail closed."""
        import torch

        _parameter, _control, _optimizer, _ordinary, adapter = setup_pair()
        for key, value in (
            ("schema_version", "future"),
            ("step", -1),
            ("reference_id", "foreign"),
            ("references", {"weight": torch.zeros((3, 3))}),
            ("config", SSTConfig(enabled=True, rank_fraction=1.0).to_dict()),
        ):
            state = adapter.state_dict()
            state[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    adapter.load_state_dict(state)
                self.assertEqual(adapter.state_dict()["step"], 0)

    def test_nonmatrix_fallback_freezes_instead_of_silent_unprotected_step(self):
        """Unsupported target shapes require explicit freeze or raise policy."""
        import torch

        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        with self.assertRaises(ValueError):
            create_sst_adapter(
                optimizer,
                {"bias": parameter},
                reference_id="fixture",
                config=SSTConfig(enabled=True),
            )
        adapter = create_sst_adapter(
            optimizer,
            {"bias": parameter},
            reference_id="fixture",
            config=SSTConfig(enabled=True, fallback="freeze"),
        )
        parameter.grad = torch.ones_like(parameter)
        report = adapter.step()
        torch.testing.assert_close(parameter, torch.ones(3))
        self.assertEqual(report["targets"]["bias"]["status"], "frozen")
        self.assertIn("matrix", report["targets"]["bias"]["reason"])

    def test_decomposition_failure_uses_declared_freeze_fallback(self):
        """An unavailable SVD never silently turns protection off."""
        import torch

        with patch.object(torch.linalg, "svd", side_effect=RuntimeError("fixture SVD failure")):
            parameter, _control, _optimizer, _ordinary, adapter = setup_pair(fallback="freeze")
        original = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        report = adapter.step()
        torch.testing.assert_close(parameter, original)
        self.assertEqual(report["targets"]["weight"]["status"], "frozen")

    def test_projection_failure_rolls_back_all_parameters_and_optimizer_state(self):
        """A failure after native Adam changes moments must not leave a half-step."""
        import torch

        parameter, _control, optimizer, _ordinary, _adapter = setup_pair()
        bias = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer.add_param_group({"params": [bias]})
        adapter = create_sst_adapter(
            optimizer,
            {"weight": parameter},
            reference_id="fixture",
            config=SSTConfig(enabled=True, rank_fraction=0.5),
        )
        parameter.grad = torch.ones_like(parameter)
        bias.grad = torch.ones_like(bias)
        adapter.step()
        original = parameter.detach().clone()
        original_bias = bias.detach().clone()
        original_state = copy.deepcopy(optimizer.state_dict())
        with patch.object(adapter, "_project", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                adapter.step()
        torch.testing.assert_close(parameter, original)
        torch.testing.assert_close(bias, original_bias)
        self.assertEqual(adapter.state_dict()["step"], 1)
        for identifier, state in original_state["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][identifier][key], value)

    def test_rectangular_matrix_projects_on_both_sides(self):
        """Reduced SVD handles unequal output and input dimensions."""
        import torch

        parameter = torch.nn.Parameter(
            torch.tensor([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        )
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        adapter = create_sst_adapter(
            optimizer,
            {"weight": parameter},
            reference_id="rectangular-fixture",
            config=SSTConfig(enabled=True, rank_fraction=0.5),
        )
        parameter.grad = torch.ones_like(parameter)
        adapter.step()
        expected = torch.tensor([[3.0, 0.0], [0.0, 0.9], [0.0, -0.1]], dtype=torch.float64)
        torch.testing.assert_close(parameter, expected)

    def test_nonfinite_gradient_fails_before_optimizer_state_changes(self):
        """NaN inputs do not advance optimizer state or modify parameters."""
        import torch

        parameter, _control, optimizer, _ordinary, adapter = setup_pair()
        original = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaises(ValueError):
            adapter.step()
        torch.testing.assert_close(parameter, original)
        self.assertFalse(optimizer.state)

    def test_unselected_parameters_keep_native_updates_and_float32_reports_cast_leakage(self):
        """Explicit target selection does not freeze other optimizer parameters."""
        import torch

        weight = torch.nn.Parameter(torch.tensor([[3.0, 1.0], [1.0, 3.0]]))
        bias = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([weight, bias], lr=0.1)
        adapter = create_sst_adapter(
            optimizer,
            {"weight": weight},
            reference_id="fixture",
            config=SSTConfig(enabled=True, rank_fraction=0.5),
        )
        before = weight.detach().clone().double()
        weight.grad = torch.tensor([[1.0, 4.0], [2.0, -3.0]])
        bias.grad = torch.ones_like(bias)
        report = adapter.step()
        torch.testing.assert_close(bias, torch.full((2,), 0.9))
        self.assertEqual(report["unselected_parameter_count"], 1)
        actual = weight.detach().double() - before
        protected = torch.tensor([[1.0], [1.0]], dtype=torch.float64) / math.sqrt(2)
        metrics = report["targets"]["weight"]
        self.assertAlmostEqual(
            metrics["left_leakage"],
            float(torch.linalg.vector_norm(protected.T @ actual)),
            places=12,
        )
        self.assertAlmostEqual(
            metrics["right_leakage"], float(torch.linalg.vector_norm(actual @ protected)), places=12
        )
