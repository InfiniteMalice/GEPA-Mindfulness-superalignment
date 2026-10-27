"""Bounds and real-optimizer checks for the authored SST matrix diagnostic."""

# Standard library
import importlib.util
import json
import unittest

# Local
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.sst_probe import run_sst_probe


class ProbeBoundaryTests(unittest.TestCase):
    """The diagnostic remains opt-in and bounded even without a tensor runtime."""

    def test_disabled_probe_does_not_load_a_runtime(self):
        """No training or runtime inspection occurs by default."""
        self.assertIsNone(run_sst_probe(seeds=object(), steps=object()))

    def test_invalid_compute_requests_fail_before_runtime_import(self):
        """The authored toy cannot become an unbounded training launcher."""
        for kwargs in (
            {"steps": True},
            {"steps": 0},
            {"steps": 1001},
            {"seeds": ()},
            {"seeds": (1, 1)},
            {"seeds": (True,)},
            {"optimizer_name": "custom"},
            {"enabled": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    run_sst_probe(**({"enabled": True} | kwargs))


@unittest.skipUnless(importlib.util.find_spec("torch"), "real PyTorch unavailable")
class ProbeTensorTests(unittest.TestCase):
    """Paired fixed seeds expose retention and blocked-correction tradeoffs."""

    def test_probe_is_deterministic_and_preserves_component_metrics(self):
        """A fixed seed reproduces diagnostic measurements without promotion claims."""
        first = run_sst_probe(seeds=(0, 1), steps=3, enabled=True)
        self.assertEqual(first, run_sst_probe(seeds=(0, 1), steps=3, enabled=True))
        self.assertEqual(len(first["pairs"]), 2)
        for pair in first["pairs"]:
            self.assertEqual(set(pair["conditions"]), {"baseline", "sst", "full_rank"})
            self.assertEqual(pair["conditions"]["full_rank"]["protected_input_drift_mse"], 0)
            self.assertLess(pair["conditions"]["sst"]["protected_input_drift_mse"], 1e-20)
            self.assertGreater(pair["conditions"]["baseline"]["protected_input_drift_mse"], 0)
            self.assertGreater(
                pair["conditions"]["full_rank"]["target_fit_mse"],
                pair["conditions"]["baseline"]["target_fit_mse"],
            )
        self.assertFalse(first["authority_granted"])
        with self.assertRaises(ValueError):
            require_training_eligible(first)
        json.dumps(first, allow_nan=False)
