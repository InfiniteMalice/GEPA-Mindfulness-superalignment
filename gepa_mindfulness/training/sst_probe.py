"""Bounded authored matrix probe; no model benchmark or constitutional certificate."""

# Standard library
from __future__ import annotations

import json

# Local
from .subspace_tuning import SSTConfig, create_sst_adapter


def run_sst_probe(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 20,
    optimizer_name: str = "AdamW",
    enabled: bool = False,
) -> dict[str, object] | None:
    """Compare ordinary, half-rank and full-rank applied updates on identical 2x2 targets.

    The fit target intentionally changes protected as well as free directions. Full
    protection blocks correction; half-rank SST can retain the protected-input map
    while learning only eligible components. These are algebraic diagnostics using
    authored fixtures, not independently held-out behavioral experiments.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("steps must be an integer in [1, 1000]")
    if (
        type(seeds) is not tuple
        or not 1 <= len(seeds) <= 16
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("bounded unique unsigned seeds required")
    if type(optimizer_name) is not str or optimizer_name not in ("SGD", "Adam", "AdamW"):
        raise ValueError("unsupported toy optimizer")

    import torch

    pairs = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        target = initial + torch.randn((2, 2), generator=generator, dtype=torch.float64)
        _left, _singular, right = torch.linalg.svd(initial, full_matrices=False)
        protected_input = right[0]
        conditions = {}
        for condition, rank in (("baseline", None), ("sst", 0.5), ("full_rank", 1.0)):
            parameter = torch.nn.Parameter(initial.clone())
            optimizer = getattr(torch.optim, optimizer_name)([parameter], lr=0.03, weight_decay=0)
            adapter = create_sst_adapter(
                optimizer,
                {"weight": parameter},
                reference_id=f"authored-toy-v1-seed-{seed}",
                config=SSTConfig(enabled=rank is not None, rank_fraction=rank or 0.0),
            )
            last_step = None
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                (parameter - target).square().mean().backward()
                if adapter is None:
                    optimizer.step()
                else:
                    last_step = adapter.step()
            with torch.no_grad():
                drift = (parameter - initial) @ protected_input
                log_prob = parameter.log_softmax(-1)
                base_log_prob = initial.log_softmax(-1)
                kl = (base_log_prob.exp() * (base_log_prob - log_prob)).sum(-1).mean()
                conditions[condition] = {
                    "protected_input_drift_mse": float(drift.square().mean()),
                    "target_fit_mse": float((parameter - target).square().mean()),
                    "row_distribution_kl": float(kl),
                    "last_step": last_step,
                }
        pairs.append({"seed": seed, "conditions": conditions})
    report = {
        "schema_version": "sst-matrix-probe-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "source_reference": "REF-SST",
        "fixture_version": "authored-matrix-v1",
        "evaluator_version": "analytic-matrix-metrics-v1",
        "runtime_version": str(torch.__version__),
        "optimizer": optimizer_name,
        "steps_per_condition": steps,
        "pairs": pairs,
    }
    json.dumps(report, allow_nan=False)
    return report
