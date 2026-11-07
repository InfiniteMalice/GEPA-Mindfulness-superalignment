"""Opt-in actual random-model method comparisons; no learned-SAE or alignment claim."""

# Standard library
from __future__ import annotations

import argparse
import json
import platform
import random
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

# Local
from docs.experiments.phase2_pr6_causal_probe import build
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.interpret.comparative import (
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
from gepa_mindfulness.interpret.native_neurons import _state_digest
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput

SEED = 79
TOKENS = {
    "generation": ((1, 2, 3), (4, 1, 2), (3, 4, 1)),
    "development": ((2, 3, 4),),
    "locked": ((1, 3, 4), (4, 2, 5), (11, 12, 13)),
}


class Clock:
    def now(self) -> datetime:
        return datetime(2026, 11, 6, tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    if not parser.parse_args(argv).enabled:
        parser.error("comparison software probe requires --enabled")
    import torch

    groups = {
        k: tuple(NativeInput(f"{k}-{i}", v, (1,) * len(v)) for i, v in enumerate(values))
        for k, values in TOKENS.items()
    }
    refs = {
        k: tuple(
            ExampleRef.from_input(v, distribution="ood" if v.example_id == "locked-2" else "id")
            for v in values
        )
        for k, values in groups.items()
    }
    manifest = SplitManifest(refs["generation"], refs["development"], refs["locked"])
    features = (TokenFeature("one", "token_is", 0, 1), TokenFeature("four", "token_is", 0, 4))
    context_features = (
        TokenFeature("previous-one", "token_is", -1, 1),
        TokenFeature("same-token", "tokens_equal", -1, other_offset=0),
    )
    results = []
    with torch.random.fork_rng(devices=[]):
        for family in ("gpt2", "llama"):
            source, native, kind = build(family)
            original = _state_digest(source)
            rng = torch.get_rng_state().clone()
            hook = native.hook(layer=0, kind=kind)
            selected = tuple(random.Random(SEED).sample(range(hook.width), 2))
            dictionary = native_dictionary(native, hook, enabled=True)
            relu = replace(
                dictionary,
                dictionary_id="constructed-relu-control",
                origin="constructed",
                nonlinearity="relu",
            )
            encoder = tuple(
                tuple(float(j == hook.width - 1 - i) for j in range(hook.width))
                for i in range(hook.width)
            )
            permuted = replace(
                dictionary,
                dictionary_id="constructed-reversed-basis",
                origin="constructed",
                encoder=encoder,
                decoder=tuple(tuple(r) for r in zip(*encoder)),
            )
            attention_hook = native.hook(layer=0, kind=HookKind.ATTENTION_RESIDUAL)
            context = native.capture(
                attention_hook,
                torch.tensor(TOKENS["generation"]),
                component_indices=tuple(range(attention_hook.width)),
                example_ids=tuple(v.example_id for v in groups["generation"]),
                token_positions=(0, 1, 2),
                enabled=True,
            )
            context_source = {
                "attention_generation": context.to_dict(),
                "local_writers_readers": [
                    native.component_weights(hook, component_index=k, enabled=True)
                    for k in selected
                ],
            }
            context_ref = EvidenceReference(
                "generation-context:" + content_digest(context_source),
                EvidenceSourceKind.EXTERNAL_RECORD,
            )
            for size, other_available in ((1, True), (2, True), (1, False)):
                prediction = EffectPrediction(-1, 1e-10, 20.0, 1e-10)
                arms = (
                    ComparisonArm(
                        "native",
                        "native",
                        dictionary,
                        selected[:size],
                        features,
                        prediction,
                        "Predeclared current-token hypothesis.",
                        0.01,
                    ),
                    ComparisonArm(
                        "context",
                        "native_context",
                        dictionary,
                        selected[:size],
                        context_features,
                        prediction,
                        "Predeclared contextual competitor.",
                        0.01,
                        context_refs=(context_ref,),
                    ),
                    ComparisonArm(
                        "sae",
                        "sparse_autoencoder",
                        relu,
                        selected[:size],
                        features,
                        prediction,
                        "Constructed ReLU control; no learned SAE supplied.",
                        0.01,
                    ),
                    ComparisonArm(
                        "other",
                        "other_decomposition",
                        permuted if other_available else None,
                        selected[:size],
                        features,
                        prediction,
                        "Constructed alternate basis control.",
                        0.01,
                        unavailable_reason=(
                            None if other_available else "No external other dictionary supplied."
                        ),
                    ),
                )
                protocol = ComparisonProtocol(
                    f"{family}-k{size}-other{other_available}",
                    native.identity.digest,
                    hook,
                    manifest,
                    arms,
                    ComparisonBudget(size, 2, 9, 1000000),
                    (0, 1, 2),
                    (0, 1),
                    LogitContrast(0, 3, 4),
                    SEED,
                    1e-4,
                )
                start = perf_counter_ns()
                commit = prepare_comparison(
                    native, protocol, groups["generation"], clock=Clock(), enabled=True
                )
                prepare_ns = perf_counter_ns() - start
                report = evaluate_comparison(native, commit, groups["locked"], enabled=True)
                results.append(
                    {
                        "family": family,
                        "selected_count": size,
                        "other_available": other_available,
                        "generation_context": context_source,
                        "context_generation_forward_calls": 1,
                        "context_cost_shared_across_family_conditions": True,
                        "preparation_forward_calls": 1,
                        "preparation_wall_time_ns": prepare_ns,
                        "report": report,
                    }
                )
            assert original == _state_digest(source) and torch.equal(rng, torch.get_rng_state())
    root = Path(__file__).resolve().parents[2]
    paths = (
        "gepa_mindfulness/interpret/comparative.py",
        "gepa_mindfulness/interpret/comparative_records.py",
        "gepa_mindfulness/interpret/activation_simulation.py",
        "gepa_mindfulness/interpret/explanation_tests.py",
        "gepa_mindfulness/interpret/native_neurons.py",
        "gepa_mindfulness/interpret/native_records.py",
        "docs/experiments/phase2_pr6_causal_probe.py",
        "docs/experiments/phase2_pr8_comparison_probe.py",
    )
    print(
        json.dumps(
            {
                "schema_version": "interpretation-comparison-software-probe-v1",
                "maturity": "experimental",
                "diagnostic_status": "diagnostic",
                "training_eligibility": "DEVELOPMENT",
                "optimizer_input": False,
                "authority_granted": False,
                "model_seed": 71,
                "control_seed": SEED,
                "python_version": platform.python_version(),
                "results": results,
                "source_models_unchanged": True,
                "torch_rng_unchanged": True,
                "source_sha256": {p: sha256((root / p).read_bytes()).hexdigest() for p in paths},
                "limitations": [
                    "Random tiny networks, artificial tokens and constructed dictionaries.",
                    "No learned SAE training or trained-model semantic claim.",
                    "Context observations are supplied; explanatory use is not established.",
                    "Conditions reuse inputs/models and are dependent observations.",
                    "No causal admission, superiority, parameter update or alignment improvement.",
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
