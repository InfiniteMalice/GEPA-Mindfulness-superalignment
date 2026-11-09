"""Reproduce diagnostic native prediction on random tiny models; not alignment research.

Run from the source checkout: python -m docs.experiments.phase2_pr4_native_probe --enabled
The script creates no trained model, downloads nothing and changes no parameters.
"""

# Standard library
from __future__ import annotations

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Local
from gepa_mindfulness.interpret.explanation_tests import (
    ExampleRef,
    SplitManifest,
    evaluate_explanation,
    freeze_explanation,
    generate_context_hypotheses,
    generate_near_misses,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel, sample_signed_regimes
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput, SamplingThresholds

SEED = 17
INPUTS = {
    "generation": ((0, 2, 3, 4), (1, 3, 4, 5), (2, 4, 3, 2), (3, 7, 8, 9)),
    "development": ((4, 4, 5, 6),),
    "locked": ((5, 1, 3, 5), (6, 2, 4, 4), (7, 17, 29, 23)),
}


class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args(argv)
    if not args.enabled:
        parser.error("random-model software probe requires --enabled")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    contexts = {
        split: tuple(
            NativeInput(f"{split}-{i}", tokens, (1,) * len(tokens)) for i, tokens in enumerate(rows)
        )
        for split, rows in INPUTS.items()
    }
    split_refs = {
        split: tuple(
            ExampleRef.from_input(
                context,
                distribution=(
                    ("id", "token-perturbation", "ood")[i] if split == "locked" else "id"
                ),
            )
            for i, context in enumerate(values)
        )
        for split, values in contexts.items()
    }
    splits = SplitManifest(
        split_refs["generation"], split_refs["development"], split_refs["locked"]
    )
    results = []
    with torch.random.fork_rng(devices=[]):
        for family in ("gpt2", "llama"):
            torch.manual_seed(SEED)
            config: GPT2Config | LlamaConfig
            source: GPT2LMHeadModel | LlamaForCausalLM
            if family == "gpt2":
                config = GPT2Config(
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
                config._attn_implementation = "eager"
                source = GPT2LMHeadModel(config)
                kind = HookKind.MLP_POST
            else:
                config = LlamaConfig(
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
                config._attn_implementation = "eager"
                source = LlamaForCausalLM(config)
                kind = HookKind.MLP_PRODUCT
            native = FrozenNativeModel(
                source,
                model_id=f"random-initialized-{family}",
                revision=f"seed-{SEED}",
                enabled=True,
            )
            hook = native.hook(layer=0, kind=kind)
            for neuron in random.Random(SEED).sample(range(32), 3):
                generation = native.capture(
                    hook,
                    torch.tensor(INPUTS["generation"]),
                    component_indices=(neuron,),
                    example_ids=tuple(i.example_id for i in contexts["generation"]),
                    token_positions=(0, 1, 2, 3),
                    enabled=True,
                )
                scale = max(abs(m.activation[0]) for m in generation.measurements) or 1.0
                sampling = sample_signed_regimes(
                    generation,
                    discovery_ids=tuple(i.example_id for i in generation.inputs),
                    seed=SEED,
                    per_regime=2,
                    thresholds=SamplingThresholds(scale * 0.01, scale * 0.25, scale * 0.75),
                    enabled=True,
                )
                hypotheses = generate_context_hypotheses(generation, splits, enabled=True)
                commits = tuple(
                    freeze_explanation(
                        h,
                        splits,
                        generation,
                        clock=Clock(),
                        max_mae=scale * 0.25,
                        activation_threshold=scale * 0.5,
                        enabled=True,
                    )
                    for h in hypotheses
                )
                # Locked activation is first captured after every candidate has been frozen.
                locked = native.capture(
                    hook,
                    torch.tensor(INPUTS["locked"]),
                    component_indices=(neuron,),
                    example_ids=tuple(i.example_id for i in contexts["locked"]),
                    token_positions=(0, 1, 2, 3),
                    enabled=True,
                )
                reports = [evaluate_explanation(c, locked, enabled=True) for c in commits]
                results.append(
                    {
                        "family": family,
                        "model": native.identity.to_dict(),
                        "configuration": json.loads(source.config.to_json_string(use_diff=False)),
                        "attention_implementation": source.config._attn_implementation,
                        "neuron": neuron,
                        "sampling": sampling,
                        "generation": generation.to_dict(),
                        "near_miss_proposals": generate_near_misses(
                            hypotheses[0],
                            generation,
                            splits,
                            replacement_tokens=(1, 2, 3),
                            enabled=True,
                        ),
                        "hypotheses_frozen_before_locked_capture": True,
                        "reports": reports,
                    }
                )
    root = Path(__file__).resolve().parents[2]
    sources = [
        "gepa_mindfulness/interpret/native_records.py",
        "gepa_mindfulness/interpret/native_neurons.py",
        "gepa_mindfulness/interpret/explanation_tests.py",
        "docs/experiments/phase2_pr4_native_probe.py",
    ]
    print(
        json.dumps(
            {
                "schema_version": "native-software-probe-v1",
                "maturity": "experimental",
                "dataset_revision": "causal-prefix-disjoint-v2",
                "diagnostic_status": "diagnostic",
                "training_eligibility": "DEVELOPMENT",
                "optimizer_input": False,
                "authority_granted": False,
                "seed": SEED,
                "python_version": platform.python_version(),
                "results": results,
                "source_sha256": {
                    name: sha256((root / name).read_bytes()).hexdigest() for name in sources
                },
                "limitations": [
                    "Random initialized models and artificial token contexts; "
                    "no trained semantic validity.",
                    "One seed, three neurons per family; "
                    "no statistical superiority or alignment claim.",
                    "All candidate and null results retained; "
                    "no selection based on locked outcomes.",
                    "No independent custody authentication, downstream KL "
                    "or causal interventions run.",
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
