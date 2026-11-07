"""Opt-in random-model software experiment with real substitution/KL controls.

python -m docs.experiments.phase2_pr5_simulation_probe --enabled > simulation.json
No trained model, model download, training update or alignment improvement is asserted.
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
from gepa_mindfulness.interpret.activation_simulation import (
    prepare_simulation_controls,
    simulate_activations,
)
from gepa_mindfulness.interpret.explanation_tests import (
    ExampleRef,
    SplitManifest,
    freeze_explanation,
    generate_context_hypotheses,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel, _state_digest
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput

SEED = 53
TOKENS = {
    "generation": ((1, 2, 3, 4), (2, 3, 4, 5), (5, 4, 3, 2), (6, 7, 8, 9)),
    "development": ((3, 4, 5, 6),),
    "locked": ((3, 7, 1, 8), (2, 9, 3, 1), (21, 13, 19, 27)),
}


class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args(argv)
    if not args.enabled:
        parser.error("simulation software probe requires --enabled")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    inputs = {
        split: tuple(
            NativeInput(f"{split}-{i}", tokens, (1,) * len(tokens))
            for i, tokens in enumerate(values)
        )
        for split, values in TOKENS.items()
    }
    refs = {
        split: tuple(
            ExampleRef.from_input(
                context, distribution=("ood" if split == "locked" and i == 2 else "id")
            )
            for i, context in enumerate(values)
        )
        for split, values in inputs.items()
    }
    manifest = SplitManifest(refs["generation"], refs["development"], refs["locked"])
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
            source_digest = _state_digest(source)
            hook = native.hook(layer=0, kind=kind)
            neurons = random.Random(SEED).sample(range(32), 3)
            batches, hypotheses = [], []
            for neuron in neurons:
                batch = native.capture(
                    hook,
                    torch.tensor(TOKENS["generation"]),
                    component_indices=(neuron,),
                    example_ids=tuple(i.example_id for i in inputs["generation"]),
                    token_positions=(0, 1, 2, 3),
                    enabled=True,
                )
                batches.append(batch)
                hypotheses.append(generate_context_hypotheses(batch, manifest, enabled=True))
            for size in (1, 3):
                for candidate in (
                    0,
                    3,
                ):  # Predeclared first lexical candidate and constant baseline.
                    commits = tuple(
                        freeze_explanation(
                            hypotheses[i][candidate],
                            manifest,
                            batches[i],
                            clock=Clock(),
                            max_mae=0.01,
                            activation_threshold=0.01,
                            enabled=True,
                        )
                        for i in range(size)
                    )
                    controls = prepare_simulation_controls(
                        commits,
                        tuple(batches[:size]),
                        seed=SEED,
                        logit_positions=(0, 1, 2),
                        enabled=True,
                    )
                    rng = torch.get_rng_state().clone()
                    report = simulate_activations(
                        native,
                        hook,
                        commits,
                        inputs["locked"],
                        controls=controls,
                        logit_positions=(0, 1, 2),
                        enabled=True,
                    )
                    assert torch.equal(rng, torch.get_rng_state())
                    assert _state_digest(source) == source_digest
                    results.append(
                        {
                            "family": family,
                            "group_size": size,
                            "candidate_index": candidate,
                            "configuration": json.loads(
                                source.config.to_json_string(use_diff=False)
                            ),
                            "source_unchanged": True,
                            "rng_unchanged": True,
                            "report": report,
                        }
                    )
    root = Path(__file__).resolve().parents[2]
    sources = [
        "gepa_mindfulness/interpret/activation_simulation.py",
        "gepa_mindfulness/interpret/activation_simulation_records.py",
        "gepa_mindfulness/interpret/native_neurons.py",
        "gepa_mindfulness/interpret/native_records.py",
        "gepa_mindfulness/interpret/explanation_tests.py",
        "docs/experiments/phase2_pr5_simulation_probe.py",
    ]
    print(
        json.dumps(
            {
                "schema_version": "simulation-software-probe-v1",
                "maturity": "experimental",
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
                    "Random tiny models and artificial tokens; "
                    "no trained semantics or alignment effects.",
                    "All predeclared candidates and controls retained; numerical effects only.",
                    "No causal mechanism admission or independent temporal-custody authentication.",
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
