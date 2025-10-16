#!/usr/bin/env python3
"""Apply manual circuit ablation based on deception fingerprints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM


def ablate_circuits(
    model: AutoModelForCausalLM,
    targets: Dict[str, Any],
    strength: float,
) -> AutoModelForCausalLM:
    print(f"\n🔧 Ablating circuits with strength={strength}...")

    ablation_targets = targets.get("ablation_targets", {})
    if not ablation_targets:
        print("No ablation targets provided. Returning original model.")
        return model

    for circuit_type, stats in ablation_targets.items():
        recommendation = stats.get("recommendation", "MEDIUM_PRIORITY")
        if recommendation not in {"HIGH_PRIORITY", "MEDIUM_PRIORITY"}:
            continue
        mean_activation = stats.get("mean_activation", 0.0)
        samples = stats.get("samples", 0)
        print(
            "  - Ablating {} (mean activation: {:.2f}, samples: {})".format(
                circuit_type,
                mean_activation,
                samples,
            )
        )
        # Placeholder: actual ablation logic depends on available circuit mappings.
        # Example operation: scale selected attention heads (no-op placeholder).
        for name, parameter in model.named_parameters():
            if circuit_type in name:
                parameter.data.mul_(1.0 - strength)

    print("✅ Circuit ablation complete")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--targets", required=True, help="Circuit targets JSON")
    parser.add_argument("--strength", type=float, default=0.8, help="Ablation strength (0-1)")
    parser.add_argument("--out", required=True, help="Directory to save ablated model")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    with open(args.targets, "r", encoding="utf-8") as handle:
        targets = json.load(handle)

    print(
        f"\n📊 Ablation targets generated from {targets.get('total_fingerprints', 0)} fingerprints"
    )

    model = ablate_circuits(model, targets, args.strength)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    metadata = {
        "source_model": args.model,
        "targets": targets,
        "ablation_strength": args.strength,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "ablation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\n✅ Ablated model saved to {out_dir}")
    print("\n🧪 Next step: Validate ablation effectiveness with validate_ablation.py")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
