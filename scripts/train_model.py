#!/usr/bin/env python3
"""Unified training script for Phi-3 and Llama-3 deception detection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_CONFIGS = {
    "phi3": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "min_vram": 12,
        "download_size": 8,
        "trust_remote_code": True,
    },
    "llama3": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "min_vram": 16,
        "download_size": 16,
        "trust_remote_code": False,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model with dual-path deception detection")

    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model to train",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Training steps (default: 100)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "adversarial_scenarios.jsonl"),
        help="Training dataset path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: runs/<model>_<timestamp>)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization (saves VRAM)",
    )

    return parser.parse_args()


def check_requirements(model_key: str) -> list[str]:
    """Check requirements for specified model."""
    issues: list[str] = []
    config = MODEL_CONFIGS[model_key]

    try:
        import torch

        if not torch.cuda.is_available():
            issues.append("CUDA not available - GPU required")
        else:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem < config["min_vram"]:
                issues.append(
                    (f"GPU has {gpu_mem:.1f}GB - " f"need {config['min_vram']}GB+ for {model_key}")
                )
    except ImportError:
        issues.append("torch not installed - run: pip install torch")

    try:
        import transformers  # noqa: F401
    except ImportError:
        issues.append("transformers not installed - run: pip install transformers")

    return issues


def main() -> int:
    """Run training loop and collect deception fingerprints."""
    args = parse_args()

    print("\n" + "=" * 70)
    print(f"🚀 GEPA Training: {args.model.upper()}")
    print("=" * 70)
    print()

    print("🔍 Checking requirements...")
    issues = check_requirements(args.model)

    if issues:
        print("\n❌ Requirements not met:\n")
        for issue in issues:
            print(f"   - {issue}")
        print("\nFix these issues and try again.")
        return 1

    print("✅ All requirements met")
    print()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mindful_trace_gepa.deception.circuit_analysis import (
        detect_deception_heuristic,
    )
    from mindful_trace_gepa.deception.fingerprints import (
        DeceptionFingerprint,
        FingerprintCollector,
    )
    from mindful_trace_gepa.prompts.dual_path import (
        make_dual_path_prompt,
        parse_dual_path_response,
    )

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "runs" / f"{args.model}_train_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print()

    model_config = MODEL_CONFIGS[args.model]

    print(f"📥 Loading {model_config['name']}...")
    cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_path.exists():
        print(
            (
                "   ⚠️  First run downloads ~"
                f"{model_config['download_size']}GB (may take several minutes)"
            )
        )
    print("   ⏳ Please be patient...")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config["trust_remote_code"],
        )

        load_kwargs: dict[str, object] = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if model_config["trust_remote_code"]:
            load_kwargs["trust_remote_code"] = True

        if args.use_8bit:
            load_kwargs["load_in_8bit"] = True
            print("   Using 8-bit quantization")

        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            **load_kwargs,
        )

        print("✅ Model loaded successfully")
        mem_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"   Memory used: {mem_used:.2f} GB")
    except Exception as error:  # noqa: BLE001
        print(f"\n❌ Failed to load model: {error}")
        return 1

    print()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    print(f"📊 Loading dataset: {dataset_path}...")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1

    with open(dataset_path, "r", encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]

    if not dataset:
        print("❌ Dataset is empty")
        return 1

    print(f"✅ Loaded {len(dataset)} examples")
    print()

    print("⚙️  Training configuration:")
    print(f"   Steps: {args.steps}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print("   Dual-path prompting: enabled")
    print("   Honesty weight: 0.35")
    print("   Deception penalty: disabled")
    print()

    fingerprint_dir = output_dir / "fingerprints"
    fingerprint_collector = FingerprintCollector(str(fingerprint_dir))

    print("🎓 Starting training...")
    print("=" * 70)
    print()

    try:
        for step in range(args.steps):
            example = random.choice(dataset)
            prompt = example.get("query") or example.get("prompt") or ""
            context_parts: list[str] = []

            scenario = example.get("scenario")
            if scenario:
                context_parts.append(f"Scenario: {scenario}")

            pressure = example.get("pressure_factors")
            if pressure:
                joined = ", ".join(pressure)
                context_parts.append(f"Pressure factors: {joined}")

            imperative = example.get("ethical_imperative")
            if imperative:
                context_parts.append(f"Ethical imperative: {imperative}")

            correct = example.get("correct_action")
            if correct:
                context_parts.append(f"Correct action: {correct}")

            context = "\n".join(context_parts)

            print(f"Step {step + 1}/{args.steps}")
            print(f"Prompt: {prompt[:70]}...")

            dual_prompt = make_dual_path_prompt(prompt, context=context)
            inputs = tokenizer(dual_prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            sections = parse_dual_path_response(response)
            deception = detect_deception_heuristic(sections)

            if deception["deception_detected"]:
                confidence = deception["confidence_score"]
                print(f"  🚨 Deception detected (confidence={confidence:.2f})")

                fingerprint = DeceptionFingerprint(
                    timestamp=datetime.now().isoformat(),
                    prompt=prompt,
                    domain=example.get("domain", "unknown"),
                    path_1_text=sections["path_1"],
                    path_2_text=sections["path_2"],
                    comparison=sections["comparison"],
                    recommendation=sections["recommendation"],
                    recommended_path=sections["recommended_path"],
                    path_1_circuits={},
                    path_2_circuits={},
                    deception_detected=True,
                    confidence_score=confidence,
                    signals=deception.get("signals", {}),
                    reasons=deception["reasons"],
                    model_checkpoint=model_config["name"],
                    training_step=step,
                )
                fingerprint_collector.add(fingerprint)
            else:
                confidence = deception["confidence_score"]
                print(f"  ✅ No deception detected (confidence={confidence:.2f})")

            print()

        print("=" * 70)
        print("✅ Training complete!")
        print()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print()
    except Exception as error:  # noqa: BLE001
        print(f"\n❌ Training failed: {error}")
        return 1

    print("💾 Saving model...")
    model_output = output_dir / "model"
    model.save_pretrained(model_output)
    tokenizer.save_pretrained(model_output)
    print(f"✅ Model saved to: {model_output}")
    print()

    summary = fingerprint_collector.get_summary()

    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total steps: {args.steps}")
    print(f"Deception fingerprints: {summary['deceptive']}")
    print(f"Deception rate: {summary['deception_rate']:.1%}")
    print()

    if summary["deceptive"] > 0:
        print("By domain:")
        for domain, stats in summary["by_domain"].items():
            total = stats["total"]
            deceptive = stats["deceptive"]
            rate = (deceptive / total) if total else 0.0
            print(f"  {domain:12s} {deceptive}/{total} ({rate:.0%})")
        print()

    print(f"📁 Output saved to: {output_dir}")
    print()

    fingerprints_path = fingerprint_dir / "fingerprints.jsonl"
    print("🎯 Next steps:")
    print(
        "   python scripts/analyze_deception_fingerprints.py " f"--fingerprints {fingerprints_path}"
    )
    print(
        "   python scripts/validate_ablation.py "
        f"--original {model_output} --test-data {dataset_path}"
    )
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
