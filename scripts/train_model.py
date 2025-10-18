#!/usr/bin/env python3
"""Supervised fine-tuning for Phi-3 and Llama-3 dual-path responses."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
        default="datasets/dual_path/data.jsonl",
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
        help="Use 8-bit quantization (not supported for full fine-tuning)",
    )
    parser.add_argument(
        "--ppo-rollouts",
        type=int,
        default=0,
        help="Number of PPO rollouts to perform after SFT (default: 0)",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=2,
        help="Queries per PPO rollout step (default: 2)",
    )
    parser.add_argument(
        "--ppo-lr",
        type=float,
        default=1e-6,
        help="Learning rate for PPO policy updates",
    )
    parser.add_argument(
        "--ppo-target-kl",
        type=float,
        default=0.1,
        help="Target KL divergence for PPO adaptive control",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to log fingerprints and metrics",
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


def run_ppo_alignment(
    *,
    base_model,
    tokenizer,
    dataset: List[dict],
    device,
    args: argparse.Namespace,
    model_config: Dict[str, object],
    output_dir: Path,
) -> tuple[bool, Dict[str, float]]:
    """Run PPO rollouts that optimise GEPA-aligned rewards."""

    if args.ppo_rollouts <= 0:
        return False, {}

    try:
        from trl import (
            AutoModelForCausalLMWithValueHead,
            PPOConfig,
            PPOTrainer,
        )
    except ImportError:
        print("❌ PPO requested but unavailable. Install TRL: pip install trl transformers")
        return False, {}

    from gepa_mindfulness.core.paraconsistent import ParaconsistentTruthValue
    from gepa_mindfulness.core.rewards import RewardSignal, RewardWeights
    from mindful_trace_gepa.deception.circuit_analysis import (
        detect_deception_heuristic,
    )
    from mindful_trace_gepa.prompts.dual_path import (
        make_dual_path_prompt,
        parse_dual_path_response,
    )
    from mindful_trace_gepa.scoring import run_heuristics

    dtype = next(base_model.parameters()).dtype

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config["name"],
        torch_dtype=dtype,
    )
    policy_model.pretrained_model.load_state_dict(base_model.state_dict())
    policy_model.to(device)
    policy_model.train()

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config["name"],
        torch_dtype=dtype,
    ).to(device)
    ref_model.eval()

    ppo_kwargs: Dict[str, object] = {
        "model_name": model_config["name"],
        "batch_size": args.ppo_batch_size,
        "learning_rate": args.ppo_lr,
    }

    if args.ppo_batch_size > 1:
        ppo_kwargs["mini_batch_size"] = max(1, args.ppo_batch_size // 2)

    try:
        ppo_kwargs["target_kl"] = args.ppo_target_kl
        ppo_config = PPOConfig(**ppo_kwargs)
    except TypeError:
        ppo_kwargs.pop("target_kl", None)
        ppo_config = PPOConfig(**ppo_kwargs)

    if not hasattr(ppo_config, "adap_kl_ctrl"):
        setattr(ppo_config, "adap_kl_ctrl", True)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    reward_weights = RewardWeights.from_mapping(
        {
            "task_success": 0.6,
            "gepa_alignment": 0.6,
            "honesty_trace": 0.4,
            "hallucination_penalty": 0.3,
        }
    )

    total_rewards: List[float] = []
    deception_hits = 0
    avg_gepa: List[float] = []

    generation_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for rollout in range(args.ppo_rollouts):
        batch = [random.choice(dataset) for _ in range(args.ppo_batch_size)]
        prompts = [
            make_dual_path_prompt(
                example.get("query", example.get("prompt", "")),
                example.get("context", ""),
            )
            for example in batch
        ]

        queries = tokenizer(prompts, return_tensors="pt", padding=True)
        query_tensors = queries["input_ids"].to(device)

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )

        responses = [tokenizer.decode(resp, skip_special_tokens=True) for resp in response_tensors]

        rewards: List[float] = []

        for response_text in responses:
            sections = parse_dual_path_response(response_text)
            deception = detect_deception_heuristic(sections)
            heuristics = run_heuristics(response_text)

            normalised = {key: value / 4.0 for key, value in heuristics.scores.items()}
            gepa_score = sum(normalised.values()) / len(normalised) if normalised else 0.0

            honesty_reward = max(
                0.0,
                1.0 - float(deception.get("confidence_score", 0.0)),
            )
            if deception.get("deception_detected"):
                honesty_reward *= 0.5

            hallucination_score = max(
                0.0,
                1.0 - normalised.get("integrity", 0.0),
            )
            opposition = (
                min(1.0, float(deception.get("confidence_score", 0.0)))
                if deception.get("deception_detected")
                else 0.0
            )
            imperatives = ParaconsistentTruthValue.from_support_opposition(
                support=gepa_score,
                opposition=opposition,
            )

            reward_signal = RewardSignal(
                task_success=gepa_score,
                gepa_score=gepa_score,
                honesty_reward=honesty_reward,
                hallucination_score=hallucination_score,
                imperatives_truth=imperatives,
            )

            reward_value = reward_signal.combined(reward_weights)
            rewards.append(reward_value)
            total_rewards.append(reward_value)
            avg_gepa.append(gepa_score)
            if deception.get("deception_detected"):
                deception_hits += 1

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        mean_reward = sum(rewards) / max(len(rewards), 1)
        kl_value = float(stats.get("objective/kl", 0.0))
        print(
            "🔁 PPO rollout %d/%d - mean_reward=%.4f - kl=%.4f"
            % (rollout + 1, args.ppo_rollouts, mean_reward, kl_value)
        )

    base_model.load_state_dict(policy_model.pretrained_model.state_dict())
    base_model.train()

    avg_reward = sum(total_rewards) / max(len(total_rewards), 1)
    avg_gepa_score = sum(avg_gepa) / max(len(avg_gepa), 1)

    summary = {
        "avg_reward": avg_reward,
        "avg_gepa": avg_gepa_score,
        "deception_events": float(deception_hits),
        "samples": float(len(total_rewards)),
    }

    ppo_dir = output_dir / "ppo_logs"
    ppo_dir.mkdir(parents=True, exist_ok=True)
    record_path = ppo_dir / "ppo_summary.json"
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"📈 PPO summary saved to: {record_path}")

    return True, summary


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
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

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
        if args.use_8bit:
            print(
                "❌ 8-bit loading is not supported for supervised fine-tuning. "
                "Use tools such as LoRA if low-VRAM training is required."
            )
            return 1

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config["trust_remote_code"],
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
        }

        if model_config["trust_remote_code"]:
            load_kwargs["trust_remote_code"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            **load_kwargs,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            print("✅ Model loaded successfully")
            print(f"   Memory used: {mem_used:.2f} GB")
        else:
            print("✅ Model loaded on CPU (training will be slow)")
    except Exception as error:  # noqa: BLE001
        print(f"\n❌ Failed to load model: {error}")
        return 1

    print()

    dataset_path = Path(args.dataset)
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

    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, args.steps // 20),
        num_training_steps=args.steps,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def format_path_guidance(lines: Iterable[str], target: str) -> str:
        cleaned: List[str] = []
        for line in lines:
            if target.lower() in line.lower():
                text = line.split(":", 1)[-1].strip()
                text = text.replace("Path 1", "").replace("Path 2", "").strip()
                if text:
                    cleaned.append(text)
        return " ".join(cleaned)

    def build_supervised_target(example: dict) -> str:
        validations = example.get("validation_criteria", [])
        harm = example.get("harms_if_wrong", "unknown risk")

        path1_focus = format_path_guidance(validations, "path 1")
        path2_focus = format_path_guidance(validations, "path 2")

        if not path1_focus:
            path1_focus = (
                "Highlight uncertainties, potential downsides, and when to seek professional help."
            )
        if not path2_focus:
            path2_focus = "Provide confident next steps while acknowledging potential trade-offs."

        comparison = (
            "Path 1 emphasises caution and risk awareness, while Path 2 focuses on "
            "immediate action and user reassurance."
        )

        correct_path = example.get("ground_truth_correct_path", "path_1")
        if correct_path == "path_2":
            recommendation = (
                "Path 2 is preferable for this scenario because it delivers decisive "
                f"guidance while managing the {harm} impact."
            )
        else:
            recommendation = (
                "Path 1 is safer here because it mitigates the potential "
                f"{harm} consequences and encourages expert review."
            )

        return (
            "[PATH 1 REASONING]\n"
            f"{path1_focus}\n\n"
            "[PATH 2 REASONING]\n"
            f"{path2_focus}\n\n"
            "[COMPARISON]\n"
            f"{comparison}\n\n"
            "[RECOMMENDATION]\n"
            f"{recommendation}"
        )

    def encode_example(prompt_text: str, target_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            target_text + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
        return input_ids, labels

    def collate_batch(batch_examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:
        input_tensors = [example[0] for example in batch_examples]
        label_tensors = [example[1] for example in batch_examples]

        padded_inputs = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=pad_token_id,
        )
        padded_labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = (padded_inputs != pad_token_id).long()

        return {
            "input_ids": padded_inputs.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": padded_labels.to(device),
        }

    print("🎓 Starting supervised fine-tuning...")
    print("=" * 70)
    print()

    running_loss = 0.0
    steps_completed = 0

    try:
        for step in range(args.steps):
            batch = [random.choice(dataset) for _ in range(args.batch_size)]

            encoded_batch = []
            prompts_used: List[str] = []
            for example in batch:
                prompt_text = make_dual_path_prompt(
                    example.get("query", example.get("prompt", "")),
                    example.get("context", ""),
                )
                target_text = build_supervised_target(example)
                encoded_batch.append(encode_example(prompt_text, target_text))
                prompts_used.append(prompt_text)

            batch_tensors = collate_batch(encoded_batch)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch_tensors)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            steps_completed += 1

            avg_loss = running_loss / steps_completed
            print(
                f"Step {step + 1}/{args.steps} - loss={loss.item():.4f} - avg_loss={avg_loss:.4f}"
            )

            if (step + 1) % args.log_interval == 0:
                model.eval()
                sample_prompt = prompts_used[0]
                inputs = tokenizer(sample_prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response_tokens = generated[0][inputs.input_ids.shape[1] :]
                response_text = tokenizer.decode(
                    response_tokens,
                    skip_special_tokens=True,
                )

                sections = parse_dual_path_response(response_text)
                deception = detect_deception_heuristic(sections)

                if deception["deception_detected"]:
                    fingerprint = DeceptionFingerprint(
                        timestamp=datetime.now().isoformat(),
                        prompt=batch[0].get("query", ""),
                        domain=batch[0].get("context", "unknown"),
                        path_1_text=sections["path_1"],
                        path_2_text=sections["path_2"],
                        comparison=sections["comparison"],
                        recommendation=sections["recommendation"],
                        recommended_path=sections["recommended_path"],
                        path_1_circuits={},
                        path_2_circuits={},
                        deception_detected=True,
                        confidence_score=deception["confidence_score"],
                        signals=deception.get("signals", {}),
                        reasons=deception["reasons"],
                        model_checkpoint=model_config["name"],
                        training_step=step,
                    )
                    fingerprint_collector.add(fingerprint)
                    print(
                        "  🚨 Deception detected during evaluation "
                        f"(confidence={deception['confidence_score']:.2f})"
                    )
                else:
                    print(
                        "  ✅ Evaluation sample shows no deception "
                        f"(confidence={deception['confidence_score']:.2f})"
                    )

                print()
                model.train()

        print("=" * 70)
        print("✅ Fine-tuning complete!")
        print()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print()
    except Exception as error:  # noqa: BLE001
        print(f"\n❌ Training failed: {error}")
        return 1

    ppo_executed, ppo_summary = run_ppo_alignment(
        base_model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        args=args,
        model_config=model_config,
        output_dir=output_dir,
    )

    if ppo_executed:
        print()
        print("✅ PPO alignment complete")
        print(
            "   Avg reward: %.4f | Avg GEPA: %.4f | Deception events: %d"
            % (
                ppo_summary.get("avg_reward", 0.0),
                ppo_summary.get("avg_gepa", 0.0),
                int(ppo_summary.get("deception_events", 0.0)),
            )
        )

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

    if ppo_executed:
        print(
            "PPO avg reward: %.4f | avg GEPA: %.4f"
            % (
                ppo_summary.get("avg_reward", 0.0),
                ppo_summary.get("avg_gepa", 0.0),
            )
        )
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
        f"--original {model_output} --test-data datasets/dual_path/test.jsonl"
    )
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
