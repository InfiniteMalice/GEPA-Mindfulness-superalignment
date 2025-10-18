#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/gepa_mindfulness/configs"
CPU_EXAMPLE_DIR="$PROJECT_ROOT/gepa_mindfulness/examples/cpu_demo"

python -c "from gepa_mindfulness.training.config import load_trainer_config; load_trainer_config('${CONFIG_DIR}/grpo/grpo_default.yaml')" \
  && echo "✅ Config validation passed"

pushd "$CPU_EXAMPLE_DIR" >/dev/null
python run_cpu_demo.py --trainer grpo
popd >/dev/null

python -m gepa_mindfulness.training.train \
  --mode grpo \
  --config "$CONFIG_DIR/default.yaml" \
  --dataset "$CPU_EXAMPLE_DIR/prompts.txt" \
  --output "$PROJECT_ROOT/runs/grpo_demo"
