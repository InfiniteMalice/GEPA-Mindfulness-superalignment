#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/gepa_mindfulness/configs"
CPU_EXAMPLE_DIR="$PROJECT_ROOT/gepa_mindfulness/examples/cpu_demo"

python -c "from gepa_mindfulness.training.configs import load_training_config; load_training_config('$CONFIG_DIR/default.yaml')" \
  && echo "✅ Config validation passed"

pushd "$CPU_EXAMPLE_DIR" >/dev/null
python run_cpu_demo.py
popd >/dev/null

python -m gepa_mindfulness.training.cli \
  --config "$CONFIG_DIR/default.yaml" \
  --dataset "$CPU_EXAMPLE_DIR/prompts.txt" \
  --adversarial-only
