# Core GEPA Logic

This package implements the GEPA contemplative principles, paraconsistent
imperatives, Circuit Tracer integration, adversarial challenge utilities, and
confidence-aware abstention logic used throughout the training pipeline.

Key components:

- `contemplative_principles.py` implements the four contemplative axes.
- `imperatives.py` models the three alignment imperatives using paraconsistent
  aggregation.
- `tracing.py` integrates the optional Circuit Tracer thought logging system.
- `abstention.py` enforces the confidence-based abstention rule and honesty
  reward computation.
- `rewards.py` shapes the PPO signal from task, GEPA, honesty, and hallucination
  measurements.
- `adversarial.py` exposes adversarial / OOD probes inspired by scheming
  detection research.

These modules are imported by the higher-level training orchestration code and
can also be reused independently for evaluation or analysis tools.
