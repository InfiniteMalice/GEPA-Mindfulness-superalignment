# Deception Circuit Ablation - Dual-Path Workflow

## Why dual-path tracing matters

Dual-path prompts force the model to surface two candidate strategies side by side. By capturing
both scratchpads and both public reasoning blocks, we can compare honest vs deceptive choices
within a single, consistent interface. Circuit tracing then targets the activation patterns
that correlate with deceptive selections, not a separate adversarial pipeline.

## The workflow

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. DUAL-PATH ELICITATION                                        │
│  Collect traces with Path 1 + Path 2 scratchpads and reasoning.  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CIRCUIT TRACING                                              │
│  Capture activations for each path and selection outcome.        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. PATTERN IDENTIFICATION                                       │
│  Identify circuits active during deceptive selections.           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. ABLATION + VERIFICATION                                      │
│  Ablate targeted circuits, re-run dual-path traces, validate.    │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Collect dual-path traces

```bash
python src/dual_path_evaluator.py \
  --scenarios datasets/dual_path/data.jsonl \
  --run runs/dual_path_baseline
```

This produces `runs/dual_path_baseline/dual_path_traces.jsonl` with the full thought-trace,
public reasoning, and FINAL ANSWER label for each scenario.

## Step 2: Trace circuits and emit fingerprints

```bash
python src/dual_path_circuit_tracer.py runs/dual_path_baseline
```

The tracer writes `fingerprints.jsonl` and a `trace_report.json` summary into the run directory.

## Step 3: Analyze fingerprints to select targets

```bash
python scripts/analyze_deception_fingerprints.py \
  --fingerprints runs/dual_path_baseline/fingerprints.jsonl \
  --out runs/dual_path_baseline/ablation_targets.json \
  --threshold 0.7
```

## Step 4: Ablate and verify

```bash
python scripts/ablate_deception_circuits.py \
  --model my-model-baseline \
  --targets runs/dual_path_baseline/ablation_targets.json \
  --strength 0.8 \
  --out models/my-model-ablated
```

Then re-run Step 1 and Step 2 on the ablated model, comparing dual-path traces across
FINAL ANSWER labels to confirm that deceptive circuits were removed.
