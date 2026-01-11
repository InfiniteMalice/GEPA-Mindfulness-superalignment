# Deception Baseline Detectors

The detectors operate on dual-path reasoning traces and other deception
artifacts emitted by the GEPA pipeline. Signals include:

- **Semantic divergence** between the public answer and the private chains.
- **Reward-hacking lexicon** hits across both chains and the public answer.
- **Situational awareness** markers that indicate the model realises it is in an eval.
- **Confidence inversion** when late-stage confidence spikes after early doubt.

`score.py` aggregates the signals into a bounded score in `[0, 1]` alongside
textual reasons. Results are stored per scenario as
`*_deception.json` and aggregated into `runs/deception.json` by the CLI.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
