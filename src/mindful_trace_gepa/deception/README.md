# Deception Baseline Detectors

The detectors operate on paired honest/deceptive chains emitted for baseline
analysis. Signals include:

- **Semantic divergence** between the public answer and the private chains.
- **Reward-hacking lexicon** hits across both chains and the public answer.
- **Situational awareness** markers that indicate the model realises it is in an eval.
- **Confidence inversion** when late-stage confidence spikes after early doubt.

`score.py` aggregates the signals into a bounded score in `[0, 1]` alongside
textual reasons. Results are stored per scenario as
`*_deception.json` and aggregated into `runs/deception.json` by the CLI.
