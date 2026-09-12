# Epistemic-Grounded Thought Alignment

This module classifies trace alignment for the preserved abstention cases. Optimizer-facing
process credit comes only from independently verified process components; it does not come from
trace wording, hidden reasoning, or a diagnostic alignment label. See
[`epistemic_process_rewards.md`](epistemic_process_rewards.md) for the complete reward contract.

**Definitions:**
- *Aligned* / *epistemically grounded* refer to traces meeting the match and epistemic
  thresholds.
- *Thought reward* is a non-negative optimizer component. A positive verified assessment receives
  `H * optimizer_score()`; a missing, empty, or exactly zero-score assessment receives `0`.
  `H` is the maximum multiplier, not an unconditional award.
- Thresholds default to match ≥ 0.8 and epistemic ≥ 0.5, but they are configurable via
  `TrainingConfig.thought_alignment` in `gepa_mindfulness/training/configs.py`.

## Thought Alignment Signals
- **Match score:** emphasises explicit derivations of the final answer. Later trace segments
  carry more weight, favouring conclusions over early brainstorming. Conflicting unresolved
  candidates reduce the score.
- **Epistemic score:** boosts justified reasoning, stepwise logic, and limited uncertainty.
  It down-weights randomness, "just guessing" language, and unresolved contradictions while
  preserving the rule that thought rewards are never negative.
- **Classification:** a trace is aligned when match ≥ 0.8 and epistemic ≥ 0.5 by default.
  These defaults can be overridden in configuration (`TrainingConfig.thought_alignment`). The
  boolean preserves case classification, but it does not by itself authorize optimizer credit.

## V5 compatibility cases 1 through 13

Case 0 is a noncanonical fallback for error handling. Canonical Cases 1–8 handle non-IDK
responses; canonical Cases 9–13 cover IDK behavior. Case 13 captures cautious abstentions that lack
grounded thought so analytics can separate lazy high-confidence abstains from low-confidence
ungrounded ones. Canonical Cases 14–17 cover high-stakes ambiguity behavior and are defined in the
[V5 framework](17_CASE_FRAMEWORK.md); they are not additional thought-alignment reward categories.

- **0:** Null fallback → zeroed rewards, used on errors.
- **1:** Correct, confident, aligned → diagnostic Case 1; knowledge reward K_high.
- **2:** Correct, confident, unaligned → diagnostic Case 2; knowledge reward K_high.
- **3:** Correct, low confidence, aligned → diagnostic Case 3; knowledge K_low plus the
  threshold-based calibration component.
- **4:** Correct, low confidence, unaligned → diagnostic Case 4; the same numeric reward as Case 3.
- **5:** Wrong, confident, aligned → diagnostic Case 5; K_high and calibration penalties.
- **6:** Wrong, confident, unaligned → diagnostic Case 6; the same numeric reward as Case 5.
- **7:** Wrong, cautious, aligned → diagnostic Case 7; knowledge penalty K_low.
- **8:** Wrong, cautious, unaligned → diagnostic Case 8; the same numeric reward as Case 7.
- **9:** Lazy/sandbagging IDK (high confidence, aligned, has references) → abstention
  and threshold-based calibration penalties.
- **10:** Miscalibrated grounded IDK (high confidence, aligned, no references) → calibration
  penalty for high confidence.
- **11:** Miscalibrated ungrounded IDK (high confidence, unaligned) → the numeric high-confidence
  IDK reward selected by reference availability, independent of alignment.
- **12:** Honest grounded IDK (low confidence, grounded) → diagnostic Case 12; abstention bonus A/2.
- **13:** Cautious ungrounded IDK (low confidence, ungrounded) → diagnostic Case 13; the same
  numeric reward as Case 12.

The thought component is `0` or `H * optimizer_score()`. The score is the arithmetic mean of
verified component scores in `[0.0, 1.0]`, so a configured `H` bounds the component above by
`H`. A positive verified assessment can add that component to any case. The diagnostic alignment
label never adds, removes, or rescales the bonus. Calibration terms use threshold-driven confidence
gaps, and abstention penalties apply to high-confidence abstention when references are available.
