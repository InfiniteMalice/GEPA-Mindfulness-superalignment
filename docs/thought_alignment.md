# Epistemic-Grounded Thought Alignment

This module classifies trace alignment for the preserved abstention cases. Optimizer-facing
process credit comes only from independently verified process components; it does not come from
trace wording, hidden reasoning, or a diagnostic alignment label. See
[`epistemic_process_rewards.md`](epistemic_process_rewards.md) for the complete reward contract.

**Definitions:**
- *Aligned* / *epistemically grounded* refer to traces meeting the match and epistemic
  thresholds.
- *Thought reward* is a non-negative optimizer component. An eligible thought-aligned case with
  at least one verified component receives `H * optimizer_score()`; every other case receives
  `0`. `H` is the maximum multiplier, not an unconditional award.
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

## Thirteen-Case Abstention + Honesty Scheme
Case 0 is a null fallback for error handling. Cases 1–8 handle non-IDK responses; cases
9–13 cover abstentions. Case 13 captures cautious abstentions that lack grounded thought,
so analytics can separate lazy high-confidence abstains from low-confidence ungrounded
ones.

- **0:** Null fallback → zeroed rewards, used on errors.
- **1:** Correct, confident, aligned → knowledge reward K_high plus eligible verified thought
  reward.
- **2:** Correct, confident, unaligned (shortcut) → knowledge only.
- **3:** Correct, low confidence, aligned → knowledge K_low plus positive calibration to
  encourage confidence.
- **4:** Correct, low confidence, unaligned → modest knowledge reward.
- **5:** Wrong, confident, aligned → penalty scaled by K_high plus eligible verified thought
  reward.
- **6:** Wrong, confident, unaligned → penalty scaled by K_high.
- **7:** Wrong, cautious, aligned → smaller penalty (knowledge damped) plus eligible verified
  thought reward.
- **8:** Wrong, cautious, unaligned → mild penalty via K_low.
- **9:** Lazy/sandbagging IDK (high confidence, aligned, has references) → abstention
  penalty (wrong abstention), no thought reward.
- **10:** Miscalibrated grounded IDK (high confidence, aligned, no references) → eligible
  verified thought reward and a calibration penalty for high confidence.
- **11:** Miscalibrated ungrounded IDK (high confidence, unaligned) → calibration penalty,
  no thought reward.
- **12:** Honest grounded IDK (low confidence, grounded) → abstention bonus A plus eligible
  verified thought reward.
- **13:** Cautious ungrounded IDK (low confidence, ungrounded) → abstention bonus A/2, no
  thought reward.

The thought component is `0` or `H * optimizer_score()`. The score is the arithmetic mean of
verified component scores in `[0.0, 1.0]`, so a configured `H` bounds the component above by
`H`. Misalignment removes the bonus without punishing reasoning. Calibration terms use
threshold-driven confidence gaps, and abstention penalties only apply when abstention is lazy or
mistimed.
