# Epistemic-Grounded Thought Alignment

This module scores traces for whether they are epistemically grounded and aligned with the
final answer. It complements the honesty-aware abstention rewards by rewarding grounded
thinking without ever penalizing thought itself.

**Definitions:**
- *Aligned* / *epistemically grounded* refer to traces meeting the match and epistemic
  thresholds.
- *Thought reward* is a non-negative bonus (+H when aligned, otherwise 0) and never a
  penalty; lower scores reduce the bonus rather than introduce punishment.
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
  These defaults can be overridden in configuration (`TrainingConfig.thought_alignment`).
  The boolean output drives thought rewards (present or zero, never negative).

## Twelve-Case Abstention + Honesty Scheme
Case 0 is a null fallback for error handling. Cases 1–8 handle non-IDK responses; cases
9–12 cover abstentions. Case 12 captures cautious abstentions that lack grounded thought,
so analytics can separate lazy high-confidence abstains from low-confidence ungrounded
ones.

- **0:** Null fallback → zeroed rewards, used on errors.
- **1:** Correct, confident, aligned → knowledge reward K_high + thought reward H.
- **2:** Correct, confident, unaligned (shortcut) → knowledge only.
- **3:** Correct, low confidence, aligned → knowledge K_low plus positive calibration to
  encourage confidence.
- **4:** Correct, low confidence, unaligned → modest knowledge reward.
- **5:** Wrong, confident, aligned → penalty scaled by K_high plus thought reward H.
- **6:** Wrong, confident, unaligned → penalty scaled by K_high.
- **7:** Wrong, cautious, aligned → smaller penalty (knowledge damped) plus thought reward H.
- **8:** Wrong, cautious, unaligned → mild penalty via K_low.
- **9:** Lazy IDK → abstention penalty (wrong abstention, high confidence).
- **10:** Miscalibrated IDK → thought reward present, calibration penalty for high confidence.
- **11:** Honest IDK → abstention bonus A plus thought reward when grounded.
- **12:** Cautious ungrounded IDK → small abstention bonus for correct IDK even without
  alignment (abstention correctness is separate from thought grounding).

Thought rewards are always {0, +H}; misalignment removes the bonus without punishing
reasoning. Calibration terms use threshold-driven confidence gaps, and abstention penalties
only apply when abstention is lazy or mistimed.
