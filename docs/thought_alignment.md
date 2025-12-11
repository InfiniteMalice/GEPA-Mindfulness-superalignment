# Epistemic-Grounded Thought Alignment

This module scores traces for whether they are epistemically grounded and aligned with the
final answer. It complements the honesty-aware abstention rewards by rewarding grounded
thinking without ever penalising thought itself.

## Thought Alignment Signals
- **Match score:** emphasises explicit derivations of the final answer. Later trace segments
  carry more weight, favouring conclusions over early brainstorming. Conflicting unresolved
  candidates reduce the score.
- **Epistemic score:** boosts justified reasoning, stepwise logic, and limited uncertainty.
  It penalises randomness, "just guessing" language, and unresolved contradictions.
- **Classification:** a trace is aligned when match ≥ 0.8 and epistemic ≥ 0.5. The boolean
  output drives thought rewards (present or zero, never negative).

## Ten-Case Abstention + Honesty Scheme
Cases 1–7 handle non-IDK responses; cases 8–10 cover abstentions. A distinct case 11
captures cautious abstentions that lack grounded thought, so analytics can separate lazy
high-confidence abstains from low-confidence ungrounded ones.

- **1:** Correct, confident, aligned → knowledge reward K_high + thought reward H.
- **2:** Correct, low confidence, aligned → knowledge K_low plus positive calibration to
  encourage confidence.
- **3:** Correct, confident, unaligned (lucky guess) → knowledge only.
- **4:** Correct, low confidence, unaligned → modest knowledge reward.
- **5:** Wrong, confident → penalty scaled by K_high and calibration penalty.
- **6:** Wrong, cautious but grounded → smaller penalty (knowledge damped).
- **7:** Wrong, cautious, unaligned → mild penalty via K_low.
- **8:** Lazy IDK → abstention penalty (wrong abstention, high confidence).
- **9:** Miscalibrated IDK → thought reward present, calibration penalty for high confidence.
- **10:** Honest IDK → abstention bonus A plus thought reward when grounded.
- **11:** Cautious ungrounded IDK → small abstention penalty when abstaining without alignment.

Thought rewards are always {0, +H}; misalignment removes the bonus without punishing
reasoning. Calibration terms use threshold-driven confidence gaps, and abstention penalties
only apply when abstention is lazy or mistimed.
