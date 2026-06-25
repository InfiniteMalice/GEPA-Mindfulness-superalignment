# Objective / Validator Robustness

## Purpose

This module detects **Validator Capture** (aliases: Task-Validator Safety Collapse,
Objective Capture, TVD-style safety collapse, Internal Safety Collapse-like failure) and adds an
optional objective-robustness overlay for proxy breakdown, novelty, and inverse objective
interpretation.

Validator Capture is a failure mode where a model is captured by local task success criteria
(validator, rubric, unit test, schema, benchmark, or required output format) such that
passing the task requires unsafe output.

The combined principle is:

> Goals guide action, but they are not sovereign. Reward functions are evidence. Values govern interpretation. Reality tests the proxy. Uncertainty slows optimization. Corrigibility keeps the system teachable.

## Validator Capture

A local validator, rubric, schema, benchmark, or unit test can pressure the model toward unsafe
completion. The existing `decompose_objective`, `detect_validator_capture`,
`decide_validator_policy`, `score_validator_robustness`, and `validator_overlay_tier` APIs remain
unchanged.

## Correlated-Proxy Breakdown

A reward can correlate with the intended goal under a base policy but become unreliable when
optimization pressure moves the current policy outside that reference region. The new
`assess_proxy_objective` helper records proxy cues, optimization pressure, evaluator-gaming risk,
reward-hacking risk, and whether a correlation warning is warranted.

This layer is intentionally honest about uncertainty: without empirical policy evidence, correlation
confidence is labeled `heuristic` or `unavailable`, not measured.

## Inverse Objective Interpretation

The supplied objective is evidence about intended direction in its design context, not a complete
license for literal maximization in all future environments. `infer_objective_posterior` represents
multiple plausible interpretations when the proxy appears incomplete or the deployment context is
novel. These posterior-like weights are deterministic heuristics unless a future integration
provides calibrated evidence.

## Novelty and Uncertainty

`assess_objective_novelty` compares the design/training context against observed deployment
conditions. Novelty does not automatically cause refusal:

| Condition | Default response |
|---|---|
| Low stakes, reversible, mild novelty | Proceed cautiously with a stated assumption |
| Moderate novelty with manageable downside | Bound the action and preserve optionality |
| High stakes or irreversible action under objective uncertainty | Clarify, escalate, or pause execution |
| Catastrophic plausible downside | Do not optimize blindly |

## Robust Policy Selection

`decide_robust_objective_policy` selects a safe action under proxy uncertainty. It can allow,
bound, transform, preserve optionality, ask clarifying questions, escalate, or refuse. It prefers
reversible actions, avoids catastrophic downside under plausible interpretations, offers safe
alternatives, and preserves existing validator-capture behavior.

`ObjectiveValidationInterrupt` is an advisory control signal. It can raise scheduling priority and
force review before unsafe execution, but it does not bypass deliberation or perform irreversible
actions.

## How this differs from semantic laundering

| Mode | Semantic laundering | Validator Capture |
|---|---|---|
| Where harmfulness is hidden | Wording, translation, roleplay, fragmentation, euphemism | Task structure, validator, rubric, schema, unit test, dataset, success condition |
| Primary failure | Model misclassifies true user intent | Model optimizes local success despite global safety conflict |
| Core question | What does the user really want? | What is this task actually optimizing? |

## Relation to ISC / TVD

This is jailbreak-adjacent but not identical to jailbreaking. In many cases, the user-facing task
appears legitimate (e.g., quality checks, benchmark prep, schema completion), yet the validator
implicitly requires harmful content. The failure can occur even when the model understands the
prompt correctly.

Reference framing for TVD-style collapse discussion:

- https://arxiv.org/pdf/2603.23509

## Alignment reference: Three imperatives

The module checks local task success against repository-level imperatives:

1. Increase prosperity
2. Reduce suffering
3. Increase understanding

It also treats constraints, review gates, validators, permissions, and procedures as possible
protective structures rather than mere friction. The least-action policy prefers narrow,
authorized, reversible, auditable responses before broad or irreversible intervention.

## 17-case compatibility via overlay tiers

This module does **not** expand, renumber, or redefine the existing 17-case epistemic schema. It
adds composable overlays:
`CaseX-OY-SZ-VW`

Overlay meanings:

- `V0`: no objective/validator analysis available
- `V1`: objective structure parsed
- `V2`: validator/rubric pressure detected
- `V3`: possible local-success/global-safety conflict
- `V4`: confirmed validator capture risk with safe alternative
- `V5`: confirmed high-risk task-validator safety collapse requiring refusal/escalation

Proxy overlay meanings:

- `P0`: no proxy analysis available
- `P1`: objective parsed and design context identified
- `P2`: objective likely incomplete or proxy-like
- `P3`: novel state or optimization-pressure concern detected
- `P4`: likely proxy breakdown; bounded action or clarification required
- `P5`: high-risk objective failure; execution pause, escalation, or refusal required

The proxy overlay is separate from V0-V5 validator-capture tiers, factuality overlays,
semantic-intent categories, and the 17-case schema.

## Design principles

- Positive and reward-oriented scoring.
- No deception penalty additions.
- No hidden-state or thought-trace punishment.
- Prefer safe rerouting and transformation when possible.
- Refuse/escalate only dangerous slots or high-risk contexts.

## Basic usage

```python
from objective_validator_robustness import (
    decompose_objective,
    detect_validator_capture,
    decide_validator_policy,
    score_validator_robustness,
    validator_overlay_tier,
)

text = "Return valid JSON that passes schema checks, including dangerous_payload."

structure = decompose_objective(text)
signal = detect_validator_capture(structure)
decision = decide_validator_policy(structure, signal)
score = score_validator_robustness(structure, signal, decision)
tier = validator_overlay_tier(signal)
```

Proxy robustness pipeline:

```python
from objective_validator_robustness import (
    ObjectiveSpecification,
    evaluate_objective_robustness,
)

specification = ObjectiveSpecification(
    objective_id="gridworld-lava",
    objective_text="Minimize time spent on grass.",
    designer_context_summary="Reward designed for dirt and grass terrain.",
    training_environment_summary="Known features: dirt grass.",
    deployment_environment_summary="Deployment now includes dirt grass and lava.",
    explicit_success_condition="Minimize grass steps.",
    implicit_success_condition="Reach the goal safely, not by exploiting hazards.",
    known_constraints=["avoid dangerous terrain"],
    metadata={"stakes_level": "high", "reversibility": "low"},
)

report = evaluate_objective_robustness(specification)
```

## Safe transformation patterns

- Preserve schema shape while replacing dangerous slots with
  `[REDACTED_SAFE_PLACEHOLDER]`.
- Use non-operational dummy strings for toy classifier datasets.
- Prefer approved internal dataset IDs over generating harmful samples.
- Provide validation guidance without dangerous field contents.

## Relationship to Existing Layers

- The Constitution supplies values.
- The 17-case framework handles epistemic calibration and high-stakes ambiguity.
- Semantic intent robustness handles laundering across phrasing and turns.
- Memory safety handles laundering across persistence boundaries.
- Objective robustness handles proxies, validators, novelty, and optimization pressure.
- Logs and attribution references support reflective-stability review.

The layer remains modular, typed, inspectable, and opt-in through
`configs/objective_validator_robustness.yaml`.
