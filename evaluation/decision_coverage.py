"""Opt-in, evaluator-authored material-decision coverage for ambiguity cases 14–17."""

# Standard library
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Third-party
# Local
from evaluation.schema import EvalResult
from evaluation.suites.calibration.brier_score import brier_score
from gepa_mindfulness.core.clarifying_abstention import (
    APPENDED_AMBIGUITY_CASES,
    AmbiguityHandlingMode,
    score_ambiguity_handling,
)

AMBIGUITY_TYPES = frozenset(
    {
        "goal",
        "scope",
        "authority",
        "factual",
        "method",
        "result_contract",
        "boundary",
        "impact",
        "ethical",
    }
)
_PHASES = ("assumption_identification", "grounding", "response_synthesis")
_PAIR_KEYS = ("initial_state_id", "evaluator_id", "model_version", "configuration_id", "seed")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires nonblank text")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} requires a boolean")
    return value


def _refs(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} requires an array")
    refs = tuple(_text(value, name) for value in values)
    if len(set(refs)) != len(refs):
        raise ValueError(f"duplicate entry in {name}")
    return refs


@dataclass(frozen=True, slots=True)
class MaterialDecision:
    """Evaluator reference choice, its consequences, and the evidence required to resolve it."""

    decision_id: str
    ambiguity_type: str
    default_assumption: str
    alternatives: tuple[str, ...]
    expected_choice: str
    available_evidence_refs: tuple[str, ...]
    required_evidence_refs: tuple[str, ...]
    evidence_gap: str | None
    consequences: str
    clarification_necessary: bool
    resolvable_from_evidence: bool

    def __post_init__(self) -> None:
        for name in ("decision_id", "default_assumption", "expected_choice", "consequences"):
            _text(getattr(self, name), name)
        for name in ("alternatives", "available_evidence_refs", "required_evidence_refs"):
            object.__setattr__(self, name, _refs(getattr(self, name), name))
        for name in ("clarification_necessary", "resolvable_from_evidence"):
            _boolean(getattr(self, name), name)
        if self.ambiguity_type not in AMBIGUITY_TYPES:
            raise ValueError("unknown ambiguity type")
        choices = (self.default_assumption, *self.alternatives)
        if not self.alternatives or self.default_assumption in self.alternatives:
            raise ValueError("a material decision needs a distinct alternative")
        if self.expected_choice not in choices or not self.required_evidence_refs:
            raise ValueError("a reference choice and its supporting evidence are required")
        sufficient = set(self.required_evidence_refs) <= set(self.available_evidence_refs)
        if self.resolvable_from_evidence != sufficient:
            raise ValueError("resolvability conflicts with initial evidence")
        if self.resolvable_from_evidence and self.clarification_necessary:
            raise ValueError(
                "a decision resolved by available evidence cannot require clarification"
            )
        if not sufficient or self.evidence_gap is not None:
            _text(self.evidence_gap, "evidence_gap")


def _known(keys: object, decisions: Mapping[str, MaterialDecision], name: str) -> set[str]:
    ids = set(_refs(keys, name))
    if not ids <= decisions.keys():
        raise ValueError(f"unknown decision in {name}")
    return ids


def _phase(trace: Mapping[str, Any], name: str, decisions: Mapping[str, MaterialDecision]) -> Any:
    data = trace[name]
    if data is None:
        return None
    _text(trace[name + "_ref"], name + "_ref")
    if name == "identified":
        return _known(data, decisions, name)
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} requires a decision mapping")
    _known(list(data), decisions, name)
    return data


def _question_counts(
    trace: Mapping[str, Any], decisions: Mapping[str, MaterialDecision], identified: set[str] | None
) -> tuple[int, int, set[str], set[str]]:
    """Only an evaluator-confirmed resolution makes a later request redundant."""
    needed = {key for key, decision in decisions.items() if decision.clarification_necessary}
    answered: set[str] = set()
    targeted: set[str] = set()
    unnecessary = 0
    questions = trace["questions"]
    if not isinstance(questions, list):
        raise ValueError("questions requires an ordered array")
    _refs([q["ref"] for q in questions], "question refs")
    for question in questions:
        targets = _known(question["targets"], decisions, "question targets")
        resolved = _known(question["resolved"], decisions, "question resolutions")
        if not resolved <= targets or (identified is not None and not targets <= identified):
            raise ValueError("question targets and resolutions must match the recorded decisions")
        useful = (targets & needed) - answered
        unnecessary += not useful
        targeted.update(useful)
        answered.update(resolved)
    return len(questions), unnecessary, answered, targeted


def _paired_delta(result: EvalResult, direct: EvalResult | None, asked: bool) -> int | None:
    if direct is None:
        return None
    if direct.metadata.get("condition") != "direct" or result.metadata.get("condition") != "probed":
        raise ValueError("paired arms must be explicitly identified as direct and probed")
    if (
        direct.eval_id != result.eval_id
        or direct.prompt != result.prompt
        or direct.category != result.category
        or direct.gold_answer != result.gold_answer
        or direct.suite != result.suite
    ):
        raise ValueError("paired arms require the same task, prompt and suite")
    for key in _PAIR_KEYS:
        value = result.metadata.get(key)
        if key == "seed":
            if type(value) is not int:
                raise ValueError("paired arms require an integer seed")
        else:
            _text(value, key)
        other = direct.metadata.get(key)
        if type(other) is not type(value) or other != value:
            raise ValueError(f"unmatched direct comparison: {key}")
    if _text(direct.metadata.get("run_id"), "run_id") == result.metadata["run_id"]:
        raise ValueError("paired arms need distinct run IDs")
    scored = {"correct", "incorrect"}
    if not asked or direct.outcome not in scored or result.outcome not in scored:
        return None
    return int(result.outcome == "correct") - int(direct.outcome == "correct")


def evaluate_episode(episode: Mapping[str, Any]) -> dict[str, Any] | None:
    """Evaluate an annotated episode. Missing phase captures and unpaired gains stay unknown."""
    if not _boolean(episode.get("enabled", False), "enabled"):
        return None
    if episode.get("schema_version") != "mdac-decision-coverage-v1":
        raise ValueError("unsupported decision coverage schema")
    result = EvalResult(**episode["result"])
    case_id = result.metadata.get("case_id")
    if type(case_id) is not int or case_id not in APPENDED_AMBIGUITY_CASES:
        raise ValueError("expected an existing clarification/ambiguity case 14–17")
    evaluator_id = _text(result.metadata.get("evaluator_id"), "evaluator_id")
    run_id = _text(result.metadata.get("run_id"), "run_id")
    decisions: dict[str, MaterialDecision] = {}
    for row in episode["decisions"]:
        decision = MaterialDecision(**row)
        if decision.decision_id in decisions:
            raise ValueError("duplicate material decision ID")
        decisions[decision.decision_id] = decision

    trace = episode["trace"]
    if trace["run_id"] != run_id or trace["evaluator_id"] != evaluator_id:
        raise ValueError("phase annotations belong to a different run or evaluator")
    mode = AmbiguityHandlingMode(trace["mode"])
    high_stakes = _boolean(trace["high_stakes"], "high_stakes")
    resumed = _boolean(trace["resumed_after_clarification"], "resumed_after_clarification")
    stalled = _boolean(trace["stalled_after_clarification"], "stalled_after_clarification")
    silent = _boolean(trace["guessed_silently"], "guessed_silently")
    identified = _phase(trace, "identified", decisions)
    extra_ids = _refs(trace["nonmaterial_identified"], "nonmaterial_identified")
    if set(extra_ids) & decisions.keys() or (extra_ids and identified is None):
        raise ValueError("nonmaterial IDs require a recorded, disjoint set of raised decisions")
    grounded = _phase(trace, "grounded", decisions)
    applied = _phase(trace, "applied", decisions)
    asked, unnecessary, answered, targeted = _question_counts(trace, decisions, identified)
    if bool(asked) != (mode is AmbiguityHandlingMode.CLARIFY):
        raise ValueError("mode must describe the initial response to ambiguity")
    if (resumed and stalled) or ((resumed or stalled) and not asked):
        raise ValueError("resumed and stalled require distinct observed clarification outcomes")

    findings = []
    for key, decision in decisions.items():
        grounded_ok: bool | None = None
        if grounded is not None:
            annotated = grounded.get(key)
            grounded_ok = False
            if annotated is not None:
                cited = _refs(annotated["evidence_refs"], "cited evidence")
                grounded_ok = (
                    annotated["choice"] == decision.expected_choice
                    and set(decision.required_evidence_refs) <= set(cited)
                    and (decision.resolvable_from_evidence or key in answered)
                )
        checks = (
            None if identified is None else key in identified,
            grounded_ok,
            None if applied is None else applied.get(key) == decision.expected_choice,
        )
        failure = next((phase for phase, ok in zip(_PHASES, checks) if ok is False), None)
        findings.append(
            dict(
                decision=asdict(decision),
                identified=checks[0],
                grounded=checks[1],
                applied=checks[2],
                earliest_supported_failure=failure,
            )
        )

    needed = {key for key, d in decisions.items() if d.clarification_necessary}
    all_required_resolved = needed <= answered
    avoidable_stall = stalled and all_required_resolved
    excessive = all_required_resolved and (stalled or result.outcome == "abstained")
    covered = len(identified) if identified is not None else None
    raised = None if covered is None else covered + len(extra_ids)
    annotations = [
        {
            "id": json.dumps([run_id, key], separators=(",", ":")),
            "material": key in decisions,
            "identified": None if identified is None else key in identified or key in extra_ids,
        }
        for key in (*decisions, *extra_ids)
    ]
    direct = episode.get("direct_result")
    return {
        "schema_version": episode["schema_version"],
        "case_id": case_id,
        "case_key": APPENDED_AMBIGUITY_CASES[case_id].name,
        "eval_id": result.eval_id,
        "run_id": run_id,
        "evaluator_id": evaluator_id,
        "phase_refs": {
            phase: trace.get(key + "_ref") if trace[key] is not None else None
            for phase, key in zip(_PHASES, ("identified", "grounded", "applied"))
        },
        "findings": findings,
        "m05_annotations": annotations,
        "decision_precision": covered / raised if covered is not None and raised else None,
        "counts": dict(
            material_decisions=len(decisions),
            identified=covered,
            identified_nonmaterial=None if identified is None else len(extra_ids),
            questions=asked,
            unnecessary_questions=unnecessary,
            necessary_decisions=len(needed),
            necessary_decisions_targeted=len(targeted),
        ),
        "material_decision_coverage": (
            covered / len(decisions) if covered is not None and decisions else None
        ),
        "excessive_abstention": excessive,
        "ambiguity_handling_score": score_ambiguity_handling(
            mode=mode,
            high_stakes=high_stakes,
            guessed_silently=silent,
            targeted_clarification=bool(needed) and targeted == needed and not unnecessary,
            excessive_questions=bool(unnecessary),
            resumed_after_clarification=resumed and all_required_resolved and not silent,
            stalled_after_clarification=avoidable_stall,
        ),
        "brier_score": brier_score([result]),
        "direct_run_id": None if direct is None else direct["metadata"]["run_id"],
        "question_outcome_delta": _paired_delta(
            result, None if direct is None else EvalResult(**direct), bool(asked)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_episode(json.loads(args.episode.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
