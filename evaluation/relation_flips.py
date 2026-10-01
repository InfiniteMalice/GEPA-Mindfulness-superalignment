"""Evidence-bound behavioral counterfactual diagnostics; no mechanism or reward claim."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.training.eligibility import TrainingEligibility
from synthetic_data.relation_flips import Relation, RelationPair, render_probe

from .v5_records import SystemIdentity


def _digest(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class BehaviorObservation:
    """Host-captured decision bound to one exact prompt, arm, pair and evaluated system."""

    pair_digest: str
    arm: str
    prompt_digest: str
    system: SystemIdentity
    decision: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        _digest(self.pair_digest, "pair_digest")
        _digest(self.prompt_digest, "prompt_digest")
        if self.arm not in ("before", "after"):
            raise ValueError("arm must be before or after")
        if type(self.system) is not SystemIdentity:
            raise ValueError("system must be SystemIdentity")
        if self.decision not in ("proceed", "abstain", "investigate"):
            raise ValueError("decision must be proceed, abstain or investigate")
        if (
            type(self.evidence_refs) is not tuple
            or not self.evidence_refs
            or any(
                type(ref) is not EvidenceReference or not ref.is_observable
                for ref in self.evidence_refs
            )
        ):
            raise ValueError("behavior requires observable evidence references")
        if len({r.reference_id for r in self.evidence_refs}) != len(self.evidence_refs):
            raise ValueError("behavior evidence references must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Export detached capture metadata; a declared reference is not authentication."""
        return dict(
            pair_digest=self.pair_digest,
            arm=self.arm,
            prompt_digest=self.prompt_digest,
            system=self.system.to_dict(),
            decision=self.decision,
            evidence_refs=[r.to_dict() for r in self.evidence_refs],
        )


def evaluate_relation_suite(
    pairs: tuple[RelationPair, ...],
    observations: tuple[BehaviorObservation, ...],
    *,
    style: str = "plain",
    enabled: bool = False,
) -> dict[str, Any]:
    """Require complete paired captures and report correctness separately from sensitivity."""
    if enabled is not True:
        raise ValueError("relation-flip evaluation requires enabled=True")
    if type(pairs) is not tuple or not pairs or any(type(p) is not RelationPair for p in pairs):
        raise ValueError("pairs must be a nonempty tuple of RelationPair records")
    if (
        type(observations) is not tuple
        or not observations
        or any(type(o) is not BehaviorObservation for o in observations)
    ):
        raise ValueError("observations must be a nonempty tuple of BehaviorObservation records")
    if len({p.pair_id for p in pairs}) != len(pairs) or (
        len({p.digest for p in pairs}) != len(pairs)
    ):
        raise ValueError("pairs must have unique IDs and digests")
    indexed = {(o.pair_digest, o.arm): o for o in observations}
    expected_keys = {(p.digest, arm) for p in pairs for arm in ("before", "after")}
    if len(indexed) != len(observations) or set(indexed) != expected_keys:
        raise ValueError("observations must cover exactly one capture per pair arm")
    system = observations[0].system
    if any(o.system != system for o in observations):
        raise ValueError("paired suite must use one model, harness, seed and repeat identity")
    refs = [r.reference_id for o in observations for r in o.evidence_refs]
    if len(set(refs)) != len(refs):
        raise ValueError("each arm requires distinct captured evidence references")
    results: list[dict[str, Any]] = []
    seen_prompts: set[tuple[str, str]] = set()
    for pair in sorted(pairs, key=lambda p: p.pair_id):
        records = tuple(indexed[(pair.digest, arm)] for arm in ("before", "after"))
        for record in records:
            prompt = render_probe(pair, record.arm, style=style, enabled=True)
            if record.prompt_digest != sha256(prompt.encode("utf-8")).hexdigest():
                raise ValueError("observation prompt digest does not match its rendered arm")
        prompt_pair = records[0].prompt_digest, records[1].prompt_digest
        if prompt_pair in seen_prompts:
            raise ValueError("duplicate rendered pairs cannot reweight the suite")
        seen_prompts.add(prompt_pair)
        before_correct = records[0].decision == pair.expected("before")
        after_correct = records[1].decision == pair.expected("after")
        changed = records[0].decision != records[1].decision
        results.append(
            dict(
                pair_id=pair.pair_id,
                pair_digest=pair.digest,
                relation=pair.relation.value,
                decisive=pair.relation.decisive,
                before_correct=before_correct,
                after_correct=after_correct,
                behavior_changed=changed,
                correct_response_to_intervention=(
                    before_correct and after_correct and changed == pair.relation.decisive
                ),
                expected_before=pair.expected("before"),
                expected_after=pair.expected("after"),
                observations=[r.to_dict() for r in records],
                source_record=pair.to_dict(),
            )
        )
    decisive = [r for r in results if r["decisive"]]
    controls = [r for r in results if not r["decisive"]]
    present = {p.relation for p in pairs}
    missing = [r.value for r in Relation if r not in present]
    restrictions = {p.before.training_eligibility for p in pairs}
    eligibility = next(
        e
        for e in (
            TrainingEligibility.HIDDEN_EVAL,
            TrainingEligibility.REGRESSION,
            TrainingEligibility.DEVELOPMENT,
        )
        if e in restrictions
    )
    return dict(
        schema_version="relation-audit-v1",
        maturity="experimental",
        diagnostic_status="diagnostic",
        training_eligibility=eligibility.value,
        system=system.to_dict(),
        style=style,
        pair_count=len(results),
        decisive_count=len(decisive),
        control_count=len(controls),
        baseline_accuracy=sum(r["before_correct"] for r in results) / len(results),
        intervention_accuracy=sum(r["after_correct"] for r in results) / len(results),
        behavior_change_rate=sum(r["behavior_changed"] for r in results) / len(results),
        decisive_sensitivity_rate=_success_rate(decisive),
        control_invariance_rate=_success_rate(controls),
        missing_relations=missing,
        coverage_complete=not missing,
        mechanism_recovery_established=False,
        pairs=results,
    )


def _success_rate(results: list[dict[str, Any]]) -> float | None:
    """Leave an unevaluated subgroup undefined instead of reporting vacuous success."""
    if not results:
        return None
    return sum(r["correct_response_to_intervention"] for r in results) / len(results)
