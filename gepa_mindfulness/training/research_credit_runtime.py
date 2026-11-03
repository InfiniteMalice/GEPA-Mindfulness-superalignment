"""Default-off, freshly host-reviewed credit composition over the existing ledger."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from math import fsum
from types import MappingProxyType
from typing import cast

from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute

from .causal_reward import _accepted, _snapshot_integrity
from .causal_reward_records import (
    _digest,
    _json_value,
    _number,
    _read_request,
    _request_wire,
    _text,
    request_digest,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .research_credit import ResearchCreditClaim, audit_research_credit
from .research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from .reward_pipeline import RewardPipeline, RewardResult
from .trajectory import Trajectory

_KINDS = frozenset({"knowledge", "progress", "process"})
HostReview = Callable[[dict[str, object]], bool]


@dataclass(frozen=True)
class ResearchRewardPolicy:
    """Operator-selected scope and research coefficients, never normative weights."""

    policy_id: str
    run_id: str
    task_id: str
    harness_version: str
    dataset_digest: str
    coefficients: tuple[tuple[str, float], ...] = ()
    max_abs_adjustment: float = 1.0
    enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("policy_id", "run_id", "task_id", "harness_version"):
            _text(getattr(self, name), name)
        _digest(self.dataset_digest)
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        _number(self.max_abs_adjustment)
        if not 0 < self.max_abs_adjustment <= 1:
            raise ValueError("max_abs_adjustment must be in (0, 1]")
        if type(self.coefficients) is not tuple or len(self.coefficients) > len(_KINDS):
            raise ValueError("coefficients require a bounded tuple")
        names = set()
        for entry in self.coefficients:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("coefficient requires kind and weight")
            kind, weight = entry
            if type(kind) is not str or kind not in _KINDS or kind in names:
                raise ValueError("unsupported or duplicate research credit kind")
            _number(weight)
            if weight <= 0:
                raise ValueError("research coefficients must be positive")
            names.add(kind)
        if self.enabled and not names:
            raise ValueError("enabled research rewards require explicit coefficients")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["coefficients"] = [list(entry) for entry in self.coefficients]
        value["schema_version"] = "research-reward-policy-v1"
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> ResearchRewardPolicy:
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or value["schema_version"] != "research-reward-policy-v1"
            or type(value["coefficients"]) is not list
            or any(type(entry) is not list for entry in value["coefficients"])
        ):
            raise ValueError("invalid research reward policy schema")
        args = {name: value[name] for name in names}
        args["coefficients"] = tuple(tuple(entry) for entry in value["coefficients"])
        return cls(**args)


@dataclass(frozen=True)
class ResearchRewardSubmission:
    """Untrusted contribution proposal; fresh reviews and accounting remain necessary."""

    claims: tuple[ResearchCreditClaim, ...]
    use: ResearchCreditUse

    def __post_init__(self) -> None:
        audit = audit_research_credit(self.claims, enabled=True)
        if type(self.use) is not ResearchCreditUse:
            raise ValueError("expected exact ResearchCreditUse")
        object.__setattr__(
            self, "claims", tuple(ResearchCreditClaim.from_dict(item) for item in audit["claims"])
        )
        object.__setattr__(self, "use", ResearchCreditUse.from_dict(self.use.to_dict()))
        validate_input_envelope(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema_version": "research-reward-submission-v1",
            "training_eligibility": "DEVELOPMENT",
            "claims": [claim.to_dict() for claim in self.claims],
            "use": self.use.to_dict(),
        }
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> ResearchRewardSubmission:
        validate_input_envelope(value)
        if (
            type(value) is not dict
            or set(value) != {"schema_version", "training_eligibility", "claims", "use"}
            or value["schema_version"] != "research-reward-submission-v1"
            or value["training_eligibility"] != "DEVELOPMENT"
            or type(value["claims"]) is not list
        ):
            raise ValueError("invalid research reward submission schema")
        return cls(
            tuple(ResearchCreditClaim.from_dict(item) for item in value["claims"]),
            ResearchCreditUse.from_dict(value["use"]),
        )


@dataclass(frozen=True, init=False)
class ResearchRewardDecision:
    """Live, target-bound optimizer access; JSON exports never restore this capability."""

    _report_json: str
    _trajectory_json: str
    _target_digest: str
    _authorized: bool
    _optimizer_total: float
    _metric_values: Mapping[str, float]
    _observable_references: tuple[EvidenceReference, ...]

    def __init__(self) -> None:
        raise TypeError("live decisions are produced only by ResearchRewardProvider.score")

    def optimizer_reward(self) -> float:
        if not self._authorized:
            raise ValueError("research optimizer access withheld")
        return self._optimizer_total

    def metrics(self) -> dict[str, float]:
        self.optimizer_reward()
        return dict(self._metric_values)

    def bind_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Bind only the exact original request, never a substituted response or source."""
        self.optimizer_reward()
        target = RewardRequest(trajectory, self._observable_references)
        if request_digest(target) != self._target_digest:
            raise ValueError("research reward target mismatch")
        return Trajectory.from_dict(_json_value(self._trajectory_json))

    def to_dict(self) -> dict[str, object]:
        return _json_value(self._report_json)

    def __reduce__(self) -> object:
        raise TypeError(
            "live research reward decisions cannot be restored; rescore original inputs"
        )


def _decision(
    report: dict[str, object], scored: Trajectory, target: RewardRequest, *, authorized: bool
) -> ResearchRewardDecision:
    validate_input_envelope(report)
    result = object.__new__(ResearchRewardDecision)
    for name, value in (
        ("_report_json", canonical_json(report)),
        ("_trajectory_json", canonical_json(scored.to_dict())),
        ("_target_digest", request_digest(target)),
        ("_authorized", authorized),
        ("_optimizer_total", cast(float, report["diagnostic_total"])),
        ("_metric_values", MappingProxyType(dict(cast(Mapping[str, float], report["metrics"])))),
        (
            "_observable_references",
            tuple(
                EvidenceReference.from_dict(ref.to_dict()) for ref in target.observable_references
            ),
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def _baseline_trajectory(
    target: RewardRequest, result: RewardResult, overlay_weight: float
) -> Trajectory:
    # Delegate negative-provider validation to the unchanged canonical boundary.
    from .engine import _bind_reward

    bound = _bind_reward(target.trajectory, result.base_result)
    if bound.reward_total != result.base_reward:
        raise ValueError("baseline provider changed its scored total")
    references = {}
    for reference in (*bound.evidence_references, *target.trajectory.evidence_references):
        previous = references.setdefault(reference.reference_id, reference)
        if previous != reference:
            raise ValueError("conflicting baseline and source evidence reference identity")
    bound = replace(bound, evidence_references=tuple(references.values()))
    overlay = result.integrity_breakdown
    if overlay is None:
        return replace(bound, reward_total=result.total)
    overlay = replace(
        overlay
    )  # Revalidate and detach the existing breakdown before host callbacks.
    components = dict(bound.reward_components)
    evidence = dict(bound.reward_component_evidence)
    provenance = dict(bound.reward_component_provenance)
    for name, raw in overlay.components.items():
        value = overlay_weight * raw * overlay.weights[name] / overlay.weights.total
        if value == 0:
            continue
        key = "reward_integrity_overlay:" + name
        if key in components:
            raise ValueError("baseline overlay component collision")
        components[key] = value
        evidence[key] = tuple(overlay.observable_evidence.get(name, ()))
        provenance[key] = replace(overlay.reward_component_provenance[name], component_name=key)
    refs = tuple(dict.fromkeys((*bound.evidence_references, *overlay.observable_references)))
    return replace(
        bound,
        reward_total=result.total,
        reward_components=components,
        reward_component_evidence=evidence,
        reward_component_provenance=provenance,
        evidence_references=refs,
    )


class ResearchRewardProvider:
    """Fresh host review and durable accounting before canonical optimizer consumption.

    Host code owns pipeline, ledger, scope and callbacks. Proposals may be generated; proposals
    never authorize themselves. Reviewers authenticate evidence, applicable admissibility, source
    admission and semantic overlap. This trusted-host adapter is not a sandbox against Python.
    """

    def __init__(
        self,
        pipeline: RewardPipeline,
        *,
        ledger: ResearchCreditLedger,
        policy: ResearchRewardPolicy,
        propose: Callable[[RewardRequest], ResearchRewardSubmission] | None = None,
        authenticate_claims: HostReview | None = None,
        authorize_optimizer_use: HostReview | None = None,
        authorize_accounting: HostReview | None = None,
    ) -> None:
        if type(policy) is not ResearchRewardPolicy:
            raise ValueError("expected exact ResearchRewardPolicy")
        self._policy = ResearchRewardPolicy.from_dict(policy.to_dict())
        if self._policy.enabled:
            if type(ledger) is not ResearchCreditLedger or not callable(propose):
                raise ValueError("enabled composition requires the existing ledger and proposer")
            if any(
                propose is callback
                for callback in (authenticate_claims, authorize_optimizer_use, authorize_accounting)
            ):
                raise ValueError("a proposer cannot certify its own submission")
        self.pipeline = pipeline
        self.ledger = ledger
        self.propose = propose
        self.authenticate_claims = authenticate_claims
        self.authorize_optimizer_use = authorize_optimizer_use
        self.authorize_accounting = authorize_accounting

    @property
    def policy(self) -> ResearchRewardPolicy:
        return self._policy

    def score(self, request: RewardRequest) -> RewardResult | ResearchRewardDecision:
        if not self.policy.enabled:
            return self.pipeline.score(request)
        policy = ResearchRewardPolicy.from_dict(self.policy.to_dict())
        source = _read_request(_request_wire(request))
        source_json = canonical_json(_request_wire(source))
        if source.trajectory.sampling_parameters.get("training_eligibility") != "TRAIN":
            raise ValueError("explicit source TRAIN declaration required")
        require_training_eligible(source.trajectory.to_dict())
        try:
            submission = self.propose(source)
        except Exception:
            raise ValueError("research proposer failed") from None
        if canonical_json(_request_wire(source)) != source_json:
            raise ValueError("proposer mutated the detached target")
        if type(submission) is not ResearchRewardSubmission:
            raise ValueError("expected exact ResearchRewardSubmission")
        submission = ResearchRewardSubmission.from_dict(submission.to_dict())
        _, selection, allocation, _ = self.ledger._prepare_allocation(
            source, submission.claims, submission.use
        )
        audit = allocation["audit"]
        if any("source_ineligible" in gaps for gaps in audit["prerequisite_gaps"].values()):
            raise ValueError("research submission retains non-training source material")
        if selection.policy_id != policy.policy_id or selection.request_digest != request_digest(
            source
        ):
            raise ValueError("research policy or target binding mismatch")
        by_id = {claim.claim_id: claim for claim in submission.claims}
        selected = []
        events = set()
        components = []
        weights = dict(policy.coefficients)
        for claim_id in selection.selected_claim_ids:
            if claim_id not in by_id or audit["prerequisite_gaps"][claim_id]:
                raise ValueError("selected research contribution is ineligible")
            claim = by_id[claim_id]
            expected = (
                policy.run_id,
                policy.task_id,
                policy.harness_version,
                policy.dataset_digest,
            )
            actual = (claim.run_id, claim.task_id, claim.harness_version, claim.dataset_digest)
            if (
                actual != expected
                or claim.action_id != source.trajectory.trajectory_id
                or claim.evaluator.evaluator_id
                in {source.trajectory.model_identifier, source.trajectory.policy_version}
            ):
                raise ValueError("research contribution scope or provenance mismatch")
            event = (claim.run_id, claim.task_id, claim.event_id)
            if event in events or claim.credit_kind not in weights:
                raise ValueError("overlapping or unconfigured research contribution")
            events.add(event)
            weighted = claim.score * weights[claim.credit_kind]
            _number(weighted)
            selected.append(claim)
            components.append(
                {
                    "claim_id": claim_id,
                    "kind": claim.credit_kind,
                    "raw": claim.score,
                    "weight": weights[claim.credit_kind],
                    "weighted": weighted,
                }
            )
        try:
            adjustment = fsum(item["weighted"] for item in components)
        except OverflowError:
            raise ValueError("research adjustment overflow") from None
        if abs(adjustment) > policy.max_abs_adjustment:
            raise ValueError("research adjustment exceeds the operator budget")
        overlay_weight = self.pipeline.overlay_weight
        baseline = self.pipeline.score(source)
        if canonical_json(_request_wire(source)) != source_json:
            raise ValueError("baseline provider mutated the detached target")
        _number(baseline.total)
        bound = _baseline_trajectory(source, baseline, overlay_weight)
        reward_components = dict(bound.reward_components)
        evidence = dict(bound.reward_component_evidence)
        provenance = dict(bound.reward_component_provenance)
        for claim, component in zip(selected, components):
            key = "research_credit:" + claim.claim_id
            if key in reward_components:
                raise ValueError("research reward component collision")
            reward_components[key] = component["weighted"]
            evidence[key] = claim.evidence_refs
            provenance[key] = RewardProvenance(
                key,
                "fresh host verification of research contribution",
                VerificationRoute.TRUSTED_EVALUATOR,
                evaluator=claim.evaluator,
            )
        total = baseline.total + adjustment
        _number(total)
        scored = replace(
            bound,
            reward_total=total,
            reward_components=reward_components,
            reward_component_evidence=evidence,
            reward_component_provenance=provenance,
        )
        envelope = {
            "target_request": _request_wire(source),
            "policy": policy.to_dict(),
            "use": selection.to_dict(),
            "audit": audit,
            "baseline_total": baseline.total,
            "baseline_base_reward": baseline.base_reward,
            "integrity_snapshot": _snapshot_integrity(baseline),
            "proposed_trajectory": scored.to_dict(),
            "components": components,
        }
        metric_values = {
            "research_baseline_reward": baseline.total,
            "research_adjustment": adjustment,
        }
        for kind in sorted(_KINDS):
            for field, suffix in (("raw", "raw"), ("weighted", "weighted")):
                try:
                    value = fsum(item[field] for item in components if item["kind"] == kind)
                except OverflowError as error:
                    raise ValueError("research metric overflow") from error
                _number(value)
                metric_values[f"research_{kind}_{suffix}"] = value
        report = {
            "schema_version": "research-reward-decision-v1",
            "metrics": metric_values,
            "training_eligibility": "DEVELOPMENT",
            "authority_granted": False,
            "status": "authorized",
            "baseline_total": baseline.total,
            "adjustment": adjustment,
            "diagnostic_total": total,
            "components": components,
            "review": envelope,
            "receipt_digest": "0" * 64,
            "reasons": ["semantic_review_rejected", "optimizer_use_rejected"],
        }
        # Reserve the full export before any reviewer runs; later substitutions only shrink it.
        validate_input_envelope(report)
        reason = None
        if not _accepted(self.authenticate_claims, envelope):
            reason = "semantic_review_rejected"
        elif not _accepted(self.authorize_optimizer_use, envelope):
            reason = "optimizer_use_rejected"
        if reason is not None:
            report.update(status="withheld", receipt_digest=None, reasons=[reason])
            return _decision(report, scored, source, authorized=False)
        receipt = self.ledger.reserve(
            source,
            submission.claims,
            selection,
            authorize_use=self.authorize_accounting,
            enabled=True,
        )
        report.update(receipt_digest=receipt["receipt_digest"], reasons=[])
        return _decision(report, scored, source, authorized=True)
