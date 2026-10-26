"""Opt-in diagnostic accounting; no reward, admission or execution authority."""

# Standard library
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from math import isfinite

# Local
from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import RewardProvenance, TrustedEvaluatorContract

from .causal_reward_records import _json_value, validate_input_envelope
from .eligibility import require_training_eligible
from .trajectory import _reward_provenance_from_dict, _reward_provenance_to_dict

_SCHEMA = "research-credit-claim-v1"
_TEXT_FIELDS = (
    "claim_id",
    "method_id",
    "component_name",
    "source_reference",
    "authority_domain",
    "run_id",
    "task_id",
    "event_id",
    "action_id",
    "model_version",
    "harness_version",
    "producer_id",
)
_KINDS = {"knowledge", "progress", "process", "teacher_agreement", "diagnostic"}


def _text(value: object) -> None:
    """Require a bounded exact UTF-8 identifier without coercion."""
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 1024:
        raise ValueError("expected nonblank text of at most 1024 UTF-8 bytes")


def _evaluator(value: object) -> TrustedEvaluatorContract:
    """Snapshot the existing evaluator identity contract after validating every field."""
    if type(value) is not TrustedEvaluatorContract:
        raise ValueError("expected exact TrustedEvaluatorContract")
    for name in ("evaluator_id", "evaluator_version", "contract_id"):
        _text(getattr(value, name))
    return TrustedEvaluatorContract(value.evaluator_id, value.evaluator_version, value.contract_id)


@dataclass(frozen=True)
class ResearchCreditClaim:
    """One caller-declared contribution proposal, always DEVELOPMENT evidence.

    Event IDs are host-normalized semantic identities, not inferred from evidence IDs.
    Provenance validates structure; hosts still authenticate issuers and evidence meaning.
    Source metadata is retained as bounded canonical JSON, never relabeled as training data.
    """

    claim_id: str
    method_id: str
    component_name: str
    credit_kind: str
    source_reference: str
    authority_domain: str
    run_id: str
    task_id: str
    event_id: str
    action_id: str
    model_version: str
    harness_version: str
    producer_id: str
    evaluator: TrustedEvaluatorContract
    dataset_digest: str
    state_digest: str
    evidence_refs: tuple[EvidenceReference, ...]
    score: float
    admissibility: str
    provenance: RewardProvenance | None
    source_metadata_json: str

    def __post_init__(self) -> None:
        """Validate bindings and freeze nested inputs as independent snapshots."""
        self._validate()
        object.__setattr__(self, "evaluator", _evaluator(self.evaluator))
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(EvidenceReference.from_dict(ref.to_dict()) for ref in self.evidence_refs),
        )
        if self.provenance is not None:
            object.__setattr__(
                self,
                "provenance",
                _reward_provenance_from_dict(_reward_provenance_to_dict(self.provenance)),
            )
        object.__setattr__(
            self, "source_metadata_json", canonical_json(_json_value(self.source_metadata_json))
        )

    def _validate(self) -> None:
        """Reject malformed provenance without authenticating its semantic claims."""
        for name in _TEXT_FIELDS:
            _text(getattr(self, name))
        if type(self.credit_kind) is not str or self.credit_kind not in _KINDS:
            raise ValueError("invalid credit_kind")
        if type(self.admissibility) is not str or self.admissibility not in {
            "admitted",
            "rejected",
            "unresolved",
        }:
            raise ValueError("invalid admissibility")
        for value in (self.dataset_digest, self.state_digest):
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("expected lowercase SHA-256 digest")
        if type(self.score) not in (int, float):
            raise ValueError("score requires finite built-in number")
        try:
            finite = isfinite(self.score)
        except OverflowError:
            finite = False
        if not finite or (self.credit_kind == "process" and self.score < 0):
            raise ValueError("nonfinite score or negative process credit")
        evaluator = _evaluator(self.evaluator)
        if self.producer_id == evaluator.evaluator_id:
            raise ValueError("producer cannot declare itself independent evaluator")
        if type(self.evidence_refs) is not tuple or not 1 <= len(self.evidence_refs) <= 256:
            raise ValueError("expected 1..256 evidence references")
        refs = []
        for ref in self.evidence_refs:
            if type(ref) is not EvidenceReference:
                raise ValueError("expected exact EvidenceReference")
            snapshot = EvidenceReference.from_dict(ref.to_dict())
            _text(snapshot.reference_id)
            if not snapshot.is_observable:
                raise ValueError("private evidence is not credit provenance")
            refs.append(snapshot)
        if len({ref.reference_id for ref in refs}) != len(refs):
            raise ValueError("duplicate evidence identity")
        if self.provenance is not None:
            if type(self.provenance) is not RewardProvenance:
                raise ValueError("expected exact RewardProvenance")
            # Check every raw field before the route-specific serializer can omit any.
            RewardProvenance(
                self.provenance.component_name,
                self.provenance.verification_method,
                self.provenance.route,
                self.provenance.evidence_refs,
                self.provenance.evaluator,
                self.provenance.public_rationale_comparison,
            )
            provenance = _reward_provenance_from_dict(_reward_provenance_to_dict(self.provenance))
            if provenance.component_name != self.component_name:
                raise ValueError("provenance component mismatch")
            if not set(provenance.evidence_refs).issubset(refs):
                raise ValueError("provenance evidence outside claim")
            if provenance.evaluator is not None and provenance.evaluator != evaluator:
                raise ValueError("provenance evaluator mismatch")
        metadata = _json_value(self.source_metadata_json)
        if type(metadata) is not dict:
            raise ValueError("source metadata must be a JSON object")

    def to_dict(self) -> dict[str, object]:
        """Return detached versioned audit data; revalidate nested caller-owned records."""
        self._validate()
        wire = {field.name: getattr(self, field.name) for field in fields(self)}
        wire.update(
            schema_version=_SCHEMA,
            training_eligibility="DEVELOPMENT",
            evaluator={
                "evaluator_id": self.evaluator.evaluator_id,
                "evaluator_version": self.evaluator.evaluator_version,
                "contract_id": self.evaluator.contract_id,
            },
            evidence_refs=[ref.to_dict() for ref in self.evidence_refs],
            provenance=(
                None if self.provenance is None else _reward_provenance_to_dict(self.provenance)
            ),
        )
        validate_input_envelope(wire)
        return wire

    @classmethod
    def from_dict(cls, value: object) -> ResearchCreditClaim:
        """Restore only exact v1 DEVELOPMENT claims, never a live authorization."""
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if type(value) is not dict or set(value) != names | {
            "schema_version",
            "training_eligibility",
        }:
            raise ValueError("unexpected or missing claim fields")
        if value["schema_version"] != _SCHEMA or value["training_eligibility"] != "DEVELOPMENT":
            raise ValueError("unsupported schema or training relabel")
        raw_evaluator = value["evaluator"]
        if type(raw_evaluator) is not dict or set(raw_evaluator) != {
            "evaluator_id",
            "evaluator_version",
            "contract_id",
        }:
            raise ValueError("invalid evaluator fields")
        if type(value["evidence_refs"]) is not list:
            raise ValueError("wire evidence must be an array")
        values = {name: value[name] for name in names}
        values["evaluator"] = TrustedEvaluatorContract(**raw_evaluator)
        values["evidence_refs"] = tuple(
            EvidenceReference.from_dict(ref) for ref in value["evidence_refs"]
        )
        values["provenance"] = (
            None
            if value["provenance"] is None
            else _reward_provenance_from_dict(value["provenance"])
        )
        return cls(**values)


def audit_research_credit(
    claims: tuple[ResearchCreditClaim, ...], *, enabled: bool = False
) -> dict[str, object] | None:
    """Report overlap and unmet prerequisites without computing or admitting reward.

    Disabled calls do not inspect claims. Semantic aliases, issuer authentication,
    evidence integrity and actual admission remain host responsibilities. This is
    transaction-local accounting, not a durable replay ledger or runtime gate.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(claims) is not tuple or not 1 <= len(claims) <= 1024:
        raise ValueError("expected 1..1024 claims")
    snapshots = []
    snapshot_bytes = 0
    for claim in claims:
        if type(claim) is not ResearchCreditClaim:
            raise ValueError("expected exact ResearchCreditClaim")
        wire = claim.to_dict()
        snapshot_bytes += len(canonical_json(wire).encode("utf-8"))
        if snapshot_bytes > 4 * 1024 * 1024:
            raise ValueError("aggregate claim snapshots exceed 4 MiB")
        snapshots.append(ResearchCreditClaim.from_dict(wire))
    if len({claim.claim_id for claim in snapshots}) != len(snapshots):
        raise ValueError("duplicate claim identity")
    events = defaultdict(list)
    evidence = defaultdict(list)
    bindings = {}
    gaps = {}
    for claim in snapshots:
        key = (claim.authority_domain, claim.run_id, claim.task_id, claim.event_id)
        binding = (claim.action_id, claim.model_version, claim.harness_version, claim.state_digest)
        if key in bindings and bindings[key] != binding:
            raise ValueError("same semantic event has contradictory action/system binding")
        bindings[key] = binding
        events[key].append(claim.claim_id)
        for ref in claim.evidence_refs:
            evidence[(claim.authority_domain, claim.run_id, ref.reference_id)].append(
                claim.claim_id
            )
        reasons = []
        if claim.admissibility != "admitted":
            reasons.append(
                "unresolved_admissibility"
                if claim.admissibility == "unresolved"
                else "admissibility_rejected"
            )
        if claim.provenance is None:
            reasons.append("unverified")
        if claim.credit_kind in {"diagnostic", "teacher_agreement"}:
            reasons.append("non_authoritative_signal")
        source = _json_value(claim.source_metadata_json)
        try:
            if source.get("training_eligibility") != "TRAIN":
                raise ValueError("explicit source TRAIN declaration required")
            require_training_eligible(source)
        except ValueError:
            reasons.append("source_ineligible")
        gaps[claim.claim_id] = reasons
    report = {
        "schema_version": "research-credit-audit-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "requires_host_review": True,
        "claims": [claim.to_dict() for claim in snapshots],
        "prerequisite_gaps": gaps,
        "event_overlaps": sorted(sorted(ids) for ids in events.values() if len(ids) > 1),
        "shared_evidence": sorted(
            {tuple(sorted(ids)) for ids in evidence.values() if len(ids) > 1}
        ),
    }
    report["shared_evidence"] = [list(ids) for ids in report["shared_evidence"]]
    validate_input_envelope(report)
    return report
