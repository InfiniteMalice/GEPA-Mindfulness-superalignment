"""Exact evaluator-only records for the text-proxy knowledge-transfer experiment."""

# Standard library
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.verification.debate_records import _digest
from gepa_mindfulness.verification.diagnostic_records import (
    DiagnosticRecord,
    choice,
    public_refs,
    restore_refs,
    strings,
)

from .improvement_records import CostEntry, ResamplingPolicy, SystemConfig

CHANNELS = ("source", "target", "application")
MAX_UNITS = 64
MAX_CELLS = 1536
NORMALIZATION = "nfc-casefold-space-terminal-period-v1"


def bounded(value: object, name: str, maximum: int = 512) -> None:
    """Bound inert text coordinates without interpreting source instructions."""
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"bounded nonempty {name} required")


def identifiers(value: object, name: str, maximum: int = MAX_UNITS) -> tuple[str, ...]:
    """Reject generic iterable coercion before restoring immutable identifier arrays."""
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError(f"bounded array required for {name}")
    result = strings(value, name)
    for item in result:
        bounded(item, name, 128)
    return result


def references(value: object, *, restore: bool = False) -> tuple[EvidenceReference, ...]:
    """Retain bounded observable source handles without certifying their contents."""
    if type(value) not in (tuple, list) or not 1 <= len(value) <= 16:
        raise ValueError("1..16 observable references required")
    result = public_refs(restore_refs(value) if restore else value, required=True)
    for ref in result:
        bounded(ref.reference_id, "reference_id")
    return result


def system_snapshot(value: object) -> SystemConfig:
    """Reuse the existing exact model/harness/configuration identity contract."""
    if type(value) is not SystemConfig:
        raise ValueError("exact SystemConfig required")
    return SystemConfig.from_dict(value.to_dict())


def identity_text(value: str) -> str:
    """Recognize ambiguous case/space/canonical-Unicode spellings without guessing aliases."""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def normalize_answer(value: str) -> str:
    """Normalize public final answers conservatively, preserving internal punctuation."""
    return identity_text(value).rstrip(".。").strip()


def _literal_leak(value: str, question: str) -> bool:
    """Detect only literal value-token sequences, never claim semantic isolation."""
    expected = re.findall(r"\w+", normalize_answer(value))
    tokens = re.findall(r"\w+", normalize_answer(question))
    if not expected:
        return normalize_answer(value) in normalize_answer(question)
    return any(tokens[i : i + len(expected)] == expected for i in range(len(tokens)))


def cost_entries(value: object, *, restore: bool = False) -> tuple[CostEntry, ...]:
    """Reuse exact cost receipts and refuse incompatible or unsupported-scale records."""
    if type(value) not in (tuple, list) or len(value) > 8:
        raise ValueError("at most eight cost entries required")
    result = []
    for entry in value:
        if not restore and type(entry) is not CostEntry:
            raise ValueError("exact CostEntry required")
        item = CostEntry.from_dict(entry if restore else entry.to_dict())
        if item.value is not None and item.value > 1e12:
            raise ValueError("unsupported cost magnitude")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class TransferProbeConfig(DiagnosticRecord):
    """Disabled evaluator settings; dependence declarations do not authenticate independence."""

    enabled: bool = False
    dependencies_known: bool = False
    resampling_policy: ResamplingPolicy = ResamplingPolicy(0, 1000, minimum_clusters=2)
    schema_version = "knowledge-transfer-config-v1"
    restorers = {"resampling_policy": ResamplingPolicy.from_dict}

    def __post_init__(self) -> None:
        """Bound conditional bootstrap work without changing the shared resampling policy."""
        if type(self.enabled) is not bool or type(self.dependencies_known) is not bool:
            raise ValueError("explicit boolean settings required")
        if type(self.resampling_policy) is not ResamplingPolicy:
            raise ValueError("exact ResamplingPolicy required")
        policy = ResamplingPolicy.from_dict(self.resampling_policy.to_dict())
        if policy.resamples > 2000:
            raise ValueError("at most 2000 bootstrap resamples supported")
        object.__setattr__(self, "resampling_policy", policy)


@dataclass(frozen=True)
class TransferUnit(DiagnosticRecord):
    """Host-authored fictional truth and sealed QA, never a training example."""

    unit_id: str
    entity_id: str
    attribute: str
    cohort: str
    value: str
    source_question: str
    target_question: str
    application_question: str
    application_answer: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version = "knowledge-transfer-unit-v1"
    restorers = {"evidence_refs": lambda v: references(v, restore=True)}

    def __post_init__(self) -> None:
        """Keep public questions distinct from evaluator truth and retain isolation uncertainty."""
        for name in ("unit_id", "entity_id", "attribute"):
            bounded(getattr(self, name), name, 128)
        choice(self.cohort, "cohort", ("alignment", "test"))
        for name in ("value", "application_answer"):
            bounded(getattr(self, name), name, 256)
            if not normalize_answer(getattr(self, name)):
                raise ValueError("reference answer normalizes to empty")
        for name in ("source_question", "target_question", "application_question"):
            bounded(getattr(self, name), name)
        if any(_literal_leak(self.value, q) for q in (self.source_question, self.target_question)):
            raise ValueError("literal answer value in recall question")
        object.__setattr__(self, "evidence_refs", references(self.evidence_refs))


@dataclass(frozen=True)
class TransferCondition(DiagnosticRecord):
    """Declared checkpoint/exposure provenance, not evidence of admission or acquired knowledge."""

    condition_id: str
    system: SystemConfig
    base_checkpoint_digest: str
    checkpoint_digest: str
    acquisition_manifest_digest: str | None
    acquired_unit_ids: tuple[str, ...]
    alignment_manifest_digest: str | None
    alignment_entities: tuple[str, ...]
    source_representation: str
    target_representation: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version = "knowledge-transfer-condition-v1"
    restorers = {
        "system": SystemConfig.from_dict,
        "acquired_unit_ids": lambda v: identifiers(v, "acquired_unit_ids"),
        "alignment_entities": lambda v: identifiers(v, "alignment_entities"),
        "evidence_refs": lambda v: references(v, restore=True),
    }

    def __post_init__(self) -> None:
        """Require explicit manifests for declared exposure while granting no new authority."""
        for name in ("condition_id", "source_representation", "target_representation"):
            bounded(getattr(self, name), name, 128)
        object.__setattr__(self, "system", system_snapshot(self.system))
        for name in ("base_checkpoint_digest", "checkpoint_digest"):
            _digest(getattr(self, name))
        for ids_name, manifest_name in (
            ("acquired_unit_ids", "acquisition_manifest_digest"),
            ("alignment_entities", "alignment_manifest_digest"),
        ):
            values = identifiers(getattr(self, ids_name), ids_name)
            object.__setattr__(self, ids_name, values)
            manifest = getattr(self, manifest_name)
            if bool(values) != (manifest is not None):
                raise ValueError("declared exposures and manifest presence must match")
            if manifest is not None:
                _digest(manifest)
        object.__setattr__(self, "evidence_refs", references(self.evidence_refs))


@dataclass(frozen=True)
class TransferObservation(DiagnosticRecord):
    """A captured public final answer/status; models cannot supply a correctness label."""

    condition_id: str
    unit_id: str
    repeat_id: str
    channel: str
    system: SystemConfig
    status: str
    final_answer: str | None
    evidence_refs: tuple[EvidenceReference, ...]
    costs: tuple[CostEntry, ...] = ()
    schema_version = "knowledge-transfer-observation-v1"
    restorers = {
        "system": SystemConfig.from_dict,
        "evidence_refs": lambda v: references(v, restore=True),
        "costs": lambda v: cost_entries(v, restore=True),
    }

    def __post_init__(self) -> None:
        """Preserve abstention/errors separately and reject invented correctness extensions."""
        for name in ("condition_id", "unit_id", "repeat_id"):
            bounded(getattr(self, name), name, 128)
        choice(self.channel, "channel", CHANNELS)
        choice(self.status, "status", ("answered", "abstained", "error"))
        object.__setattr__(self, "system", system_snapshot(self.system))
        if self.status == "answered":
            bounded(self.final_answer, "final_answer", 4096)
        elif self.final_answer is not None:
            raise ValueError("non-answer statuses require null final_answer")
        object.__setattr__(self, "evidence_refs", references(self.evidence_refs))
        object.__setattr__(self, "costs", cost_entries(self.costs))
