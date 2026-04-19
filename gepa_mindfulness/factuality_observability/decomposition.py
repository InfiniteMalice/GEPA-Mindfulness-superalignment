"""Atomic-fact decomposition and localized answer repair helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import (
    AtomicFactRecord,
    HallucinationPrimaryType,
    VerifiabilityClass,
    VerificationStatus,
)


@dataclass(slots=True)
class AtomicDecompositionResult:
    """Per-answer decomposition output for evaluation and repair."""

    atomic_fact_list: list[str]
    evidence_per_fact: list[list[str]]
    fact_verdict_per_fact: list[VerificationStatus]
    clause_to_fact_mapping: dict[int, list[int]]
    repaired_answer: str
    attribution_precision_score: float
    support_coverage_score: float
    unsupported_fact_indices: list[int]
    contradiction_fact_indices: list[int]
    fact_risk_score: float
    per_fact_hallucination_type: list[HallucinationPrimaryType]
    per_fact_verifiability_class: list[VerifiabilityClass]


def split_into_clauses(answer: str) -> list[str]:
    """Split a long answer into rough molecular clauses."""

    parts = re.split(r"(?:\n+|(?<=[.!?;])\s+)", answer.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_clause_with_separators(clause: str) -> tuple[list[str], list[str]]:
    split_parts = re.split(r"(\s+\b(?:and|but|while)\b\s+)", clause)
    facts: list[str] = []
    separators: list[str] = []

    for index, part in enumerate(split_parts):
        if index % 2 == 0:
            fact = part.strip()
            if fact:
                facts.append(fact)
        elif facts:
            separators.append(part)

    while len(separators) < max(0, len(facts) - 1):
        separators.append(" ")
    return facts, separators


def decompose_clauses_to_atomic_facts(clauses: list[str]) -> tuple[list[str], dict[int, list[int]]]:
    """Decompose each clause into rough atomic claims using conjunction splits."""

    facts: list[str] = []
    mapping: dict[int, list[int]] = {}
    for clause_index, clause in enumerate(clauses):
        mini_facts, _separators = _split_clause_with_separators(clause)
        mapping[clause_index] = []
        for mini_fact in mini_facts:
            mapping[clause_index].append(len(facts))
            facts.append(mini_fact)
    return facts, mapping


def verify_atomic_facts(
    atomic_facts: list[str],
    evidence_lookup: dict[str, list[str]],
    contradiction_lookup: set[str] | None = None,
) -> list[AtomicFactRecord]:
    """Attach evidence and lightweight verdicts for each fact."""

    contradiction_lookup = contradiction_lookup or set()
    records: list[AtomicFactRecord] = []
    for fact in atomic_facts:
        evidence = evidence_lookup.get(fact, [])
        if fact in contradiction_lookup:
            verdict = VerificationStatus.CONTRADICTED
            hallucination_type = HallucinationPrimaryType.FACTUAL_ERROR
            risk_score = 1.0
        elif evidence:
            verdict = VerificationStatus.EXTERNALLY_VERIFIED
            hallucination_type = HallucinationPrimaryType.NONE
            risk_score = 0.1
        else:
            verdict = VerificationStatus.UNVERIFIED
            hallucination_type = HallucinationPrimaryType.UNSUPPORTED_BRIDGE
            risk_score = 0.7

        records.append(
            AtomicFactRecord(
                fact=fact,
                evidence=evidence,
                verdict=verdict,
                verifiability_class=VerifiabilityClass.DIRECT,
                hallucination_type=hallucination_type,
                risk_score=risk_score,
            )
        )
    return records


def repair_answer_from_atomic_facts(
    clauses: list[str], mapping: dict[int, list[int]], records: list[AtomicFactRecord]
) -> str:
    """Reconstruct answer while removing unsupported or contradicted fact fragments."""

    kept_clauses: list[str] = []
    for clause_index, clause in enumerate(clauses):
        fact_indices = mapping.get(clause_index, [])
        local_facts, local_separators = _split_clause_with_separators(clause)
        if len(local_facts) != len(fact_indices):
            local_facts = [records[index].fact for index in fact_indices]
            local_separators = [" and "] * max(0, len(local_facts) - 1)

        supported_indices = [
            index
            for index in fact_indices
            if records[index].verdict
            not in {VerificationStatus.UNVERIFIED, VerificationStatus.CONTRADICTED}
        ]
        if not supported_indices and fact_indices:
            continue
        if len(supported_indices) == len(fact_indices):
            kept_clauses.append(clause)
            continue

        if supported_indices:
            local_supported_positions = [
                local_idx
                for local_idx, global_index in enumerate(fact_indices)
                if global_index in supported_indices
            ]
            rebuilt_clause = local_facts[local_supported_positions[0]]
            for pos in local_supported_positions[1:]:
                separator = local_separators[pos - 1] if pos - 1 < len(local_separators) else " "
                rebuilt_clause = f"{rebuilt_clause}{separator}{local_facts[pos]}"
            kept_clauses.append(rebuilt_clause)
    return " ".join(kept_clauses).strip() or "I don't know based on current verifiable evidence."


def decompose_verify_and_repair(
    answer: str,
    evidence_lookup: dict[str, list[str]],
    contradiction_lookup: set[str] | None = None,
) -> AtomicDecompositionResult:
    """Run decomposition, verification, and localized repair in one call."""

    clauses = split_into_clauses(answer)
    atomic_facts, mapping = decompose_clauses_to_atomic_facts(clauses)
    records = verify_atomic_facts(atomic_facts, evidence_lookup, contradiction_lookup)
    repaired_answer = repair_answer_from_atomic_facts(clauses, mapping, records)

    contradicted = [
        i for i, record in enumerate(records) if record.verdict is VerificationStatus.CONTRADICTED
    ]
    unsupported = [
        i for i, record in enumerate(records) if record.verdict is VerificationStatus.UNVERIFIED
    ]
    supported = [
        i
        for i, record in enumerate(records)
        if record.verdict is VerificationStatus.EXTERNALLY_VERIFIED
    ]

    coverage = len(supported) / max(1, len(records))
    attribution_precision = sum(1 for r in records if r.evidence) / max(1, len(records))
    fact_risk_score = sum(record.risk_score for record in records) / max(1, len(records))

    return AtomicDecompositionResult(
        atomic_fact_list=atomic_facts,
        evidence_per_fact=[record.evidence for record in records],
        fact_verdict_per_fact=[record.verdict for record in records],
        clause_to_fact_mapping=mapping,
        repaired_answer=repaired_answer,
        attribution_precision_score=attribution_precision,
        support_coverage_score=coverage,
        unsupported_fact_indices=unsupported,
        contradiction_fact_indices=contradicted,
        fact_risk_score=fact_risk_score,
        per_fact_hallucination_type=[record.hallucination_type for record in records],
        per_fact_verifiability_class=[record.verifiability_class for record in records],
    )
