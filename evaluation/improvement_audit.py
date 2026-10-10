"""Offline improvement accounting; hashes never authenticate host evidence."""

# Standard library
from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

# Third-party
# Local
from .causal_records import content_digest
from .improvement_records import (
    AttemptJournal,
    DatasetManifest,
    ExposureRecord,
    ImprovementProtocol,
    record_digest,
)


def _index(items: tuple, key: str) -> dict[str, Any]:
    result = {}
    for item in items:
        name = getattr(item, key)
        if name in result:
            raise ValueError(f"duplicate {key}")
        result[name] = item
    return result


def _acyclic(parents: dict[str, tuple[str, ...]]) -> None:
    children = defaultdict(list)
    degrees = {name: len(values) for name, values in parents.items()}
    for child, values in parents.items():
        for parent in values:
            if parent not in parents:
                raise ValueError("unknown parent")
            children[parent].append(child)
    ready = deque(name for name, degree in degrees.items() if degree == 0)
    visited = 0
    while ready:
        visited += 1
        for child in children[ready.popleft()]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if visited != len(parents):
        raise ValueError("cyclic ancestry")


def validate_manifest(manifest: DatasetManifest) -> dict[str, str]:
    manifest = DatasetManifest.from_dict(manifest.to_dict())
    cases = _index(manifest.cases, "case_id")
    if not cases:
        raise ValueError("manifest requires cases")
    _acyclic({name: c.parent_ids for name, c in cases.items()})
    roots = {name: name for name in cases}

    def root(name: str) -> str:
        while roots[name] != name:
            roots[name] = roots[roots[name]]
            name = roots[name]
        return name

    def join(left: str, right: str) -> None:
        a, b = sorted((root(left), root(right)))
        roots[b] = a

    owners = {}
    definitions = {content_digest(json.loads(d)) for d in manifest.ood_definitions}
    for name, case in sorted(cases.items()):
        if case.purpose == "ood_combinations" and case.combination_digest not in definitions:
            raise ValueError("OOD case requires a bound combination definition")
        for parent in case.parent_ids:
            join(name, parent)
        keys = [("family", case.family_id), ("content", case.content_digest)]
        keys += [("chain", chain) for chain in case.transformation_ids]
        keys += [("dependency", dep) for dep in case.dependency_ids]
        for key in keys:
            if key in owners:
                join(name, owners[key])
            owners[key] = name
    groups = defaultdict(set)
    result = {name: root(name) for name in sorted(cases)}
    for name, cluster in result.items():
        groups[cluster].add(cases[name].purpose)
    if any(len(purposes) != 1 for purposes in groups.values()):
        raise ValueError("cross-partition lineage, family, content or dependency overlap")
    return result


def partition_digest(manifest: DatasetManifest, purpose: str) -> str:
    """Include ancestor-only nodes, not merely observed cases."""
    return content_digest(
        [
            c.to_dict()
            for c in sorted(manifest.cases, key=lambda c: c.case_id)
            if c.purpose == purpose
        ]
    )


def _protocol_indexes(protocol: ImprovementProtocol) -> tuple[dict, dict, dict]:
    protocol.__post_init__()
    candidates = _index(protocol.candidates, "candidate_id")
    metrics = _index(protocol.metrics, "metric_id")
    slots = _index(protocol.slots, "slot_id")
    _acyclic(
        {
            n: (c.parent_candidate_id,) if c.parent_candidate_id else ()
            for n, c in candidates.items()
        }
    )
    for c in candidates.values():
        if (
            c.parent_candidate_id
            and candidates[c.parent_candidate_id].freeze_ordinal > c.freeze_ordinal
        ):
            raise ValueError("candidate ancestry contradicts freeze ordering")
    coordinates = set()
    for slot in slots.values():
        if slot.candidate_id not in candidates or slot.metric_id not in metrics:
            raise ValueError("slot references unknown candidate or metric")
        coordinate = (
            slot.candidate_id,
            slot.case_id,
            slot.metric_id,
            slot.arm,
            slot.repeat_id,
            slot.condition_id,
        )
        if coordinate in coordinates:
            raise ValueError("duplicate evaluation coordinates")
        coordinates.add(coordinate)
    return candidates, metrics, slots


def summarize_journal(protocol: ImprovementProtocol, journal: AttemptJournal) -> dict[str, Any]:
    protocol = ImprovementProtocol.from_dict(protocol.to_dict())
    journal = AttemptJournal.from_dict(journal.to_dict())
    if journal.protocol_digest != record_digest(protocol):
        raise ValueError("journal protocol binding mismatch")
    candidates, _, slots = _protocol_indexes(protocol)
    _index(journal.events, "event_id")
    proposed, started, finished, decisions = set(), {}, set(), {}
    rounds, candidate_rounds = set(), defaultdict(set)
    previous = -1
    charges = defaultdict(list)
    measured = set()
    for event in journal.events:
        cid = event.candidate_id
        if event.ordinal <= previous or cid not in candidates:
            raise ValueError("invalid event ordering or candidate")
        previous = event.ordinal
        if cid in decisions:
            raise ValueError("event after terminal decision")
        candidate_rounds[cid].add(event.round_id)
        if event.kind == "proposal":
            if cid in proposed:
                raise ValueError("duplicate proposal")
            proposed.add(cid)
        else:
            if cid not in proposed:
                raise ValueError("event before proposal")
            if event.kind == "decision":
                decisions[cid] = event.terminal_status
            else:
                slot = slots.get(event.slot_id)
                if slot is None or slot.candidate_id != cid:
                    raise ValueError("evaluation slot/candidate mismatch")
                if event.ordinal < candidates[cid].freeze_ordinal:
                    raise ValueError("evaluation before candidate freeze")
                if event.kind == "evaluation_started":
                    if slot.slot_id in started:
                        raise ValueError("duplicate evaluation start; use a distinct repeat")
                    started[slot.slot_id] = event
                    rounds.add(event.round_id)
                else:
                    start = started.get(slot.slot_id)
                    if (
                        start is None
                        or slot.slot_id in finished
                        or start.round_id != event.round_id
                    ):
                        raise ValueError("unmatched or repeated evaluation finish")
                    finished.add(slot.slot_id)
        for cost in event.costs:
            charges[(cost.unit, cost.currency, cost.basis)].append(cost.value)
        if event.slot_id and event.costs:
            measured.add(event.slot_id)
    attempted = {e.candidate_id for e in started.values()}
    cost_groups = []
    for (unit, currency, basis), values in sorted(charges.items(), key=lambda item: str(item[0])):
        known = [v for v in values if v is not None]
        cost_groups.append(
            dict(
                unit=unit,
                currency=currency,
                basis=basis,
                known_total=sum(known) if known else None,
                known_count=len(known),
                unknown_count=len(values) - len(known),
                partial=len(known) != len(values),
            )
        )
    return dict(
        candidate_count=len(attempted),
        evaluation_attempt_count=len(started),
        selection_round_count=len(rounds),
        planned_unattempted=sorted(set(candidates) - attempted),
        candidates=[
            dict(
                candidate_id=cid,
                status=decisions.get(cid, "pending"),
                attempted=cid in attempted,
                rounds=sorted(candidate_rounds[cid]),
                spec=candidates[cid].to_dict(),
            )
            for cid in sorted(candidates)
        ],
        starts={name: e.to_dict() for name, e in sorted(started.items())},
        finished=sorted(finished),
        events=[e.to_dict() for e in journal.events],
        cost=dict(groups=cost_groups, unmeasured_attempts=len(set(started) - measured)),
    )


def independence_status(
    protocol: ImprovementProtocol,
    candidate_id: str,
    purpose: str,
    exposures: tuple[ExposureRecord, ...],
    *,
    journal: AttemptJournal | None = None,
) -> dict[str, Any]:
    candidates, _, _ = _protocol_indexes(protocol)
    if candidate_id not in candidates:
        raise ValueError("unknown independence target")
    ancestors = set()
    cursor = candidate_id
    while cursor is not None:
        ancestors.add(cursor)
        cursor = candidates[cursor].parent_candidate_id
    round_members = defaultdict(set)
    if journal is not None:
        for event in journal.events:
            round_members[event.round_id].add(event.candidate_id)
    reasons = []
    for exposure in exposures:
        exposure.__post_init__()
        if set(exposure.candidate_ids) - set(candidates):
            raise ValueError("unknown exposure recipient")
        if set(exposure.round_ids) - set(round_members):
            raise ValueError("exposure round requires matching journal")
        recipients = set(exposure.candidate_ids)
        for round_id in exposure.round_ids:
            recipients.update(round_members[round_id])
        if exposure.purpose == purpose and exposure.use != "evaluation_only":
            if not recipients or recipients & ancestors:
                reasons.append(dict(ordinal=exposure.ordinal, use=exposure.use))
    return dict(independent=not reasons, reasons=reasons, authentication="not_assessed")
