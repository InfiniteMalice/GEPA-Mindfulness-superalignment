"""Default-off family planning and journal diagnostics over existing improvement contracts."""

# Standard library
from __future__ import annotations

from collections import defaultdict

# Third-party
# Local
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope

from .causal_records import canonical_json, content_digest
from .family_search_records import (
    FamilyMembership,
    FamilySearchConfig,
    FamilySearchPlan,
    identity_key,
    unique_ids,
)
from .improvement_audit import _protocol_indexes, summarize_journal, validate_manifest
from .improvement_records import (
    AttemptJournal,
    CandidateSpec,
    DatasetManifest,
    EvaluationSlot,
    MetricSpec,
    record_digest,
)


def _config(value: FamilySearchConfig) -> FamilySearchConfig:
    """Snapshot only the operator flag before considering task or evidence inputs."""
    if type(value) is not FamilySearchConfig:
        raise ValueError("exact FamilySearchConfig required")
    return FamilySearchConfig.from_dict(value.to_dict())


def _bounded(value: dict, maximum: int = 4 * 1024 * 1024) -> None:
    """Use the existing JSON boundary and a smaller plan-specific size bound."""
    validate_input_envelope(value)
    if len(canonical_json(value).encode("utf-8")) > maximum:
        raise ValueError("family-search payload exceeds byte capacity")


def _lineage(
    plan: FamilySearchPlan, candidates: dict[str, CandidateSpec]
) -> tuple[dict[str, FamilyMembership], list[dict]]:
    """Validate scheduled single-parent transfers without inventing a second ancestry graph."""
    members = {m.candidate_id: m for m in plan.members}
    families = {f.family_id for f in plan.families}
    if set(members) != set(candidates) or any(m.family_id not in families for m in plan.members):
        raise ValueError("exact known candidate/family membership required")
    rounds, reverse, roots, transfers = {}, {}, set(), []
    for member in plan.members:
        previous = rounds.setdefault(member.round_index, member.round_id)
        index = reverse.setdefault(identity_key(member.round_id), member.round_index)
        if previous != member.round_id or index != member.round_index:
            raise ValueError("round index/identity mismatch")
        candidate = candidates[member.candidate_id]
        if candidate.parent_candidate_id is None:
            if member.round_index == 0:
                roots.add(member.family_id)
            continue
        parent = candidates[candidate.parent_candidate_id]
        source = members[parent.candidate_id]
        if (
            source.round_index >= member.round_index
            or parent.freeze_ordinal >= candidate.freeze_ordinal
        ):
            raise ValueError("parent round/freeze must strictly precede descendant")
        if source.family_id != member.family_id:
            offset = member.round_index - plan.independent_rounds
            if offset < 0 or offset % plan.sharing_interval:
                raise ValueError("cross-family transfer outside declared sharing schedule")
            transfers.append(
                dict(
                    candidate_id=member.candidate_id,
                    donor_candidate_id=parent.candidate_id,
                    from_family=source.family_id,
                    to_family=member.family_id,
                    round_index=member.round_index,
                    round_id=member.round_id,
                )
            )
    if roots != families:
        raise ValueError("every family requires an independent round-zero root")
    return members, transfers


def _rosters(
    plan: FamilySearchPlan,
    manifest: DatasetManifest,
    candidates: dict[str, CandidateSpec],
    metrics: dict[str, MetricSpec],
    slots: dict[str, EvaluationSlot],
) -> frozenset[tuple[str, str, int, int]]:
    """Require complete pairs and identical planned task/seed/budget coverage per candidate."""
    if not 1 <= len(manifest.cases) <= 128:
        raise ValueError("bounded existing task manifest required")
    cases = {c.case_id: c for c in manifest.cases}
    expected_cases = {
        c.case_id for c in manifest.cases if c.evaluated and c.purpose != "synthetic_training"
    }
    grouped = defaultdict(dict)
    for slot in slots.values():
        case = cases.get(slot.case_id)
        if case is None or not case.evaluated or case.purpose == "synthetic_training":
            raise ValueError("slots require declared evaluated non-training tasks")
        if (
            slot.seed_policy != "shared"
            or slot.seed is None
            or not 0 <= slot.seed < 2**32
            or not 0 <= slot.repeat_id <= 7
        ):
            raise ValueError("shared bounded seed and repeat coordinates required")
        if slot.budget_digest != plan.inference_budget_digest:
            raise ValueError("evaluation budget differs from fixed commitment")
        key = slot.case_id, slot.metric_id, slot.repeat_id, slot.seed
        arms = grouped[slot.candidate_id].setdefault(key, {})
        if slot.arm in arms:
            raise ValueError("duplicate paired coordinate")
        arms[slot.arm] = slot
    baseline = next(iter(candidates.values())).baseline
    roster = None
    for candidate in candidates.values():
        if candidate.baseline != baseline:
            raise ValueError("all candidates require the common baseline")
        own = grouped[candidate.candidate_id]
        for arms in own.values():
            if set(arms) != {"baseline", "candidate"}:
                raise ValueError("incomplete baseline/candidate pair")
            if arms["baseline"].condition_id != arms["candidate"].condition_id:
                raise ValueError("paired conditions differ")
        keys = frozenset(own)
        if (
            not keys
            or {k[0] for k in keys} != expected_cases
            or {k[1] for k in keys} != set(metrics)
        ):
            raise ValueError("complete declared task/metric roster required")
        if roster is not None and keys != roster:
            raise ValueError("candidate comparison rosters differ")
        roster = keys
    seeds = {key[3] for key in roster}
    if not 2 <= len(seeds) <= 16:
        raise ValueError("requires 2..16 shared seeds")
    seed_rosters = {frozenset(k[:3] for k in roster if k[3] == seed) for seed in seeds}
    if len(seed_rosters) != 1:
        raise ValueError("task/metric/repeat coverage differs across seeds")
    return roster


def prepare_family_search(
    plan: FamilySearchPlan,
    manifest: DatasetManifest,
    *,
    config: FamilySearchConfig = FamilySearchConfig(),
) -> dict | None:
    """Validate an optional research plan without creating candidates, slots or authority.

    Surface names, ancestry and hashes are declarations. Existing host controls authenticate
    their truth, protect evaluation partitions and determine eligibility of actual changes.
    No evaluator, generation callback, model runtime or acceptance policy is added here.
    """
    if not _config(config).enabled:
        return None
    if type(plan) is not FamilySearchPlan or type(manifest) is not DatasetManifest:
        raise ValueError("exact family plan and existing task manifest required")
    plan = FamilySearchPlan.from_dict(plan.to_dict())
    manifest = DatasetManifest.from_dict(manifest.to_dict())
    _bounded(plan.to_dict(), 1024 * 1024)
    _bounded(manifest.to_dict())
    if plan.protocol.manifest_digest != record_digest(manifest):
        raise ValueError("task manifest/protocol binding mismatch")
    validate_manifest(manifest)
    candidates, metrics, slots = _protocol_indexes(plan.protocol)
    unique_ids(tuple(candidates))
    members, transfers = _lineage(plan, candidates)
    roster = _rosters(plan, manifest, candidates, metrics, slots)
    families = []
    for family in plan.families:
        own = [m.candidate_id for m in plan.members if m.family_id == family.family_id]
        count = sum(s.candidate_id in own for s in slots.values())
        if len(own) > plan.max_candidates_per_family or count > plan.max_evaluations_per_family:
            raise ValueError("per-family candidate/evaluation cap exceeded")
        families.append(
            dict(
                family_id=family.family_id,
                surface=family.surface,
                hypothesis=family.hypothesis,
                candidate_ids=own,
                planned_evaluations=count,
                root_ids=[
                    cid
                    for cid in own
                    if candidates[cid].parent_candidate_id is None and members[cid].round_index == 0
                ],
            )
        )
    result = dict(
        schema_version="family-search-preparation-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        promotion_eligible=False,
        execute_runtime=False,
        model_effectiveness_established=False,
        comparability="matched_planned_roster_and_caps_only",
        plan=plan.to_dict(),
        protocol_digest=record_digest(plan.protocol),
        manifest_digest=record_digest(manifest),
        plan_digest=record_digest(plan),
        families=families,
        transfers=transfers,
        paired_cells_per_candidate=len(roster),
    )
    result["result_digest"] = content_digest(result)
    _bounded(result)
    return result


def summarize_family_search(
    plan: FamilySearchPlan,
    manifest: DatasetManifest,
    journal: AttemptJournal,
    *,
    config: FamilySearchConfig = FamilySearchConfig(),
) -> dict | None:
    """Retain declared attempt/cost histories by family without scoring or selecting outcomes.

    Existing audit_improvement independently authenticates diagnostic captures and exposure
    history. This report neither consumes outcome flags nor certifies admissibility, metering,
    independence, success, runtime authorization, optimizer feedback or promotion.
    """
    if not _config(config).enabled:
        return None
    prepared = prepare_family_search(plan, manifest, config=config)
    plan = FamilySearchPlan.from_dict(prepared["plan"])
    if type(journal) is not AttemptJournal or len(journal.events) > 4096:
        raise ValueError("bounded existing attempt journal required")
    journal = AttemptJournal.from_dict(journal.to_dict())
    _bounded(journal.to_dict())
    summary = summarize_journal(plan.protocol, journal)
    members = {m.candidate_id: m for m in plan.members}
    candidates = {c.candidate_id: c for c in plan.protocol.candidates}
    proposed, last_round = set(), -1
    for event in journal.events:
        if event.kind != "proposal":
            continue
        member = members[event.candidate_id]
        if event.round_id != member.round_id or member.round_index < last_round:
            raise ValueError("proposal journal contradicts declared round schedule")
        parent_id = candidates[event.candidate_id].parent_candidate_id
        if parent_id is not None and (
            parent_id not in proposed or event.ordinal < candidates[parent_id].freeze_ordinal
        ):
            raise ValueError("child proposal precedes recorded/frozen parent")
        proposed.add(event.candidate_id)
        last_round = member.round_index
    families = []
    for family in prepared["families"]:
        ids = set(family["candidate_ids"])
        own_events = tuple(e for e in journal.events if e.candidate_id in ids)
        own = summarize_journal(
            plan.protocol,
            AttemptJournal(
                journal.journal_id, journal.protocol_digest, own_events, journal.evidence_refs
            ),
        )
        families.append(
            dict(
                **family,
                proposal_count=sum(e.kind == "proposal" for e in own_events),
                attempted_candidate_count=own["candidate_count"],
                evaluation_attempt_count=own["evaluation_attempt_count"],
                completed_evaluations=len(own["finished"]),
                incomplete_evaluations=len(own["starts"]) - len(own["finished"]),
                planned_unattempted=[cid for cid in own["planned_unattempted"] if cid in ids],
                statuses={
                    c["candidate_id"]: c["status"]
                    for c in own["candidates"]
                    if c["candidate_id"] in ids
                },
                cost=own["cost"],
            )
        )
    result = dict(
        schema_version="family-search-report-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        promotion_eligible=False,
        execute_runtime=False,
        model_effectiveness_established=False,
        preparation_digest=prepared["result_digest"],
        plan=prepared["plan"],
        protocol_digest=prepared["protocol_digest"],
        families=families,
        transfers=prepared["transfers"],
        journal=summary,
        outcome_qualification="not_assessed_use_existing_independent_audit",
        cost_scope="provided_journal_entries_only",
    )
    result["result_digest"] = content_digest(result)
    _bounded(result)
    return result
