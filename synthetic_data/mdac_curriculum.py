"""Plan development work from diagnostic evidence and export reviewed curricular examples."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from evaluation.causal_records import canonical_json
from gepa_mindfulness.verification.mdac import MDACRecord
from gepa_mindfulness.verification.mdac_metrics import MetricResult

from .cumulative_curriculum import export_cumulative_curriculum
from .curriculum_records import (
    BoundaryCandidate,
    CurationPolicy,
    GenerationReview,
    JsonObject,
    read_curriculum_object,
)
from .lesson_curation import HostCheck, report_document, require_enabled

CHANNELS = (
    "targeted_corrective",
    "retention_replay",
    "novel_generalization",
    "rare_severe",
    "diagnostic_exploration",
    "baseline_capability",
)
# When the budget cannot cover every active channel, protect observed severe failures first.
_RESERVE_ORDER = (CHANNELS[3], CHANNELS[4], CHANNELS[1], CHANNELS[5], CHANNELS[0], CHANNELS[2])
_DEFAULT_WEIGHTS = dict(zip(CHANNELS, (40, 15, 15, 10, 10, 10)))
LESSONS = MappingProxyType(
    {
        "M01": "Retain decisive information across long contexts and compression.",
        "M02": "Retrieve evidence, verify sources, and ground multi-hop answers.",
        "M03": "Check memory applicability and avoid interference from irrelevant lessons.",
        "M04": "Preserve semantic intent across paraphrases and resist intent laundering.",
        "M05": "Discover hidden assumptions and identify missing material decisions.",
        "M06": "Calibrate confidence and abstain or clarify when evidence is insufficient.",
        "M07": "Verify reasoning steps and compare alternative solution paths.",
        "M08": "Track temporal dependencies and delayed outcomes across turns.",
        "M09": "Replay constitutional principles during capability learning.",
        "M10": "Use Sensitive Debate, pluralistic perspectives, and value decomposition.",
        "M11": "Compare PEO predictions, uncertainty, and counterfactual outcomes.",
        "M12": "Apply generalized principles to novel moral dilemmas.",
        "M13": "Test objective invariance and sensitivity to material changes.",
        "M14": "Test proxy robustness against independently verified outcomes.",
        "M15": "Practice positive honesty and transparent correction.",
        "M16": "Cooperate with oversight and distinguish valid from invalid corrections.",
    }
)
_EXTERNAL = {
    "tool": "engineering",
    "runtime": "engineering",
    "harness": "engineering",
    "environment": "engineering",
    "evaluator": "engineering",
    "governance": "governance",
    "authorization": "governance",
}


def _unit(value: float, name: str) -> None:
    if type(value) not in (float, int) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True)
class AllocationPolicy:
    """One budget unit requests one reviewed candidate pair."""

    budget: int = 32
    group_cap: int = 8
    regression_weight: float = 1.0
    severe_threshold: float = 0.8
    rare_frequency: float = 0.1
    confidence_floor: float = 0.5
    weights: Mapping[str, int] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        if type(self.budget) is not int or not 0 <= self.budget <= 1024:
            raise ValueError("budget must be an integer from 0 to 1024")
        if type(self.group_cap) is not int or not 1 <= self.group_cap <= 1024:
            raise ValueError("group_cap must be an integer from 1 to 1024")
        weight = self.regression_weight
        if type(weight) not in (float, int) or not 0 <= weight <= 1024:
            raise ValueError("regression_weight must be finite in [0,1024]")
        for name in ("severe_threshold", "rare_frequency", "confidence_floor"):
            _unit(getattr(self, name), name)
        if set(self.weights) != set(CHANNELS) or any(
            type(w) is not int or not 1 <= w <= 1024 for w in self.weights.values()
        ):
            raise ValueError("weights require a positive integer up to 1024 for each channel")
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def to_dict(self) -> JsonObject:
        return {**vars(self), "weights": dict(self.weights)}


@dataclass(frozen=True, kw_only=True)
class CurriculumCriterion:
    """Host-declared development cohort, failure origin, and one normalized criterion."""

    record: MDACRecord
    mechanism: str | None
    origin: str
    partition: str
    current: MetricResult | None = None
    target: float = 1.0
    previous: MetricResult | None = None
    previous_record: MDACRecord | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        _unit(self.target, "target")
        if self.partition not in ("training", "diagnostic_development"):
            raise ValueError("curriculum requires a training or diagnostic_development partition")
        if type(self.record) is not MDACRecord:
            raise ValueError("record must be an MDAC observation")
        if self.origin not in ("unknown", "model", *_EXTERNAL):
            raise ValueError("unknown failure origin")
        if self.group is not None and (type(self.group) is not str or not self.group.strip()):
            raise ValueError("group requires a nonempty string")
        mechanisms = {a.mechanism for a in self.record.assessments}
        if self.mechanism not in mechanisms and not (self.mechanism is None and not mechanisms):
            raise ValueError("mechanism must be assessed by this diagnostic")
        for metric in (self.current, self.previous):
            if metric is None:
                continue
            if type(metric) is not MetricResult or metric.mechanism_id != self.mechanism:
                raise ValueError("metric mechanism must match the diagnostic assessment")
            if metric.normalized is not None:
                _unit(metric.normalized, "normalized metric")
                if type(metric.denominator) is not int or metric.denominator <= 0:
                    raise ValueError("normalized scores require measured opportunities")
        prior = self.previous_record
        if (self.previous is None) != (prior is None):
            raise ValueError("previous metric and observation must be supplied together")
        if self.previous is not None and prior is not None:
            if (
                type(prior) is not MDACRecord
                or self.current is None
                or self.previous.name != self.current.name
                or self.previous.normalization != self.current.normalization
                or prior.diagnostic_id == self.record.diagnostic_id
                or self.mechanism not in {a.mechanism for a in prior.assessments}
                or self.record.task_id != prior.task_id
                or self.record.case_id != prior.case_id
                or not self.record.dataset_version
                or self.record.dataset_version != prior.dataset_version
                or not self.record.evaluator_version
                or self.record.evaluator_version != prior.evaluator_version
            ):
                raise ValueError(
                    "regression requires comparable observations of the same criterion"
                )

    def to_dict(self) -> JsonObject:
        return {
            **vars(self),
            "record": self.record.to_dict(),
            "previous_record": self.previous_record.to_dict() if self.previous_record else None,
            "current": self.current.to_dict() if self.current else None,
            "previous": self.previous.to_dict() if self.previous else None,
        }


def _score(c: CurriculumCriterion, policy: AllocationPolicy) -> JsonObject:
    record = c.record
    assessment = next((a for a in record.assessments if a.mechanism == c.mechanism), None)
    confidence = assessment.confidence if assessment else None
    freq = record.frequency
    current = c.current.normalized if c.current else None
    previous = c.previous.normalized if c.previous else None
    gap = None if current is None else max(0.0, c.target - current)
    regression = None if current is None or previous is None else max(0.0, previous - current)
    priority = None
    if None not in (record.severity, confidence, freq, gap):
        assert record.severity is not None and confidence is not None
        assert freq is not None and gap is not None
        priority = (
            record.severity
            * confidence
            * freq
            * (gap + policy.regression_weight * (regression or 0.0))
        )
    uncertain = (
        c.origin == "unknown"
        or priority is None
        or assessment is None
        or assessment.state not in ("supported", "strongly_supported", "resolved", "regressed")
        or confidence is None
        or confidence < policy.confidence_floor
        or bool(assessment.contradictory)
    )
    row: JsonObject = dict(
        diagnostic_id=record.diagnostic_id,
        mechanism=c.mechanism,
        group=c.group or record.diagnostic_id,
        origin=c.origin,
        gap=gap,
        regression=regression,
        priority=priority,
        channels=[],
    )
    channels = row["channels"]
    if c.origin in _EXTERNAL:
        return row
    if uncertain:
        channels.append("diagnostic_exploration")
    elif gap == 0 and not regression:
        channels.append("retention_replay")
    elif priority is not None and priority > 0:
        channels.extend(("targeted_corrective", "novel_generalization"))
    else:
        channels.append("diagnostic_exploration")
    if (
        not uncertain
        and freq is not None
        and 0 < freq <= policy.rare_frequency
        and record.severity is not None
        and record.severity >= policy.severe_threshold
    ):
        channels.append("rare_severe")
    return row


def plan_curriculum(
    criteria: tuple[CurriculumCriterion, ...],
    *,
    policy: AllocationPolicy = AllocationPolicy(),
    enabled: bool = False,
) -> JsonObject:
    """Allocate exactly the declared budget while bounding related diagnostic work."""
    require_enabled(enabled)
    if type(policy) is not AllocationPolicy:
        raise ValueError("expected AllocationPolicy")
    if type(criteria) is not tuple or len(criteria) > 1024:
        raise ValueError("criteria must be a tuple of at most 1024 records")
    if any(type(c) is not CurriculumCriterion for c in criteria):
        raise ValueError("expected CurriculumCriterion")
    ordered = sorted(criteria, key=lambda c: (c.record.diagnostic_id, c.mechanism or ""))
    if len({(c.record.diagnostic_id, c.mechanism) for c in ordered}) != len(ordered):
        raise ValueError("duplicate diagnostic and mechanism pair")
    rows = [_score(c, policy) for c in ordered]
    # Multiple labels from one diagnostic must share a cap and the original evidence.
    sources: dict[str, tuple[str, MDACRecord]] = {}
    pools: dict[str, dict[str, list[int]]] = {ch: defaultdict(list) for ch in CHANNELS}
    for i, (row, criterion) in enumerate(zip(rows, ordered)):
        source = row["group"], criterion.record
        if sources.setdefault(row["diagnostic_id"], source) != source:
            raise ValueError("one diagnostic must bind one correlation group and observation")
        for channel in row["channels"]:
            pools[channel][row["group"]].append(i)

    counts = dict.fromkeys(CHANNELS, 0)
    group_counts: Counter[str] = Counter()
    visits: Counter[int] = Counter()
    slots = []
    for index in range(policy.budget):
        available = [
            ch
            for ch in _RESERVE_ORDER
            if ch == "baseline_capability"
            or any(group_counts[g] < policy.group_cap for g in pools[ch])
        ]
        # Reserve one place per active channel, then allocate by relative channel weight.
        channel = next((ch for ch in available if counts[ch] == 0), None)
        if channel is None:
            channel = min(available, key=lambda ch: (counts[ch] + 1) / policy.weights[ch])
        slot: JsonObject = dict(
            index=index, channel=channel, group=None, mechanism=None, diagnostic_id=None
        )
        slot["lesson"] = "Maintain baseline capability on reviewed development tasks."
        if channel != "baseline_capability":
            pool = pools[channel]

            def weight(i: int) -> float:
                if channel in ("targeted_corrective", "novel_generalization"):
                    return rows[i]["priority"]
                return 1.0

            # Max pooling prevents duplicating labels from increasing a group's priority.
            group = min(
                (g for g in pool if group_counts[g] < policy.group_cap),
                key=lambda g: (-max(weight(i) for i in pool[g]) / (1 + group_counts[g]), g),
            )
            chosen = min(pool[group], key=lambda i: -weight(i) / (1 + visits[i]))
            row = rows[chosen]
            slot.update({k: row[k] for k in ("group", "diagnostic_id", "mechanism")})
            slot["lesson"] = LESSONS.get(row["mechanism"], "Identify discriminating evidence.")
            group_counts[group] += 1
            visits[chosen] += 1
        slots.append(slot)
        counts[channel] += 1

    routes = [
        dict(
            diagnostic_id=row["diagnostic_id"],
            mechanism=row["mechanism"],
            destination=_EXTERNAL[row["origin"]],
            origin=row["origin"],
        )
        for row in rows
        if row["origin"] in _EXTERNAL
    ]
    return report_document(
        "mdac-allocation-v1",
        dict(policy=policy.to_dict(), criteria=[c.to_dict() for c in ordered]),
        criteria=rows,
        routes=routes,
        channel_counts=counts,
        slots=slots,
    )


Generated = tuple[BoundaryCandidate, GenerationReview | None]


def generate_curriculum(
    criteria: tuple[CurriculumCriterion, ...],
    *,
    generate: Callable[[JsonObject], Generated | None],
    curation: CurationPolicy,
    as_of: str,
    policy: AllocationPolicy = AllocationPolicy(),
    authenticate: HostCheck | None = None,
    enabled: bool = False,
) -> JsonObject:
    """Request candidates per planned slot, then pass them through normal reviewed export."""
    plan = plan_curriculum(criteria, policy=policy, enabled=enabled)
    bindings: dict[str, JsonObject] = {}
    candidates, reviews = [], []
    generated = dict.fromkeys(CHANNELS, 0)
    exported = dict.fromkeys(CHANNELS, 0)
    for slot in plan["slots"]:
        request = dict(slot, plan_digest=plan["result_digest"])
        result = generate(dict(request))
        if result is None:
            continue
        candidate, review = result
        if type(candidate) is not BoundaryCandidate or candidate.candidate_id in bindings:
            raise ValueError("every generated slot requires a unique boundary candidate")
        config = read_curriculum_object(candidate.generator_config_json)
        if canonical_json(config.get("mdac_request")) != canonical_json(request):
            raise ValueError("candidate must bind the exact allocation request")
        candidates.append(candidate)
        if review is not None:
            reviews.append(review)
        bindings[candidate.candidate_id] = request
        generated[slot["channel"]] += 1
    export = export_cumulative_curriculum(
        tuple(candidates),
        tuple(reviews),
        policy=curation,
        as_of=as_of,
        authenticate=authenticate,
        enabled=True,
    )
    for example in export["examples"]:
        exported[bindings[example["candidate_id"]]["channel"]] += 1
    return report_document(
        "mdac-generated-curriculum-v1",
        dict(plan_digest=plan["result_digest"], export_digest=export["result_digest"]),
        plan=plan,
        export=export,
        bindings=bindings,
        generated_counts=generated,
        exported_counts=exported,
    )
