"""Opt-in, bounded defensive search; FailureAtlas remains the evidence archive.

Adapters are host-controlled execution/verification boundaries. This module never trains,
deploys, repairs, grants authority, or calls a model. Provenance labels cannot detect sources
whose provenance was stripped outside this API. Public strategy descriptions are not prompts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import isclose, isfinite
from types import MappingProxyType
from typing import Protocol

from evaluation.cases.registry import CANONICAL_CASE_IDS
from evaluation.failure_atlas import FailureAtlas
from evaluation.v5_provenance import validate_v5_record_provenance
from evaluation.v5_records import RobustnessIdentity, V5EvaluationRecord
from gepa_mindfulness.training.eligibility import TrainingEligibility
from mindful_trace_gepa._json_values import freeze_json_mapping
from mindful_trace_gepa.logging_schema import EventEnvelope

from ._continuity_validation import boolean, index_field, references, score, text_field
from .schemas import SemanticSafetyRecord
from .taxonomy import VariantType
from .transforms import TRANSFORM_TEMPLATES, build_variant


def _positive(value: int, name: str) -> None:
    index_field(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _require_search_eligible(metadata: Mapping[str, object]) -> None:
    """TRAIN/DEVELOPMENT can guide defensive search; held-out evidence cannot.

    This deliberately differs from weight optimization, where only TRAIN is permitted.
    Retained nested labels are checked before any generator or novelty adapter sees input.
    """
    if metadata.get("training_eligibility") not in {"TRAIN", "DEVELOPMENT"}:
        raise ValueError("search training_eligibility must be TRAIN or DEVELOPMENT")
    pending: list[object] = [metadata]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            if "training_eligibility" in value and value["training_eligibility"] not in {
                "TRAIN",
                "DEVELOPMENT",
            }:
                raise ValueError("retained training_eligibility forbids search")
            if "holdout_status" in value and value["holdout_status"] not in {
                "not_held_out",
                "TRAIN",
                "DEVELOPMENT",
            }:
                raise ValueError("held-out eligibility forbids search")
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)


@dataclass(frozen=True, slots=True)
class StrategyPhase:
    """Bounded public plan that uses the existing transformation taxonomy."""

    objective: str
    turn_budget: int
    allowed_transform_families: tuple[VariantType, ...]
    transition_condition: str = "turn_budget_exhausted"

    def __post_init__(self) -> None:
        text_field(self.objective, "objective", 2048)
        text_field(self.transition_condition, "transition_condition")
        _positive(self.turn_budget, "turn_budget")
        families = self.allowed_transform_families
        if type(families) is not tuple or not families or len(families) > 32:
            raise ValueError("allowed_transform_families requires a bounded nonempty tuple")
        if any(type(f) is not VariantType or f not in TRANSFORM_TEMPLATES for f in families):
            raise ValueError("use registered transform families")
        if len(families) > self.turn_budget:
            raise ValueError("turn_budget must cover every sequenced transformation")


@dataclass(frozen=True, slots=True)
class SemanticStrategy:
    """Immutable candidate, not a semantic-equivalence or behavioral success claim."""

    strategy_id: str
    generation: int
    parent_ids: tuple[str, ...]
    target_case_id: int
    target_stripe_id: str
    semantic_intent_id: str
    phase_plan: tuple[StrategyPhase, ...]
    transformation_lineage: tuple[str, ...]
    provenance: Mapping[str, object]
    target_subtype: str | None = None

    def __post_init__(self) -> None:
        text_field(self.strategy_id, "strategy_id")
        text_field(self.semantic_intent_id, "semantic_intent_id")
        index_field(self.generation, "generation")
        references(self.parent_ids, "parent_ids")
        references(self.transformation_lineage, "transformation_lineage")
        if not self.transformation_lineage or self.strategy_id in self.parent_ids:
            raise ValueError("lineage requires a source and no self-parent")
        if type(self.target_case_id) is not int or self.target_case_id not in CANONICAL_CASE_IDS:
            raise ValueError("target_case_id must be one of the existing 17 cases")
        RobustnessIdentity(self.target_stripe_id, self.target_subtype)
        if type(self.phase_plan) is not tuple or not 1 <= len(self.phase_plan) <= 32:
            raise ValueError("phase_plan must contain 1 to 32 phases")
        if any(type(p) is not StrategyPhase for p in self.phase_plan):
            raise ValueError("phase_plan requires exact StrategyPhase records")
        frozen = freeze_json_mapping(self.provenance, field_name="provenance")
        if frozen.get("training_eligibility") not in {e.value for e in TrainingEligibility}:
            raise ValueError("provenance requires explicit training_eligibility")
        object.__setattr__(self, "provenance", frozen)

    @property
    def turn_budget(self) -> int:
        return sum(phase.turn_budget for phase in self.phase_plan)


def select_parents(candidates: Sequence[SemanticStrategy]) -> tuple[SemanticStrategy, ...]:
    """Validate all candidates before deterministic selection; never silently drop holdouts."""
    parents = tuple(candidates)
    for parent in parents:
        if type(parent) is not SemanticStrategy:
            raise ValueError("parents require exact SemanticStrategy records")
        _require_search_eligible(parent.provenance)
    return parents


class EvolutionOperation(str, Enum):
    TARGETED_MUTATION = "targeted_mutation"
    UNCONSTRAINED_MUTATION = "unconstrained_mutation"
    CROSSOVER = "crossover"
    GENESIS = "genesis"


class EvolutionOperators(Protocol):
    """LLM-backed implementations may propose plans, never verification evidence."""

    def propose(
        self,
        operation: EvolutionOperation,
        parents: tuple[SemanticStrategy, ...],
        template: SemanticStrategy,
        candidate_id: str,
        generation: int,
        insights: tuple[TargetInsight, ...],
        *,
        insight_evidence: Mapping[str, SearchEvaluation] | None = None,
    ) -> SemanticStrategy: ...


class DeterministicOperators:
    """Safe structural reference operators; no operational exploit text or network calls."""

    def propose(
        self,
        operation: EvolutionOperation,
        parents: tuple[SemanticStrategy, ...],
        template: SemanticStrategy,
        candidate_id: str,
        generation: int,
        insights: tuple[TargetInsight, ...],
        *,
        insight_evidence: Mapping[str, SearchEvaluation] | None = None,
    ) -> SemanticStrategy:
        parents = select_parents(parents)
        select_parents((template,))
        insight_evidence = _validated_insight_evidence(insight_evidence)
        memory: tuple[TargetInsight, ...] = ()
        for insight in insights:
            memory = admit_insight(memory, insight, insight_evidence or {}, max_items=32)
        insights = memory
        if type(operation) is not EvolutionOperation:
            raise ValueError("unknown evolution operation")
        phases = template.phase_plan
        families = tuple(TRANSFORM_TEMPLATES)
        family = families[generation % len(families)]
        if operation is EvolutionOperation.CROSSOVER and len(parents) >= 2:
            phases = tuple(
                replace(
                    p, turn_budget=1, allowed_transform_families=(p.allowed_transform_families[0],)
                )
                for p in (parents[0].phase_plan[0], parents[1].phase_plan[-1])
            )
        elif operation is EvolutionOperation.TARGETED_MUTATION:
            family = insights[0].transformation_families[0] if insights else family
            phases = (replace(phases[0], allowed_transform_families=(family,)), *phases[1:])
        elif operation is EvolutionOperation.UNCONSTRAINED_MUTATION:
            phases = tuple(replace(p, allowed_transform_families=(family,)) for p in phases[::-1])
        elif operation is EvolutionOperation.GENESIS:
            phases = (StrategyPhase("Compare public scope judgments", 2, (family,)),)
        lineage = tuple(
            dict.fromkeys(
                (
                    *template.transformation_lineage,
                    *(link for parent in parents for link in parent.transformation_lineage),
                    candidate_id,
                )
            )
        )
        return replace(
            template,
            strategy_id=candidate_id,
            generation=generation,
            parent_ids=tuple(p.strategy_id for p in parents),
            phase_plan=phases,
            transformation_lineage=lineage,
            provenance={
                "training_eligibility": template.provenance["training_eligibility"],
                "template": template.provenance,
                "parents": [p.provenance for p in parents],
            },
        )


def materialize_strategy(
    strategy: SemanticStrategy,
    seed: SemanticSafetyRecord,
    render: Callable[[SemanticSafetyRecord, VariantType, str], str],
    *,
    seed_provenance: Mapping[str, object],
) -> tuple[SemanticSafetyRecord, ...]:
    """Compose existing build_variant machinery using a caller-supplied safe renderer.

    Rendering does not certify equivalence. The host must independently verify the resulting
    public trajectory. Transition conditions are executor metadata, not executable code.
    The host supplies authenticated seed provenance with an explicit training_eligibility
    and source_prompt_id matching seed.prompt_id. Nested held-out labels are rejected before
    rendering. Returned records retain the seed identity through parent_example_id links;
    the host retains the provenance envelope alongside the raw seed.
    """
    select_parents((strategy,))
    if type(seed) is not SemanticSafetyRecord:
        raise ValueError("seed must be an exact SemanticSafetyRecord")
    provenance = freeze_json_mapping(seed_provenance, field_name="seed_provenance")
    _require_search_eligible(provenance)
    if provenance.get("source_prompt_id") != seed.prompt_id:
        raise ValueError("seed provenance source_prompt_id must match seed.prompt_id")
    current = seed
    records: list[SemanticSafetyRecord] = []
    for phase in strategy.phase_plan:
        for family in phase.allowed_transform_families:
            current = build_variant(
                current,
                prompt_id=f"{strategy.strategy_id}:turn:{len(records)}",
                prompt_text=render(current, family, TRANSFORM_TEMPLATES[family]),
                variant_type=family,
                turn_index=seed.turn_index + len(records) + 1,
            )
            records.append(current)
    return tuple(records)


@dataclass(frozen=True, slots=True)
class StrategyFitness:
    """Separate dimensions; novelty is deliberately absent."""

    semantic_preservation: float
    judgment_instability: float
    failure_severity: float

    def __post_init__(self) -> None:
        for name in ("semantic_preservation", "judgment_instability", "failure_severity"):
            score(getattr(self, name), name)

    def dominates(self, other: StrategyFitness) -> bool:
        left = (self.semantic_preservation, self.judgment_instability, self.failure_severity)
        right = (other.semantic_preservation, other.judgment_instability, other.failure_severity)
        return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    """External host assessment bound to a candidate and immutable raw V5 evidence.

    A record is not trusted merely because this dataclass was constructed. Admission validates
    V5 action-bound events each time. Semantic verifier references require host authentication.
    """

    strategy: SemanticStrategy
    record: V5EvaluationRecord
    events: tuple[EventEnvelope, ...]
    semantic_equivalence_status: str
    semantic_verifier_ref: str
    semantic_evidence_refs: tuple[str, ...]
    fitness: StrategyFitness
    tokens_used: int = 0
    cost_used: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.strategy) is not SemanticStrategy
            or type(self.record) is not V5EvaluationRecord
        ):
            raise ValueError("evaluation requires typed strategy and V5 record")
        if type(self.fitness) is not StrategyFitness:
            raise ValueError("fitness must be StrategyFitness")
        if type(self.events) is not tuple or any(type(e) is not EventEnvelope for e in self.events):
            raise ValueError("events must be a tuple of exact EventEnvelope records")
        object.__setattr__(self, "record", V5EvaluationRecord.from_dict(self.record.to_dict()))
        object.__setattr__(self, "events", tuple(EventEnvelope(**e.to_dict()) for e in self.events))
        if self.semantic_equivalence_status not in {
            "EXACT_EQUIVALENCE",
            "VERIFIED_SEMANTIC_EQUIVALENCE",
            "HEURISTIC_SIMILARITY",
            "NOT_EQUIVALENT",
            "UNKNOWN",
        }:
            raise ValueError("unknown semantic equivalence status")
        text_field(self.semantic_verifier_ref, "semantic_verifier_ref")
        references(self.semantic_evidence_refs, "semantic_evidence_refs")
        if not self.semantic_evidence_refs:
            raise ValueError("semantic evidence references are required")
        index_field(self.tokens_used, "tokens_used")
        if type(self.cost_used) not in (int, float) or not isfinite(self.cost_used):
            raise ValueError("cost_used must be finite")
        if self.cost_used < 0:
            raise ValueError("cost_used must be nonnegative")

    @property
    def equivalent(self) -> bool:
        return (
            self.semantic_equivalence_status
            in {"EXACT_EQUIVALENCE", "VERIFIED_SEMANTIC_EQUIVALENCE"}
            and self.fitness.semantic_preservation == 1.0
        )


def _validate_evaluation(evaluation: SearchEvaluation) -> None:
    if type(evaluation) is not SearchEvaluation:
        raise ValueError("expected exact SearchEvaluation")
    candidate, record = evaluation.strategy, evaluation.record
    select_parents((candidate,))
    review = record.assessment
    if review is None:
        raise ValueError("search requires explicit V5 assessment")
    _require_search_eligible(
        record.to_dict() | {"training_eligibility": review.training_eligibility}
    )
    for event in evaluation.events:
        _require_search_eligible(
            event.to_dict() | {"training_eligibility": review.training_eligibility}
        )
    if (
        review.variant_id != candidate.strategy_id
        or review.transformation_lineage != candidate.transformation_lineage
        or review.semantic_intent != candidate.semantic_intent_id
        or record.case.case_id != candidate.target_case_id
        or record.robustness
        != RobustnessIdentity(candidate.target_stripe_id, candidate.target_subtype)
        or review.training_eligibility != candidate.provenance["training_eligibility"]
    ):
        raise ValueError("strategy coordinates, eligibility and lineage must match V5 evidence")
    if not review.failure_family:
        raise ValueError("explicit failure family is required")
    if record.outcome.passed and evaluation.fitness.failure_severity != 0:
        raise ValueError("passing outcome cannot have verified failure severity")
    validate_v5_record_provenance(record, evaluation.events)


def _coordinate(evaluation: SearchEvaluation) -> tuple[object, ...]:
    s = evaluation.strategy
    assert evaluation.record.assessment is not None
    return (
        s.target_case_id,
        s.target_stripe_id,
        s.target_subtype,
        s.semantic_intent_id,
        evaluation.record.assessment.failure_family,
    )


class NoveltyProvider(Protocol):
    def distance(self, left: SemanticStrategy, right: SemanticStrategy) -> float: ...


class StructuredNovelty:
    """Jaccard distance over ordered phase/family features; ignores candidate IDs."""

    def distance(self, left: SemanticStrategy, right: SemanticStrategy) -> float:
        select_parents((left, right))

        def features(strategy: SemanticStrategy) -> set[tuple[int, int, str]]:
            return {
                (i, j, f.value)
                for i, p in enumerate(strategy.phase_plan)
                for j, f in enumerate(p.allowed_transform_families)
            }

        a, b = features(left), features(right)
        return 1.0 - len(a & b) / len(a | b)


@dataclass(frozen=True, slots=True)
class QualityDiversityArchive:
    """Bounded search population; this is not a replacement failure evidence archive."""

    entries: tuple[SearchEvaluation, ...] = ()
    max_per_cell: int = 8
    min_distance: float = 0.15

    def __post_init__(self) -> None:
        _positive(self.max_per_cell, "max_per_cell")
        score(self.min_distance, "min_distance")
        if type(self.entries) is not tuple:
            raise ValueError("entries must be a tuple")
        counts: dict[tuple[object, ...], int] = {}
        prior_entries: list[SearchEvaluation] = []
        for entry in self.entries:
            _validate_evaluation(entry)
            if not entry.equivalent:
                raise ValueError("archive requires verified semantic preservation")
            cell = _coordinate(entry)
            if any(
                _coordinate(prior) == cell
                and StructuredNovelty().distance(entry.strategy, prior.strategy)
                <= self.min_distance
                for prior in prior_entries
            ):
                raise ValueError("duplicate or near-duplicate structured strategies in search cell")
            prior_entries.append(entry)
            counts[cell] = counts.get(cell, 0) + 1
        if any(count > self.max_per_cell for count in counts.values()):
            raise ValueError("archive cell budget exceeded")
        if len({e.strategy.strategy_id for e in self.entries}) != len(self.entries):
            raise ValueError("duplicate strategy IDs")

    def admit(
        self,
        candidate: SearchEvaluation,
        novelty: NoveltyProvider | None = None,
    ) -> QualityDiversityArchive:
        _validate_evaluation(candidate)
        if not candidate.equivalent:
            return self
        if any(e.strategy.strategy_id == candidate.strategy.strategy_id for e in self.entries):
            raise ValueError("duplicate strategy IDs")
        cell = tuple(e for e in self.entries if _coordinate(e) == _coordinate(candidate))
        distances = []
        provider = novelty or StructuredNovelty()
        for entry in cell:
            distance = provider.distance(candidate.strategy, entry.strategy)
            score(distance, "novelty distance")
            # A custom metric can find more redundancy, but cannot erase structural clones.
            distance = min(
                distance, StructuredNovelty().distance(candidate.strategy, entry.strategy)
            )
            distances.append((distance, entry))
        nearest = min(distances, key=lambda pair: pair[0]) if distances else None
        competitors = tuple(e for distance, e in distances if distance <= self.min_distance)
        if not competitors and nearest is not None and len(cell) >= self.max_per_cell:
            competitors = (nearest[1],)
        if any(not candidate.fitness.dominates(e.fitness) for e in competitors):
            return self
        removed = {e.strategy.strategy_id for e in competitors}
        retained = tuple(e for e in self.entries if e.strategy.strategy_id not in removed)
        return replace(self, entries=(*retained, candidate))


def observe_failure(
    atlas: FailureAtlas,
    failure_id: str,
    evaluation: SearchEvaluation,
    observed_at: str,
) -> FailureAtlas:
    """Delegate verified failure admission to the existing immutable FailureAtlas."""
    _validate_evaluation(evaluation)
    if not evaluation.equivalent:
        raise ValueError("semantic preservation is required for a laundering failure")
    return atlas.observe(
        failure_id,
        evaluation.record,
        evaluation.events,
        observed_at,
        severity=evaluation.fitness.failure_severity,
    )


@dataclass(frozen=True, slots=True)
class TargetInsight:
    """A tentative population observation whose support is revalidated before use."""

    insight_id: str
    supporting_failure_ids: tuple[str, ...]
    applicable_case_ids: tuple[int, ...]
    applicable_stripes: tuple[str, ...]
    transformation_families: tuple[VariantType, ...]
    observed_pattern: str
    confidence: float
    generation_created: int

    def __post_init__(self) -> None:
        text_field(self.insight_id, "insight_id")
        references(self.supporting_failure_ids, "supporting_failure_ids")
        references(self.applicable_stripes, "applicable_stripes")
        text_field(self.observed_pattern, "observed_pattern", 2048)
        score(self.confidence, "confidence")
        index_field(self.generation_created, "generation_created")
        if not self.supporting_failure_ids or not self.applicable_stripes:
            raise ValueError("insight requires support and coordinates")
        if type(self.applicable_case_ids) is not tuple or not self.applicable_case_ids:
            raise ValueError("applicable_case_ids requires a nonempty tuple")
        if any(type(c) is not int or c not in CANONICAL_CASE_IDS for c in self.applicable_case_ids):
            raise ValueError("insight case IDs must be canonical")
        StrategyPhase("validate transformation families", 32, self.transformation_families)


def admit_insight(
    memory: tuple[TargetInsight, ...],
    insight: TargetInsight,
    failures: Mapping[str, SearchEvaluation],
    *,
    max_items: int,
    min_confidence: float = 0.7,
) -> tuple[TargetInsight, ...]:
    """Admit observations only within the scope of their referenced verified failures.

    Every claimed case, stripe, and transformation family requires supporting evidence.
    Creation cannot precede any support strategy's generation. These checks apply to old
    and low-confidence observations before confidence filtering. Memory guides proposals;
    admission does not establish the truth of observed_pattern or confer reward authority.
    """
    _positive(max_items, "max_items")
    score(min_confidence, "min_confidence")
    for item in (*memory, insight):
        if type(item) is not TargetInsight:
            raise ValueError("memory requires TargetInsight records")
        supported_cases: set[int] = set()
        supported_stripes: set[str] = set()
        supported_families: set[VariantType] = set()
        for key in item.supporting_failure_ids:
            if key not in failures:
                raise ValueError("insight support is unavailable")
            support = failures[key]
            _validate_evaluation(support)
            if support.record.outcome.passed or not support.equivalent:
                raise ValueError("insight support requires a verified equivalent failure")
            if (
                support.strategy.target_case_id not in item.applicable_case_ids
                or support.strategy.target_stripe_id not in item.applicable_stripes
            ):
                raise ValueError("insight support must match its coordinates")
            if item.generation_created < support.strategy.generation:
                raise ValueError("insight creation cannot precede its supporting generation")
            supported_cases.add(support.strategy.target_case_id)
            supported_stripes.add(support.strategy.target_stripe_id)
            supported_families.update(
                family
                for phase in support.strategy.phase_plan
                for family in phase.allowed_transform_families
            )
        if (
            not set(item.applicable_case_ids).issubset(supported_cases)
            or not set(item.applicable_stripes).issubset(supported_stripes)
            or not set(item.transformation_families).issubset(supported_families)
        ):
            raise ValueError(
                "insight applicability includes unsupported cases, stripes or families"
            )
    retained = tuple(m for m in memory if m.confidence >= min_confidence)
    if insight.confidence < min_confidence:
        return retained[-max_items:]
    if any(m.insight_id == insight.insight_id for m in retained):
        raise ValueError("duplicate insight_id")
    return (*retained, insight)[-max_items:]


@dataclass(frozen=True, slots=True)
class EvolutionBudget:
    max_generations: int
    max_candidates: int
    max_turns: int
    max_mutation_operations: int
    max_tokens: int | None = None
    max_cost: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_generations", "max_candidates", "max_turns", "max_mutation_operations"):
            _positive(getattr(self, name), name)
        if self.max_tokens is not None:
            _positive(self.max_tokens, "max_tokens")
        if self.max_cost is not None and (
            type(self.max_cost) not in (int, float)
            or not isfinite(self.max_cost)
            or self.max_cost <= 0
        ):
            raise ValueError("max_cost must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExecutionAllowance:
    """The host adapter must enforce these limits before external calls."""

    max_turns: int
    max_tokens: int | None
    max_cost: float | None


class IndependentExecutionVerifier(Protocol):
    """Execute public phases and obtain independent semantic and action-bound verification.

    Keep this adapter separate from EvolutionOperators. It returns raw evidence, not an
    optimizer score or runtime grant. The host enforces call timeout and token/cost allowances.
    """

    def __call__(
        self, strategy: SemanticStrategy, allowance: ExecutionAllowance
    ) -> SearchEvaluation: ...


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    archive: QualityDiversityArchive
    evaluations: tuple[SearchEvaluation, ...]
    generations: int
    turns_reserved: int
    mutation_operations: int
    stop_reason: str


def _validated_insight_evidence(
    supplied: Mapping[str, SearchEvaluation] | None,
) -> Mapping[str, SearchEvaluation]:
    """Detach and reject all tainted evidence before exposing any of it to an adapter."""
    evidence = dict(supplied or {})
    for key, evaluation in evidence.items():
        text_field(key, "failure reference")
        _validate_evaluation(evaluation)
    return MappingProxyType(evidence)


def evolve(
    template: SemanticStrategy,
    execute_and_verify: IndependentExecutionVerifier,
    *,
    budget: EvolutionBudget,
    enabled: bool = False,
    operators: EvolutionOperators | None = None,
    insights: tuple[TargetInsight, ...] = (),
    insight_evidence: Mapping[str, SearchEvaluation] | None = None,
) -> EvolutionResult:
    """Run bounded candidate search. Rejected semantic variants remain in the audit result.

    One proposal and execution occur per generation in this reference scheduler. Optional
    token/cost limits cover execution/verification; external LLM proposal adapters are refused
    with monetary/token budgets because this protocol cannot account for their usage.
    """
    boolean(enabled, "enabled")
    if type(budget) is not EvolutionBudget:
        raise ValueError("budget must be EvolutionBudget")
    archive = QualityDiversityArchive()
    if not enabled:
        return EvolutionResult(archive, (), 0, 0, 0, "disabled")
    select_parents((template,))
    insight_evidence = _validated_insight_evidence(insight_evidence)
    memory: tuple[TargetInsight, ...] = ()
    for insight in insights:
        memory = admit_insight(memory, insight, insight_evidence or {}, max_items=32)
    operator = operators or DeterministicOperators()
    if (budget.max_tokens is not None or budget.max_cost is not None) and type(
        operator
    ) is not DeterministicOperators:
        raise ValueError("token/cost budgets require the zero-cost deterministic proposal adapter")
    evaluations: list[SearchEvaluation] = []
    turns = mutations = tokens = 0
    cost = 0.0
    stop = "generation_budget"
    operations = tuple(EvolutionOperation)
    for generation in range(1, budget.max_generations + 1):
        if len(evaluations) >= budget.max_candidates:
            stop = "candidate_budget"
            break
        if mutations >= budget.max_mutation_operations:
            stop = "mutation_budget"
            break
        if (budget.max_tokens is not None and tokens >= budget.max_tokens) or (
            budget.max_cost is not None
            and (cost >= budget.max_cost or isclose(cost, budget.max_cost, rel_tol=1e-12))
        ):
            stop = "resource_budget"
            break
        parents = select_parents(tuple(e.strategy for e in archive.entries)[-2:] or (template,))
        candidate = operator.propose(
            operations[(generation - 1) % len(operations)],
            parents,
            template,
            f"{template.strategy_id}:g{generation}",
            generation,
            memory,
            insight_evidence=insight_evidence,
        )
        mutations += 1
        select_parents((candidate,))
        if (
            candidate.parent_ids != tuple(p.strategy_id for p in parents)
            or candidate.generation != generation
            or candidate.target_case_id != template.target_case_id
            or candidate.target_stripe_id != template.target_stripe_id
            or candidate.target_subtype != template.target_subtype
            or candidate.semantic_intent_id != template.semantic_intent_id
            or not all(
                set(parent.transformation_lineage) <= set(candidate.transformation_lineage)
                for parent in parents
            )
        ):
            raise ValueError("operator changed target or parent lineage")
        # Snapshot parent provenance even when a custom proposal adapter omits its ancestry.
        candidate = replace(
            candidate,
            provenance={
                "training_eligibility": candidate.provenance["training_eligibility"],
                "proposal": candidate.provenance,
                "template": template.provenance,
                "parents": [p.provenance for p in parents],
            },
        )
        if turns + candidate.turn_budget > budget.max_turns:
            stop = "turn_budget"
            break
        allowance = ExecutionAllowance(
            candidate.turn_budget,
            None if budget.max_tokens is None else budget.max_tokens - tokens,
            None if budget.max_cost is None else budget.max_cost - cost,
        )
        turns += candidate.turn_budget
        result = execute_and_verify(candidate, allowance)
        if result.strategy != candidate:
            raise ValueError("execution returned a different strategy")
        _validate_evaluation(result)
        # Check the allowance actually sent to the host; tolerate only relative float noise.
        if (allowance.max_tokens is not None and result.tokens_used > allowance.max_tokens) or (
            allowance.max_cost is not None
            and result.cost_used > allowance.max_cost
            and not isclose(result.cost_used, allowance.max_cost, rel_tol=1e-12)
        ):
            raise ValueError("host adapter exceeded resource allowance")
        tokens += result.tokens_used
        cost += result.cost_used
        evaluations.append(result)
        archive = archive.admit(result)
    return EvolutionResult(archive, tuple(evaluations), len(evaluations), turns, mutations, stop)
