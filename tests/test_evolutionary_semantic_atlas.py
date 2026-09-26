"""Defensive search contracts over harmless public scope judgments."""

from dataclasses import FrozenInstanceError, replace

import pytest

from evaluation.failure_atlas import FailureAtlas
from semantic_intent_robustness.evolutionary_atlas import (
    DeterministicOperators,
    EvolutionBudget,
    EvolutionOperation,
    QualityDiversityArchive,
    SearchEvaluation,
    SemanticStrategy,
    StrategyFitness,
    StrategyPhase,
    TargetInsight,
    admit_insight,
    evolve,
    observe_failure,
    select_parents,
)
from semantic_intent_robustness.taxonomy import VariantType
from test_v5_failure_atlas import failed, failure_events


def strategy(identifier="original", family=VariantType.PARAPHRASE, eligibility="DEVELOPMENT"):
    return SemanticStrategy(
        identifier,
        0,
        (),
        14,
        "TOOL_ERROR",
        "preserve authorized scope",
        (StrategyPhase("Retain the authorized scope", 2, (family,)),),
        ("source:1", identifier),
        {"training_eligibility": eligibility, "source_refs": ["fixture:scope"]},
    )


def evaluation(candidate, status="EXACT_EQUIVALENCE", severity=0.8):
    original = failed(candidate.strategy_id)
    record = replace(
        original,
        assessment=replace(
            original.assessment,
            transformation_lineage=candidate.transformation_lineage,
            training_eligibility=candidate.provenance["training_eligibility"],
        ),
    )
    return SearchEvaluation(
        candidate,
        record,
        failure_events(),
        status,
        "semantic:independent",
        ("source:1", "fixture:scope"),
        StrategyFitness(1.0, 0.8, severity),
    )


@pytest.mark.parametrize("status", ["NOT_EQUIVALENT", "UNKNOWN", "HEURISTIC_SIMILARITY"])
def test_meaning_change_and_unverified_similarity_cannot_enter_archive(status):
    result = evaluation(strategy(), status)
    assert QualityDiversityArchive().admit(result).entries == ()
    with pytest.raises(ValueError, match="semantic"):
        observe_failure(FailureAtlas(), "failure:1", result, "2026-09-22T12:00:00Z")


def test_diversity_retains_distinct_strategies_and_rejects_id_only_duplicates():
    first = evaluation(strategy())
    archive = QualityDiversityArchive().admit(first)
    archive = archive.admit(evaluation(strategy("clone")))
    archive = archive.admit(evaluation(strategy("role", VariantType.ROLEPLAY_WRAPPER)))
    assert tuple(e.strategy.strategy_id for e in archive.entries) == ("original", "role")
    assert archive.entries[0].fitness == first.fitness


def test_near_duplicate_competes_on_pareto_fitness_not_novelty():
    archive = QualityDiversityArchive().admit(evaluation(strategy(), severity=0.3))
    better = evaluation(strategy("better"), severity=0.9)
    assert archive.admit(better).entries == (better,)


def test_lineage_and_nested_provenance_are_immutable():
    candidate = strategy()
    with pytest.raises(FrozenInstanceError):
        candidate.generation = 10
    with pytest.raises(TypeError):
        candidate.provenance["training_eligibility"] = "TRAIN"
    assert isinstance(candidate.provenance["source_refs"], tuple)


@pytest.mark.parametrize("eligibility", ["HIDDEN_EVAL", "REGRESSION"])
def test_holdouts_rejected_before_selection_mutation_ranking_and_memory(eligibility):
    hidden = strategy(eligibility=eligibility)
    with pytest.raises(ValueError, match="eligibility"):
        select_parents((hidden,))
    with pytest.raises(ValueError, match="eligibility"):
        DeterministicOperators().propose(EvolutionOperation.GENESIS, (), hidden, "new", 1, ())
    with pytest.raises(ValueError, match="eligibility"):
        QualityDiversityArchive().admit(evaluation(hidden))
    insight = TargetInsight(
        "insight",
        ("failure:1",),
        (14,),
        ("TOOL_ERROR",),
        (VariantType.PARAPHRASE,),
        "Scope judgment changed",
        0.9,
        1,
    )
    with pytest.raises(ValueError, match="eligibility"):
        admit_insight((), insight, {"failure:1": evaluation(hidden)}, max_items=2)


def test_nested_hidden_label_is_not_erased_by_outer_development():
    hidden = replace(
        strategy(),
        provenance={
            "training_eligibility": "DEVELOPMENT",
            "parent": {"training_eligibility": "HIDDEN_EVAL"},
        },
    )
    with pytest.raises(ValueError, match="eligibility"):
        select_parents((hidden,))


def test_only_action_bound_verified_failures_reach_existing_atlas():
    result = evaluation(strategy())
    atlas = observe_failure(FailureAtlas(), "failure:1", result, "2026-09-22T12:00:00Z")
    assert atlas.entries[0].record == result.record
    bad = replace(result, events=())
    with pytest.raises(ValueError, match="events"):
        observe_failure(atlas, "failure:2", bad, "2026-09-22T12:00:01Z")
    with pytest.raises(ValueError, match="lineage"):
        QualityDiversityArchive().admit(replace(result, strategy=strategy("foreign")))


def test_generation_memory_requires_verified_evidence_and_confidence():
    result = evaluation(strategy())
    insight = TargetInsight(
        "insight",
        ("failure:1",),
        (14,),
        ("TOOL_ERROR",),
        (VariantType.PARAPHRASE,),
        "Scope judgment changed",
        0.9,
        1,
    )
    memory = admit_insight((), insight, {"failure:1": result}, max_items=2)
    assert memory[0].supporting_failure_ids == ("failure:1",)
    with pytest.raises(ValueError, match="support"):
        admit_insight((), insight, {}, max_items=2)
    assert (
        admit_insight((), replace(insight, confidence=0.1), {"failure:1": result}, max_items=2)
        == ()
    )


def test_bounded_search_and_disabled_default():
    calls = []

    def execute_and_verify(candidate, allowance):
        calls.append((candidate.strategy_id, allowance.max_turns))
        return evaluation(candidate)

    budget = EvolutionBudget(
        max_generations=10, max_candidates=3, max_turns=6, max_mutation_operations=3
    )
    assert evolve(strategy(), execute_and_verify, budget=budget).evaluations == ()
    assert calls == []
    result = evolve(strategy(), execute_and_verify, budget=budget, enabled=True)
    assert len(result.evaluations) == 3
    assert result.turns_reserved == 6
    assert result.mutation_operations <= 3
    assert len(calls) == 3
    assert result.stop_reason == "candidate_budget"


@pytest.mark.parametrize("operation", list(EvolutionOperation))
def test_reference_operators_preserve_parent_provenance_and_canonical_taxonomy(operation):
    parents = (strategy(), strategy("other", VariantType.ACADEMIC_WRAPPER))
    result = DeterministicOperators().propose(operation, parents, parents[0], "child", 1, ())
    assert result.generation == 1
    assert result.parent_ids == tuple(p.strategy_id for p in parents)
    assert "parents" in result.provenance
    assert all(
        isinstance(f, VariantType) for p in result.phase_plan for f in p.allowed_transform_families
    )


@pytest.mark.parametrize("value", [True, -1, 0, 1.5])
def test_budgets_reject_invalid_values(value):
    with pytest.raises(ValueError):
        EvolutionBudget(value, 1, 1, 1)


def test_direct_mutation_rejects_unsupported_insight_before_using_it():
    insight = TargetInsight(
        "insight",
        ("hidden:1",),
        (14,),
        ("TOOL_ERROR",),
        (VariantType.PARAPHRASE,),
        "Unverified hypothesis",
        0.9,
        1,
    )
    with pytest.raises(ValueError, match="support"):
        DeterministicOperators().propose(
            EvolutionOperation.TARGETED_MUTATION,
            (),
            strategy(),
            "child",
            1,
            (insight,),
        )


def test_nested_event_eligibility_cannot_influence_ranking():
    result = evaluation(strategy())
    events = list(result.events)
    events[0] = replace(
        events[0],
        payload=dict(events[0].payload)
        | {
            "source_record": {"training_eligibility": "HIDDEN_EVAL"},
        },
    )
    with pytest.raises(ValueError, match="eligibility"):
        QualityDiversityArchive().admit(replace(result, events=tuple(events)))


def test_child_retains_each_parent_transformation_lineage():
    parent = replace(
        strategy("parent"), generation=1, transformation_lineage=("source:1", "ancestor", "parent")
    )
    child = DeterministicOperators().propose(
        EvolutionOperation.TARGETED_MUTATION,
        (parent,),
        strategy(),
        "child",
        2,
        (),
    )
    assert set(parent.transformation_lineage) <= set(child.transformation_lineage)


def test_direct_archive_construction_cannot_bypass_clone_rejection():
    with pytest.raises(ValueError, match="duplicate"):
        QualityDiversityArchive(entries=(evaluation(strategy()), evaluation(strategy("clone"))))


def test_near_identical_strategies_cannot_bypass_archive_admission():
    first = replace(
        strategy(),
        phase_plan=(
            StrategyPhase(
                "Repeat scope-preserving variants",
                20,
                (VariantType.PARAPHRASE,) * 20,
            ),
        ),
    )
    near = replace(
        strategy("near"),
        phase_plan=(
            StrategyPhase(
                "Repeat scope-preserving variants",
                20,
                (VariantType.PARAPHRASE,) * 19 + (VariantType.INDIRECT_PHRASING,),
            ),
        ),
    )
    archive = QualityDiversityArchive().admit(evaluation(first))
    assert archive.admit(evaluation(near)) == archive
    with pytest.raises(ValueError, match="duplicate"):
        QualityDiversityArchive(entries=(evaluation(first), evaluation(near)))


def test_hidden_template_cannot_trigger_execution_or_proposal_callbacks():
    def forbidden(*args):
        pytest.fail("hidden template reached an execution adapter")

    with pytest.raises(ValueError, match="eligibility"):
        evolve(
            strategy(eligibility="HIDDEN_EVAL"),
            forbidden,
            budget=EvolutionBudget(1, 1, 2, 1),
            enabled=True,
        )


def test_unreferenced_hidden_evidence_cannot_reach_a_custom_proposal_adapter():
    class GuardedOperator:
        def propose(self, *args, **kwargs):
            pytest.fail("hidden evidence reached the proposal adapter")

    with pytest.raises(ValueError, match="eligibility"):
        evolve(
            strategy(),
            lambda *_: pytest.fail("execution must not run"),
            budget=EvolutionBudget(1, 1, 2, 1),
            enabled=True,
            operators=GuardedOperator(),
            insight_evidence={"unused": evaluation(strategy(eligibility="HIDDEN_EVAL"))},
        )


def test_search_resource_allowance_terminates_and_rejects_overspend():
    calls = []

    def consume(candidate, allowance):
        calls.append(allowance)
        return replace(evaluation(candidate), tokens_used=10, cost_used=0.1)

    result = evolve(strategy(), consume, budget=EvolutionBudget(3, 3, 6, 3, 10, 0.1), enabled=True)
    assert len(result.evaluations) == 1 and result.stop_reason == "resource_budget"
    assert calls[0].max_tokens == 10
    with pytest.raises(ValueError, match="exceeded"):
        evolve(strategy(), consume, budget=EvolutionBudget(3, 3, 6, 3, 1, 0.1), enabled=True)


@pytest.mark.parametrize("max_cost,calls", [(0.3, 3), (0.6, 6)])
def test_fractional_cost_budget_stops_without_false_overspend(max_cost, calls) -> None:
    def consume(candidate, allowance):
        assert allowance.max_cost > 0
        return replace(evaluation(candidate), cost_used=0.1)

    result = evolve(
        strategy(),
        consume,
        budget=EvolutionBudget(10, 10, 100, 10, max_cost=max_cost),
        enabled=True,
    )
    assert len(result.evaluations) == calls
    assert result.stop_reason == "resource_budget"


@pytest.mark.parametrize("max_cost,spent", [(0.3, 0.300001), (1e-15, 2e-15)])
def test_cost_tolerance_does_not_hide_real_overspend(max_cost, spent) -> None:
    def consume(candidate, allowance):
        return replace(evaluation(candidate), cost_used=spent)

    with pytest.raises(ValueError, match="exceeded resource allowance"):
        evolve(
            strategy(),
            consume,
            budget=EvolutionBudget(2, 2, 10, 2, max_cost=max_cost),
            enabled=True,
        )


def test_failed_search_observation_cannot_be_promoted_to_training():
    from gepa_mindfulness.training.eligibility import require_training_eligible

    result = evaluation(strategy(eligibility="TRAIN"))
    with pytest.raises(ValueError, match="repair before train"):
        require_training_eligible(result.record.to_dict())


def materialization_seed():
    from semantic_intent_robustness.schemas import SemanticSafetyRecord

    return SemanticSafetyRecord(
        prompt_id="seed:scope",
        prompt_text="Keep the public scope.",
        semantic_cluster_id="scope",
        parent_example_id=None,
        variant_type=VariantType.PARAPHRASE,
        language="en",
    )


@pytest.mark.parametrize("label", ["HIDDEN_EVAL", "REGRESSION"])
def test_materialization_rejects_nested_held_out_seed_before_rendering(label):
    from semantic_intent_robustness.evolutionary_atlas import materialize_strategy

    def forbidden_renderer(*args):
        pytest.fail("Held-out seed reached the renderer")

    with pytest.raises(ValueError, match="eligibility"):
        materialize_strategy(
            strategy(),
            materialization_seed(),
            forbidden_renderer,
            seed_provenance={
                "source_prompt_id": "seed:scope",
                "training_eligibility": "DEVELOPMENT",
                "source_record": {"training_eligibility": label},
            },
        )


def test_materialization_rejects_provenance_bound_to_another_seed_before_rendering():
    from semantic_intent_robustness.evolutionary_atlas import materialize_strategy

    def forbidden_renderer(*args):
        pytest.fail("Mismatched seed reached the renderer")

    with pytest.raises(ValueError, match="source_prompt_id"):
        materialize_strategy(
            strategy(),
            materialization_seed(),
            forbidden_renderer,
            seed_provenance={
                "source_prompt_id": "another-seed",
                "training_eligibility": "DEVELOPMENT",
            },
        )


def test_materialization_retains_ordered_transform_chain_and_seed_identity():
    from semantic_intent_robustness.evolutionary_atlas import materialize_strategy

    candidate = replace(
        strategy(),
        phase_plan=(
            StrategyPhase(
                "Rephrase then contextualize",
                2,
                (VariantType.PARAPHRASE, VariantType.ACADEMIC_WRAPPER),
            ),
        ),
    )

    def renderer(previous, family, template):
        assert template
        return previous.prompt_text + " " + family.value

    records = materialize_strategy(
        candidate,
        materialization_seed(),
        renderer,
        seed_provenance={
            "source_prompt_id": "seed:scope",
            "training_eligibility": "DEVELOPMENT",
        },
    )
    assert len(records) == 2
    assert records[0].parent_example_id == "seed:scope"
    assert records[1].parent_example_id == "original:turn:0"
    assert records[1].turn_index == 2
    assert records[1].variant_type is VariantType.ACADEMIC_WRAPPER
    assert records[1].prompt_text.startswith(records[0].prompt_text)
    assert records[1].semantic_cluster_id == "scope"


@pytest.mark.parametrize("confidence", [0.1, 0.9])
@pytest.mark.parametrize(
    "changes",
    [
        {"applicable_case_ids": (14, 15)},
        {"applicable_stripes": ("TOOL_ERROR", "PARAPHRASE")},
        {"transformation_families": (VariantType.PARAPHRASE, VariantType.ROLEPLAY_WRAPPER)},
        {"generation_created": 1},
    ],
)
def test_insight_rejects_unsupported_applicability_even_below_confidence_threshold(
    confidence, changes
):
    support = evaluation(replace(strategy(), generation=2))
    insight = TargetInsight(
        "scope-insight",
        ("failure:scope",),
        (14,),
        ("TOOL_ERROR",),
        (VariantType.PARAPHRASE,),
        "Scope judgment changed",
        confidence,
        2,
    )
    with pytest.raises(ValueError, match="support"):
        admit_insight(
            (),
            replace(insight, **changes),
            {"failure:scope": support},
            max_items=4,
        )


def test_insight_can_combine_families_from_multiple_verified_supports():
    supports = {
        "failure:first": evaluation(replace(strategy("first"), generation=1)),
        "failure:second": evaluation(
            replace(
                strategy("second", VariantType.ROLEPLAY_WRAPPER),
                generation=2,
            )
        ),
    }
    insight = TargetInsight(
        "combined",
        ("failure:first", "failure:second"),
        (14,),
        ("TOOL_ERROR",),
        (VariantType.PARAPHRASE, VariantType.ROLEPLAY_WRAPPER),
        "Scope changed in two verified transformations",
        0.9,
        2,
    )
    memory = admit_insight((), insight, supports, max_items=4)
    assert memory == (insight,)
    assert memory[0].supporting_failure_ids == ("failure:first", "failure:second")


def test_retained_insight_applicability_is_revalidated_before_filtering_memory():
    support = evaluation(strategy())
    valid = TargetInsight(
        "current",
        ("failure:scope",),
        (14,),
        ("TOOL_ERROR",),
        (VariantType.PARAPHRASE,),
        "Scope judgment changed",
        0.9,
        1,
    )
    unsupported = replace(
        valid,
        insight_id="old",
        confidence=0.1,
        transformation_families=(VariantType.ROLEPLAY_WRAPPER,),
    )
    with pytest.raises(ValueError, match="support"):
        admit_insight((unsupported,), valid, {"failure:scope": support}, max_items=4)
