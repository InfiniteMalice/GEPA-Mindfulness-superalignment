"""Tests for graph-aware factuality and unlearning evaluation."""

# Standard library
from __future__ import annotations

# Local
from evaluation.suites.factuality.structured_unlearning import (
    GraphAwareUnlearningCase,
    evaluate_graph_aware_unlearning,
)
from factuality_certification import EvidenceItem, FactualityCertificationConfig, certify_answer
from factuality_certification.structured_knowledge import (
    ClaimRelation,
    StructuredKnowledgeGraph,
    StructuredKnowledgeGraphBuilder,
    estimate_reconstructability,
)
from factuality_certification.types import AtomicClaim, ClaimSupport
from semantic_intent_robustness.capability_graph import build_capability_disclosure_graph
from semantic_intent_robustness.disclosure_events import DisclosureEvent, DisclosureSourceType


def _claim(claim_id: str, text: str) -> AtomicClaim:
    return AtomicClaim(id=claim_id, text=text)


def _support(
    claim_id: str,
    label: str,
    score: float,
    evidence_ids: list[str] | None = None,
) -> ClaimSupport:
    return ClaimSupport(
        claim_id=claim_id,
        support_label=label,
        support_score=score,
        contradiction_score=0.0,
        evidence_ids=evidence_ids or [],
    )


def test_directly_supported_claim_remains_certified() -> None:
    claim = _claim("c1", "The report says METRIC_A.")
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=[claim],
        claim_support=[_support("c1", "supported", 0.9, ["e1"])],
        evidence=[EvidenceItem(id="e1", text="METRIC_A")],
    )

    assessment = estimate_reconstructability(graph=graph, target_claim_id="c1")

    assert assessment.direct_support_score == 0.9
    assert assessment.requires_review is False


def test_unsupported_isolated_claim_remains_unsupported() -> None:
    claim = _claim("c1", "Unsupported claim.")
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=[claim],
        claim_support=[_support("c1", "unsupported", 0.0)],
        evidence=[],
    )

    assessment = estimate_reconstructability(graph=graph, target_claim_id="c1")

    assert assessment.indirect_support_score == 0.0
    assert assessment.inferable_despite_missing_direct_support is False


def test_indirectly_inferable_claim_is_flagged() -> None:
    claims = [_claim("c1", "A implies B."), _claim("c2", "B is inferable.")]
    supports = [_support("c1", "supported", 0.9, ["e1"]), _support("c2", "unsupported", 0.0)]
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=claims,
        claim_support=supports,
        evidence=[EvidenceItem(id="e1", text="A implies B.")],
    )
    relation = ClaimRelation("c1", "c2", "implies", 0.9)
    graph = StructuredKnowledgeGraph(
        nodes=graph.nodes,
        relations=(relation,),
        inference_paths=graph.inference_paths,
    )

    assessment = estimate_reconstructability(graph=graph, target_claim_id="c2")

    assert assessment.inferable_despite_missing_direct_support is True
    assert assessment.requires_review is True


def test_multiple_weak_paths_combine_conservatively() -> None:
    claims = [_claim("c1", "Fragment A."), _claim("c2", "Fragment B."), _claim("c3", "Target.")]
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=claims,
        claim_support=[
            _support("c1", "supported", 0.5, ["e1"]),
            _support("c2", "supported", 0.5, ["e2"]),
            _support("c3", "unsupported", 0.0),
        ],
        evidence=[EvidenceItem(id="e1", text="A"), EvidenceItem(id="e2", text="B")],
    )
    graph = StructuredKnowledgeGraph(
        nodes=graph.nodes,
        relations=(
            ClaimRelation("c1", "c3", "correlated_with", 0.5),
            ClaimRelation("c2", "c3", "correlated_with", 0.5),
        ),
        inference_paths=(),
    )

    assessment = estimate_reconstructability(graph=graph, target_claim_id="c3")

    assert round(assessment.reconstructability_score, 4) == 0.75


def test_contradicted_claim_with_strong_indirect_path_requires_review() -> None:
    claims = [_claim("c1", "Supported premise."), _claim("c2", "Contradicted target.")]
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=claims,
        claim_support=[
            _support("c1", "supported", 0.95, ["e1"]),
            ClaimSupport(
                claim_id="c2",
                support_label="contradicted",
                support_score=0.1,
                contradiction_score=0.8,
            ),
        ],
        evidence=[EvidenceItem(id="e1", text="premise")],
    )
    graph = StructuredKnowledgeGraph(
        nodes=graph.nodes,
        relations=(ClaimRelation("c1", "c2", "implies", 0.9),),
        inference_paths=(),
    )

    assessment = estimate_reconstructability(graph=graph, target_claim_id="c2")

    assert assessment.requires_review is True
    assert assessment.recommended_action == "manual_review"


def test_multi_hop_inference_path_uses_supported_source() -> None:
    claims = [
        _claim("c1", "Supported source."),
        _claim("c2", "Intermediate bridge."),
        _claim("c3", "Unsupported target."),
    ]
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=claims,
        claim_support=[
            _support("c1", "supported", 0.9, ["e1"]),
            _support("c2", "unsupported", 0.0),
            _support("c3", "unsupported", 0.0),
        ],
        evidence=[EvidenceItem(id="e1", text="source")],
    )
    graph = StructuredKnowledgeGraph(
        nodes=graph.nodes,
        relations=(
            ClaimRelation("c1", "c2", "implies", 0.9),
            ClaimRelation("c2", "c3", "implies", 0.9),
        ),
        inference_paths=(),
    )

    assessment = estimate_reconstructability(graph=graph, target_claim_id="c3", max_depth=2)

    assert ("c1", "c2", "c3") in {path.claim_ids for path in assessment.inference_paths}
    assert assessment.reconstructability_score > 0.70


def test_missing_target_claim_raises_clear_error() -> None:
    graph = StructuredKnowledgeGraphBuilder().build(
        claims=[_claim("c1", "A.")],
        claim_support=[_support("c1", "supported", 0.9)],
        evidence=[],
    )

    try:
        estimate_reconstructability(graph=graph, target_claim_id="missing")
    except ValueError as exc:
        assert "target_claim_id not found in graph" in str(exc)
    else:
        raise AssertionError("missing target should raise ValueError")


def test_structured_knowledge_disabled_by_default() -> None:
    cfg = FactualityCertificationConfig()

    assert cfg.enable_structured_knowledge is False
    assert cfg.structured_knowledge_mode == "off"


def test_existing_atomic_claim_behavior_unchanged_by_default() -> None:
    answer = "Mars has 3 moons."
    before = certify_answer("Q", answer, [])
    after = certify_answer("Q", answer, [], config=FactualityCertificationConfig())

    assert before.recommended_action == after.recommended_action
    assert "structured_knowledge_enabled" not in before.logs


def test_structured_knowledge_advisory_adds_logs() -> None:
    cfg = FactualityCertificationConfig(
        enable_structured_knowledge=True,
        structured_knowledge_mode="advisory",
    )
    res = certify_answer("Q", "A implies B. B is inferable.", [], config=cfg)

    assert res.logs["structured_knowledge_enabled"] is True
    assert "structured_knowledge_graph_summary" in res.logs


def test_surface_forgetting_gap_detects_superficial_unlearning() -> None:
    case = GraphAwareUnlearningCase(
        case_id="case",
        direct_recall_blocked=True,
        indirect_reconstructability_score=0.8,
        correlated_knowledge_leaked=True,
        benign_control=False,
    )

    report = evaluate_graph_aware_unlearning((case,))

    assert report.surface_forgetting_gap > 0.0
    assert report.graph_aware_unlearning_success_rate == 0.0


def test_shared_graph_interface_supports_kv_and_factuality_graphs() -> None:
    fact_graph = StructuredKnowledgeGraphBuilder().build(
        claims=[_claim("c1", "A."), _claim("c2", "B.")],
        claim_support=[_support("c1", "supported", 0.9), _support("c2", "unsupported", 0.0)],
        evidence=[],
    )
    capability_graph = build_capability_disclosure_graph(
        (
            DisclosureEvent(
                event_id="e1",
                conversation_id="conv",
                turn_index=0,
                source_type=DisclosureSourceType.RAG_RETRIEVAL,
                content_summary="COMPONENT_A",
                fragment_ids=("COMPONENT_A",),
                capability_domain="abstract",
                operational_specificity=0.4,
                executionality=0.2,
                provenance_retained=True,
                trust_level="retrieved",
                cross_session_relevance=False,
            ),
        )
    )

    assert fact_graph.nodes()
    assert fact_graph.edges() == ()
    assert capability_graph.nodes()
    assert capability_graph.edges() == ()
