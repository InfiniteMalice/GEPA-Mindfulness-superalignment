"""Tests for epistemically qualified failure graphs."""

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.failure_graph import (
    FailureEdge,
    FailureGraph,
    FailureLocalization,
    FailureNode,
    FailureRelation,
)


def _observable(reference_id: str = "observation:failure") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _private(reference_id: str = "reasoning:failure") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.PRIVATE_REASONING)


def _node(
    failure_id: str,
    *,
    observed_at: str = "2026-09-10T12:00:00Z",
    evidence_refs: tuple[EvidenceReference, ...] | list[EvidenceReference] | None = None,
) -> FailureNode:
    references = (_observable(f"observation:{failure_id}"),)
    return FailureNode(
        failure_id=failure_id,
        event_id=f"event:{failure_id}",
        summary=f"Observed {failure_id}",
        observed_at=observed_at,
        evidence_refs=references if evidence_refs is None else cast(Any, evidence_refs),
    )


def _edge(
    source_id: str,
    target_id: str,
    relation: FailureRelation = FailureRelation.CAUSAL,
    verifier_refs: tuple[str, ...] | list[str] | None = None,
) -> FailureEdge:
    references = (f"verifier:{source_id}-to-{target_id}",)
    if verifier_refs is None:
        verifier_refs = () if relation is FailureRelation.HYPOTHESIZED else references
    return FailureEdge(source_id, target_id, relation, cast(Any, verifier_refs))


def _localization(**changes: object) -> FailureLocalization:
    values: dict[str, object] = {
        "first_anomaly": "anomaly",
        "root_cause": "root",
        "decisive_failure": "decisive",
        "symptoms": ("symptom",),
        "recoverable_until": "anomaly",
    }
    values.update(changes)
    return FailureLocalization(**cast(Any, values))


def _supported_graph() -> FailureGraph:
    nodes = (
        _node("symptom", observed_at="2026-09-10T12:00:01Z"),
        _node("decisive", observed_at="2026-09-10T12:00:02Z"),
        _node("anomaly", observed_at="2026-09-10T12:00:03Z"),
        _node("root", observed_at="2026-09-10T12:00:04Z"),
    )
    edges = (
        _edge("root", "anomaly"),
        _edge("anomaly", "decisive"),
        _edge("decisive", "symptom", FailureRelation.CONTRIBUTING),
    )
    return FailureGraph(nodes, edges, _localization())


def test_localization_keeps_anomaly_cause_decisive_failure_and_symptom_distinct() -> None:
    """Catch localization collapsing four different failure roles into one label."""

    graph = _supported_graph()

    assert graph.localization.first_anomaly == "anomaly"
    assert graph.localization.root_cause == "root"
    assert graph.localization.decisive_failure == "decisive"
    assert graph.localization.symptoms == ("symptom",)
    assert graph.localization.root_cause not in graph.localization.symptoms


def test_localization_preserves_recovery_boundary() -> None:
    """Catch loss of the last failure stage at which recovery remained possible."""

    assert _supported_graph().localization.recoverable_until == "anomaly"


def test_causal_edges_require_nonempty_verifier_provenance() -> None:
    """Catch an asserted causal link with no independently auditable verifier."""

    with pytest.raises(ValueError, match="causal.*verifier_refs"):
        FailureEdge("root", "anomaly", FailureRelation.CAUSAL)


def test_hypothesized_edges_cannot_claim_verifier_support() -> None:
    """Catch a hypothesis being serialized as both conjectural and verifier-supported."""

    with pytest.raises(ValueError, match="hypothesized.*verifier_refs"):
        _edge(
            "root",
            "anomaly",
            FailureRelation.HYPOTHESIZED,
            ("verifier:claim",),
        )


def test_graph_rejects_dangling_and_self_edges() -> None:
    """Catch relations that cannot identify two distinct nodes in the graph."""

    nodes = (_node("root"), _node("anomaly"))

    with pytest.raises(ValueError, match="dangling"):
        FailureGraph(
            nodes,
            (_edge("root", "missing"),),
            _localization(
                first_anomaly="anomaly",
                root_cause="root",
                decisive_failure="anomaly",
                symptoms=(),
            ),
        )
    with pytest.raises(ValueError, match="itself"):
        FailureEdge(
            "root",
            "root",
            FailureRelation.CAUSAL,
            ("verifier:self",),
        )


def test_graph_rejects_duplicate_node_ids_and_duplicate_edges() -> None:
    """Catch ambiguous node lookup and duplicate relation records."""

    localization = _localization(
        first_anomaly="anomaly",
        root_cause="root",
        decisive_failure="anomaly",
        symptoms=(),
    )
    edge = _edge("root", "anomaly")

    with pytest.raises(ValueError, match="failure IDs.*unique"):
        FailureGraph((_node("root"), _node("root")), (), localization)
    with pytest.raises(ValueError, match="edges.*unique"):
        FailureGraph((_node("root"), _node("anomaly")), (edge, edge), localization)


def test_graph_rejects_duplicate_event_ids() -> None:
    """Catch two failure identities claiming the same canonical source event."""

    first = _node("first")
    second = _node("second")
    object.__setattr__(second, "event_id", first.event_id)

    with pytest.raises(ValueError, match="event IDs.*unique"):
        FailureGraph(
            (first, second),
            (),
            _localization(
                first_anomaly="first",
                root_cause=None,
                decisive_failure=None,
                symptoms=(),
                recoverable_until=None,
            ),
        )


def test_only_supported_causal_edges_participate_in_cycle_rejection() -> None:
    """Catch cycles in supported causality without rejecting noncausal relation cycles."""

    nodes = (_node("one"), _node("two"), _node("three"))
    localization = _localization(
        first_anomaly="two",
        root_cause="one",
        decisive_failure="two",
        symptoms=("three",),
        recoverable_until=None,
    )

    with pytest.raises(ValueError, match="causal edges.*cycle"):
        FailureGraph(
            nodes,
            (
                _edge("one", "two"),
                _edge("two", "one"),
            ),
            localization,
        )

    graph = FailureGraph(
        nodes,
        (
            _edge("one", "two"),
            _edge("two", "one", FailureRelation.CORRELATED, ()),
            _edge("two", "three", FailureRelation.PRECEDING, ()),
            _edge("three", "two", FailureRelation.HYPOTHESIZED, ()),
        ),
        localization,
    )
    assert graph.supported_topological_order() == ("one", "three", "two")


def test_root_cause_requires_supported_or_hypothesized_path_to_decisive_failure() -> None:
    """Catch a root-cause label inferred only from correlation or chronology."""

    nodes = (_node("root"), _node("decisive"))
    localization = _localization(
        first_anomaly="decisive",
        root_cause="root",
        decisive_failure="decisive",
        symptoms=(),
        recoverable_until=None,
    )

    for relation in (
        FailureRelation.CONTRIBUTING,
        FailureRelation.PRECEDING,
        FailureRelation.CORRELATED,
    ):
        with pytest.raises(ValueError, match="unsupported root_cause"):
            FailureGraph(nodes, (_edge("root", "decisive", relation, ()),), localization)


def test_root_cause_cannot_be_supported_by_a_zero_length_path() -> None:
    """Catch node identity alone being treated as verifier-supported root causality."""

    with pytest.raises(ValueError, match="unsupported root_cause"):
        FailureGraph(
            (_node("failure"),),
            (),
            _localization(
                first_anomaly="failure",
                root_cause="failure",
                decisive_failure="failure",
                symptoms=(),
                recoverable_until=None,
            ),
        )


def test_hypothesized_root_serializes_its_epistemic_status_honestly() -> None:
    """Catch insufficient causal evidence being serialized as supported causality."""

    graph = FailureGraph(
        (_node("possible-root"), _node("decisive")),
        (
            _edge(
                "possible-root",
                "decisive",
                FailureRelation.HYPOTHESIZED,
                (),
            ),
        ),
        _localization(
            first_anomaly="decisive",
            root_cause="possible-root",
            decisive_failure="decisive",
            symptoms=(),
            recoverable_until=None,
        ),
    )

    payload = graph.to_dict()

    assert graph.root_cause_status == "hypothesized"
    assert payload["root_cause_status"] == "hypothesized"
    assert cast(list[dict[str, object]], payload["edges"])[0] == {
        "source_id": "possible-root",
        "target_id": "decisive",
        "relation": "hypothesized",
        "verifier_refs": [],
    }


def test_mixed_hypothesized_and_supported_path_remains_hypothesized() -> None:
    """Catch one supported downstream link laundering an upstream hypothesis."""

    graph = FailureGraph(
        (_node("root"), _node("anomaly"), _node("decisive")),
        (
            _edge("root", "anomaly", FailureRelation.HYPOTHESIZED, ()),
            _edge("anomaly", "decisive"),
        ),
        _localization(symptoms=(), recoverable_until=None),
    )

    assert graph.root_cause_status == "hypothesized"


def test_supported_topological_order_ignores_timestamps_and_noncausal_edges() -> None:
    """Catch event time or association being silently promoted into causal ordering."""

    graph = _supported_graph()

    assert graph.root_cause_status == "supported"
    assert graph.supported_topological_order() == (
        "root",
        "anomaly",
        "decisive",
        "symptom",
    )
    assert graph.topological_order() == graph.supported_topological_order()


def test_independent_nodes_have_deterministic_identifier_order() -> None:
    """Catch caller insertion order changing the supported graph traversal."""

    graph = FailureGraph(
        (_node("zeta"), _node("alpha"), _node("middle")),
        (),
        _localization(
            first_anomaly="middle",
            root_cause=None,
            decisive_failure=None,
            symptoms=(),
            recoverable_until=None,
        ),
    )

    assert graph.supported_topological_order() == ("alpha", "middle", "zeta")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("failure_id", ""),
        ("failure_id", 7),
        ("event_id", " "),
        ("summary", True),
        ("observed_at", "2026-09-10T12:00:00"),
    ],
)
def test_failure_node_rejects_noncanonical_scalar_fields(
    field_name: str,
    value: object,
) -> None:
    """Catch absent, coercible, or timezone-ambiguous node identity and observations."""

    values: dict[str, object] = {
        "failure_id": "failure",
        "event_id": "event:failure",
        "summary": "Observed failure",
        "observed_at": "2026-09-10T12:00:00Z",
        "evidence_refs": (_observable(),),
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        FailureNode(**cast(Any, values))


def test_failure_node_requires_observable_canonical_evidence() -> None:
    """Catch failure observations based only on private or noncanonical evidence objects."""

    class ReferenceSubclass(EvidenceReference):
        pass

    subclass = ReferenceSubclass("observation:failure", EvidenceSourceKind.OBSERVABLE_OUTPUT)

    with pytest.raises(ValueError, match="observable evidence"):
        _node("failure", evidence_refs=[])
    with pytest.raises(ValueError, match="observable evidence"):
        _node("failure", evidence_refs=[_private()])
    with pytest.raises(ValueError, match="exact EvidenceReference"):
        _node("failure", evidence_refs=cast(Any, [subclass]))


def test_failure_edge_rejects_hostile_scalars_enums_and_reference_collections() -> None:
    """Catch scalar subclasses, enum strings, and unordered provenance at the edge boundary."""

    class HostileString(str):
        def strip(self, chars: str | None = None) -> str:
            return "pretend-valid"

    with pytest.raises(ValueError, match="source_id"):
        FailureEdge(
            HostileString(""),
            "target",
            FailureRelation.CAUSAL,
            ("verifier:one",),
        )
    with pytest.raises(ValueError, match="relation"):
        FailureEdge("source", "target", cast(Any, "causal"), ("verifier:one",))
    with pytest.raises(ValueError, match="verifier_refs"):
        FailureEdge(
            "source",
            "target",
            FailureRelation.CAUSAL,
            cast(Any, {"verifier:one"}),
        )


def test_localization_rejects_unknown_duplicate_or_noncanonical_references() -> None:
    """Catch ambiguous localization references before graph traversal."""

    with pytest.raises(ValueError, match="symptoms.*unique"):
        _localization(symptoms=("symptom", "symptom"))
    with pytest.raises(ValueError, match="first_anomaly"):
        _localization(first_anomaly=7)
    with pytest.raises(ValueError, match="symptoms"):
        _localization(symptoms={"symptom"})

    with pytest.raises(ValueError, match="dangling localization"):
        FailureGraph(
            (_node("anomaly"),),
            (),
            _localization(
                first_anomaly="missing",
                root_cause=None,
                decisive_failure=None,
                symptoms=(),
                recoverable_until=None,
            ),
        )


def test_records_snapshot_caller_collections_and_nested_records() -> None:
    """Catch caller mutation changing a previously validated failure graph."""

    evidence = _observable("observation:root")
    evidence_refs = [evidence]
    root = _node("root", evidence_refs=evidence_refs)
    decisive = _node("decisive")
    edge = _edge("root", "decisive", verifier_refs=["verifier:root"])
    symptoms: list[str] = []
    localization = _localization(
        first_anomaly="decisive",
        root_cause="root",
        decisive_failure="decisive",
        symptoms=cast(Any, symptoms),
        recoverable_until=None,
    )
    graph = FailureGraph([root, decisive], [edge], localization)

    evidence_refs.clear()
    symptoms.append("rewritten")
    object.__setattr__(evidence, "reference_id", "rewritten")
    object.__setattr__(root, "failure_id", "rewritten")
    object.__setattr__(edge, "source_id", "rewritten")
    object.__setattr__(localization, "first_anomaly", "rewritten")

    assert graph.nodes[0].failure_id == "root"
    assert graph.edges[0].source_id == "root"
    assert graph.localization.first_anomaly == "decisive"
    assert graph.nodes[0].evidence_refs == (_observable("observation:root"),)


def test_use_time_mutation_is_revalidated_before_serialization_or_traversal() -> None:
    """Catch frozen-record bypasses entering JSON or graph localization results."""

    serialized = _supported_graph()
    traversed = _supported_graph()
    object.__setattr__(serialized.nodes[0], "failure_id", " ")
    object.__setattr__(traversed.edges[0], "relation", "causal")

    with pytest.raises(ValueError, match="failure_id"):
        serialized.to_dict()
    with pytest.raises(ValueError, match="relation"):
        traversed.supported_topological_order()


def test_failure_graph_json_round_trip_is_exact_and_frozen() -> None:
    """Catch lossy persistence or mutability in the public failure-graph records."""

    graph = _supported_graph()
    restored = FailureGraph.from_dict(json.loads(json.dumps(graph.to_dict())))

    assert restored == graph
    assert restored.root_cause_status == "supported"
    for record in (
        restored,
        restored.nodes[0],
        restored.edges[0],
        restored.localization,
    ):
        assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        restored.nodes = ()


def test_graph_deserializer_validates_root_status_before_comparison() -> None:
    """Catch a hostile scalar executing equality while root status is being restored."""

    class HostileStatus:
        def __ne__(self, other: object) -> bool:
            raise RuntimeError("comparison should not run")

    payload = _supported_graph().to_dict()
    payload["root_cause_status"] = HostileStatus()

    with pytest.raises(ValueError, match="root_cause_status"):
        FailureGraph.from_dict(payload)


@pytest.mark.parametrize(
    ("constructor", "payload"),
    [
        (FailureNode.from_dict, []),
        (FailureNode.from_dict, {"failure_id": "failure"}),
        (FailureEdge.from_dict, []),
        (FailureEdge.from_dict, {"source_id": "source"}),
        (FailureLocalization.from_dict, []),
        (FailureLocalization.from_dict, {"first_anomaly": "anomaly"}),
        (FailureGraph.from_dict, []),
        (FailureGraph.from_dict, {"nodes": []}),
    ],
)
def test_deserializers_reject_wrong_shapes_and_partial_records(
    constructor: Any,
    payload: object,
) -> None:
    """Catch partial persisted records silently acquiring defaults."""

    with pytest.raises(ValueError):
        constructor(payload)


def test_public_package_exports_failure_graph_contract_without_breaking_prior_exports() -> None:
    """Catch the new public records being unreachable or replacing earlier contracts."""

    from gepa_mindfulness import verification

    assert verification.FailureGraph is FailureGraph
    assert verification.FailureNode is FailureNode
    assert verification.FailureRelation is FailureRelation
    assert verification.EvidenceState.__name__ == "EvidenceState"
    assert verification.LocalVerificationResult.__name__ == "LocalVerificationResult"


def test_relation_values_remain_distinct_in_serialized_output() -> None:
    """Catch noncausal epistemic qualifiers collapsing into the causal relation."""

    assert [relation.value for relation in FailureRelation] == [
        "causal",
        "contributing",
        "preceding",
        "correlated",
        "hypothesized",
    ]
