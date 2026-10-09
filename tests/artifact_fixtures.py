"""Authored artifact fixtures with explicit host identities and unverified interpretations."""

# Standard library
from dataclasses import replace
from hashlib import sha256

# Third-party
# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.artifact_records import (
    ArtifactLocation,
    ArtifactRecord,
    ArtifactSnapshot,
    DerivedInterpretation,
    SourceFragment,
    artifact_digest,
)
from gepa_mindfulness.verification.artifact_topology import EvidenceTopology, SupportRoute
from gepa_mindfulness.verification.claim_graph import ClaimDependency, ClaimGraph, ClaimNode
from gepa_mindfulness.verification.evidence_use import EvidenceQuality, EvidenceUsePolicy
from gepa_mindfulness.verification.state import ArtifactObservation, EvidenceClaim, EvidenceState
from semantic_intent_robustness.memory_safety import (
    MemorySourceType,
    MemoryTrustLevel,
    RetrievedMemory,
)
from semantic_intent_robustness.taxonomy import CapabilityTransferRisk

NOW = "2026-10-09T12:00:00Z"


def memory_for(claim, source):
    """Bind a reviewed memory to the exact claim without granting authority."""
    return RetrievedMemory(
        claim.claim_id,
        claim.proposition,
        MemorySourceType.EXTERNAL_CONTENT,
        MemoryTrustLevel.REVIEWED,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        CapabilityTransferRisk.LOW,
        source,
    )


def make_snapshot():
    """Two independently observed sources can support one separately judged conclusion."""
    artifacts, sources = [], []
    for key in ("a", "b"):
        refs = (EvidenceReference("record:" + key, EvidenceSourceKind.EXTERNAL_RECORD),)
        text = "Inspection establishes a material risk."
        obs = ArtifactObservation(
            "observation:" + key, "artifact:" + key, sha256(text.encode()).hexdigest(), NOW, refs
        )
        artifacts.append(
            ArtifactRecord(
                key.upper(), "1", obs, ("site",), "available", TrainingEligibility.DEVELOPMENT
            )
        )
        claim = EvidenceClaim(key, text, refs, "observed")
        sources.append(
            SourceFragment(
                key,
                (key.upper(), "1"),
                obs.digest,
                ("site",),
                ArtifactLocation("page", "1"),
                text,
                claim,
                memory_for(claim, "artifact:" + key),
                EvidenceQuality(NOW, 0.9, 0.0, "intact", "information_only", ("inspection",)),
                (),
            )
        )
    goal = EvidenceClaim("goal", "Disclose the material risk.", (), "unverified")
    return ArtifactSnapshot(
        ("site",),
        tuple(artifacts),
        tuple(sources),
        (),
        EvidenceState(tuple(s.claim for s in sources) + (goal,)),
    )


def make_derived_snapshot():
    """An old summary depends on both sources, even if one is later unavailable."""
    s = make_snapshot()
    refs = tuple(r for source in s.sources for r in source.claim.evidence_refs)
    claim = EvidenceClaim("goal", "The combined inspection indicates risk.", refs, "unverified")
    item = DerivedInterpretation(
        "summary",
        ("site",),
        claim,
        memory_for(claim, "transform:summary"),
        tuple((source.item_id, artifact_digest(source)) for source in s.sources),
        "summarize",
        "1",
        NOW,
        content_digest(claim.to_dict()),
    )
    return replace(
        s,
        interpretations=(item,),
        state=EvidenceState(tuple(source.claim for source in s.sources) + (claim,)),
    )


def make_topology(snapshot, layout):
    """Declare structural routes without claiming their semantic sufficiency."""
    nodes = tuple(
        ClaimNode(c, "host", None, "LEGACY_UNSPECIFIED", 1, 1) for c in snapshot.state.claims
    )
    refs = snapshot.sources[0].claim.evidence_refs
    edges = [("goal", "a", "requires")]
    routes = (SupportRoute("via-a", "goal", ("a",), ("a",)),)
    if layout == "redundant":
        edges = [("goal", "a", "supports"), ("goal", "b", "supports")]
        routes += (SupportRoute("via-b", "goal", ("b",), ("b",)),)
    elif layout == "synthesis":
        edges += [("goal", "b", "requires")]
        routes = (SupportRoute("joint", "goal", ("a", "b"), ("a", "b")),)
    elif layout == "bridge":
        edges = [("goal", "b", "requires"), ("b", "a", "requires")]
        routes = (SupportRoute("bridge", "goal", ("a", "b"), ("a", "b")),)
    elif layout != "single":
        raise ValueError("unknown fixture layout")
    graph = ClaimGraph(nodes, tuple(ClaimDependency(p, c, k, refs) for p, c, k in edges))
    return EvidenceTopology(artifact_digest(snapshot), graph, routes)


def make_query(snapshot):
    """A fixed-time scoped request, separate from any expected answer or evaluator label."""
    from gepa_mindfulness.verification.artifact_evidence import ArtifactQuery

    return ArtifactQuery(
        "request:attack:1",
        "What should the report disclose?",
        "principal-secret",
        "scope-secret",
        snapshot.entity_ids,
        (),
        NOW,
        "access-v1",
        EvidenceUsePolicy(60, 0.8, 0.1),
        "current",
        (),
    )
