"""Current access and ancestor validity precede every artifact producer projection."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from artifact_fixtures import make_derived_snapshot, make_query, make_snapshot, make_topology

# Local
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.artifact_evidence import (
    ArtifactRetrieval,
    retrieve_artifact_evidence,
)
from gepa_mindfulness.verification.state import EvidenceState
from semantic_intent_robustness.memory_safety import MemoryTrustLevel


def retrieve(s=None, *, query=None, authorize=lambda request: True, mode="artifact_index"):
    """Use an explicit fixture ACL with no semantic judge."""
    s = make_snapshot() if s is None else s
    return retrieve_artifact_evidence(
        s,
        make_topology(s, "redundant"),
        make_query(s) if query is None else query,
        mode=mode,
        authorize=authorize,
        enabled=True,
    )


def change_source(s, **changes):
    """Keep the exact state claim consistent when changing one authored source."""
    source = replace(s.sources[0], **changes)
    claims = tuple(
        source.claim if c.claim_id == source.claim.claim_id else c for c in s.state.claims
    )
    return replace(s, sources=(source,) + s.sources[1:], state=EvidenceState(claims))


def test_disabled_has_no_callbacks():
    """Disabled entry points never execute ACL callbacks."""
    calls = []
    s = make_snapshot()
    with pytest.raises(ValueError, match="enabled"):
        retrieve_artifact_evidence(
            s, make_topology(s, "single"), make_query(s), authorize=lambda r: calls.append(r)
        )
    assert calls == []


def test_access_rechecked_after_revocation():
    """Previously allowed output cannot act as an access token on a second call."""
    s = make_snapshot()
    allowed = retrieve(s)
    restored = ArtifactRetrieval.from_dict(allowed.to_dict())
    assert restored == allowed
    calls = []
    denied = retrieve(s, authorize=lambda r: calls.append(r.artifact_key) or False)
    assert denied.selected_item_ids == ()
    assert set(calls) == {("A", "1"), ("B", "1")}
    assert allowed.selected_item_ids == ("a", "b")
    assert retrieve(s, authorize=None).selected_item_ids == ()


@pytest.mark.parametrize("failure", ["integer", "exception", "mutation", "false"])
def test_access_callback_failures(failure):
    """Only exact True with an unchanged descriptor allows source text."""

    def authorize(request):
        if failure == "exception":
            raise RuntimeError("SECRET-CREDENTIAL")
        if failure == "mutation":
            object.__setattr__(request, "principal_id", "different")
            return True
        return 1 if failure == "integer" else False

    result = retrieve(authorize=authorize)
    assert result.selected_item_ids == ()
    assert "SECRET-CREDENTIAL" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("block", ["denied", "deleted", "restricted", "unknown", "excluded"])
def test_transitive_summary_exclusion(block):
    """A mixed-source summary is withheld whole; an independent permitted original survives."""
    s = make_derived_snapshot()
    if block in {"deleted", "restricted", "unknown"}:
        s = replace(s, artifacts=(replace(s.artifacts[0], availability=block),) + s.artifacts[1:])
    q = make_query(s)
    if block == "excluded":
        q = replace(q, excluded_artifacts=(("A", "1"),))
    result = retrieve(
        s, query=q, authorize=lambda r: block != "denied" or r.artifact_key != ("A", "1")
    )
    assert result.selected_item_ids == ("b",)
    assert "summary" not in result.selected_item_ids
    assert s.interpretations[0].claim.proposition not in json.dumps(result.producer_view)


def test_exact_entity_matching():
    """One display name or shared artifact cannot merge distinct entity IDs."""
    s = make_snapshot()
    s = replace(s, entity_ids=("site", "other-site"))
    q = replace(make_query(s), entity_ids=("other-site",))
    assert retrieve(s, query=q).selected_item_ids == ()
    with pytest.raises(ValueError):
        retrieve(s, query=replace(q, entity_ids=("unknown",)))
    source = replace(s.sources[0], entity_ids=("site", "other-site"))
    artifact = replace(s.artifacts[0], entity_ids=("site", "other-site"))
    s = replace(s, sources=(source,) + s.sources[1:], artifacts=(artifact,) + s.artifacts[1:])
    result = retrieve(s, query=replace(q, entity_ids=("site",)))
    assert result.selected_item_ids == ("b",)


@pytest.mark.parametrize("seconds,expected", [(59, ("a", "b")), (60, ("a", "b")), (61, ())])
def test_validity_boundaries(seconds, expected):
    """The age threshold is inclusive and uses original observation timestamps."""
    s = make_snapshot()
    time = "2026-10-09T12:00:59Z" if seconds == 59 else f"2026-10-09T12:01:0{seconds - 60}Z"
    q = replace(make_query(s), assessed_at=time)
    assert retrieve(s, query=q).selected_item_ids == expected
    if seconds == 61:
        historical = retrieve(s, query=replace(q, purpose="historical"))
        assert historical.producer_view["items"][0]["channel"] == "historical"
        assert historical.producer_view["items"][0]["observed_at"] == ["2026-10-09T12:00:00Z"]


def test_future_and_derived_age():
    """Newly generated summaries cannot make old observations current."""
    s = make_derived_snapshot()
    future = replace(
        s, interpretations=(replace(s.interpretations[0], created_at="2026-10-10T12:00:00Z"),)
    )
    with pytest.raises(ValueError):
        retrieve(future)
    with pytest.raises(ValueError):
        retrieve(s, query=replace(make_query(s), assessed_at="2026-10-08T12:00:00Z"))
    q = replace(make_query(s), assessed_at="2026-10-10T12:00:00Z")
    assert retrieve(s, query=q).selected_item_ids == ()
    historical = retrieve(s, query=replace(q, purpose="historical"))
    assert {r["channel"] for r in historical.producer_view["items"]} == {"historical"}


@pytest.mark.parametrize(
    "status", ["stale", "contradicted", "superseded", "unavailable", "unverified"]
)
def test_declared_validity_channels(status):
    """History retains its label; no old claim silently becomes its replacement."""
    s = make_snapshot()
    claim = replace(
        s.sources[0].claim, status=status, superseded_by="b" if status == "superseded" else None
    )
    s = change_source(s, claim=claim)
    current = retrieve(s)
    if status == "unverified":
        assert next(r for r in current.item_rows if r["item_id"] == "a")["channel"] == "unverified"
    else:
        assert current.selected_item_ids == ("b",)
    history = retrieve(s, query=replace(make_query(s), purpose="historical"))
    if status not in {"unavailable", "unverified"}:
        assert next(r for r in history.item_rows if r["item_id"] == "a")["channel"] == "historical"


@pytest.mark.parametrize("flag", ["attempts_protected_override", "changes_goal_or_policy"])
def test_memory_policy(flag):
    """Unsafe memory cannot re-enter a producer view through source provenance."""
    s = make_snapshot()
    s = change_source(s, memory=replace(s.sources[0].memory, **{flag: True}))
    assert retrieve(s).selected_item_ids == ("b",)
    s = make_snapshot()
    s = change_source(
        s, memory=replace(s.sources[0].memory, trust_level=MemoryTrustLevel.UNTRUSTED)
    )
    assert next(r for r in retrieve(s).item_rows if r["item_id"] == "a")["channel"] == "untrusted"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_reliability", None),
        ("source_reliability", 0.7),
        ("compression_distortion", None),
        ("compression_distortion", 0.2),
        ("integrity", "unknown"),
        ("authority", "unknown"),
    ],
)
def test_unknown_or_failed_quality(field, value):
    """Unknown quality is explicit withheld evidence, never an assumed pass."""
    s = make_snapshot()
    s = change_source(s, quality=replace(s.sources[0].quality, **{field: value}))
    assert retrieve(s).selected_item_ids == ("b",)


@pytest.mark.parametrize("mode", ["artifact_index", "artifact_topology"])
def test_complete_producer_allowlist(mode):
    """The complete public projection contains no host labels, admission or denied content."""
    s = make_snapshot()
    s = replace(
        s,
        artifacts=(
            replace(s.artifacts[0], source_training_eligibility=TrainingEligibility.HIDDEN_EVAL),
        )
        + s.artifacts[1:],
    )
    s = change_source(s, quotation="SECRET_FROM_A")
    descriptors = []

    def authorize(r):
        descriptors.append(r.to_dict())
        return r.artifact_key != ("A", "1")

    result = retrieve(s, authorize=authorize, mode=mode)
    public = json.dumps(result.producer_view)
    for forbidden in (
        "SECRET_FROM_A",
        "principal-secret",
        "scope-secret",
        "attack",
        "HIDDEN_EVAL",
        "training_eligibility",
        "expected_actions",
        "record:b",
    ):
        assert forbidden not in public
    assert set(result.producer_view) == {"public_query", "items"}
    assert result.producer_view["items"][0]["handle"] == "item-0"
    assert all("quotation" not in d and "expected_actions" not in d for d in descriptors)
    assert all("SECRET_FROM_A" not in json.dumps(d) for d in descriptors)
    assert result.selected_item_ids == ("b",)
    data = result.to_dict()
    data["item_rows"][0]["authority"] = True
    with pytest.raises(ValueError):
        ArtifactRetrieval.from_dict(data)


def test_topology_mode_filters_incomplete_routes():
    """The topology treatment adds structural filtering without claiming semantic verification."""
    s = make_snapshot()
    t = make_topology(s, "synthesis")
    q = replace(make_query(s), excluded_artifacts=(("A", "1"),))
    indexed = retrieve_artifact_evidence(s, t, q, authorize=lambda r: True, enabled=True)
    topological = retrieve_artifact_evidence(
        s, t, q, mode="artifact_topology", authorize=lambda r: True, enabled=True
    )
    assert indexed.selected_item_ids == ("b",)
    assert topological.selected_item_ids == ()
