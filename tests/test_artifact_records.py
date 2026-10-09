"""Artifact identity and ancestry cannot be rewritten by derived memory."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from artifact_fixtures import NOW, make_derived_snapshot, make_snapshot, memory_for

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.artifact_records import (
    ArtifactSnapshot,
    artifact_digest,
    source_ancestors,
)
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState


def test_snapshot_roundtrip():
    """Exact JSON is detached and carries non-training admission."""
    s = make_derived_snapshot()
    assert ArtifactSnapshot.from_dict(s.to_dict()) == s
    assert source_ancestors(s, "summary") == ("a", "b")
    assert s.interpretations[0].claim.status == "unverified"
    data = s.to_dict()
    assert data["training_eligibility"] == "DEVELOPMENT"
    data["sources"][0]["quotation"] = "changed"
    assert s.sources[0].quotation != "changed"
    for field in ("authority", "verified"):
        data = s.to_dict()
        data[field] = True
        with pytest.raises(ValueError):
            ArtifactSnapshot.from_dict(data)


@pytest.mark.parametrize("change", ["duplicate", "digest", "lineage_digest", "claim"])
def test_artifact_identity_replay(change):
    """IDs cannot hide changed version contents, source claims or transformation inputs."""
    s = make_derived_snapshot()
    with pytest.raises(ValueError):
        if change == "duplicate":
            replace(s, artifacts=s.artifacts + (replace(s.artifacts[0], availability="deleted"),))
        elif change == "digest":
            replace(s, sources=(replace(s.sources[0], artifact_digest="0" * 64),) + s.sources[1:])
        elif change == "lineage_digest":
            item = replace(s.interpretations[0], inputs=(("a", "0" * 64),))
            replace(s, interpretations=(item,))
        else:
            source = s.sources[0]
            c = replace(source.claim, proposition="Different claim")
            replace(
                s, sources=(replace(source, claim=c, memory=memory_for(c, "a")),) + s.sources[1:]
            )


@pytest.mark.parametrize("change", ["entity", "memory_id", "memory_content", "memory_flag"])
def test_entity_and_memory_binding(change):
    """Entity identities and exact memory fields remain bound to the source."""
    s = make_snapshot()
    with pytest.raises(ValueError):
        if change == "entity":
            replace(s, entity_ids=("different",))
        else:
            source = s.sources[0]
            args = {
                "memory_id": {"memory_id": "other"},
                "memory_content": {"content_summary": "other"},
                "memory_flag": {"provenance_retained": 1},
            }[change]
            replace(source, memory=replace(source.memory, **args))


def test_lineage_cycles_and_diamond():
    """Shared ancestry is deduplicated, while cycles and unknown inputs fail closed."""
    s = make_derived_snapshot()
    first = s.interpretations[0]
    c = replace(first.claim, claim_id="second")
    second = replace(
        first,
        item_id="second",
        claim=c,
        memory=memory_for(c, "second"),
        transform_id="second",
        output_digest=content_digest(c.to_dict()),
        inputs=(("summary", artifact_digest(first)), ("a", artifact_digest(s.sources[0]))),
    )
    diamond = replace(
        s, interpretations=(first, second), state=EvidenceState(s.state.claims + (c,))
    )
    assert source_ancestors(diamond, "second") == ("a", "b")
    for inputs in ((("summary", "0" * 64),), (("absent", "0" * 64),)):
        with pytest.raises(ValueError):
            replace(s, interpretations=(replace(first, inputs=inputs),))
    # Two-node cycle also cannot be laundered with invented digests.
    with pytest.raises(ValueError):
        replace(
            diamond,
            interpretations=(replace(first, inputs=(("second", artifact_digest(second)),)), second),
        )
    with pytest.raises(ValueError):
        replace(diamond, interpretations=(first, replace(second, transform_id=first.transform_id)))


@pytest.mark.parametrize("change", ["source_time", "derived_before_source", "bad_time"])
def test_original_timestamps(change):
    """Source age is bound to observation time and cannot be refreshed by a transform."""
    s = make_derived_snapshot()
    with pytest.raises(ValueError):
        if change == "source_time":
            src = s.sources[0]
            src = replace(src, quality=replace(src.quality, recorded_at="2026-10-10T12:00:00Z"))
            replace(s, sources=(src,) + s.sources[1:])
        else:
            time = "2026-10-08T12:00:00Z" if change == "derived_before_source" else "yesterday"
            replace(s, interpretations=(replace(s.interpretations[0], created_at=time),))
    assert s.sources[0].quality.recorded_at == NOW


def test_record_limits_and_admission():
    """The approved maxima are inclusive; admission and overflow cannot be coerced."""
    s = make_snapshot()
    artifacts = tuple(replace(s.artifacts[0], artifact_id=str(i)) for i in range(62))
    assert len(replace(s, artifacts=s.artifacts + artifacts).artifacts) == 64
    with pytest.raises(ValueError):
        replace(
            s, artifacts=s.artifacts + artifacts + (replace(s.artifacts[0], artifact_id="extra"),)
        )
    sources = tuple(replace(s.sources[0], item_id=str(i)) for i in range(254))
    assert len(replace(s, sources=s.sources + sources).sources) == 256
    with pytest.raises(ValueError):
        replace(s, sources=s.sources + sources + (replace(s.sources[0], item_id="extra"),))
    claims = tuple(EvidenceClaim(str(i), "A declaration", (), "unverified") for i in range(253))
    assert len(replace(s, state=EvidenceState(s.state.claims + claims)).state.claims) == 256
    with pytest.raises(ValueError):
        replace(
            s,
            state=EvidenceState(
                s.state.claims + claims + (EvidenceClaim("extra", "X", (), "unverified"),)
            ),
        )
    with pytest.raises(ValueError):
        replace(s.artifacts[0], source_training_eligibility=TrainingEligibility.TRAIN)
    hidden = replace(s.artifacts[0], source_training_eligibility=TrainingEligibility.HIDDEN_EVAL)
    assert hidden.to_dict()["source_training_eligibility"] == "HIDDEN_EVAL"
