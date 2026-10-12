"""Strict input and source-envelope boundaries for the offline curriculum."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from curriculum_fixtures import curation_inputs
from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from synthetic_data.curriculum_records import (
    EpisodeSource,
    LessonReview,
    curriculum_digest,
    read_curriculum_object,
    validate_curriculum_envelope,
    validate_episode_roster,
)


def test_record_round_trip_detaches_sources():
    for value in curation_inputs().values():
        for record in value if isinstance(value, tuple) else (value,):
            if type(record).__module__ != "synthetic_data.curriculum_records":
                continue
            wire = record.to_dict()
            restored = type(record).from_dict(wire)
            assert restored == record
            assert curriculum_digest(restored) == curriculum_digest(record)
            wire["schema_version"] = "mutated"
            assert record.to_dict()["schema_version"] != "mutated"
    episode = curation_inputs()["episodes"][0]
    source = read_curriculum_object(episode.source_json)
    source["output"] = "changed"
    assert read_curriculum_object(episode.source_json)["output"] == "observed"


def test_nested_json_depth_and_size_bounds():
    read_curriculum_object('{"text":"[\\"{]"}')
    for depth in (32, 20000):
        with pytest.raises(ValueError):
            read_curriculum_object('{"x":' * depth + "0" + "}" * depth)
    read_curriculum_object('{"x":' * 31 + "0" + "}" * 31)
    small = canonical_json({"x": "x" * 1500000})
    with pytest.raises(ValueError):
        validate_curriculum_envelope({"sources": [small] * 3})
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{ "x":1}', '{"x":9007199254740992}'):
        with pytest.raises(ValueError):
            read_curriculum_object(text)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        validate_curriculum_envelope(cycle)


def test_duplicate_ids_and_ancestry_rejected():
    episode = curation_inputs()["episodes"][0]
    for episodes in (
        (episode, episode),
        (replace(episode, parent_episode_ids=("missing",)),),
        (replace(episode, parent_episode_ids=(episode.episode_id,)),),
    ):
        with pytest.raises(ValueError):
            validate_episode_roster(episodes)


def test_review_time_interval_is_exact():
    review = curation_inputs()["reviews"][0]
    for value in (review.valid_from, "2026-10-11", "2026-10-11T00:00:00+00:00"):
        with pytest.raises(ValueError):
            replace(review, valid_until=value)
    wire = review.to_dict()
    wire["human_required"] = 0
    with pytest.raises(ValueError):
        LessonReview.from_dict(wire)


def test_private_evidence_is_rejected():
    episode = curation_inputs()["episodes"][0]
    with pytest.raises(ValueError):
        replace(
            episode,
            evidence_refs=(EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING),),
        )


@pytest.mark.parametrize("field,value", [("episode_id", "x" * 257), ("terminal_status", "success")])
def test_invalid_fields_rejected(field, value):
    episode = curation_inputs()["episodes"][0]
    with pytest.raises(ValueError):
        replace(episode, **{field: value})
    wire = episode.to_dict()
    wire["extra"] = True
    with pytest.raises(ValueError):
        EpisodeSource.from_dict(wire)
