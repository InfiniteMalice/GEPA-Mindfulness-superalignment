"""Skill text separation, lifecycle identity, and foundational protection."""

from dataclasses import replace

import pytest
from test_verified_skill_lifecycle import _ref, _store

from gepa_mindfulness.skill_bank import SkillBank, SkillCard, SkillChangeKind
from gepa_mindfulness.skill_lifecycle import SkillLifecycleState, transition_skill
from gepa_mindfulness.verification.failure_layers import FailureLayer


def card(tmp_path, *, foundational=False):
    store = _store(tmp_path)
    history = store.create_source("EvidenceConflictResolution", "v1", (_ref(),))
    return history, SkillCard.from_history(
        history,
        "When observable sources disagree.",
        "Preserve both sources and obtain independent verification.",
        foundational=foundational,
    )


def test_skill_card_preserves_when_how_and_lifecycle_identity(tmp_path):
    history, item = card(tmp_path)
    assert item.routing_description.startswith("When")
    assert item.operational_guidance.startswith("Preserve")
    assert item.artifact_id == history.current().artifact_id
    assert item.matches_history(history)
    assert item.to_dict()["confers_authority"] is False


@pytest.mark.parametrize("kind", list(SkillChangeKind))
def test_foundational_skill_cannot_be_autonomously_changed_or_retired(tmp_path, kind):
    _, item = card(tmp_path, foundational=True)
    bank = SkillBank((item,))
    proposal = bank.propose(item.skill_id, kind, "Improved performance", "replacement")
    assert proposal.blocked
    assert proposal.reason == "foundational_norm_requires_human_governance"
    assert bank.cards == (item,)


@pytest.mark.parametrize(
    "layer,kind",
    [
        (FailureLayer.ROUTING, SkillChangeKind.ROUTING_DESCRIPTION),
        (FailureLayer.KNOWLEDGE_SKILL, SkillChangeKind.OPERATIONAL_GUIDANCE),
    ],
)
def test_layer_specific_skill_proposals_do_not_mutate_lifecycle(tmp_path, layer, kind):
    history, item = card(tmp_path)
    before = history.snapshot()
    proposal = SkillBank((item,)).propose_for_layer(item.skill_id, layer, "review:1", "new text")
    assert proposal.kind == kind
    assert not proposal.blocked
    assert proposal.to_dict()["requires_held_out_validation"] is True
    assert history.snapshot() == before
    assert item.operational_guidance.startswith("Preserve")


@pytest.mark.parametrize(
    "layer",
    [
        layer
        for layer in FailureLayer
        if layer
        not in {
            FailureLayer.ROUTING,
            FailureLayer.KNOWLEDGE_SKILL,
        }
    ],
)
def test_other_layers_do_not_become_skill_training(tmp_path, layer):
    _, item = card(tmp_path)
    with pytest.raises(ValueError, match="skill text"):
        SkillBank((item,)).propose_for_layer(item.skill_id, layer, "review:1", "new text")


def test_duplicate_ids_empty_text_and_changed_artifact_are_rejected(tmp_path):
    history, item = card(tmp_path)
    with pytest.raises(ValueError):
        SkillBank((item, item))
    with pytest.raises(ValueError):
        replace(item, operational_guidance=" ")
    assert not replace(item, artifact_id="another-artifact").matches_history(history)


def test_lifecycle_transition_makes_old_card_stale(tmp_path):
    history, item = card(tmp_path)
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    assert not item.matches_history(history)


def test_operational_retirement_remains_only_a_review_proposal(tmp_path):
    history, item = card(tmp_path)
    bank = SkillBank((item,))
    before = history.snapshot()
    proposal = bank.propose(item.skill_id, SkillChangeKind.RETIRE, "review:performance")
    assert not proposal.blocked
    assert not proposal.to_dict()["confers_authority"]
    assert bank.cards == (item,)
    assert history.snapshot() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"foundational": 1},
        {"artifact_digest": "bad"},
        {"skill_id": ""},
        {"routing_description": 12},
    ],
)
def test_malformed_card_is_rejected(tmp_path, changes):
    _, item = card(tmp_path)
    with pytest.raises(ValueError):
        replace(item, **changes)


def test_unknown_skill_or_kind_cannot_make_a_proposal(tmp_path):
    _, item = card(tmp_path)
    bank = SkillBank((item,))
    with pytest.raises(ValueError):
        bank.propose("absent", SkillChangeKind.RETIRE, "review")
    with pytest.raises(ValueError):
        bank.propose(item.skill_id, "retire", "review")
