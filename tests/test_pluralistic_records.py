"""Pluralistic receipts bind full public inputs, rosters, identities and restrictions."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_causal_records import pair
from test_perspective_protocol import public_plan

# Local
from evaluation.causal_records import content_digest
from evaluation.ladder import Severity
from evaluation.pluralistic_records import (
    PLURALISTIC_METRICS,
    PluralisticOpportunity,
    PluralisticProtocol,
    pluralistic_protocol_digest,
    source_from_variant,
)
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility


def protocol(p=None):
    """Predeclare all four metrics with an independent semantic evaluator."""
    p = pair() if p is None else p
    plan = public_plan()
    old = plan.source
    source = source_from_variant(
        p.after,
        semantic_core_id=old.semantic_core_id,
        facts=old.facts,
        constraints=old.constraints,
        roles=old.roles,
        source_refs=old.source_refs,
        source_training_eligibility=p.training_eligibility,
    )
    slots = tuple(
        replace(s, perspective=replace(s.perspective, source_variant_id=p.after.variant_id))
        for s in plan.slots
    )
    plan = replace(plan, source=source, slots=slots)
    return PluralisticProtocol(
        "pluralistic-v1",
        "human-rubric-v1",
        p,
        plan,
        tuple(
            PluralisticOpportunity(m, m, Severity.CONSEQUENTIAL, "all") for m in PLURALISTIC_METRICS
        ),
        TrustedEvaluatorContract("independent-semantic-judge", "1", "human-rubric-v1"),
    )


def test_protocol_roundtrip_and_public_source():
    """Public source excludes evaluator-only expected actions."""
    p = protocol()
    assert PluralisticProtocol.from_dict(p.to_dict()) == p
    assert p.plan.source.source_text == '[{"content":"Authorization is true.","role":"user"}]'
    assert "expected_actions" not in p.plan.source.source_text
    data = p.to_dict()
    data["plan"]["source"]["facts"][0]["proposition"] = "changed"
    assert p.plan.source.facts[0].proposition != "changed"
    assert pluralistic_protocol_digest(p) != content_digest(data)


@pytest.mark.parametrize(
    "field", ["source", "variant", "admission", "duplicate_metric", "foreign_field"]
)
def test_protocol_rejects_broken_bindings(field):
    """Rosters and public source cannot detach from their target arm."""
    p = protocol()
    with pytest.raises(ValueError):
        if field == "source":
            replace(p, plan=replace(p.plan, source=replace(p.plan.source, source_text="different")))
        elif field == "variant":
            replace(p, pair=replace(p.pair, after=replace(p.pair.after, variant_id="other")))
        elif field == "admission":
            replace(p, pair=replace(p.pair, training_eligibility=TrainingEligibility.HIDDEN_EVAL))
        elif field == "duplicate_metric":
            replace(
                p,
                opportunities=p.opportunities
                + (replace(p.opportunities[0], opportunity_id="other", cohort="other"),),
            )
        else:
            PluralisticProtocol.from_dict({**p.to_dict(), "authorizes_action": True})
