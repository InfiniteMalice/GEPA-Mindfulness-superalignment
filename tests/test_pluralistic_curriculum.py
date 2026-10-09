"""Contrastive exports retain source admission and remain evaluator-only artifacts."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_laundering_families import family_metadata

# Local
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from synthetic_data.laundering_families import generate_controlled_laundering_families
from synthetic_data.pluralistic_curriculum import export_pluralistic_curriculum


def test_exports_preserve_holdout_and_all_contrastive_behaviors():
    """A forged outer TRAIN label cannot erase source restrictions or prompt lineage."""
    meta = family_metadata()
    key = "authority-reframing:attack:after"
    meta[key] = replace(meta[key], training_eligibility=TrainingEligibility.HIDDEN_EVAL)
    families = generate_controlled_laundering_families(cell_metadata=meta, enabled=True)
    rows = export_pluralistic_curriculum(families, curriculum_version="test-v1", enabled=True)
    assert len(rows) == 21
    row = next(r for r in rows if r["pair_id"] == "authority-reframing:attack")
    assert row["source_record"]["training_eligibility"] == "HIDDEN_EVAL"
    assert row["curriculum_version"] == "test-v1"
    assert row["arms"]["after"]["v5"]["robustness"]["subtype"] == "AUTHORITY_REFRAMING"
    assert {r["behavior"] for r in row["contrastive_examples"]} == {
        "stable_conclusion",
        "decisive_update",
        "resist_rationalization",
        "third_party_interest",
        "avoid_reflexive_agreement",
        "avoid_reflexive_disagreement",
        "safe_useful_benign",
    }
    assert all(c["preferred"] != c["dispreferred"] for c in row["contrastive_examples"])
    for r in rows:
        assert r["training_eligibility"] == "DEVELOPMENT"
        with pytest.raises(ValueError):
            require_training_eligible(r)
        with pytest.raises(ValueError):
            require_training_eligible({**r, "training_eligibility": "TRAIN"})
    with pytest.raises(ValueError):
        export_pluralistic_curriculum(families, curriculum_version="test-v1")
