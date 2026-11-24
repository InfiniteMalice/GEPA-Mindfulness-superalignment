import json
from dataclasses import replace

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.mdac import (
    MECHANISMS,
    MDACRecord,
    MechanismAssessment,
    load_diagnostic,
)


def test_registry_and_unknown_cause():
    assert set(MECHANISMS) == {f"M{i:02d}" for i in range(1, 17)}
    record = MDACRecord("run1:obs1", "answer omitted a constraint")
    assert record.assessments == ()
    assert record.frequency is None
    assert load_diagnostic(None) is None
    assert load_diagnostic(record.to_dict(), enabled=True) == record


def test_multilabel_roundtrip_and_measurement_availability():
    evidence = EvidenceReference("replay-1", EvidenceSourceKind.EXTERNAL_RECORD)
    first = MechanismAssessment("M01", "supported", supporting=(evidence,))
    second = MechanismAssessment("M02", "inconclusive", confidence=0)
    record = MDACRecord(
        "trial:1", "missed fact", assessments=(first, second), occurrences=0, opportunities=4
    )
    assert record.frequency == 0
    assert replace(record, occurrences=0, opportunities=0).frequency is None
    assert MDACRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    assert record.to_dict()["training_eligibility"] == "DEVELOPMENT"


@pytest.mark.parametrize(
    "changes",
    [
        {"opportunities": 0, "occurrences": 1},
        {"opportunities": True, "occurrences": 0},
        {"opportunities": 2},
        {"case_id": 18},
        {"severity": float("nan")},
        {"assessments": (MechanismAssessment("M01"),) * 2},
    ],
)
def test_invalid_record(changes):
    with pytest.raises(ValueError):
        MDACRecord("trial:2", "missing fact", **changes)


def test_evidence_and_authority_validation():
    with pytest.raises(ValueError):
        MechanismAssessment("M99")
    with pytest.raises(ValueError):
        MechanismAssessment("M15", "supported")
    encoded = MDACRecord("trial:2", "missing fact").to_dict()
    with pytest.raises(ValueError):
        MDACRecord.from_dict(encoded | {"training_eligibility": "OPTIMIZER"})
    with pytest.raises(ValueError):
        MDACRecord.from_dict(encoded | {"reward": -1})
    with pytest.raises(ValueError):
        load_diagnostic(encoded, enabled="false")


def test_schema_with_local_model():
    """Optional capture smoke for a locally supplied research checkpoint."""
    import os

    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL for model-backed capture")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    observed = tokenizer.decode(output[0, inputs.input_ids.shape[1] :])
    record = MDACRecord("local:1", observed.strip(), checkpoint_id=checkpoint)
    assert load_diagnostic(record.to_dict(), enabled=True) == record


@pytest.mark.parametrize("state", ["supported", "strongly_supported"])
def test_deception_hypothesis_requires_convergence_and_alternatives(state):
    first = EvidenceReference("matched-oversight", EvidenceSourceKind.EXTERNAL_RECORD)
    second = EvidenceReference("counterfactual-replay", EvidenceSourceKind.EXTERNAL_RECORD)
    alternatives = ("retrieval failure",)
    with pytest.raises(ValueError, match="M15"):
        MechanismAssessment("M15", state, supporting=(first,), alternatives=alternatives)
    with pytest.raises(ValueError, match="M15"):
        MechanismAssessment("M15", state, supporting=(first, second))
    assessment = MechanismAssessment(
        "M15", state, supporting=(first, second), alternatives=alternatives
    )
    assert MechanismAssessment.from_dict(assessment.to_dict()) == assessment
