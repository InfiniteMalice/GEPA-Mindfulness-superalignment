"""CPU-safe negative tests for cross-method diagnostic credit accounting."""

# Standard library
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.research_credit import ResearchCreditClaim, audit_research_credit


def claim(**changes):
    """One proposed knowledge contribution; all evidence is caller-declared."""
    ref = EvidenceReference("observation", EvidenceSourceKind.EXTERNAL_RECORD)
    values = dict(
        claim_id="claim-1",
        method_id="KnowGPO",
        component_name="knowledge",
        credit_kind="knowledge",
        source_reference="operator-spec:2026-10-26",
        authority_domain="host",
        run_id="run",
        task_id="task",
        event_id="semantic-event",
        action_id="action",
        model_version="student-v1",
        harness_version="harness-v1",
        producer_id="student",
        evaluator=TrustedEvaluatorContract("independent", "v1", "contract-v1"),
        dataset_digest="a" * 64,
        state_digest="b" * 64,
        evidence_refs=(ref,),
        score=0.5,
        admissibility="unresolved",
        provenance=RewardProvenance(
            "knowledge", "observed", VerificationRoute.OBSERVABLE_EVIDENCE, (ref,)
        ),
        source_metadata_json=json.dumps({"training_eligibility": "TRAIN"}),
    )
    values.update(changes)
    return ResearchCreditClaim(**values)


class ResearchCreditTests(unittest.TestCase):
    def test_disabled_does_not_inspect_input(self):
        """Disabled does not inspect input."""
        self.assertIsNone(audit_research_credit(object()))
        with self.assertRaises(ValueError):
            audit_research_credit((), enabled=1)

    def test_exact_roundtrip_and_detached_wire(self):
        """Exact roundtrip and detached wire."""
        original = claim()
        wire = original.to_dict()
        self.assertEqual(ResearchCreditClaim.from_dict(json.loads(json.dumps(wire))), original)
        wire["evidence_refs"][0]["reference_id"] = "changed"
        self.assertEqual(original.evidence_refs[0].reference_id, "observation")

    def test_report_never_authorizes_optimizer_or_execution(self):
        """Report never authorizes optimizer or execution."""
        report = audit_research_credit((claim(admissibility="admitted"),), enabled=True)
        self.assertFalse(report["authority_granted"])
        self.assertNotIn("total", report)
        self.assertNotIn("optimizer_reward", report)
        for wire in (report, claim().to_dict(), {"training_eligibility": "TRAIN", "x": report}):
            with self.subTest(wire=wire):
                with self.assertRaises(ValueError):
                    require_training_eligible(wire)

    def test_unresolved_and_unverified_remain_separate(self):
        """Unresolved and unverified remain separate."""
        report = audit_research_credit((claim(provenance=None),), enabled=True)
        self.assertEqual(
            report["prerequisite_gaps"]["claim-1"], ["unresolved_admissibility", "unverified"]
        )

    def test_rejected_admissibility_is_not_a_rule_violation(self):
        """Rejected admissibility is not a rule violation."""
        report = audit_research_credit((claim(admissibility="rejected"),), enabled=True)
        self.assertIn("admissibility_rejected", report["prerequisite_gaps"]["claim-1"])
        self.assertNotIn("moral_violation", json.dumps(report))

    def test_same_event_credit_is_reported_across_components(self):
        """Same event credit is reported across components."""
        first = claim()
        second = claim(
            claim_id="claim-2", method_id="PRM4RL", component_name="progress", provenance=None
        )
        report = audit_research_credit((first, second), enabled=True)
        self.assertEqual(report["event_overlaps"], [["claim-1", "claim-2"]])
        self.assertEqual(len(report["claims"]), 2)

    def test_different_events_sharing_evidence_require_review(self):
        """Different events sharing evidence require review."""
        second = claim(claim_id="claim-2", event_id="another-fact")
        report = audit_research_credit((claim(), second), enabled=True)
        self.assertEqual(report["event_overlaps"], [])
        self.assertEqual(report["shared_evidence"], [["claim-1", "claim-2"]])
        self.assertTrue(report["requires_host_review"])

    def test_event_scope_prevents_foreign_run_collision(self):
        """Event scope prevents foreign run collision."""
        second = claim(claim_id="claim-2", run_id="other-run")
        report = audit_research_credit((claim(), second), enabled=True)
        self.assertEqual(report["event_overlaps"], [])

    def test_same_event_cannot_change_action_or_system_binding(self):
        """Same event cannot change action or system binding."""
        for changes in ({"action_id": "other"}, {"model_version": "other"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    audit_research_credit(
                        (claim(), claim(claim_id="claim-2", **changes)), enabled=True
                    )

    def test_duplicate_claim_identity_rejected(self):
        """Duplicate claim identity rejected."""
        with self.assertRaises(ValueError):
            audit_research_credit((claim(), claim()), enabled=True)

    def test_nested_holdout_provenance_is_retained_and_flagged(self):
        """Nested holdout provenance is retained and flagged."""
        metadata = {
            "training_eligibility": "TRAIN",
            "source_record": {"holdout_status": "held_out"},
        }
        original = claim(source_metadata_json=json.dumps(metadata))
        report = audit_research_credit((original,), enabled=True)
        self.assertIn("source_ineligible", report["prerequisite_gaps"]["claim-1"])
        self.assertEqual(json.loads(original.source_metadata_json), metadata)

    def test_missing_train_source_is_not_legacy_admitted(self):
        """Missing train source is not legacy admitted."""
        report = audit_research_credit((claim(source_metadata_json="{}"),), enabled=True)
        self.assertIn("source_ineligible", report["prerequisite_gaps"]["claim-1"])

    def test_private_evidence_and_unbound_provenance_rejected(self):
        """Private evidence and unbound provenance rejected."""
        private = EvidenceReference("thought", EvidenceSourceKind.PRIVATE_REASONING)
        with self.assertRaises(ValueError):
            claim(evidence_refs=(private,))
        with self.assertRaises(ValueError):
            claim(evidence_refs=(EvidenceReference("other", EvidenceSourceKind.EXTERNAL_RECORD),))

    def test_component_and_evaluator_identity_must_match(self):
        """Component and evaluator identity must match."""
        with self.assertRaises(ValueError):
            claim(component_name="different")
        trusted = RewardProvenance(
            "knowledge",
            "review",
            VerificationRoute.TRUSTED_EVALUATOR,
            evaluator=TrustedEvaluatorContract("foreign", "v1", "contract-v1"),
        )
        with self.assertRaises(ValueError):
            claim(provenance=trusted)
        with self.assertRaises(ValueError):
            claim(producer_id="independent")

    def test_matching_trusted_evaluator_route_roundtrips(self):
        """Matching trusted evaluator route roundtrips."""
        evaluator = claim().evaluator
        provenance = RewardProvenance(
            "knowledge", "review", VerificationRoute.TRUSTED_EVALUATOR, evaluator=evaluator
        )
        value = claim(provenance=provenance)
        self.assertEqual(ResearchCreditClaim.from_dict(value.to_dict()), value)

    def test_finite_scores_exact_digests_and_status(self):
        """Finite scores exact digests and status."""
        for score in (True, float("nan"), float("inf")):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    claim(score=score)
        for changes in (
            {"dataset_digest": "bad"},
            {"state_digest": "A" * 64},
            {"admissibility": True},
            {"claim_id": " "},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    claim(**changes)

    def test_unknown_version_fields_and_train_relabel_rejected(self):
        """Unknown version fields and train relabel rejected."""
        for key, value in (
            ("schema_version", "future"),
            ("extra", 1),
            ("training_eligibility", "TRAIN"),
        ):
            wire = claim().to_dict()
            wire[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    ResearchCreditClaim.from_dict(wire)

    def test_hostile_metadata_is_rejected(self):
        """Hostile metadata is rejected."""
        for text in ('{"x": NaN}', '{"x":1,"x":2}', "[]", "{", "[" * 40 + "]" * 40):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    claim(source_metadata_json=text)

    def test_nested_provenance_mutation_revalidated(self):
        """Nested provenance mutation revalidated."""
        value = claim()
        object.__setattr__(value.provenance, "component_name", "foreign")
        with self.assertRaises(ValueError):
            audit_research_credit((value,), enabled=True)

    def test_trusted_route_cannot_hide_injected_observable_evidence(self):
        """Trusted route cannot hide injected observable evidence."""
        evaluator = claim().evaluator
        provenance = RewardProvenance(
            "knowledge", "review", VerificationRoute.TRUSTED_EVALUATOR, evaluator=evaluator
        )
        value = claim(provenance=provenance)
        object.__setattr__(value.provenance, "evidence_refs", value.evidence_refs)
        with self.assertRaises(ValueError):
            audit_research_credit((value,), enabled=True)

    def test_negative_process_credit_rejected(self):
        """Negative process credit rejected."""
        with self.assertRaises(ValueError):
            replace(claim(), credit_kind="process", score=-0.1, provenance=None)

    def test_contradictory_route_fields_cannot_disappear_during_snapshot(self):
        """Contradictory route fields cannot disappear during snapshot."""
        value = claim(admissibility="admitted")
        object.__setattr__(
            value.provenance, "evaluator", TrustedEvaluatorContract("student", "v1", "self")
        )
        with self.assertRaises(ValueError):
            audit_research_credit((value,), enabled=True)

    def test_aggregate_budget_stops_before_restoring_all_claims(self):
        """Aggregate budget stops before restoring all claims."""
        metadata = json.dumps({"training_eligibility": "TRAIN", "padding": "x" * 2_200_000})
        values = tuple(claim(claim_id=str(i), source_metadata_json=metadata) for i in range(3))
        with patch.object(
            ResearchCreditClaim, "from_dict", wraps=ResearchCreditClaim.from_dict
        ) as restore:
            with self.assertRaises(ValueError):
                audit_research_credit(values, enabled=True)
            self.assertEqual(restore.call_count, 1)


if __name__ == "__main__":
    unittest.main()
