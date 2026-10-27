"""CPU admission, concurrency and replay tests for reviewed credit accounting."""

# Standard library
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.causal_reward_records import request_digest
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.trajectory import Trajectory
from test_research_credit import claim


def request():
    """Construct a real typed training target with observable evidence."""
    ref = EvidenceReference("observation", EvidenceSourceKind.EXTERNAL_RECORD)
    trajectory = Trajectory(
        "trajectory",
        None,
        "question",
        "answer",
        evidence_references=(ref,),
        sampling_parameters={"training_eligibility": "TRAIN"},
        policy_version="student-v1",
    )
    return RewardRequest(trajectory, (ref,))


def proposal(target=None, **changes):
    """Bind one independently reviewed claim to exact target bytes."""
    target = target or request()
    metadata = json.dumps(
        {"training_eligibility": "TRAIN", "target_request_digest": request_digest(target)}
    )
    return claim(admissibility="admitted", source_metadata_json=metadata, **changes)


def use(target=None, **changes):
    """Declare every existing reward layer's overlap explicitly."""
    values = dict(
        use_id="use-1",
        policy_id="host-policy-v1",
        request_digest=request_digest(target or request()),
        selected_claim_ids=("claim-1",),
        layers=(
            (
                "claim-1",
                (
                    ("provider", "not_counted"),
                    ("reward_integrity", "not_counted"),
                    ("causal_reward", "not_counted"),
                ),
            ),
        ),
    )
    values.update(changes)
    return ResearchCreditUse(**values)


class ResearchCreditLedgerTests(unittest.TestCase):
    """Host review and durable accounting remain separate from reward authority."""

    def setUp(self):
        """Give each test its own host-owned journal."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "credits.sqlite"
        self.ledger = ResearchCreditLedger(self.path, authority_domain="host")

    def reserve(self, **changes):
        """Submit the complete baseline proposal unless overridden by a test."""
        values = dict(
            request=request(),
            claims=(proposal(),),
            use=use(),
            authorize_use=lambda envelope: True,
            enabled=True,
        )
        values.update(changes)
        return self.ledger.reserve(**values)

    def test_default_off_leaves_input_and_journal_untouched(self):
        """Importing or disabled invocation does not consume an event."""
        self.assertIsNone(self.ledger.reserve(object(), object(), object()))
        self.assertEqual(self.reserve()["status"], "reserved")

    def test_success_is_durable_development_evidence(self):
        """An allocation receipt is never a scalar reward or execution grant."""
        receipt = self.reserve()
        self.assertFalse(receipt["authority_granted"])
        self.assertNotIn("total", receipt)
        with self.assertRaises(ValueError):
            require_training_eligible(receipt)
        self.assertEqual(self.ledger.read(receipt["receipt_digest"]), receipt)
        receipt["status"] = "forged"
        self.assertEqual(self.ledger.read(receipt["receipt_digest"])["status"], "reserved")

    def test_replay_after_restart_is_rejected(self):
        """Changing use/policy identity cannot pay for the same event again."""
        self.reserve()
        reopened = ResearchCreditLedger(self.path, authority_domain="host")
        for candidate in (use(), use(use_id="other", policy_id="other")):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    reopened.reserve(
                        request(),
                        (proposal(),),
                        candidate,
                        authorize_use=lambda envelope: True,
                        enabled=True,
                    )

    def test_domain_is_pinned_on_reopen(self):
        """One journal cannot silently change its authority namespace."""
        with self.assertRaises(ValueError):
            ResearchCreditLedger(self.path, authority_domain="foreign")

    def test_missing_rejected_and_exceptional_host_review_do_not_reserve(self):
        """A producer's admitted label never replaces fresh host review."""

        def broken(envelope):
            """Simulate a failing host reviewer without exposing its error text."""
            raise RuntimeError("private verifier detail")

        for callback in (None, lambda envelope: False, lambda envelope: 1, broken):
            with self.subTest(callback=callback):
                with self.assertRaises(ValueError) as error:
                    self.reserve(authorize_use=callback)
                self.assertNotIn("private verifier detail", str(error.exception))
        self.assertEqual(self.reserve()["status"], "reserved")

    def test_mutating_review_cannot_change_allocation(self):
        """The complete callback envelope is immutable by content binding."""

        def mutate(envelope):
            """Attempt to alter the reviewed selection before returning approval."""
            envelope["use"]["selected_claim_ids"] = []
            return True

        with self.assertRaises(ValueError):
            self.reserve(authorize_use=mutate)
        self.reserve()

    def test_request_and_claim_target_bindings_are_both_required(self):
        """A valid claim cannot be moved to a different rollout."""
        foreign = replace(request(), trajectory=replace(request().trajectory, response="other"))
        with self.assertRaises(ValueError):
            self.reserve(request=foreign)
        with self.assertRaises(ValueError):
            self.reserve(claims=(claim(admissibility="admitted"),))
        self.reserve()

    def test_evidence_and_model_version_bind_to_request(self):
        """Foreign evidence or a stale policy version cannot be allocated."""
        for target in (
            RewardRequest(request().trajectory, ()),
            RewardRequest(
                replace(request().trajectory, policy_version="foreign"),
                request().observable_references,
            ),
        ):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    self.reserve(request=target, use=use(target), claims=(proposal(target),))

    def test_ineligible_source_and_unknown_admissibility_are_rejected(self):
        """Selected proposals must pass every existing structural prerequisite."""
        for value in (
            replace(proposal(), admissibility="unresolved"),
            replace(proposal(), provenance=None),
            replace(proposal(), credit_kind="teacher_agreement"),
            replace(proposal(), source_metadata_json='{"training_eligibility":"HIDDEN_EVAL"}'),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.reserve(claims=(value,))

    def test_missing_overlap_or_already_counted_layer_is_rejected(self):
        """A layer omission cannot be interpreted as zero prior credit."""
        for layers in (
            (("provider", "not_counted"),),
            (
                ("provider", "already_counted"),
                ("reward_integrity", "not_counted"),
                ("causal_reward", "not_counted"),
            ),
        ):
            with self.subTest(layers=layers):
                with self.assertRaises(ValueError):
                    self.reserve(use=use(layers=(("claim-1", layers),)))

    def test_only_one_selected_claim_per_semantic_event(self):
        """Knowledge and progress proposals cannot both consume one verified event."""
        second = replace(proposal(), claim_id="claim-2")
        both = use(
            selected_claim_ids=("claim-1", "claim-2"),
            layers=use().layers + (("claim-2", use().layers[0][1]),),
        )
        with self.assertRaises(ValueError):
            self.reserve(claims=(proposal(), second), use=both)
        receipt = self.reserve(claims=(proposal(), second))
        self.assertEqual(len(receipt["envelope"]["audit"]["claims"]), 2)

    def test_use_wire_is_exact_versioned_and_non_authoritative(self):
        """Serialized proposals cannot restore an allocation grant."""
        wire = use().to_dict()
        self.assertEqual(ResearchCreditUse.from_dict(wire), use())
        for field, value in (
            ("schema_version", "future"),
            ("training_eligibility", "TRAIN"),
            ("extra", True),
        ):
            with self.subTest(field=field):
                changed = {**wire, field: value}
                with self.assertRaises(ValueError):
                    ResearchCreditUse.from_dict(changed)

    def test_concurrent_consumers_have_one_winner(self):
        """Separate connections serialize conflicting reservations."""
        barrier = Barrier(2)

        def compete(index):
            """Race a separate host connection for the same semantic event."""
            ledger = ResearchCreditLedger(self.path, authority_domain="host")
            barrier.wait()
            try:
                ledger.reserve(
                    request(),
                    (proposal(),),
                    use(use_id=str(index)),
                    authorize_use=lambda envelope: True,
                    enabled=True,
                )
                return True
            except ValueError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(compete, (0, 1))), [False, True])

    def test_conflicting_evidence_kinds_do_not_reach_host_review(self):
        """One scoped evidence identity cannot acquire a second capture kind."""
        first = proposal()
        second = replace(
            first,
            claim_id="claim-2",
            event_id="second-event",
            provenance=None,
            evidence_refs=(EvidenceReference("observation", EvidenceSourceKind.OBSERVABLE_OUTPUT),),
        )
        calls = []
        with self.assertRaises(ValueError):
            self.reserve(
                claims=(first, second), authorize_use=lambda envelope: calls.append(1) or True
            )
        self.assertEqual(calls, [])

    def test_atomic_failure_does_not_partially_reserve(self):
        """Any SQLite failure rolls back the allocation and all event rows."""
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TRIGGER reject_credit BEFORE INSERT ON credit_events "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
            )
        with self.assertRaises(sqlite3.DatabaseError):
            self.reserve()
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TRIGGER reject_credit")
        self.reserve()

    def test_reservation_does_not_change_existing_reward_pipeline(self):
        """Recording an allocation cannot silently add research credit to reward."""

        class Provider:
            """A stable existing outcome-only provider."""

            def score(self, target):
                """Return the same base outcome score before and after reservation."""
                return 1.5

        pipeline = RewardPipeline(Provider())
        before = pipeline.score(request())
        self.reserve()
        after = pipeline.score(request())
        self.assertEqual(before.total, after.total)
        self.assertEqual(after.total, 1.5)

    def test_corrupted_receipt_fails_integrity_check(self):
        """A changed stored receipt is not returned as accepted audit evidence."""
        receipt = self.reserve()
        changed = {**receipt, "status": "forged"}
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE credit_reservations SET payload=?", (json.dumps(changed),))
        with self.assertRaises(ValueError):
            self.ledger.read(receipt["receipt_digest"])

    def test_foreign_authority_and_unknown_claim_are_rejected(self):
        """A selection cannot escape the journal's domain or invent claim identity."""
        for changes in (
            {"claims": (replace(proposal(), authority_domain="foreign"),)},
            {
                "use": use(
                    selected_claim_ids=("unknown",), layers=(("unknown", use().layers[0][1]),)
                )
            },
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.reserve(**changes)

    def test_distinct_events_can_be_allocated_atomically(self):
        """Independently reviewed distinct facts can share a source reference."""
        second = replace(proposal(), claim_id="claim-2", event_id="second-fact")
        selected = use(
            selected_claim_ids=("claim-1", "claim-2"),
            layers=use().layers + (("claim-2", use().layers[0][1]),),
        )
        receipt = self.reserve(claims=(proposal(), second), use=selected)
        self.assertEqual(receipt["envelope"]["audit"]["event_overlaps"], [])
        self.assertEqual(receipt["envelope"]["audit"]["shared_evidence"], [["claim-1", "claim-2"]])
        with self.assertRaises(ValueError):
            self.reserve(
                claims=(second,),
                use=use(
                    use_id="retry",
                    selected_claim_ids=("claim-2",),
                    layers=(("claim-2", use().layers[0][1]),),
                ),
            )


if __name__ == "__main__":
    unittest.main()
