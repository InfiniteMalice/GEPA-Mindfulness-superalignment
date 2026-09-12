"""Contracts for distinct local-execution and relational-evidence verification."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast, get_type_hints

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import EvidenceClaim, EvidenceState
from gepa_mindfulness.verification.interfaces import (
    LocalExecutionVerifier,
    LocalVerificationResult,
    RelationalEvidenceVerifier,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
    VerificationLevel,
    make_local_verification_event,
    make_relational_verification_event,
)
from mindful_trace_gepa.action_bound_events import ActionRecord
from mindful_trace_gepa.logging_schema import EventEnvelope


def _observable(reference_id: str = "output:action-1") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _private(reference_id: str = "private:action-1") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.PRIVATE_REASONING)


def _binding(
    field_name: str,
    evidence_refs: tuple[EvidenceReference, ...],
) -> VerificationEvidenceBinding:
    return VerificationEvidenceBinding(field_name, evidence_refs)


def _local(
    *,
    action_id: str = "action-1",
    executed: bool = True,
    arguments_valid: bool = True,
    schema_valid: bool = True,
    authorization_valid: bool = True,
    intended_operation_observed: bool = True,
    irreversible_action_permitted: bool | None = None,
    evidence_refs: tuple[EvidenceReference, ...] = (_observable(),),
    evidence_bindings: tuple[VerificationEvidenceBinding, ...] | None = None,
) -> LocalVerificationResult:
    if evidence_bindings is None:
        fields = {
            "executed": executed,
            "arguments_valid": arguments_valid,
            "schema_valid": schema_valid,
            "authorization_valid": authorization_valid,
            "intended_operation_observed": intended_operation_observed,
            "irreversible_action_permitted": irreversible_action_permitted is True,
        }
        evidence_bindings = tuple(
            _binding(field_name, evidence_refs)
            for field_name, affirmative in fields.items()
            if affirmative is True
        )
    return LocalVerificationResult(
        action_id,
        executed,
        arguments_valid,
        schema_valid,
        authorization_valid,
        intended_operation_observed,
        irreversible_action_permitted,
        evidence_refs,
        evidence_bindings,
    )


def _relational(
    *,
    action_id: str = "action-1",
    task_fit: bool = False,
    dependencies_satisfied: bool = False,
    contradiction_status: str = "unknown",
    provenance_intact: bool = False,
    authorization_scope_valid: bool = False,
    claimed_outcome_supported: bool = False,
    repeated_failed_route: bool = False,
    evidence_refs: tuple[EvidenceReference, ...] = (),
    evidence_bindings: tuple[VerificationEvidenceBinding, ...] | None = None,
) -> RelationalVerificationResult:
    if evidence_bindings is None:
        fields = {
            "task_fit": task_fit,
            "dependencies_satisfied": dependencies_satisfied,
            "provenance_intact": provenance_intact,
            "authorization_scope_valid": authorization_scope_valid,
            "claimed_outcome_supported": claimed_outcome_supported,
            "repeated_failed_route": repeated_failed_route,
            "contradiction_status": contradiction_status in {"none", "contradicted"},
        }
        evidence_bindings = tuple(
            _binding(field_name, evidence_refs)
            for field_name, affirmative in fields.items()
            if affirmative is True
        )
    return RelationalVerificationResult(
        action_id,
        task_fit,
        dependencies_satisfied,
        contradiction_status,
        provenance_intact,
        authorization_scope_valid,
        claimed_outcome_supported,
        repeated_failed_route,
        evidence_refs,
        evidence_bindings,
    )


def test_local_success_does_not_imply_relational_support() -> None:
    """Catch a local result being treated as evidence that a claimed outcome is supported."""

    local = _local()
    relational = _relational()

    assert local.executed is True
    assert local.intended_operation_observed is True
    assert relational.claimed_outcome_supported is False
    assert type(local) is not type(relational)
    assert not hasattr(local, "claimed_outcome_supported")
    assert not hasattr(relational, "executed")


def test_relational_result_rejects_fabricated_local_execution_field() -> None:
    """Catch relational deserialization accepting an assertion about local execution."""

    payload = _relational().to_dict()
    payload["executed"] = True

    with pytest.raises(ValueError, match="exactly"):
        RelationalVerificationResult.from_dict(payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "executed",
        "arguments_valid",
        "schema_valid",
        "authorization_valid",
        "intended_operation_observed",
        "irreversible_action_permitted",
    ],
)
def test_local_affirmative_fields_require_observable_evidence(field_name: str) -> None:
    """Catch an affirmative local finding supported only by private reasoning."""

    values: dict[str, Any] = {
        "executed": False,
        "arguments_valid": False,
        "schema_valid": False,
        "authorization_valid": False,
        "intended_operation_observed": False,
        "irreversible_action_permitted": None,
        "evidence_refs": (_private(),),
    }
    values[field_name] = True

    with pytest.raises(ValueError, match="observable evidence"):
        _local(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "task_fit",
        "dependencies_satisfied",
        "provenance_intact",
        "authorization_scope_valid",
        "claimed_outcome_supported",
        "repeated_failed_route",
    ],
)
def test_relational_affirmative_fields_require_observable_evidence(field_name: str) -> None:
    """Catch an affirmative relational finding supported only by latent state."""

    values: dict[str, Any] = {field_name: True, "evidence_refs": (_private(),)}

    with pytest.raises(ValueError, match="observable evidence"):
        _relational(**values)


def test_negative_results_can_preserve_nonobservable_diagnostic_references() -> None:
    """Catch negative verifier results discarding diagnostic provenance unnecessarily."""

    local = _local(
        executed=False,
        arguments_valid=False,
        schema_valid=False,
        authorization_valid=False,
        intended_operation_observed=False,
        evidence_refs=(_private(),),
    )
    relational = _relational(evidence_refs=(_private(),))

    assert local.evidence_refs[0].source_kind is EvidenceSourceKind.PRIVATE_REASONING
    assert relational.evidence_refs[0].source_kind is EvidenceSourceKind.PRIVATE_REASONING


@pytest.mark.parametrize(
    "value",
    [1, 0, "true", None],
)
@pytest.mark.parametrize(
    "field_name",
    [
        "executed",
        "arguments_valid",
        "schema_valid",
        "authorization_valid",
        "intended_operation_observed",
    ],
)
def test_local_result_requires_exact_boolean_fields(field_name: str, value: object) -> None:
    """Catch truthy and falsy stand-ins changing the local verification contract."""

    values: dict[str, Any] = {field_name: value}

    with pytest.raises(ValueError, match=field_name):
        _local(**values)


@pytest.mark.parametrize("value", [1, 0, "true", object()])
def test_local_result_requires_exact_optional_boolean(value: object) -> None:
    """Catch an invalid stand-in for the optional irreversible permission decision."""

    with pytest.raises(ValueError, match="irreversible_action_permitted"):
        _local(irreversible_action_permitted=cast(Any, value))


@pytest.mark.parametrize(
    "field_name",
    [
        "task_fit",
        "dependencies_satisfied",
        "provenance_intact",
        "authorization_scope_valid",
        "claimed_outcome_supported",
        "repeated_failed_route",
    ],
)
@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_relational_result_requires_exact_boolean_fields(
    field_name: str,
    value: object,
) -> None:
    """Catch truthy and falsy stand-ins changing the relational contract."""

    values: dict[str, Any] = {field_name: value}

    with pytest.raises(ValueError, match=field_name):
        _relational(**values)


@pytest.mark.parametrize("constructor", [_local, _relational])
@pytest.mark.parametrize("action_id", ["", " ", 7, True])
def test_verifier_results_require_exact_nonblank_action_ids(
    constructor: Any,
    action_id: object,
) -> None:
    """Catch ambiguous or noncanonical action identity at the verifier boundary."""

    with pytest.raises(ValueError, match="action_id"):
        constructor(action_id=cast(Any, action_id))


def test_results_reject_hostile_action_id_string_subclass() -> None:
    """Catch a string subclass whose strip method disguises an empty action ID."""

    class HostileEmptyString(str):
        def strip(self, chars: str | None = None) -> str:
            return "pretend-valid"

    with pytest.raises(ValueError, match="action_id"):
        _local(action_id=HostileEmptyString(""))


@pytest.mark.parametrize("contradiction_status", ["", " ", "clear", 0, True, None])
def test_relational_result_requires_exact_nonblank_contradiction_status(
    contradiction_status: object,
) -> None:
    """Catch an absent or coercible contradiction finding."""

    with pytest.raises(ValueError, match="contradiction_status"):
        _relational(contradiction_status=cast(Any, contradiction_status))


@pytest.mark.parametrize("contradiction_status", ["none", "contradicted"])
def test_affirmative_contradiction_status_requires_field_bound_observable_evidence(
    contradiction_status: str,
) -> None:
    """Catch a categorical contradiction finding without its own observable provenance."""

    with pytest.raises(ValueError, match="contradiction_status.*observable evidence"):
        _relational(
            contradiction_status=contradiction_status,
            evidence_refs=(_observable(), _private()),
            evidence_bindings=(_binding("contradiction_status", (_private(),)),),
        )


def test_unknown_contradiction_status_is_not_an_affirmative_finding() -> None:
    """Catch unknown contradiction state being presented as an evidence-backed outcome."""

    result = _relational(contradiction_status="unknown")

    assert result.evidence_bindings == ()


def test_each_affirmative_field_requires_its_own_evidence_binding() -> None:
    """Catch one aggregate reference laundering support across distinct affirmative fields."""

    reference = _observable()

    with pytest.raises(ValueError, match="arguments_valid.*evidence binding"):
        _local(
            executed=True,
            arguments_valid=True,
            schema_valid=False,
            authorization_valid=False,
            intended_operation_observed=False,
            evidence_refs=(reference,),
            evidence_bindings=(_binding("executed", (reference,)),),
        )


def test_each_affirmative_binding_requires_observable_evidence() -> None:
    """Catch unrelated aggregate evidence masking a field binding to private reasoning."""

    observable = _observable()
    private = _private()

    with pytest.raises(ValueError, match="executed.*observable evidence"):
        _local(
            executed=True,
            arguments_valid=False,
            schema_valid=False,
            authorization_valid=False,
            intended_operation_observed=False,
            evidence_refs=(observable, private),
            evidence_bindings=(_binding("executed", (private,)),),
        )


def test_evidence_bindings_must_be_unique_allowed_and_subset_of_result_evidence() -> None:
    """Catch ambiguous keys and evidence smuggled outside the aggregate result boundary."""

    reference = _observable()
    other = _observable("output:other")
    base = {
        "executed": False,
        "arguments_valid": False,
        "schema_valid": False,
        "authorization_valid": False,
        "intended_operation_observed": False,
        "evidence_refs": (reference,),
    }

    with pytest.raises(ValueError, match="allowed field"):
        _local(**base, evidence_bindings=(_binding("claimed_outcome_supported", (reference,)),))
    with pytest.raises(ValueError, match="unique"):
        _local(
            **base,
            evidence_bindings=(
                _binding("executed", (reference,)),
                _binding("executed", (reference,)),
            ),
        )
    with pytest.raises(ValueError, match="subset"):
        _local(**base, evidence_bindings=(_binding("executed", (other,)),))


def test_evidence_binding_rejects_fields_without_a_verification_finding() -> None:
    """Catch evidence attached to an optional local or relational finding that was not made."""

    reference = _observable()

    with pytest.raises(ValueError, match="irreversible_action_permitted.*no finding"):
        _local(
            executed=False,
            arguments_valid=False,
            schema_valid=False,
            authorization_valid=False,
            intended_operation_observed=False,
            irreversible_action_permitted=None,
            evidence_refs=(reference,),
            evidence_bindings=(_binding("irreversible_action_permitted", (reference,)),),
        )
    with pytest.raises(ValueError, match="contradiction_status.*no finding"):
        _relational(
            contradiction_status="unknown",
            evidence_refs=(reference,),
            evidence_bindings=(_binding("contradiction_status", (reference,)),),
        )


def test_evidence_binding_snapshots_and_json_round_trips() -> None:
    """Catch caller mutation or lossy serialization changing field-keyed provenance."""

    reference = _observable()
    references = [reference]
    binding = VerificationEvidenceBinding("executed", cast(Any, references))
    references.clear()
    object.__setattr__(reference, "reference_id", "rewritten")
    restored = VerificationEvidenceBinding.from_dict(json.loads(json.dumps(binding.to_dict())))

    assert restored == _binding("executed", (_observable(),))
    assert not hasattr(restored, "__dict__")


@pytest.mark.parametrize(
    ("field_name", "evidence_refs"),
    [
        ("", (_observable(),)),
        (" ", (_observable(),)),
        (7, (_observable(),)),
        ("executed", ()),
        ("executed", ("output:action-1",)),
        ("executed", {_observable()}),
        ("nonexistent", (_observable(),)),
    ],
)
def test_evidence_binding_rejects_malformed_keys_and_references(
    field_name: object,
    evidence_refs: object,
) -> None:
    """Catch malformed field-keyed provenance before it reaches a result."""

    with pytest.raises(ValueError):
        VerificationEvidenceBinding(cast(Any, field_name), cast(Any, evidence_refs))


def test_result_snapshots_binding_collection_and_objects() -> None:
    """Catch later caller mutation changing the result's field-to-evidence relation."""

    reference = _observable()
    binding = _binding("executed", (reference,))
    bindings = [binding]
    result = _local(
        executed=True,
        arguments_valid=False,
        schema_valid=False,
        authorization_valid=False,
        intended_operation_observed=False,
        evidence_refs=(reference,),
        evidence_bindings=cast(Any, bindings),
    )
    bindings.clear()
    object.__setattr__(binding, "field_name", "rewritten")

    assert result.evidence_bindings == (_binding("executed", (_observable(),)),)
    assert result.evidence_bindings[0] is not binding


def test_result_rejects_unordered_evidence_binding_collection() -> None:
    """Catch nondeterministic binding order entering serialized verifier output."""

    reference = _observable()

    with pytest.raises(ValueError, match="evidence_bindings"):
        _local(
            executed=False,
            arguments_valid=False,
            schema_valid=False,
            authorization_valid=False,
            intended_operation_observed=False,
            evidence_refs=(reference,),
            evidence_bindings=cast(Any, {_binding("executed", (reference,))}),
        )


def test_results_revalidate_evidence_bindings_at_serialization_time() -> None:
    """Catch object-level mutation of a binding entering a serialized verifier result."""

    result = _local()
    object.__setattr__(result.evidence_bindings[0], "field_name", "nonexistent")

    with pytest.raises(ValueError, match="verification finding"):
        result.to_dict()


def test_results_snapshot_evidence_references_and_caller_collections() -> None:
    """Catch later mutation of caller evidence changing either verifier result."""

    reference = _observable()
    references = [reference]
    local = _local(evidence_refs=cast(Any, references))
    relational = _relational(
        claimed_outcome_supported=True,
        contradiction_status="none",
        evidence_refs=cast(Any, references),
    )
    references.clear()
    object.__setattr__(reference, "reference_id", "rewritten")

    assert local.evidence_refs == (_observable(),)
    assert relational.evidence_refs == (_observable(),)
    assert local.evidence_refs[0] is not reference
    assert relational.evidence_refs[0] is not reference


def test_results_reject_subclassed_or_corrupted_evidence_references() -> None:
    """Catch noncanonical evidence objects crossing the verifier boundary."""

    class ReferenceSubclass(EvidenceReference):
        pass

    subclass = ReferenceSubclass("output:action-1", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    corrupted = _observable()
    object.__setattr__(corrupted, "source_kind", "observable_output")

    for reference in (subclass, corrupted):
        with pytest.raises(ValueError, match="evidence_refs"):
            _local(evidence_refs=cast(Any, (reference,)))
        with pytest.raises(ValueError, match="evidence_refs"):
            _relational(evidence_refs=cast(Any, (reference,)))


def test_results_refuse_to_serialize_use_time_corruption() -> None:
    """Catch object-level mutation being persisted after construction validation."""

    local = _local()
    relational = _relational()
    object.__setattr__(local, "executed", 1)
    object.__setattr__(relational, "action_id", " ")

    with pytest.raises(ValueError, match="executed"):
        local.to_dict()
    with pytest.raises(ValueError, match="action_id"):
        relational.to_dict()


def test_results_are_frozen_slotted_and_json_round_trip_exactly() -> None:
    """Catch mutable or lossy verifier result records at the JSON boundary."""

    local = _local(irreversible_action_permitted=True)
    relational = _relational(
        task_fit=True,
        dependencies_satisfied=True,
        contradiction_status="none",
        provenance_intact=True,
        authorization_scope_valid=True,
        claimed_outcome_supported=True,
        evidence_refs=(_observable(),),
    )

    restored_local = LocalVerificationResult.from_dict(json.loads(json.dumps(local.to_dict())))
    restored_relational = RelationalVerificationResult.from_dict(
        json.loads(json.dumps(relational.to_dict()))
    )

    assert restored_local == local
    assert restored_relational == relational
    assert not hasattr(restored_local, "__dict__")
    assert not hasattr(restored_relational, "__dict__")
    with pytest.raises(FrozenInstanceError):
        restored_local.executed = False
    with pytest.raises(FrozenInstanceError):
        restored_relational.claimed_outcome_supported = False


@pytest.mark.parametrize(
    ("constructor", "payload"),
    [
        (LocalVerificationResult.from_dict, []),
        (LocalVerificationResult.from_dict, {"action_id": "action-1"}),
        (RelationalVerificationResult.from_dict, []),
        (RelationalVerificationResult.from_dict, {"action_id": "action-1"}),
    ],
)
def test_result_deserializers_reject_wrong_shapes_and_fields(
    constructor: Any,
    payload: object,
) -> None:
    """Catch partial records silently changing the persisted verifier contract."""

    with pytest.raises(ValueError):
        constructor(payload)


def test_protocols_use_canonical_action_state_and_distinct_results() -> None:
    """Catch protocol annotations drifting to interchangeable or competing record types."""

    local_hints = get_type_hints(LocalExecutionVerifier.verify_local)
    relational_hints = get_type_hints(RelationalEvidenceVerifier.verify_relational)

    assert local_hints == {"action": ActionRecord, "return": LocalVerificationResult}
    assert relational_hints == {
        "action": ActionRecord,
        "evidence_state": EvidenceState,
        "return": RelationalVerificationResult,
    }


def test_verification_levels_are_exact_and_non_interchangeable() -> None:
    """Catch a renamed or aliased verifier level losing its serialized distinction."""

    assert VerificationLevel.LOCAL_EXECUTION.value == "local_execution"
    assert VerificationLevel.RELATIONAL_EVIDENCE.value == "relational_evidence"
    assert VerificationLevel.LOCAL_EXECUTION is not VerificationLevel.RELATIONAL_EVIDENCE


def test_event_adapters_preserve_level_structured_result_and_links() -> None:
    """Catch verifier events reducing structured evidence to one ambiguous success scalar."""

    common = {
        "event_id": "verification-event-1",
        "timestamp": "2026-09-10T12:00:04Z",
        "run_id": "run-1",
        "repeat_id": 0,
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "parent_event_ids": ("observation-event-1",),
        "verifier_refs": ("verifier:independent-1",),
    }
    local = _local()
    relational = _relational()

    local_event = make_local_verification_event(local, **common)
    common["event_id"] = "verification-event-2"
    relational_event = make_relational_verification_event(relational, **common)

    assert isinstance(local_event, EventEnvelope)
    assert local_event.event_type == "verification_result"
    assert local_event.action_id == "action-1"
    assert local_event.evidence_refs == ("output:action-1",)
    assert local_event.payload == {
        "verification_level": "local_execution",
        "result": local.to_dict(),
        "verifier_refs": ["verifier:independent-1"],
    }
    assert relational_event.payload == {
        "verification_level": "relational_evidence",
        "result": relational.to_dict(),
        "verifier_refs": ["verifier:independent-1"],
    }
    assert "verified" not in local_event.payload
    assert "verified" not in relational_event.payload
    assert json.loads(json.dumps(local_event.to_dict()))["payload"] == local_event.payload


def test_event_adapters_reject_wrong_result_level_and_conflicting_links() -> None:
    """Catch adapters accepting a swapped result or rewriting its action/evidence linkage."""

    local = _local()
    relational = _relational()

    with pytest.raises(TypeError, match="LocalVerificationResult"):
        make_local_verification_event(
            cast(Any, relational), verifier_refs=("verifier:independent-1",)
        )
    with pytest.raises(TypeError, match="RelationalVerificationResult"):
        make_relational_verification_event(
            cast(Any, local), verifier_refs=("verifier:independent-1",)
        )
    with pytest.raises(ValueError, match="action_id"):
        make_local_verification_event(
            local,
            action_id="other-action",
            verifier_refs=("verifier:independent-1",),
        )
    with pytest.raises(ValueError, match="evidence_refs"):
        make_local_verification_event(
            local,
            evidence_refs=("other-evidence",),
            verifier_refs=("verifier:independent-1",),
        )


def test_event_adapters_revalidate_result_after_use_time_mutation() -> None:
    """Catch a corrupted frozen result crossing the event serialization boundary."""

    local = _local()
    object.__setattr__(local.evidence_refs[0], "reference_id", " ")

    with pytest.raises(ValueError, match="evidence_refs"):
        make_local_verification_event(local, verifier_refs=("verifier:independent-1",))


@pytest.mark.parametrize(
    "verifier_refs",
    [None, (), ("",), (" ",), (7,), "verifier:independent-1", {"verifier:set"}],
)
def test_event_adapters_require_nonempty_exact_verifier_provenance(
    verifier_refs: object,
) -> None:
    """Catch absent or malformed verifier identity on a verification event."""

    with pytest.raises(ValueError, match="verifier_refs"):
        make_local_verification_event(_local(), verifier_refs=cast(Any, verifier_refs))


def test_event_adapter_rejects_missing_verifier_provenance() -> None:
    """Catch a verification event created without an independently auditable verifier."""

    with pytest.raises(TypeError, match="verifier_refs"):
        make_local_verification_event(_local())


def test_event_adapters_snapshot_correlated_verifier_provenance() -> None:
    """Catch later caller-list mutation changing envelope or payload verifier provenance."""

    verifier_refs = ["verifier:independent-1"]
    event = make_relational_verification_event(
        _relational(),
        verifier_refs=cast(Any, verifier_refs),
    )
    verifier_refs[0] = "verifier:rewritten"

    assert event.verifier_refs == ("verifier:independent-1",)
    assert event.payload["verifier_refs"] == ("verifier:independent-1",)


def test_claimed_success_text_does_not_create_execution_or_relational_support() -> None:
    """Catch model prose being promoted into either verifier contract."""

    claim = EvidenceClaim("claim-1", "I fixed the bug", (), "unverified")
    evidence_state = EvidenceState((claim,))

    assert evidence_state.resolve("claim-1").status == "unverified"
    with pytest.raises(TypeError):
        make_local_verification_event(cast(Any, claim), verifier_refs=("verifier:independent-1",))
    with pytest.raises(TypeError):
        make_relational_verification_event(
            cast(Any, claim), verifier_refs=("verifier:independent-1",)
        )
