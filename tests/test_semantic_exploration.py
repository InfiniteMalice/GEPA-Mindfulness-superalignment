"""Matched controls and trust boundaries for semantic inquiry proposals."""

import json
from dataclasses import replace
from itertools import permutations

import pytest

from evaluation.experimental_overlays import ExperimentalOverlayConfig
from evaluation.experimental_records import experimental_record_from_dict
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from gepa_mindfulness.verification.semantic_exploration import (
    ExplorationCandidate,
    ExplorationPolicy,
    ExplorationRequest,
    propose_exploration,
)
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

ON = ExperimentalOverlayConfig(expected_information_gain_inquiry=True)


def refs():
    """Keep provenance observable and external to the actor payload."""
    return (EvidenceReference("PRIVATE_EVIDENCE", EvidenceSourceKind.EXTERNAL_RECORD),)


def request(**changes):
    """Declare a host request with distinct uncertainty classes and integer compute units."""
    return replace(
        ExplorationRequest(
            "request",
            EpistemicContext("PRIVATE_RUN", 0, EvaluatedSystemVersion("m", "h")),
            1,
            0.9,
            0.7,
            0.1,
            0.2,
            0.7,
            0.8,
            10,
            "tokens",
            "normalized-gain-v1",
            True,
            refs(),
        ),
        **changes,
    )


def candidate(identifier="a", **changes):
    """Supply one attributable public inquiry with a host estimate of gain and cost."""
    return replace(
        ExplorationCandidate(
            identifier,
            "SEARCH",
            "world",
            1,
            "Which observation separates them?",
            0.6,
            2,
            0.9,
            refs(),
        ),
        **changes,
    )


def propose(req=None, candidates=None, **kwargs):
    """Call the pure selector with its experimental flag explicitly enabled."""
    return propose_exploration(
        request() if req is None else req,
        (candidate(),) if candidates is None else candidates,
        config=ON,
        **kwargs,
    )


def test_information_gain_selection_reuses_legacy_contract_without_authority():
    """One eligible inquiry is a non-TRAIN diagnostic, never execution permission."""
    result = propose(candidates=(candidate(), candidate("b", expected_information_gain=0.8)))
    assert result["selected"]["id"] == "b"
    assert result["reason"] == "selected"
    assert result["authority_granted"] is False
    assert result["training_eligibility"] == "DEVELOPMENT"
    record = experimental_record_from_dict(result["diagnostic"])
    assert record.expected_information_gain == 0.8
    assert record.uncertainty == 0.9
    with pytest.raises(ValueError):
        require_training_eligible(result)


def test_uncertainty_alone_never_widens_distance():
    """Matched evidence gap/diversity changes, not uncertainty alone, permit breadth."""
    broad = candidate(distance=5)
    policy = ExplorationPolicy(max_distance=5)
    req = request(world_uncertainty=1, evidence_gap=0.1, hypothesis_diversity=0.1)
    assert propose(req, (broad,), policy=policy)["distance_cap"] == 1
    assert propose(replace(req, evidence_gap=0.7), (broad,), policy=policy)["distance_cap"] == 3
    assert (
        propose(replace(req, evidence_gap=0.7, hypothesis_diversity=0.8), (broad,), policy=policy)[
            "selected"
        ]["distance"]
        == 5
    )
    assert propose(request(), (broad,))["selected"] is None  # default host cap remains 3


def test_stakes_reversibility_and_budget_constrain_selection():
    """Changing only stakes or remaining compute changes the allowable inquiry."""
    choices = (
        candidate("wide", distance=3, expected_information_gain=0.9),
        candidate("local", reversibility=0.7, compute_cost=1),
        candidate("reversible", reversibility=0.9, compute_cost=3),
    )
    assert propose(candidates=choices)["selected"]["id"] == "wide"
    high = request(stakes=0.8)
    assert propose(high, choices)["selected"]["id"] == "reversible"
    assert propose(replace(high, remaining_compute=2), choices)["selected"] is None
    assert propose(high, choices)["distance_cap"] == 2


@pytest.mark.parametrize("mode", ["SEARCH", "DEBATE", "SPARK"])
@pytest.mark.parametrize("distance", range(1, 6))
def test_all_semantic_levels_have_action_specific_guidance(mode, distance):
    """The proposal states how to explore instead of changing sampling temperature."""
    result = propose(
        candidates=(candidate(mode=mode, distance=distance),),
        policy=ExplorationPolicy(max_distance=5),
    )
    selected = result["selected"]
    assert selected["mode"] == mode and selected["distance"] == distance
    assert selected["semantic_scope"]
    assert "temperature" not in result


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"evidence_changed": False}, "unchanged_evidence"),
        ({"remaining_compute": 0}, "budget_exhausted"),
        ({"monitor_uncertainty": None}, "monitor_unavailable"),
    ],
)
def test_request_level_deferral(changes, reason):
    """No event change, budget or monitor basis yields no actionable proposal."""
    result = propose(request(**changes))
    assert result["selected"] is None and result["diagnostic"] is None
    assert result["reason"] == reason


def test_monitor_inquiry_precedes_world_or_model_inquiry():
    """A high monitor diagnostic cannot be hidden by a lower world uncertainty."""
    candidates = (candidate(expected_information_gain=1), candidate("monitor", target="monitor"))
    result = propose(request(monitor_uncertainty=0.8), candidates)
    assert result["selected"]["id"] == "monitor"
    assert result["rejections"] == [{"id": "a", "reason": "monitor_priority"}]
    assert propose(request(world_uncertainty=None))["selected"] is None
    assert propose(request(world_uncertainty=0.1))["selected"] is None
    assert (
        propose(request(world_uncertainty=0.1), (candidate(target="model"),))["selected"]
        is not None
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"expected_information_gain": None}, "gain_unavailable"),
        ({"expected_information_gain": 0}, "insufficient_gain"),
        ({"expected_information_gain": 0.01}, "insufficient_gain"),
        ({"compute_cost": 11}, "over_budget"),
        ({"distance": 4}, "distance_limit"),
        ({"reversibility": None}, "reversibility_unavailable"),
        ({"reversibility": 0.1}, "insufficient_reversibility"),
    ],
)
def test_candidate_rejections_are_explicit(changes, reason):
    """Missing or infeasible values never receive optimistic defaults."""
    result = propose(candidates=(candidate(**changes),))
    assert result["selected"] is None and result["reason"] == "no_eligible_candidate"
    assert result["rejections"] == [{"id": "a", "reason": reason}]


def test_rank_is_deterministic_by_gain_cost_distance_then_id():
    """Permutation and all tiebreaks yield the same proposal without float ratios."""
    choices = (
        candidate("b"),
        candidate("a"),
        candidate("c", compute_cost=3),
        candidate("d", distance=2),
    )
    for order in permutations(choices):
        assert propose(candidates=order)["selected"]["id"] == "a"
    assert (
        propose(candidates=(candidate("a"), candidate("b", compute_cost=1)))["selected"]["id"]
        == "b"
    )


@pytest.mark.parametrize("question", ["Why?\n", "Why?\u00a0", "\u754c" * 170 + "\u00a0"])
def test_public_payload_preserves_text_measurement_semantics_and_privacy(question):
    """The actor can interpret scores without receiving external provenance."""
    req = request(training_eligibility=TrainingEligibility.HIDDEN_EVAL)
    result = propose(req, (candidate(question=question),))
    assert result["selected"]["question"] == question
    assert json.loads(result["diagnostic"]["question"]) == question
    assert "PRIVATE" not in json.dumps(result)
    assert result["measurement"] == {
        "compute_unit": "tokens",
        "gain_protocol_id": "normalized-gain-v1",
    }
    assert result["training_eligibility"] == "HIDDEN_EVAL"
    result["measurement"]["compute_unit"] = "other"
    assert req.compute_unit == "tokens"


def test_nested_records_are_detached_and_never_call_instance_serializers():
    """Mutation and serializer injection cannot forge observable provenance."""
    evidence = refs()[0]
    evidence.__dict__["to_dict"] = lambda: pytest.fail("untrusted serializer called")
    evidence.__dict__["__dataclass_fields__"] = {}
    req = replace(request(), evidence_refs=(evidence,))
    object.__setattr__(evidence, "reference_id", "changed")
    assert req.evidence_refs[0].reference_id == "PRIVATE_EVIDENCE"
    object.__setattr__(req.evidence_refs[0], "source_kind", EvidenceSourceKind.PRIVATE_REASONING)
    with pytest.raises(ValueError):
        propose(req)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 1.1, "0.5"])
def test_strict_unit_values(value):
    """Every declared unit metric rejects coercion and out-of-range inputs."""
    for field in (
        "world_uncertainty",
        "model_uncertainty",
        "monitor_uncertainty",
        "stakes",
        "evidence_gap",
        "hypothesis_diversity",
    ):
        with pytest.raises(ValueError):
            request(**{field: value})
    for field in ("expected_information_gain", "reversibility"):
        with pytest.raises(ValueError):
            candidate(**{field: value})


@pytest.mark.parametrize("value", [True, 1.5, -1, 2**53])
def test_exact_compute_integers(value):
    """Compute amounts are positive/safe integers; no lossy cost comparisons."""
    with pytest.raises(ValueError):
        candidate(compute_cost=value)
    with pytest.raises(ValueError):
        request(remaining_compute=value)


def test_flags_types_bounds_and_training_rejection():
    """Invalid inputs fail before selection regardless of the candidate's apparent gain."""
    with pytest.raises(ValueError):
        propose_exploration(request(), (candidate(),))
    bad = ExperimentalOverlayConfig(expected_information_gain_inquiry=True)
    object.__setattr__(bad, "expected_information_gain_inquiry", 1)
    with pytest.raises(ValueError):
        propose_exploration(request(), (candidate(),), config=bad)
    for candidates in (
        [],
        (),
        (candidate(), candidate()),
        tuple(candidate(str(i)) for i in range(33)),
    ):
        with pytest.raises(ValueError):
            propose(candidates=candidates)
    for change in (
        {"mode": "OTHER"},
        {"target": "other"},
        {"distance": True},
        {"distance": 6},
        {"question": "x" * 513},
        {"id": "\ud800"},
        {"compute_cost": 0},
    ):
        with pytest.raises(ValueError):
            candidate(**change)
    with pytest.raises(ValueError):
        request(training_eligibility=TrainingEligibility.TRAIN)
    with pytest.raises(ValueError):
        request(evidence_changed=1)
    with pytest.raises(ValueError):
        request(source_case_id=True)
    with pytest.raises(ValueError):
        ExplorationPolicy(max_distance=0)
    with pytest.raises(ValueError):
        ExplorationPolicy(breadth_threshold=None)
    assert propose(candidates=tuple(candidate(str(i)) for i in range(32)))["selected"]


def test_mutated_records_are_revalidated():
    """Frozen Python objects still need boundary validation against object.__setattr__."""
    item = candidate()
    object.__setattr__(item, "expected_information_gain", float("nan"))
    with pytest.raises(ValueError):
        propose(candidates=(item,))
    policy = ExplorationPolicy()
    object.__setattr__(policy, "max_distance", 5.0)
    with pytest.raises(ValueError):
        propose(policy=policy)


def test_zero_monitor_uncertainty_does_not_block_world_inquiry_at_zero_threshold():
    """A permissive threshold does not manufacture monitor uncertainty from zero."""
    result = propose(
        request(monitor_uncertainty=0), policy=ExplorationPolicy(uncertainty_threshold=0)
    )
    assert result["selected"]["target"] == "world"


def test_exact_record_and_provenance_boundaries():
    """Public boundaries reject subclasses, missing provenance and private nested mutations."""
    for obj, cls in ((request(), ExplorationRequest), (candidate(), ExplorationCandidate)):
        with pytest.raises(ValueError):
            replace(obj, evidence_refs=())
        with pytest.raises(ValueError):
            replace(
                obj,
                evidence_refs=tuple(
                    EvidenceReference(str(i), EvidenceSourceKind.EXTERNAL_RECORD) for i in range(33)
                ),
            )
        subclass = type("Untrusted", (cls,), {})
        with pytest.raises(ValueError):
            if cls is ExplorationRequest:
                propose(subclass(**{f: getattr(obj, f) for f in obj.__dataclass_fields__}))
            else:
                propose(
                    candidates=(subclass(**{f: getattr(obj, f) for f in obj.__dataclass_fields__}),)
                )
    with pytest.raises(ValueError):
        request(training_eligibility="DEVELOPMENT")
    req = request()
    object.__setattr__(req.context.system, "model_version", 1)
    with pytest.raises(ValueError):
        propose(req)


def test_repeated_calls_do_not_spend_budget_or_modify_history():
    """The host must reserve/charge compute and mark unchanged evidence between calls."""
    req, choices = request(remaining_compute=2), (candidate(),)
    first = propose(req, choices)
    second = propose(req, choices)
    assert first == second and req.remaining_compute == 2
    assert propose(replace(req, evidence_changed=False), choices)["selected"] is None
