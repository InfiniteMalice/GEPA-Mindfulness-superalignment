"""Verified behavior, admission and real optimization contracts for PR-13."""

import json
from dataclasses import replace

import pytest

from evaluation.dynamic_uncertainty import DecisionBackend, compare_decisions
from gepa_mindfulness.core.epistemic_process import (
    EpistemicProcessAssessment,
    VerifiedProcessComponent,
)
from gepa_mindfulness.core.epistemic_process import (
    EpistemicProcessComponent as Component,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.training.dynamic_uncertainty import (
    ACTIONS,
    Behavior,
    DecisionVerifier,
    TrajectoryExample,
    prepare_trajectories,
    train_decisions,
)
from mindful_trace_gepa.confidence import ConfidenceSource
from tests.test_epistemic_reconciliation import reconciliation, sequence

CONTRACT = TrustedEvaluatorContract("fixture-decisions", "v1", "observable-decisions-v1")
SCENARIOS = (
    (
        "The sensor disagreed with its committed prediction; verify before repeating.",
        Action.DECOMPOSE_AND_VERIFY,
        Component.RECOVERY,
    ),
    (
        "Three independently audited sensors support this same scoped claim.",
        Action.ACCEPT,
        Component.BELIEF_UPDATE,
    ),
    (
        "Three reports copy one unverified sensor; acquire independent evidence.",
        Action.RETRIEVE_MORE,
        Component.BELIEF_UPDATE,
    ),
    (
        "Two authenticated measurements contradict each other; resolve the conflict.",
        Action.DECOMPOSE_AND_VERIFY,
        Component.CONTRADICTION_HANDLING,
    ),
    (
        "The required measurement is missing and can be retrieved.",
        Action.RETRIEVE_MORE,
        Component.MISSING_EVIDENCE_DETECTION,
    ),
    (
        "Evidence remains unresolved and the verification budget is exhausted.",
        Action.ABSTAIN,
        Component.JUSTIFIED_ABSTENTION,
    ),
    (
        "The unresolved measurement controls an irreversible high-stakes release.",
        Action.ROUTE_EXTERNAL,
        Component.JUSTIFIED_ABSTENTION,
    ),
    (
        "The successful prediction tested this sensor only; the new sensor is untested.",
        Action.RETRIEVE_MORE,
        Component.CONSEQUENCE_PREDICTION,
    ),
)


def catalog(split="DEVELOPMENT", prefix="fixture"):
    """Build authored contract fixtures, not an effectiveness benchmark or admitted dataset."""
    return tuple(
        TrajectoryExample(
            example_id=f"{prefix}-{index}",
            source_group=prefix,
            behavior=behavior,
            context=scenario[0],
            events=tuple(sequence()),
            source_record={"training_eligibility": split, "origin": "authored-test"},
        )
        for index, (behavior, scenario) in enumerate(zip(Behavior, SCENARIOS))
    )


def assessments(example):
    """Stand in for a trusted host evaluator using authored observable decision targets."""
    _, target, component = SCENARIOS[list(Behavior).index(example.behavior)]
    return tuple(
        EpistemicProcessAssessment(
            (
                VerifiedProcessComponent(
                    component,
                    float(action == target),
                    RewardProvenance(
                        component.value,
                        "authored observable decision fixture",
                        VerificationRoute.TRUSTED_EVALUATOR,
                        evaluator=CONTRACT,
                    ),
                ),
            )
        )
        for action in ACTIONS
    )


VERIFIER = DecisionVerifier(CONTRACT, assessments)


def test_preparation_projects_public_history_and_snapshots_provenance():
    """Only scoped public evidence and diagnostics reach the policy input."""
    examples = catalog()
    prepared = prepare_trajectories(examples, for_training=False)
    item = prepared[0]
    history = json.loads(item.input.history_json)
    assert history[0]["prediction"] == [20]
    assert history[0]["observation"] == [22]
    assert history[0]["residuals"] == [2.0]
    assert history[0]["posterior"]["model_uncertainty"] is None
    assert history[0]["posterior"]["estimator_version"] == "diagnostic-v1"
    assert "sensor-log" not in item.input.history_json
    assert "fixture-0" not in repr(item.input)
    examples[0].source_record["origin"] = "mutated"
    assert "mutated" not in item.source_json


@pytest.mark.parametrize("split", ["DEVELOPMENT", "REGRESSION", "HIDDEN_EVAL", None])
def test_training_rejects_nontrain_and_missing_split(split):
    """Explicit TRAIN is required in addition to recursive admission."""
    with pytest.raises(ValueError):
        prepare_trajectories(catalog(split), for_training=True)


def test_nested_holdout_is_rejected_before_callbacks():
    """Outer TRAIN cannot launder hidden examples into a single optimizer update."""
    examples = list(catalog("TRAIN"))
    examples[-1].source_record["nested"] = {"training_eligibility": "HIDDEN_EVAL"}
    verifier = DecisionVerifier(CONTRACT, lambda _: pytest.fail("verifier called"))
    with pytest.raises(ValueError, match="forbids"):
        train_decisions(
            examples, lambda _: pytest.fail("scorer called"), None, verifier=verifier, enabled=True
        )


@pytest.mark.parametrize("fault", ["residual", "chronology", "verification", "trailing"])
def test_corrupted_trajectory_fails_closed(fault):
    """Causal validation is mandatory even when a caller supplies a valid label."""
    example = catalog()[0]
    events = list(example.events)
    if fault == "residual":
        record = reconciliation()
        binding = record.bindings[0]
        record = replace(
            record,
            bindings=(
                replace(binding, innovation=replace(binding.innovation, predicted_measurement=19)),
            ),
        )
        events = sequence(record)
    elif fault == "chronology":
        events[3] = replace(events[3], timestamp="2026-09-30T11:00:00Z")
    elif fault == "verification":
        payload = dict(events[4].payload)
        payload["verified"] = False
        events[4] = replace(events[4], payload=payload)
    else:
        events = events[:-1]
    with pytest.raises(ValueError):
        prepare_trajectories((replace(example, events=tuple(events)),), for_training=False)


def test_duplicate_identity_or_public_input_is_rejected():
    """Renaming an example must not silently duplicate its effective policy input."""
    item = catalog()[0]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_trajectories((item, replace(item, example_id="renamed")), for_training=False)


def test_evaluation_reports_all_behaviors_without_uncertainty_reward():
    """Verified next decisions improve over an always-accept baseline across all strata."""
    by_context = {context: target for context, target, _ in SCENARIOS}
    result = compare_decisions(
        catalog(),
        {
            "baseline": DecisionBackend("v1", lambda _: Action.ACCEPT),
            "fixture": DecisionBackend("v1", lambda item: by_context[item.context]),
        },
        verifier=VERIFIER,
        enabled=True,
    )
    assert result["complete_behavior_coverage"] is True
    assert result["split_check"] == "unavailable"
    assert result["backends"]["baseline"]["mean_verified_score"] == 1 / 8
    assert result["backends"]["fixture"]["mean_verified_score"] == 1
    assert result["confers_authority"] is False
    assert len(result["diagnostics"]) == 8


def test_evaluation_rejects_overlap_before_any_callback():
    """Changing IDs and groups does not hide reused normalized policy inputs."""
    verifier = DecisionVerifier(CONTRACT, lambda _: pytest.fail("verifier called"))
    with pytest.raises(ValueError, match="overlap"):
        compare_decisions(
            catalog(),
            {"baseline": DecisionBackend("v1", lambda _: Action.ACCEPT)},
            verifier=verifier,
            training_examples=catalog("TRAIN", "other"),
            enabled=True,
        )


def test_actual_optimizer_improves_verified_decisions():
    """A tiny tabular policy learns fixture actions; this only tests optimizer mechanics."""
    torch = pytest.importorskip("torch")
    contexts = {context: index for index, (context, _, _) in enumerate(SCENARIOS)}
    weights = torch.nn.Parameter(torch.zeros((8, len(ACTIONS))))
    optimizer = torch.optim.SGD([weights], lr=4)

    def scorer(item):
        return weights[contexts[item.context]]

    before = weights.detach().clone()
    report = train_decisions(
        catalog("TRAIN"), scorer, optimizer, verifier=VERIFIER, epochs=10, enabled=True
    )
    assert not torch.equal(before, weights)
    assert report["losses"][-1] < report["losses"][0]
    assert sum(report["updates_by_behavior"].values()) == 80
    assert all(
        ACTIONS[int(scorer(item.input).argmax())] == SCENARIOS[i][1]
        for i, item in enumerate(prepare_trajectories(catalog(), for_training=False))
    )


def test_default_disabled_prevents_callbacks():
    """Opt-in is required even with otherwise valid records and callbacks."""
    with pytest.raises(ValueError, match="enabled=True"):
        train_decisions(catalog("TRAIN"), lambda _: None, None, verifier=VERIFIER)
    with pytest.raises(ValueError, match="enabled=True"):
        compare_decisions(catalog(), {}, verifier=VERIFIER)


def test_nested_evaluation_restriction_is_preserved():
    """Reports must retain restrictions anywhere in retained source provenance."""
    examples = catalog()
    examples[0].source_record["parent"] = {"training_eligibility": "HIDDEN_EVAL"}
    report = compare_decisions(examples, {"missing": None}, verifier=VERIFIER, enabled=True)
    assert report["training_eligibility"] == "HIDDEN_EVAL"


def test_uninformative_tables_do_not_claim_best_action_success():
    """An all-zero verified table contains no evidence of a successful decision."""
    verifier = DecisionVerifier(
        CONTRACT,
        lambda item: tuple(
            replace(
                a, verified_components=tuple(replace(c, score=0) for c in a.verified_components)
            )
            for a in assessments(item)
        ),
    )
    report = compare_decisions(
        catalog()[:1],
        {"baseline": DecisionBackend("v1", lambda _: Action.ACCEPT)},
        verifier=verifier,
        enabled=True,
    )
    assert report["backends"]["baseline"]["best_action_rate"] is None
    assert report["missing_behaviors"] == [b.value for b in list(Behavior)[1:]]


@pytest.mark.parametrize("fault", ["empty", "length", "contract", "component", "different"])
def test_bad_verifier_catalog_never_reaches_policy(fault):
    """Assessment validation completes before any policy callback or optimizer mutation."""

    def invalid(item):
        values = assessments(item)
        if fault == "empty":
            return (EpistemicProcessAssessment(),) * len(ACTIONS)
        if fault == "length":
            return values[:-1]
        if fault == "contract":
            component = values[0].verified_components[0]
            component = replace(
                component,
                provenance=replace(
                    component.provenance,
                    evaluator=TrustedEvaluatorContract("wrong", "v1", "contract"),
                ),
            )
        else:
            name = Component.GROUNDING if fault == "component" else Component.CALIBRATION
            component = VerifiedProcessComponent(
                name,
                0,
                RewardProvenance(
                    name.value, "fixture", VerificationRoute.TRUSTED_EVALUATOR, evaluator=CONTRACT
                ),
            )
        return (EpistemicProcessAssessment((component,)), *values[1:])

    with pytest.raises(ValueError):
        compare_decisions(
            catalog(),
            {"baseline": DecisionBackend("v1", lambda _: pytest.fail())},
            verifier=DecisionVerifier(CONTRACT, invalid),
            enabled=True,
        )


@pytest.mark.parametrize("seed", [-1, True, 2**32, 1.5])
def test_invalid_seed_is_rejected(seed):
    """Seeds have one canonical bounded integer representation."""
    with pytest.raises(ValueError, match="seed"):
        compare_decisions(catalog(), {"none": None}, verifier=VERIFIER, seed=seed, enabled=True)


@pytest.mark.parametrize("epochs", [0, True, 1001, 1.5])
def test_invalid_epochs_is_rejected(epochs):
    """Invalid loop bounds cannot start an optimizer."""
    with pytest.raises(ValueError, match="epochs"):
        train_decisions(
            catalog("TRAIN"), lambda _: None, None, verifier=VERIFIER, epochs=epochs, enabled=True
        )


def test_partial_training_catalog_is_rejected():
    """Training exposure must cover every requested behavioral stratum."""
    with pytest.raises(ValueError, match="eight"):
        train_decisions(
            catalog("TRAIN")[:-1], lambda _: None, None, verifier=VERIFIER, enabled=True
        )


def test_policies_receive_identical_seeded_order_and_immutable_inputs():
    """All arms see the same order without IDs or mutable assessment state."""
    seen = [[], []]

    def policy(index, item):
        seen[index].append(item)
        with pytest.raises(AttributeError):
            item.context = "changed"
        return Action.ACCEPT

    report = compare_decisions(
        catalog(),
        {
            "first": DecisionBackend("v1", lambda item: policy(0, item)),
            "second": DecisionBackend("v2", lambda item: policy(1, item)),
            "absent": None,
        },
        verifier=VERIFIER,
        enabled=True,
        seed=18,
    )
    assert seen[0] == seen[1]
    assert report["backends"]["absent"] == {"available": False}
    assert report["presentation_order"] != [e.example_id for e in catalog()]


@pytest.mark.parametrize("fault", ["shape", "nan", "detached", "foreign", "infinite_parameters"])
def test_invalid_torch_outputs_or_gradients_stop_training(fault):
    """Invalid or disconnected policy outputs must not update caller parameters."""
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(len(ACTIONS)))
    optimizer = torch.optim.SGD([parameter], lr=1)
    before = parameter.detach().clone()
    if fault == "infinite_parameters":
        with torch.no_grad():
            parameter.fill_(float("inf"))

    def score(_):
        if fault == "shape":
            return parameter[:2]
        if fault == "nan":
            return parameter * float("nan")
        if fault == "detached":
            return parameter.detach()
        return torch.zeros(len(ACTIONS), requires_grad=True)

    with pytest.raises(ValueError):
        train_decisions(catalog("TRAIN"), score, optimizer, verifier=VERIFIER, enabled=True)
    if fault != "infinite_parameters":
        assert torch.equal(parameter, before)


def test_evaluation_accepts_disjoint_declared_training_catalog():
    """Disjoint public contexts and source groups permit an explicit split check."""
    training = tuple(
        replace(item, context="Other deployment: " + item.context)
        for item in catalog("TRAIN", "training")
    )
    result = compare_decisions(
        catalog(), {"missing": None}, verifier=VERIFIER, training_examples=training, enabled=True
    )
    assert result["split_check"] == "passed"


def test_bound_measurement_order_keeps_units_with_values():
    """Reordering bindings must not attach a measurement to another dimension."""
    record = reconciliation()
    second = replace(
        record.update.measurements[0], measurement_id="second", target_dimension="other"
    )
    binding = replace(
        record.bindings[0],
        measurement_id="second",
        innovation=replace(record.bindings[0].innovation, innovation_id="second"),
    )
    record = replace(
        record,
        bindings=(binding, *record.bindings),
        update=replace(record.update, measurements=(*record.update.measurements, second)),
    )
    example = replace(catalog()[0], events=tuple(sequence(record)))
    item = prepare_trajectories((example,), for_training=False)[0]
    assert json.loads(item.input.history_json)[0]["dimensions"] == ["other", "temperature"]


def test_uncertainty_changes_diagnostics_but_not_verified_reward():
    """A smaller uncertainty estimate never independently earns decision credit."""
    example = catalog()[0]
    record = reconciliation()
    record = replace(
        record,
        update=replace(
            record.update,
            posterior_state=replace(record.update.posterior_state, world_uncertainty=0.01),
        ),
    )
    changed = replace(example, events=tuple(sequence(record)))
    backend = {"baseline": DecisionBackend("v1", lambda _: Action.ACCEPT)}
    first = compare_decisions((example,), backend, verifier=VERIFIER, enabled=True)
    second = compare_decisions((changed,), backend, verifier=VERIFIER, enabled=True)
    assert first["backends"] == second["backends"]
    assert first["diagnostics"] != second["diagnostics"]


def test_existing_longitudinal_world_export_is_accepted_but_cannot_train():
    """The existing simulator's complete retained export stays DEVELOPMENT-only."""
    from gepa_mindfulness.verification.epistemic_state import EpistemicContext
    from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
    from mindful_trace_gepa.logging_schema import EventEnvelope
    from synthetic_data.world_peo import EpisodeStep, build_episode
    from synthetic_data.worlds import generate_world

    episode = build_episode(
        generate_world(seed=1, enabled=True),
        (EpisodeStep("inspect", 0.8, 0.6), EpisodeStep("release", 0.7, 0.6)),
        context=EpistemicContext("fixture", 0, EvaluatedSystemVersion("test", "v1")),
        episode_id="longitudinal",
        start_timestamp="2026-10-01T00:00:00Z",
        enabled=True,
    )
    episode = json.loads(json.dumps(episode))
    item = replace(
        catalog()[0],
        events=tuple(EventEnvelope(**event) for event in episode["events"]),
        source_record=episode,
    )
    result = prepare_trajectories((item,), for_training=False)[0]
    assert len(json.loads(result.input.history_json)) == 2
    with pytest.raises(ValueError, match="forbids"):
        prepare_trajectories(
            (replace(item, source_record={"training_eligibility": "TRAIN", "parent": episode}),),
            for_training=True,
        )


def test_private_measurement_cannot_become_decision_evidence():
    """Private evidence remains diagnostic-only even when a generic verifier says success."""
    record = reconciliation()
    refs = (EvidenceReference("sensor-log", EvidenceSourceKind.PRIVATE_REASONING),)
    update = record.update
    record = replace(
        record,
        update=replace(
            update,
            prior_state=replace(update.prior_state, evidence_refs=refs),
            posterior_state=replace(update.posterior_state, evidence_refs=refs),
            evidence_refs=refs,
            measurements=(
                replace(
                    update.measurements[0], evidence_refs=refs, source=ConfidenceSource.TOOL_RESULT
                ),
            ),
        ),
        bindings=(
            replace(
                record.bindings[0],
                verifier_event_id=None,
                innovation=replace(record.bindings[0].innovation, evidence_refs=refs),
            ),
        ),
    )
    with pytest.raises(ValueError, match="observable"):
        prepare_trajectories(
            (replace(catalog()[0], events=tuple(sequence(record))),), for_training=False
        )
