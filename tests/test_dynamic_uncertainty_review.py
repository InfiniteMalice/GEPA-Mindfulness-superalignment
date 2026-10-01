"""Regression tests for strict rewards, callback isolation and optimizer compatibility."""

from dataclasses import replace

import pytest

from evaluation.dynamic_uncertainty import DecisionBackend, compare_decisions
from gepa_mindfulness.core.epistemic_process import (
    EpistemicProcessAssessment,
    VerifiedProcessComponent,
)
from gepa_mindfulness.core.reward_provenance import RewardProvenance, TrustedEvaluatorContract
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.training.dynamic_uncertainty import (
    ACTIONS,
    DecisionVerifier,
    prepare_trajectories,
    train_decisions,
)
from tests.test_dynamic_uncertainty import CONTRACT, SCENARIOS, VERIFIER, assessments, catalog


@pytest.mark.parametrize("kind", ["component", "provenance", "contract"])
def test_nested_reward_subclasses_are_rejected_before_policy(kind):
    """Overridden dataclass validation cannot create optimizer-eligible scores."""

    class BadComponent(VerifiedProcessComponent):
        def __post_init__(self):
            pass

    class BadProvenance(RewardProvenance):
        def __post_init__(self):
            pass

    class BadContract(TrustedEvaluatorContract):
        def __eq__(self, other):
            return True

    def invalid(item):
        values = list(assessments(item))
        component = values[0].verified_components[0]
        if kind == "component":
            component = BadComponent(component.component, 99, component.provenance)
        elif kind == "provenance":
            provenance = component.provenance
            component = replace(
                component,
                provenance=BadProvenance(
                    provenance.component_name,
                    provenance.verification_method,
                    provenance.route,
                    evaluator=CONTRACT,
                ),
            )
        else:
            component = replace(
                component,
                provenance=replace(
                    component.provenance, evaluator=BadContract("spoof", "v0", "fake")
                ),
            )
        values[0] = EpistemicProcessAssessment((component,))
        return tuple(values)

    with pytest.raises(ValueError, match="canonical"):
        compare_decisions(
            catalog(),
            {"never": DecisionBackend("v1", lambda _: pytest.fail())},
            verifier=DecisionVerifier(CONTRACT, invalid),
            enabled=True,
        )


def test_callback_records_have_no_mutable_instance_dictionary():
    """Frozen callback values cannot be edited through vars()."""
    item = prepare_trajectories(catalog(), for_training=False)[0]
    for record in (item, item.input):
        with pytest.raises(TypeError):
            vars(record)


def test_policies_cannot_mutate_other_arms_or_report_identity():
    """Even deliberate low-level mutation only changes a disposable callback input."""
    contract = TrustedEvaluatorContract("fixture-decisions", "v1", "observable-decisions-v1")
    observed = []

    def mutate(item):
        object.__setattr__(item, "context", "changed")
        vars(contract)["evaluator_version"] = "changed"
        return Action.ACCEPT

    def observe(item):
        observed.append(item.context)
        return Action.ACCEPT

    report = compare_decisions(
        catalog(),
        {
            "mutator": DecisionBackend("v1", mutate),
            "observer": DecisionBackend("v1", observe),
        },
        verifier=DecisionVerifier(contract, assessments),
        enabled=True,
    )
    assert set(observed) == {context for context, _, _ in SCENARIOS}
    assert report["evaluator"]["evaluator_version"] == "v1"


def test_verifier_cannot_modify_retained_inputs():
    """The assessment callback receives a disposable evaluator snapshot."""
    observed = []

    def mutate(item):
        values = assessments(item)
        object.__setattr__(item.input, "context", "verifier changed")
        object.__setattr__(item, "source_digest", "forged")
        return values

    def observe(item):
        observed.append(item.context)
        return Action.ACCEPT

    report = compare_decisions(
        catalog(),
        {"observer": DecisionBackend("v1", observe)},
        verifier=DecisionVerifier(CONTRACT, mutate),
        enabled=True,
    )
    baseline = compare_decisions(catalog(), {"observer": None}, verifier=VERIFIER, enabled=True)
    assert set(observed) == {context for context, _, _ in SCENARIOS}
    assert report["dataset_digest"] == baseline["dataset_digest"]
    assert report["assessment_digest"] == baseline["assessment_digest"]


def test_mixed_precision_scores_preserve_verified_preference():
    """BF16 policy output must not round distinct verified targets to an equal table."""
    torch = pytest.importorskip("torch")
    weights = torch.nn.Parameter(torch.zeros(len(ACTIONS)))

    def close_scores(item):
        return tuple(
            replace(
                a,
                verified_components=(
                    replace(a.verified_components[0], score=0.501 if i == 1 else 0.5),
                ),
            )
            for i, a in enumerate(assessments(item))
        )

    report = train_decisions(
        catalog("TRAIN"),
        lambda _: weights.to(torch.bfloat16),
        torch.optim.SGD([weights], lr=10),
        verifier=DecisionVerifier(CONTRACT, close_scores),
        epochs=3,
        enabled=True,
    )
    assert report["losses"][0] < -0.5
    assert weights[1].item() > weights[0].item()


def test_sparse_optimizer_and_repeated_input_isolation():
    """Sparse finite gradients update parameters and every visit gets a fresh input."""
    torch = pytest.importorskip("torch")
    embedding = torch.nn.Embedding(8, len(ACTIONS), sparse=True)
    with torch.no_grad():
        embedding.weight.zero_()
    contexts = {context: index for index, (context, _, _) in enumerate(SCENARIOS)}

    def score(item):
        index = contexts[item.context]
        object.__setattr__(item, "context", "changed")
        return embedding(torch.tensor(index))

    report = train_decisions(
        catalog("TRAIN"),
        score,
        torch.optim.SparseAdam(embedding.parameters(), lr=0.1),
        verifier=VERIFIER,
        epochs=2,
        enabled=True,
    )
    assert len(report["losses"]) == 16
    assert torch.count_nonzero(embedding.weight).item() > 0


def test_required_closure_optimizer_rejected_before_callbacks():
    """A required-closure optimizer fails before evaluator or model callbacks."""
    torch = pytest.importorskip("torch")
    weights = torch.nn.Parameter(torch.zeros(len(ACTIONS)))
    verifier = DecisionVerifier(CONTRACT, lambda _: pytest.fail("verifier called"))
    with pytest.raises(ValueError, match="required arguments"):
        train_decisions(
            catalog("TRAIN"),
            lambda _: pytest.fail("scorer called"),
            torch.optim.LBFGS([weights]),
            verifier=verifier,
            enabled=True,
        )
