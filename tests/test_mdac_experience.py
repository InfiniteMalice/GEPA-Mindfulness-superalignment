"""Contextual experience fitting and replay interference."""

import json
from dataclasses import replace
from sys import float_info

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.mdac_experience import (
    Experience,
    ExperienceDomain,
    ExperienceLibraries,
    Query,
    UsageOutcome,
    measure_gain,
)


def reference(name):
    return EvidenceReference(name, EvidenceSourceKind.EXTERNAL_RECORD)


def record(domain=ExperienceDomain.UNCERTAINTY):
    return Experience(
        "help-1",
        domain,
        {"conflicting_sources": True},
        ("qa",),
        (reference("observed-3"),),
        "Disclose the uncertainty",
        "qualify the answer",
        "qualify the answer",
        "observed continuation",
        True,
        "honest uncertainty",
        (reference("run-3"),),
        ("requires conflicting evidence",),
        "v1",
    )


def query(domain=ExperienceDomain.UNCERTAINTY):
    return Query("q1", domain, "qa", {"conflicting_sources": True}, "v1", "judge", "horizon-5")


@pytest.mark.parametrize("domain", list(ExperienceDomain))
def test_verified_record_in_separate_domain_library(domain):
    item, context = record(domain), query(domain)
    trusted = {item.digest}
    library = ExperienceLibraries(resolve=lambda e: e.digest in trusted)
    library.remember(item)
    assert library.recall(context) == (item,)
    for wrong in (
        replace(context, state={"conflicting_sources": False}),
        replace(context, state={"conflicting_sources": 1}),
        replace(context, scope="other"),
        replace(context, policy_version="v2"),
    ):
        assert not library.recall(wrong)
    other = next(d for d in ExperienceDomain if d != domain)
    assert not library.recall(replace(context, domain=other))
    trusted.clear()
    assert not library.recall(context)


def test_failed_and_unverified_experience_do_not_become_advice():
    item = record()
    library = ExperienceLibraries(
        resolve=lambda e: e.verification_method == "observed continuation"
    )
    failed = replace(item, succeeded=False, observed_effect="guessed")
    library.remember(failed)
    assert library.history(item.domain) == (failed,)
    assert not library.recall(query())
    with pytest.raises(ValueError):
        library.remember(replace(item, verification_method="model generated assurance"))
    with pytest.raises(ValueError):
        library.remember(item)  # same local identity, different content


def test_matched_gain_counts_interference_and_regressions():
    item, context = record(), query()
    library = ExperienceLibraries(resolve=lambda e: e == item)
    library.remember(item)

    def evaluate(call):
        scores = {1: (0.0, 1.0), 2: (1.0, 0.0), 3: (0.5, 0.5)}
        value = scores[call.seed][call.experience is not None]
        return UsageOutcome(call.digest, value, value == 1.0, reference(call.digest))

    result = measure_gain(library, context, item.record_id, seeds=(1, 2, 3), sandbox=evaluate)
    assert result.gains == (1.0, -1.0, 0.0)
    assert result.mean_gain == 0.0 and result.harmed == 1 and result.regressed == 1
    assert len(result.pairs) == 3
    with pytest.raises(ValueError):
        measure_gain(
            library,
            context,
            item.record_id,
            seeds=(1,),
            sandbox=lambda call: replace(evaluate(call), request_digest="other"),
        )
    with pytest.raises(ValueError):
        measure_gain(
            library, replace(context, state={}), item.record_id, seeds=(1,), sandbox=evaluate
        )


def test_mean_gain_handles_large_finite_gains():
    item = record()
    library = ExperienceLibraries(resolve=lambda e: e == item)
    library.remember(item)

    def evaluate(call):
        value = 0.0 if call.experience is None else 1e308
        return UsageOutcome(call.digest, value, False, reference(call.digest))

    result = measure_gain(library, query(), item.record_id, seeds=(1, 2), sandbox=evaluate)
    assert result.gains == (1e308, 1e308)
    assert result.mean_gain == 1e308


def test_mean_cancels_unrepresentable_paired_differences():
    item = record()
    library = ExperienceLibraries(resolve=lambda e: e == item)
    library.remember(item)

    def evaluate(call):
        if call.seed == 3:
            return UsageOutcome(call.digest, 0.0, False, reference(call.digest))
        baseline = -1e308 if call.seed == 1 else 1e308
        value = baseline if call.experience is None else -baseline
        return UsageOutcome(call.digest, value, False, reference(call.digest))

    result = measure_gain(library, query(), item.record_id, seeds=(1, 2), sandbox=evaluate)
    assert result.mean_gain == 0.0
    assert result.harmed == 1
    assert json.loads(json.dumps(result.mean_gain, allow_nan=False)) == 0.0
    with pytest.raises(OverflowError):
        _ = result.gains
    positive = measure_gain(library, query(), item.record_id, seeds=(1,), sandbox=evaluate)
    with pytest.raises(OverflowError):
        _ = positive.mean_gain
    diluted = measure_gain(library, query(), item.record_id, seeds=(1, 3), sandbox=evaluate)
    assert diluted.mean_gain == 1e308

    with pytest.raises(ValueError, match="requested replay"):
        measure_gain(
            library,
            query(),
            item.record_id,
            seeds=(1, 2),
            sandbox=lambda call: evaluate(replace(call, seed=3 - call.seed)),
        )


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), 10**400, -(10**400)]
)
def test_outcome_rejects_values_outside_finite_float_domain(invalid):
    with pytest.raises(ValueError, match="finite"):
        UsageOutcome("request", invalid, False, reference("source"))


def test_extrema_and_subnormal_gains_have_finite_json_representations():
    item = record()
    library = ExperienceLibraries(resolve=lambda e: e == item)
    library.remember(item)
    values = (float_info.max, -float_info.max, float.fromhex("0x0.0000000000001p-1022"))

    def evaluate(call):
        value = 0.0 if call.experience is None else values[call.seed]
        return UsageOutcome(call.digest, value, False, reference(call.digest))

    for seed, value in enumerate(values):
        result = measure_gain(library, query(), item.record_id, seeds=(seed,), sandbox=evaluate)
        report = json.loads(json.dumps([result.gains, result.mean_gain], allow_nan=False))
        assert report == [[value], value]
