"""Contextual experience fitting and replay interference."""

from dataclasses import replace

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
