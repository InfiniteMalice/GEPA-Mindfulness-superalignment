"""Source-bound reproducibility statistics, protected failures and finite reconstruction."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from test_research_archive import Clock, append, source

# Local
from evaluation.improvement_records import MetricSpec, ResamplingPolicy
from evaluation.phase2_reporting import (
    ModelExperimentManifest,
    PairedSource,
    ProtectedRegression,
    ResearchComparisonProtocol,
    ResearchHostContext,
    SourceSelector,
    finite_reconstruction_trial,
    report_research_comparison,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.verification.research_archive import ResearchArchive
from synthetic_data.cva_fixtures import make_security_fixtures, qualify_security_fixture


def archive_json(archive, name, payload):
    raw = json.dumps(payload, sort_keys=True).encode()
    proposal = source(raw, artifact_id=name)
    after, _ = append(archive, proposal, raw)
    return SourceSelector(name, "1", proposal.source_digest, "/value", "number")


def comparison_fixture(tmp_path, kind="model", passed=True):
    archive = ResearchArchive(tmp_path, "reports", enabled=True)
    pairs = []
    for n in range(4):
        baseline = archive_json(
            archive, f"b{n}", {"schema_version": "metric-v1", "capture_id": f"b{n}", "value": 0.1}
        )
        candidate = archive_json(
            archive,
            f"c{n}",
            {"schema_version": "metric-v1", "capture_id": f"c{n}", "value": 0.3 + 0.1 * n},
        )
        pairs.append(
            PairedSource(f"p{n}", f"case{n}", "condition", 0, f"cluster{n}", baseline, candidate)
        )
    regression = ProtectedRegression(
        "1" * 64 if kind == "model" else None,
        "2" * 64,
        passed,
        kind,
        TrustedEvaluatorContract("protected-checker", "1", "capability-regression"),
        (EvidenceReference("protected-receipt", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    selected = archive_json(archive, "protected", regression.to_dict())
    selected = replace(selected, pointer="", kind="protected")
    manifest = ModelExperimentManifest(
        "phase2-test",
        "c92ba24" + "0" * 33,
        "0" * 64,
        "1" * 64,
        "candidate-v1",
        "matched-harness-v1",
        "3" * 64,
        (79,),
        "behaviorally_trained",
        Clock().now().isoformat(),
    )
    protocol = ResearchComparisonProtocol(
        "paired-research",
        manifest,
        MetricSpec("success", "metric-v1", "value", "4" * 64),
        tuple(pairs),
        ResamplingPolicy(79, resamples=256, confidence_level=0.95, minimum_clusters=2),
        True,
        selected,
    )
    host = ResearchHostContext(
        TrustedEvaluatorContract("independent-stat-checker", "1", "source-rubric-custody"),
        "researcher",
        lambda packet: True,
    )
    return archive, protocol, host


def report(archive, protocol, host=None):
    return report_research_comparison(
        protocol,
        archive,
        archive.catalog,
        trusted_catalog_digest=archive.catalog.digest,
        context=host,
        enabled=True,
    )


def test_source_based_cluster_intervals_are_deterministic_and_remain_diagnostic(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    first = report(archive, p, host)
    second = report(archive, p, host)
    assert first == second
    assert first["estimate"]["raw_delta"] == pytest.approx(0.35)
    assert len(first["estimate"]["cluster_ids"]) == 4
    assert first["estimate"]["interval"] is not None
    assert first["interpretation"] == "OBSERVED_METRIC_IMPROVEMENT_WITH_PROTECTED_CHECK"
    assert first["alignment_improvement"] == "NOT_ESTABLISHED"
    assert first["promotion"] == "NOT_AUTHORIZED"
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(first)


def test_missing_authentication_and_dependencies_cannot_create_confidence(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    unknown = report(archive, p)
    assert unknown["estimate"]["interval"] is None
    assert unknown["interpretation"] == "UNVERIFIED_SOURCES"
    assert unknown["observations"]
    dependent = report(archive, replace(p, dependencies_known=False), host)
    assert dependent["estimate"]["interval"] is None
    assert "unknown_dependencies" in dependent["estimate"]["reasons"]
    undersampled = report(archive, replace(p, policy=replace(p.policy, minimum_clusters=20)), host)
    assert "insufficient_clusters" in undersampled["estimate"]["reasons"]


def test_model_regression_failure_is_never_averaged_away(tmp_path):
    archive, p, host = comparison_fixture(tmp_path, passed=False)
    result = report(archive, p, host)
    assert result["estimate"]["improvement"] > 0
    assert result["interpretation"] == "WITHHELD_PROTECTED_REGRESSION"
    assert result["protected_regression"]["status"] == "FAILED"


def test_software_tests_cannot_stand_in_for_model_capability_regression(tmp_path):
    archive, p, host = comparison_fixture(tmp_path, kind="software")
    result = report(archive, p, host)
    assert result["interpretation"] == "WITHHELD_MODEL_REGRESSION_UNRUN"
    with pytest.raises(ValueError, match="diagnostic|model"):
        replace(p.manifest, condition="diagnostic_instrumented")


def test_mutated_auth_packets_self_review_and_source_aliases_are_rejected(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)

    def mutate(packet):
        object.__setattr__(packet.protocol.manifest, "model_version", "changed")
        return True

    denied = report(archive, p, replace(host, authenticate=mutate))
    assert denied["interpretation"] == "UNVERIFIED_SOURCES"
    with pytest.raises(ValueError, match="self"):
        replace(host, actor_id=host.evaluator.evaluator_id)
    with pytest.raises(ValueError, match="reused|alias"):
        replace(
            p, pairs=(p.pairs[0], replace(p.pairs[1], baseline=p.pairs[0].baseline)) + p.pairs[2:]
        )


def test_missing_pair_and_changed_source_digest_remain_visible(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    missing = replace(
        p.pairs[-1],
        candidate=replace(p.pairs[-1].candidate, artifact_id="not-run", source_digest="a" * 64),
    )
    result = report(archive, replace(p, pairs=p.pairs[:-1] + (missing,)), host)
    assert "incomplete_pairs" in result["estimate"]["reasons"]
    assert result["missing_pair_ids"] == [missing.pair_id]
    bad = replace(p.pairs[0], baseline=replace(p.pairs[0].baseline, source_digest="b" * 64))
    with pytest.raises(ValueError, match="digest"):
        report(archive, replace(p, pairs=(bad,) + p.pairs[1:]), host)


def test_finite_reconstruction_retains_original_evidence_and_actual_fidelity(tmp_path):
    archive = ResearchArchive(tmp_path, "resource", enabled=True)
    fixture = make_security_fixtures(enabled=True)[1]
    payload = {
        "schema_version": "finite-cva-source-v1",
        "fixture": fixture.to_dict(),
        "qualification": qualify_security_fixture(fixture, enabled=True),
    }
    selected = replace(archive_json(archive, "finite", payload), pointer="", kind="document")
    result = finite_reconstruction_trial(
        archive,
        archive.catalog,
        selected,
        trusted_catalog_digest=archive.catalog.digest,
        enabled=True,
    )
    assert result["verification_fidelity"] is True
    assert result["source_unchanged"] is True
    assert result["compact"]["retained_state_bytes"] < result["full"]["retained_state_bytes"]
    assert result["compact"]["reconstruction_wall_time_ns"] >= 0
    assert result["mathematical_theorem_claim"] is False
    assert result["measurement_scope"] == "serialized checkpoint state; original archive retained"


def test_lower_direction_and_rate_transform_preserve_interval_sign(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    p = replace(p, metric=replace(p.metric, direction="lower", transform="one_minus"))
    result = report(archive, p, host)
    assert result["estimate"]["raw_delta"] == pytest.approx(-0.35)
    assert result["estimate"]["improvement"] == pytest.approx(0.35)
    assert result["estimate"]["interval"][1] < 0
    assert result["improvement_interval"][0] > 0
    assert result["interpretation"] == "OBSERVED_METRIC_IMPROVEMENT_WITH_PROTECTED_CHECK"


def test_repeated_cases_are_not_independent_clusters(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    pairs = []
    for n in range(3):
        left = archive_json(
            archive,
            f"repeat-b{n}",
            {"schema_version": "metric-v1", "capture_id": f"rb{n}", "value": 0.1},
        )
        right = archive_json(
            archive,
            f"repeat-c{n}",
            {"schema_version": "metric-v1", "capture_id": f"rc{n}", "value": 0.5},
        )
        pairs.append(
            PairedSource(f"repeat{n}", "one-case", "same-condition", n, "one-cluster", left, right)
        )
    result = report(archive, replace(p, pairs=tuple(pairs)), host)
    assert len(result["estimate"]["cluster_ids"]) == 1
    assert result["estimate"]["matched_count"] == 3
    assert result["estimate"]["interval"] is None
    assert "insufficient_clusters" in result["estimate"]["reasons"]


def test_mutated_host_context_is_revalidated_before_authentication(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    object.__setattr__(host, "actor_id", host.evaluator.evaluator_id)
    with pytest.raises(ValueError, match="self"):
        report(archive, p, host)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[" * 40 + b"0" + b"]" * 40])
def test_ambiguous_or_nonfinite_source_json_is_rejected(raw):
    from evaluation.phase2_reporting import _load_json

    with pytest.raises(ValueError):
        _load_json(raw, "json")


def test_protected_receipt_for_another_candidate_fails_closed(tmp_path):
    archive, p, host = comparison_fixture(tmp_path)
    mismatched = replace(p, manifest=replace(p.manifest, candidate_model_digest="9" * 64))
    with pytest.raises(ValueError, match="candidate"):
        report(archive, mismatched, host)


@pytest.mark.parametrize("digit", ["٠", "０", "𝟘"])
def test_unicode_array_aliases_cannot_manufacture_independent_pairs(tmp_path, digit):
    archive, p, host = comparison_fixture(tmp_path)
    original = archive_json(
        archive, "shared", {"schema_version": "metric-v1", "values": [0.1, 0.9]}
    )
    b = replace(original, pointer="/values/0")
    c = replace(original, pointer="/values/1")
    aliased_b = replace(original, pointer="/values/" + digit)
    aliased_c = replace(original, pointer="/values/" + chr(ord(digit) + 1))
    pairs = (
        PairedSource("first", "case0", "condition", 0, "cluster0", b, c),
        PairedSource("second", "case1", "condition", 0, "cluster1", aliased_b, aliased_c),
    )
    with pytest.raises(ValueError, match="pointer|canonical"):
        report(archive, replace(p, pairs=pairs), host)
