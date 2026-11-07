"""Opt-in finite resource trial with actual action-scoped synthetic host custody."""

# Standard library
import argparse
import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Local
from evaluation.phase2_reporting import SourceSelector, finite_reconstruction_trial
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.research_archive import (
    ArchiveHostContext,
    ResearchArchive,
    propose_research_source,
)
from gepa_mindfulness.verification.runtime_governance import (
    ActionAuthorityPolicy,
    AuthorityGrant,
    AuthorityGrantRegistry,
    RuntimeCapability,
    RuntimeRole,
    action_record_digest,
    authorize_action,
)
from mindful_trace_gepa.action_bound_events import ActionRecord
from synthetic_data.cva_fixtures import make_security_fixtures, qualify_security_fixture


class Clock:
    def now(self):
        return datetime.now(timezone.utc)


def run(output: Path, *, enabled: bool = False) -> dict:
    if enabled is not True:
        raise ValueError("finite resource probe is disabled")
    output.mkdir(exist_ok=False)
    archive = ResearchArchive(output, "phase2-resource-demo", enabled=True)
    trials = []
    clock = Clock()
    for fixture in make_security_fixtures(enabled=True):
        payload = json.dumps(
            {
                "schema_version": "finite-cva-source-v1",
                "fixture": fixture.to_dict(),
                "qualification": qualify_security_fixture(fixture, enabled=True),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
        source = propose_research_source(
            fixture.fixture_id,
            "1",
            payload,
            actor_id="synthetic-researcher",
            evidence_refs=(
                EvidenceReference(
                    "finite:" + fixture.fixture_id, EvidenceSourceKind.EXTERNAL_RECORD
                ),
            ),
            clock=clock,
            enabled=True,
        )
        before = archive.catalog
        scope = archive.append_scope(before, source)
        name = "append-" + str(before.revision)
        action = ActionRecord(name, "research_source_append", True, scope, "prediction:" + name)
        digest = action_record_digest(action)
        policy = ActionAuthorityPolicy("policy", name, digest, scope, RuntimeCapability.WRITE)
        grant = AuthorityGrant(
            "grant",
            "synthetic-host",
            RuntimeRole.EXECUTOR,
            (RuntimeCapability.WRITE,),
            name,
            digest,
            scope,
        )
        registry = AuthorityGrantRegistry.enroll((grant,), (policy,))
        authorization = authorize_action(
            action,
            principal_id="synthetic-host",
            role=RuntimeRole.EXECUTOR,
            capability=RuntimeCapability.WRITE,
            policy_id="policy",
            grant_registry=registry,
            grant_ids=("grant",),
            clock=clock,
        )
        context = ArchiveHostContext(
            TrustedEvaluatorContract("synthetic-source-checker", "1", "finite-demo-custody"),
            lambda packet: packet.source.source_digest == sha256(payload).hexdigest()
            and packet.source.actor_id == "synthetic-researcher",
        )
        catalog = archive.append(
            before,
            source,
            payload,
            context=context,
            action=action,
            authorization=authorization,
            grant_registry=registry,
            clock=clock,
            enabled=True,
        )
        selector = SourceSelector(
            source.artifact_id, source.version, source.source_digest, "", "document"
        )
        trials.append(
            finite_reconstruction_trial(
                archive, catalog, selector, trusted_catalog_digest=catalog.digest, enabled=True
            )
        )
    return {
        "schema_version": "phase2-resource-probe-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "authentication": "SYNTHETIC_SOFTWARE_FIXTURE",
        "seed": 0,
        "python": platform.python_version(),
        "created_at": clock.now().isoformat(),
        "catalog": archive.catalog.to_dict(),
        "trials": trials,
        "model_experiments": "NOT_RUN",
        "alignment_improvement": "NOT_ESTABLISHED",
        "limitations": [
            "Serialized checkpoint bytes, not peak resident memory.",
            "Wall times include different checked operations; no speedup claim.",
            "Synthetic host credentials do not authorize production admissions.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.enabled:
        parser.error("requires --enabled")
    print(
        json.dumps(run(args.archive_root, enabled=True), indent=2, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
