"""Raw research storage reuses artifact/authority contracts without self-certification."""

# Standard library
from dataclasses import replace
from datetime import datetime, timezone

# Third-party
import pytest

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import require_training_eligible
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
    consume_authorization,
)
from mindful_trace_gepa.action_bound_events import ActionRecord


class Clock:
    def now(self):
        return datetime(2026, 11, 6, tzinfo=timezone.utc)


def source(payload=b'{"value":1}', artifact_id="result", version="1"):
    return propose_research_source(
        artifact_id,
        version,
        payload,
        actor_id="researcher",
        evidence_refs=(
            EvidenceReference(
                "capture:" + artifact_id + ":" + version, EvidenceSourceKind.EXTERNAL_RECORD
            ),
        ),
        clock=Clock(),
        enabled=True,
    )


def context(callback=lambda packet: True):
    return ArchiveHostContext(
        TrustedEvaluatorContract("host-source-checker", "1", "custody"), callback
    )


def authority(scope, action_class="research_source_append", name="append"):
    action = ActionRecord(name, action_class, True, scope, "prediction:" + name)
    d = action_record_digest(action)
    policy = ActionAuthorityPolicy("policy", name, d, scope, RuntimeCapability.WRITE)
    grant = AuthorityGrant(
        "grant", "host", RuntimeRole.EXECUTOR, (RuntimeCapability.WRITE,), name, d, scope
    )
    registry = AuthorityGrantRegistry.enroll((grant,), (policy,))
    authorization = authorize_action(
        action,
        principal_id="host",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.WRITE,
        policy_id="policy",
        grant_registry=registry,
        grant_ids=("grant",),
        clock=Clock(),
    )
    return dict(action=action, authorization=authorization, grant_registry=registry, clock=Clock())


def append(archive, proposal, payload, ctx=None):
    before = archive.catalog
    args = authority(archive.append_scope(before, proposal))
    after = archive.append(
        before, proposal, payload, context=context() if ctx is None else ctx, enabled=True, **args
    )
    return after, args


def test_archive_preserves_all_versions_and_requires_exact_one_use_write(tmp_path):
    archive = ResearchArchive(tmp_path, "research", enabled=True)
    before = archive.catalog
    proposal = source()
    after, args = append(archive, proposal, b'{"value":1}')
    assert before.revision == 0 and after.revision == 1
    with pytest.raises(PermissionError, match="consumed"):
        consume_authorization(
            args["authorization"],
            args["action"],
            grant_registry=args["grant_registry"],
            clock=Clock(),
        )
    assert after.snapshot.state.claims == () and after.snapshot.sources == ()
    assert (
        archive.read(after, "result", "1", trusted_catalog_digest=after.digest, enabled=True)
        == b'{"value":1}'
    )
    with pytest.raises((ValueError, PermissionError)):
        archive.append(before, proposal, b'{"value":1}', context=context(), enabled=True, **args)
    after2, _ = append(archive, source(b'{"value":2}', version="2"), b'{"value":2}')
    assert (
        archive.read(after2, "result", "1", trusted_catalog_digest=after2.digest, enabled=True)
        == b'{"value":1}'
    )
    assert (
        archive.read(after2, "result", "2", trusted_catalog_digest=after2.digest, enabled=True)
        == b'{"value":2}'
    )
    with pytest.raises(ValueError, match="duplicate|version"):
        archive.append_scope(after2, proposal)
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(after2.to_dict())
    reopened = ResearchArchive(
        tmp_path, "research", catalog=after2, trusted_catalog_digest=after2.digest, enabled=True
    )
    assert reopened.catalog == after2


@pytest.mark.parametrize("callback", [None, lambda packet: False, lambda packet: 1])
def test_missing_or_truthy_source_authentication_fails_before_io(tmp_path, callback):
    archive = ResearchArchive(tmp_path, "research", enabled=True)
    proposal = source()
    before = archive.catalog
    args = authority(archive.append_scope(before, proposal))
    with pytest.raises(PermissionError):
        archive.append(
            before, proposal, b'{"value":1}', context=context(callback), enabled=True, **args
        )
    assert list(tmp_path.iterdir()) == [] and archive.catalog == before


def test_mutated_source_and_wrong_authority_cannot_publish(tmp_path):
    archive = ResearchArchive(tmp_path, "research", enabled=True)
    proposal = source()
    before = archive.catalog
    args = authority(archive.append_scope(before, proposal))
    with pytest.raises(ValueError, match="bytes|digest"):
        archive.append(before, proposal, b"changed", context=context(), enabled=True, **args)
    with pytest.raises(PermissionError):
        archive.append(
            before,
            proposal,
            b'{"value":1}',
            context=context(),
            enabled=True,
            **{**args, "authorization": None},
        )

    def mutate(packet):
        object.__setattr__(packet.source, "actor_id", "changed")
        return True

    with pytest.raises(PermissionError):
        archive.append(
            before, proposal, b'{"value":1}', context=context(mutate), enabled=True, **args
        )
    self_auth = replace(proposal, actor_id="host-source-checker")
    with pytest.raises(PermissionError, match="self"):
        archive.append(
            before,
            self_auth,
            b'{"value":1}',
            context=context(),
            enabled=True,
            **authority(archive.append_scope(before, self_auth)),
        )
    assert list(tmp_path.iterdir()) == []


def test_pinned_catalog_and_source_tampering_fail_closed(tmp_path):
    archive = ResearchArchive(tmp_path, "research", enabled=True)
    after, _ = append(archive, source(), b'{"value":1}')
    with pytest.raises(PermissionError):
        archive.read(after, "result", "1", trusted_catalog_digest="0" * 64, enabled=True)
    blob = next(tmp_path.glob("*.blob"))
    blob.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest|source"):
        archive.read(after, "result", "1", trusted_catalog_digest=after.digest, enabled=True)
    blob.unlink()
    with pytest.raises(ValueError, match="missing|source"):
        archive.read(after, "result", "1", trusted_catalog_digest=after.digest, enabled=True)


def test_scope_is_bound_to_physical_archive_root(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    first = ResearchArchive(one, "same-logical-store", enabled=True)
    second = ResearchArchive(two, "same-logical-store", enabled=True)
    proposal = source()
    args = authority(first.append_scope(first.catalog, proposal))
    with pytest.raises(PermissionError):
        second.append(
            second.catalog, proposal, b'{"value":1}', context=context(), enabled=True, **args
        )
    assert list(two.iterdir()) == []


def test_source_symlinks_and_partial_publication_cannot_advance_catalog(tmp_path, monkeypatch):
    archive = ResearchArchive(tmp_path, "research", enabled=True)
    proposal = source()
    before = archive.catalog
    original = archive._write_file

    def fail_catalog(name, payload):
        if name.startswith("catalog-"):
            raise OSError("simulated metadata I/O failure")
        return original(name, payload)

    monkeypatch.setattr(archive, "_write_file", fail_catalog)
    with pytest.raises(OSError):
        append(archive, proposal, b'{"value":1}')
    assert archive.catalog == before
    blob = next(tmp_path.glob("*.blob"))
    observed = blob.stat().st_mtime_ns
    assert blob.read_bytes() == b'{"value":1}'
    monkeypatch.setattr(archive, "_write_file", original)
    after, _ = append(archive, proposal, b'{"value":1}')
    assert blob.stat().st_mtime_ns == observed
    target = tmp_path / "outside"
    target.write_bytes(b"not-the-source")
    blob.unlink()
    blob.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe|source"):
        archive.read(after, "result", "1", trusted_catalog_digest=after.digest, enabled=True)
    assert target.read_bytes() == b"not-the-source"
