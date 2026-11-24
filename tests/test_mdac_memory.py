"""Lifecycle of host-approved, revision-bound memory in deterministic CPU fixtures."""

from dataclasses import replace

import pytest

from gepa_mindfulness.verification import evidence_workspace as workspace

# isort: split
from artifact_fixtures import NOW, make_derived_snapshot, make_query, make_topology


def fixture():
    from gepa_mindfulness.verification import mdac_memory as memory

    snapshot = make_derived_snapshot()
    nodes = tuple(
        workspace.WorkspaceNode(item.item_id, item.item_id)
        for item in snapshot.sources + snapshot.interpretations
    )
    index = workspace.prepare_evidence_workspace(
        snapshot, nodes, snapshot.sources[0].claim.evidence_refs, enabled=True
    )
    query = make_query(snapshot)
    context = memory.MemoryContext("session1", "inspection", "obligations-v1")
    tags = tuple(
        memory.MemoryEntry(
            item_id=item.item_id,
            item_digest=memory.artifact_digest(item),
            state_kind=kind,
            subject=subject,
            principal_id=query.principal_id,
            scope_id=query.scope_id,
            session_id="session1",
            topic_id=None if subject == "obligation" else "inspection",
            retention="cross_session" if subject == "obligation" else "session",
            expires_at="2026-11-09T12:00:00Z",
            authorization_version=query.policy_version,
            obligation_version="obligations-v1",
        )
        for item, kind, subject in zip(
            snapshot.sources + snapshot.interpretations,
            ("prescribed", "observed", "inferred"),
            ("obligation", "authorization", "state"),
        )
    )
    catalog = memory.revise_memory(
        None, snapshot, upsert=tags, as_of=NOW, retain=lambda entry: True, enabled=True
    )
    frame = memory.MemoryFrame(catalog, snapshot, index, make_topology(snapshot, "synthesis"))
    return memory, frame, query, context


def read(frame, query, context, *, authorize=lambda request: True, detail="full"):
    from gepa_mindfulness.verification.mdac_memory import retrieve_memory

    return retrieve_memory(
        lambda: frame, query, context, detail=detail, authorize=authorize, enabled=True
    )["items"]


def test_disabled_does_not_contact_store_or_parse_inputs():
    from gepa_mindfulness.verification import mdac_memory as memory

    assert memory.retrieve_memory(lambda: 1 / 0, object(), object()) is None
    assert memory.revise_memory(object(), object()) is None


def test_compression_and_live_restoration_preserve_provenance_and_privacy():
    memory, frame, query, context = fixture()
    pointers = read(frame, query, context, detail="pointer")
    assert {row["state_kind"] for row in pointers} == {"prescribed", "observed", "inferred"}
    assert all("text" not in row for row in pointers)
    assert "principal-secret" not in str(pointers)
    full = read(frame, query, context)
    summary = next(row for row in full if row["state_kind"] == "inferred")
    assert summary["text"] == "The combined inspection indicates risk."
    assert summary["channel"] == "unverified" and len(summary["source_handles"]) == 2
    assert read(frame, query, context, authorize=lambda request: False) == []
    assert read(frame, query, context, authorize=None) == []
    assert memory.MemoryCatalog.from_dict(frame.catalog.to_dict()) == frame.catalog
    assert "material risk" not in str(frame.catalog.to_dict())


def test_long_history_topic_switch_session_and_expiry():
    memory, frame, query, context = fixture()
    # Unrelated session-scoped notes are explicitly revised in a long history.
    note = next(entry for entry in frame.catalog.entries if entry.item_id == "b")
    for tick in range(100):
        context = replace(context, topic_id=f"other-{tick}")
        catalog = memory.revise_memory(
            frame.catalog,
            frame.snapshot,
            upsert=(replace(note, topic_id=context.topic_id),),
            as_of=NOW,
            retain=lambda _: True,
            enabled=True,
        )
        frame = replace(frame, catalog=catalog)
    rows = read(frame, query, replace(context, session_id="session2"))
    assert [row["subject"] for row in rows] == ["obligation"]
    assert read(frame, replace(query, policy_version="revoked"), context) == []
    assert read(frame, query, replace(context, obligation_version="obligations-v2")) == []
    assert read(frame, replace(query, assessed_at="2026-11-09T12:00:00Z"), context) == []
    assert read(frame, replace(query, principal_id="other"), context) == []


def test_forget_and_update_invalidate_summaries_preserve_independent_support():
    memory, frame, query, context = fixture()
    options = {"enabled": True, "as_of": NOW, "retain": lambda entry: True}
    catalog = memory.revise_memory(frame.catalog, frame.snapshot, delete=("a",), **options)
    rows = read(replace(frame, catalog=catalog), query, context)
    assert [row["state_kind"] for row in rows] == ["observed"]
    assert {entry.item_id for entry in catalog.entries} == {"b"}
    old = next(entry for entry in frame.catalog.entries if entry.item_id == "b")
    update = replace(old, authorization_version="revoked")
    revised = memory.revise_memory(frame.catalog, frame.snapshot, upsert=(update,), **options)
    assert {entry.item_id for entry in revised.entries} == {"a", "b"}
    assert [row["subject"] for row in read(replace(frame, catalog=revised), query, context)] == [
        "obligation"
    ]


def test_external_load_rechecks_current_revision_and_retention():
    memory, frame, query, context = fixture()
    current = frame

    def load():
        return current

    def retrieve():
        return memory.retrieve_memory(
            load, query, context, authorize=lambda req: True, enabled=True
        )

    assert len(retrieve()["items"]) == 3
    empty = memory.revise_memory(
        None,
        frame.snapshot,
        upsert=frame.catalog.entries,
        as_of=NOW,
        retain=lambda _: False,
        enabled=True,
    )
    current = replace(frame, catalog=empty)
    assert retrieve()["items"] == []
    current = replace(frame, catalog=replace(frame.catalog, snapshot_digest="0" * 64))
    with pytest.raises(ValueError, match="snapshot"):
        retrieve()


def test_prediction_does_not_become_observation_or_permission():
    memory, frame, query, context = fixture()
    prior = next(entry for entry in frame.catalog.entries if entry.item_id == "summary")
    options = {"enabled": True, "as_of": NOW, "retain": lambda _: True}
    with pytest.raises(ValueError, match="derived"):
        memory.revise_memory(
            frame.catalog,
            frame.snapshot,
            upsert=(replace(prior, state_kind="observed"),),
            **options,
        )
    predicted = memory.revise_memory(
        frame.catalog, frame.snapshot, upsert=(replace(prior, state_kind="predicted"),), **options
    )
    rows = read(replace(frame, catalog=predicted), query, context)
    assert any(row["state_kind"] == "predicted" and row["channel"] == "unverified" for row in rows)


def test_physical_source_removal_or_revision_prunes_old_pointers():
    from gepa_mindfulness.verification.state import EvidenceState

    memory, frame, _, _ = fixture()
    old = frame.snapshot
    survivor = old.sources[1]
    removed = replace(
        old,
        artifacts=(old.artifacts[1],),
        sources=(survivor,),
        interpretations=(),
        state=EvidenceState((survivor.claim,)),
    )
    catalog = memory.revise_memory(frame.catalog, removed, as_of=NOW, enabled=True)
    assert [entry.item_id for entry in catalog.entries] == ["b"]
    changed = replace(survivor, quality=replace(survivor.quality, compression_distortion=0.5))
    revised = replace(removed, sources=(changed,))
    assert memory.revise_memory(catalog, revised, as_of=NOW, enabled=True).entries == ()
    with pytest.raises(ValueError, match="exact live"):
        memory.revise_memory(
            None, revised, upsert=catalog.entries, as_of=NOW, retain=lambda _: True, enabled=True
        )


def test_restore_pointer_rechecks_exact_item_after_revision():
    memory, frame, query, context = fixture()
    pointers = read(frame, query, context, detail="pointer")
    handle = next(row["handle"] for row in pointers if row["state_kind"] == "prescribed")

    def restore(current, authorize=lambda _: True):
        return memory.retrieve_memory(
            lambda: current,
            query,
            context,
            restore_handle=handle,
            detail="full",
            enabled=True,
            authorize=authorize,
        )["items"]

    assert [row["text"] for row in restore(frame)] == ["Inspection establishes a material risk."]
    changed = memory.revise_memory(
        frame.catalog, frame.snapshot, delete=("a",), as_of=NOW, enabled=True
    )
    assert restore(replace(frame, catalog=changed)) == []
    assert restore(frame, authorize=lambda _: False) == []


def test_expiry_prunes_metadata_on_refresh_even_with_retention_approval():
    memory, frame, _, _ = fixture()
    revised = memory.revise_memory(
        frame.catalog,
        frame.snapshot,
        upsert=frame.catalog.entries,
        as_of="2026-11-09T12:00:00Z",
        retain=lambda _: True,
        enabled=True,
    )
    assert revised.entries == ()


def test_optional_local_model_reads_only_current_memory():
    import os

    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL for local model integration")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, frame, query, context = fixture()
    allowed = read(frame, query, context)
    prompt = query.public_query + "\n" + "\n".join(row["text"] for row in allowed)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    assert output.shape[1] == inputs.input_ids.shape[1] + 1
    assert read(frame, query, context, authorize=lambda _: False) == []
