"""Host-reviewed, durable event accounting without reward or execution authority."""

# Standard library
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

# Local
from evaluation.causal_records import canonical_json, content_digest

from .causal_reward import _accepted
from .causal_reward_records import (
    _digest,
    _json_value,
    _read_request,
    _request_wire,
    _text,
    request_digest,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .research_credit import ResearchCreditClaim, audit_research_credit

_USE_SCHEMA = "research-credit-use-v1"
_LEDGER_SCHEMA = "research-credit-ledger-v1"
_RECEIPT_SCHEMA = "research-credit-reservation-v1"
_REQUIRED_LAYERS = {"provider", "reward_integrity", "causal_reward"}


@dataclass(frozen=True)
class ResearchCreditUse:
    """A proposed allocation, never authorization to optimize or execute."""

    use_id: str
    policy_id: str
    request_digest: str
    selected_claim_ids: tuple[str, ...]
    layers: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]

    def __post_init__(self) -> None:
        """Keep all selection and overlap coordinates exact, bounded and immutable."""
        self._validate()

    def _validate(self) -> None:
        """Validate caller declarations without filling unknown overlap statuses."""
        _text(self.use_id, "use_id")
        _text(self.policy_id, "policy_id")
        _digest(self.request_digest)
        if (
            type(self.selected_claim_ids) is not tuple
            or not 1 <= len(self.selected_claim_ids) <= 1024
        ):
            raise ValueError("expected 1..1024 selected claims")
        for claim_id in self.selected_claim_ids:
            _text(claim_id, "claim_id")
        if len(set(self.selected_claim_ids)) != len(self.selected_claim_ids):
            raise ValueError("duplicate selected claim")
        if type(self.layers) is not tuple or len(self.layers) != len(self.selected_claim_ids):
            raise ValueError("expected accounting for every selected claim")
        accounted = set()
        for entry in self.layers:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("accounting entry requires claim ID and layers")
            claim_id, layers = entry
            _text(claim_id, "accounting claim_id")
            if claim_id in accounted:
                raise ValueError("duplicate accounting claim")
            accounted.add(claim_id)
            if type(layers) is not tuple or not 1 <= len(layers) <= 32:
                raise ValueError("expected 1..32 layers")
            names = set()
            for layer in layers:
                if type(layer) is not tuple or len(layer) != 2:
                    raise ValueError("layer requires name and disposition")
                name, disposition = layer
                _text(name, "layer")
                if (
                    name in names
                    or type(disposition) is not str
                    or disposition not in {"not_counted", "already_counted", "unknown"}
                ):
                    raise ValueError("invalid or duplicate accounting layer")
                names.add(name)
        if accounted != set(self.selected_claim_ids):
            raise ValueError("selected and accounted claims differ")

    def to_dict(self) -> dict[str, object]:
        """Export a detached DEVELOPMENT proposal, never a reusable authority token."""
        self._validate()
        value = {
            "schema_version": _USE_SCHEMA,
            "training_eligibility": "DEVELOPMENT",
            "use_id": self.use_id,
            "policy_id": self.policy_id,
            "request_digest": self.request_digest,
            "selected_claim_ids": list(self.selected_claim_ids),
            "layers": [
                [claim_id, [list(layer) for layer in layers]] for claim_id, layers in self.layers
            ],
        }
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> ResearchCreditUse:
        """Restore only exact v1 proposals while preserving declared unknowns."""
        validate_input_envelope(value)
        if type(value) is not dict or set(value) != {
            "schema_version",
            "training_eligibility",
            "use_id",
            "policy_id",
            "request_digest",
            "selected_claim_ids",
            "layers",
        }:
            raise ValueError("unexpected use fields")
        if value["schema_version"] != _USE_SCHEMA or value["training_eligibility"] != "DEVELOPMENT":
            raise ValueError("unsupported use schema or relabel")
        if type(value["selected_claim_ids"]) is not list or type(value["layers"]) is not list:
            raise ValueError("wire selections and layers must be arrays")
        layers = []
        for entry in value["layers"]:
            if type(entry) is not list or len(entry) != 2 or type(entry[1]) is not list:
                raise ValueError("invalid accounting array")
            if any(type(layer) is not list for layer in entry[1]):
                raise ValueError("wire layer must be an array")
            layers.append((entry[0], tuple(tuple(layer) for layer in entry[1])))
        return cls(
            value["use_id"],
            value["policy_id"],
            value["request_digest"],
            tuple(value["selected_claim_ids"]),
            tuple(layers),
        )


class ResearchCreditLedger:
    """Host-owned accounting journal; it cannot issue reward or action grants.

    The host authenticates review, evidence and semantic event identities. A SQLite
    reservation is conservative and permanent here; downstream failure does not
    refund an event. Keep the file and callback outside candidate/model control.
    """

    def __init__(self, path: str | Path, *, authority_domain: str) -> None:
        """Create/reopen a journal pinned to a schema and one authority domain."""
        _text(authority_domain, "authority_domain")
        if str(path) == ":memory:":
            raise ValueError("a durable journal path is required")
        self._path = str(Path(path).absolute())
        self.authority_domain = authority_domain
        with closing(sqlite3.connect(self._path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS credit_metadata "
                    "(id INTEGER PRIMARY KEY CHECK(id=1), schema_version TEXT, domain TEXT)"
                )
                connection.execute(
                    "INSERT OR IGNORE INTO credit_metadata VALUES (1, ?, ?)",
                    (_LEDGER_SCHEMA, authority_domain),
                )
                self._check(connection)
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS credit_reservations "
                    "(use_id TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL, "
                    "payload TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS credit_events "
                    "(run_id TEXT, task_id TEXT, event_id TEXT, receipt_digest TEXT, "
                    "PRIMARY KEY(run_id, task_id, event_id))"
                )

    def _check(self, connection: sqlite3.Connection) -> None:
        """Fail closed when a journal's pinned metadata changes."""
        if connection.execute("SELECT schema_version, domain FROM credit_metadata").fetchall() != [
            (_LEDGER_SCHEMA, self.authority_domain)
        ]:
            raise ValueError("journal authority/schema mismatch")

    def reserve(
        self,
        request: RewardRequest,
        claims: tuple[ResearchCreditClaim, ...],
        use: ResearchCreditUse,
        *,
        authorize_use: Callable[[dict[str, object]], bool] | None = None,
        enabled: bool = False,
    ) -> dict[str, object] | None:
        """Atomically record a fresh host-reviewed allocation once per event.

        Structural eligibility never substitutes for host review. The callback sees
        the complete request, raw audit, selection and prior-layer declarations. All
        refusals happen before writes; raw database failures roll back every row.
        """
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if not enabled:
            return None
        if type(request) is not RewardRequest or type(use) is not ResearchCreditUse:
            raise ValueError("expected exact request and use records")
        target = _read_request(_request_wire(request))
        selection = ResearchCreditUse.from_dict(use.to_dict())
        if request_digest(target) != selection.request_digest:
            raise ValueError("target request digest mismatch")
        require_training_eligible(target.trajectory.to_dict())
        audit = audit_research_credit(claims, enabled=True)
        snapshots = [ResearchCreditClaim.from_dict(value) for value in audit["claims"]]
        by_id = {claim.claim_id: claim for claim in snapshots}
        evidence_kinds = {}
        for claim in snapshots:
            if claim.authority_domain != self.authority_domain:
                raise ValueError("foreign claim authority")
            for ref in claim.evidence_refs:
                key = (claim.authority_domain, claim.run_id, ref.reference_id)
                if key in evidence_kinds and evidence_kinds[key] != ref.source_kind:
                    raise ValueError("contradictory evidence source kind")
                evidence_kinds[key] = ref.source_kind
        events = set()
        for claim_id, layers in selection.layers:
            if claim_id not in by_id or audit["prerequisite_gaps"][claim_id]:
                raise ValueError("selected claim is unresolved or structurally ineligible")
            claim = by_id[claim_id]
            if (
                _json_value(claim.source_metadata_json).get("target_request_digest")
                != selection.request_digest
            ):
                raise ValueError("claim target binding mismatch")
            if claim.model_version != target.trajectory.policy_version:
                raise ValueError("claim policy version mismatch")
            if not set(claim.evidence_refs).issubset(target.observable_references):
                raise ValueError("claim evidence outside target request")
            if not _REQUIRED_LAYERS.issubset(dict(layers)) or any(
                disposition != "not_counted" for _, disposition in layers
            ):
                raise ValueError("missing, unknown or previously counted reward layer")
            event = (claim.run_id, claim.task_id, claim.event_id)
            if event in events:
                raise ValueError("one semantic event cannot select multiple contributions")
            events.add(event)
        envelope = {
            "schema_version": "research-credit-review-v1",
            "authority_domain": self.authority_domain,
            "target_request": _request_wire(target),
            "use": selection.to_dict(),
            "audit": audit,
        }
        validate_input_envelope(envelope)
        # Pin the database connection before invoking trusted external host code.
        with closing(sqlite3.connect(self._path, timeout=5)) as connection:
            self._check(connection)
            if not _accepted(authorize_use, envelope):
                raise ValueError("independent host credit review rejected or unavailable")
            receipt = {
                "schema_version": _RECEIPT_SCHEMA,
                "training_eligibility": "DEVELOPMENT",
                "authority_granted": False,
                "status": "reserved",
                "envelope": envelope,
            }
            receipt["receipt_digest"] = content_digest(receipt)
            validate_input_envelope(receipt)
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                self._check(connection)
                if connection.execute(
                    "SELECT 1 FROM credit_reservations WHERE use_id=?", (selection.use_id,)
                ).fetchone():
                    raise ValueError("credit use identity already reserved")
                for event in events:
                    if connection.execute(
                        "SELECT 1 FROM credit_events WHERE "
                        "run_id=? AND task_id=? AND event_id=?",
                        event,
                    ).fetchone():
                        raise ValueError("semantic event already reserved")
                connection.execute(
                    "INSERT INTO credit_reservations VALUES (?, ?, ?)",
                    (selection.use_id, receipt["receipt_digest"], canonical_json(receipt)),
                )
                connection.executemany(
                    "INSERT INTO credit_events VALUES (?, ?, ?, ?)",
                    [(*event, receipt["receipt_digest"]) for event in sorted(events)],
                )
        return receipt

    def read(self, receipt_digest: str) -> dict[str, object]:
        """Read a detached, digest-checked accounting receipt, not a live grant."""
        _digest(receipt_digest)
        with closing(sqlite3.connect(self._path, timeout=5)) as connection:
            self._check(connection)
            row = connection.execute(
                "SELECT payload FROM credit_reservations WHERE digest=?", (receipt_digest,)
            ).fetchone()
        if row is None:
            raise ValueError("unknown reservation receipt")
        value = _json_value(row[0])
        if type(value) is not dict or value.get("receipt_digest") != receipt_digest:
            raise ValueError("invalid reservation receipt")
        payload = {key: item for key, item in value.items() if key != "receipt_digest"}
        if (
            content_digest(payload) != receipt_digest
            or value.get("schema_version") != _RECEIPT_SCHEMA
        ):
            raise ValueError("reservation receipt integrity mismatch")
        return value
