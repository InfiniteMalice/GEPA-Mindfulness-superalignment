"""Bounded public debate through trusted application callbacks, disabled by default."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from gepa_mindfulness.verification.check_records import CheckResult
from gepa_mindfulness.verification.debate_records import (
    ArgumentSnapshot,
    BoundCheckResult,
    DebateChallenge,
    DebateRound,
    DebateSession,
    _record,
)
from gepa_mindfulness.verification.diagnostic_records import DiagnosticRecord

from .causal_records import content_digest
from .debate_records import (
    DebateContext,
    DebateProtocol,
    DebateVerification,
    debate_protocol_digest,
)


def authenticated(candidate: DiagnosticRecord, callback: Callable[..., bool] | None) -> bool:
    """Accept only unchanged, detached, exactly-True host receipts; exceptions stay unresolved."""
    if callback is None:
        return False
    try:
        original = candidate.to_dict()
        detached = type(candidate).from_dict(original)
        accepted = callback(detached)
        return accepted is True and detached.to_dict() == original
    except Exception:
        return False


def verification_envelope(
    protocol: DebateProtocol,
    round_: DebateRound,
    check: BoundCheckResult,
) -> DebateVerification:
    """Reconstruct an exact host-authentication envelope from retained public evidence."""
    if round_.before is None or round_.challenge is None:
        raise ValueError("verification requires before snapshot and challenge")
    return DebateVerification(
        debate_protocol_digest(protocol),
        round_.round_index,
        content_digest(round_.before.to_dict()),
        content_digest(round_.challenge.to_dict()),
        check,
        protocol.evaluator,
    )


def validate_debate_round(protocol: DebateProtocol, round_: DebateRound) -> None:
    """Reject stale, cross-role or foreign check bindings before using any judgment."""
    roles = {a.role: a.actor_id for a in protocol.actors}
    if round_.round_index >= protocol.max_rounds:
        raise ValueError("round outside protocol budget")
    for snapshot in (round_.before, round_.after):
        if snapshot is not None and snapshot.actor_id != roles["defender"]:
            raise ValueError("snapshot must belong to declared defender")
    before, ch = round_.before, round_.challenge
    if ch is None:
        return
    if before is None or before.graph is None:
        raise ValueError("challenge requires a decomposable snapshot")
    if ch.actor_id != roles["challenger"] or ch.snapshot_digest != content_digest(before.to_dict()):
        raise ValueError("challenge role or snapshot digest mismatch")
    ids = {n.claim.claim_id for n in before.graph.nodes}
    if not set(ch.target_claim_ids) <= ids:
        raise ValueError("challenge targets unknown claims")
    slots = {s.check_id for s in protocol.check_slots if s.round_index == round_.round_index}
    requests = {r.check_id: r for r in ch.requests}
    if not requests.keys() <= slots:
        raise ValueError("request outside preregistered round slots")
    for bound in round_.results:
        r = bound.result
        req = requests.get(r.check_id)
        if req is None:
            raise ValueError("result has no submitted request")
        if bound.snapshot_digest != ch.snapshot_digest or bound.request_digest != content_digest(
            req.to_dict()
        ):
            raise ValueError("result snapshot or request digest mismatch")
        if (
            r.claim_id != req.claim_id
            or r.action_id != req.action_id
            or r.verifier_id != roles["verifier"]
        ):
            raise ValueError("result claim/action/verifier mismatch")
    if any(r.human_required for r in round_.results) and round_.after is not None:
        raise ValueError("pending human review cannot have a revision")


def validate_debate_session(protocol: DebateProtocol, session: DebateSession) -> None:
    """Revalidate persisted transcripts without granting any verifier authority."""
    if (
        session.session_id != protocol.session_id
        or session.protocol_digest != debate_protocol_digest(protocol)
    ):
        raise ValueError("session protocol binding mismatch")
    if len(session.rounds) > protocol.max_rounds:
        raise ValueError("session exceeds protocol budget")
    seen: set[str] = set()
    for round_ in session.rounds:
        validate_debate_round(protocol, round_)
        if round_.challenge is not None:
            if round_.challenge.challenge_id in seen:
                raise ValueError("duplicate challenge ID")
            seen.add(round_.challenge.challenge_id)
    last = session.rounds[-1]
    if session.stop_reason in ("host_stopped", "budget_exhausted"):
        if last.status != "observed":
            raise ValueError("completed capture requires observed final round")
        if session.stop_reason == "budget_exhausted" and len(session.rounds) != protocol.max_rounds:
            raise ValueError("budget stop before round budget exhausted")
    elif last.status != session.stop_reason:
        raise ValueError("partial round and session stop disagree")


class _ProducerFailure(Exception):
    """Sanitized callback failure; arbitrary exception text never enters public evidence."""


def _call(stage: str, callback: Callable[..., Any], *args: Any) -> Any:
    try:
        return callback(*(deepcopy(arg) for arg in args))
    except Exception as exc:
        raise _ProducerFailure(f"{stage}: {type(exc).__name__}") from None


def run_sensitive_debate(
    protocol: DebateProtocol,
    *,
    defend: Callable[[DebateContext], ArgumentSnapshot | None],
    challenge: Callable[[DebateContext, ArgumentSnapshot], DebateChallenge | None],
    verify: Callable[
        [DebateContext, ArgumentSnapshot, DebateChallenge], tuple[BoundCheckResult, ...]
    ],
    revise: Callable[
        [DebateContext, ArgumentSnapshot, DebateChallenge, tuple[CheckResult, ...]],
        ArgumentSnapshot | None,
    ],
    authenticate: Callable[[DebateVerification], bool] | None = None,
    stop: Callable[[DebateSession], bool] | None = None,
    enabled: bool = False,
) -> DebateSession:
    """Capture at most eight rounds, retaining uncertainty without choosing a truth winner.

    Producer exceptions become partial transcripts. Structurally invalid returned data raises
    ValueError. Authentication errors remain unresolved. Only trusted caller-provided Python
    callbacks run; procedure text and model output are inert data.
    """
    if enabled is not True:
        raise ValueError("debate requires enabled=True")
    protocol = _record(protocol, DebateProtocol)
    rounds: list[DebateRound] = []
    digest = debate_protocol_digest(protocol)

    def finish(reason: str) -> DebateSession:
        session = DebateSession(protocol.session_id, digest, tuple(rounds), reason)
        validate_debate_session(protocol, session)
        return session

    for index in range(protocol.max_rounds):
        context = DebateContext(
            protocol.session_id,
            index,
            tuple(rounds),
            tuple(s for s in protocol.check_slots if s.round_index == index),
            protocol.subject.turns,
        )
        current = DebateRound(index, None, None, (), None, "missing", "defend not captured")
        try:
            before = _call("defend", defend, context)
            if before is None:
                rounds.append(current)
                return finish("missing")
            before = _record(before, ArgumentSnapshot)
            current = replace(current, before=before, reason="challenge not captured")
            validate_debate_round(protocol, current)
            if before.decomposition_status == "undecomposable":
                rounds.append(
                    replace(current, status="undecomposable", reason="defender cannot decompose")
                )
                return finish("undecomposable")
            ch = _call("challenge", challenge, context, before)
            if ch is None:
                rounds.append(current)
                return finish("missing")
            ch = _record(ch, DebateChallenge)
            current = replace(current, challenge=ch, reason="verification not captured")
            validate_debate_round(protocol, current)
            values = _call("verify", verify, context, before, ch)
            if type(values) is not tuple or any(type(v) is not BoundCheckResult for v in values):
                raise ValueError("verify must return tuple of BoundCheckResult")
            current = replace(current, results=values, reason="revision not captured")
            validate_debate_round(protocol, current)
            accepted = tuple(
                (
                    bound.result
                    if authenticated(verification_envelope(protocol, current, bound), authenticate)
                    and not bound.human_required
                    else replace(bound.result, verdict="unresolved", revision_claim_id=None)
                )
                for bound in current.results
            )
            if any(bound.human_required for bound in current.results):
                rounds.append(
                    replace(
                        current, status="human_pending", reason="independent human review required"
                    )
                )
                return finish("human_pending")
            after = _call("revise", revise, context, before, ch, accepted)
            if after is None:
                rounds.append(current)
                return finish("missing")
            after = _record(after, ArgumentSnapshot)
            current = replace(
                current, after=after, status="observed", reason="public phases captured"
            )
            validate_debate_round(protocol, current)
            if after.decomposition_status == "undecomposable":
                rounds.append(
                    replace(current, status="undecomposable", reason="revision cannot decompose")
                )
                return finish("undecomposable")
            rounds.append(current)
            candidate = finish("host_stopped")
            if stop is not None and _call("stop", stop, candidate) is True:
                return candidate
        except _ProducerFailure as exc:
            failed = replace(current, status="callback_error", reason=str(exc))
            if len(rounds) > index:
                rounds[-1] = failed
            else:
                rounds.append(failed)
            return finish("callback_error")
    return finish("budget_exhausted")
