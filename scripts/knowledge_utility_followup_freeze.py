"""Pure offline acceptance preparation and explicitly declared freeze construction."""

# Standard library
from copy import deepcopy
from hashlib import sha256

# Third-party
# Local
from evaluation.causal_records import content_digest
from scripts import knowledge_utility_followup_freeze_profile as profile
from scripts.knowledge_utility_followup_records import (
    MAX_INPUT_BYTES,
    SUPPORTED_PACKET_DIGEST,
    JsonObject,
    bounded_text,
    decode_json_text,
    object_fields,
    validate_json_value,
)

MAX_RECORD_BYTES = 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MARKDOWN_ROLES = frozenset({"review_readme", "review_report", "targeted_recheck"})


def _digest(value: JsonObject, field: str) -> str:
    return content_digest({key: item for key, item in value.items() if key != field})


def _packet(value: object, expected: str) -> JsonObject:
    validate_json_value(value, max_bytes=MAX_INPUT_BYTES)
    if type(value) is not dict or value.get("packet_digest") != expected:
        raise ValueError("unsupported packet")
    if _digest(value, "packet_digest") != expected:
        raise ValueError("packet digest mismatch")
    return value


def _evidence(evidence: dict[str, bytes]) -> tuple[list[JsonObject], JsonObject]:
    if type(evidence) is not dict or set(evidence) != set(profile.EVIDENCE_PINS):
        raise ValueError("unsupported evidence roles")
    total = 0
    metadata: list[JsonObject] = []
    predecessor: JsonObject = {}
    for role in sorted(profile.EVIDENCE_PINS):
        data = evidence[role]
        if type(data) is not bytes or len(data) > MAX_INPUT_BYTES:
            raise ValueError("invalid evidence bytes")
        total += len(data)
        if total > MAX_EVIDENCE_BYTES:
            raise ValueError("evidence exceeds total limit")
        digest = sha256(data).hexdigest()
        if (digest, len(data)) != profile.EVIDENCE_PINS[role]:
            raise ValueError("unsupported evidence content")
        try:
            text = data.decode("utf-8")
        except UnicodeError as error:
            raise ValueError("invalid evidence encoding") from error
        if role not in MARKDOWN_ROLES:
            decoded = decode_json_text(text, max_bytes=MAX_INPUT_BYTES)
            if type(decoded) is not dict:
                raise ValueError("evidence object required")
            if role == "predecessor_packet":
                predecessor = decoded
        metadata.append(dict(role=role, file_sha256=digest, size_bytes=len(data)))
    return metadata, _packet(predecessor, profile.PREDECESSOR_DIGEST)


def _correction_scope(old: JsonObject, packet: JsonObject) -> None:
    """Reverse only the documented v3 corrections and require exact predecessor content."""
    restored = deepcopy(packet)
    old_cases = {case["task_id"]: case for case in old["evaluator_cases"]}
    if len(old_cases) != 48 or len(packet["evaluator_cases"]) != 48:
        raise ValueError("invalid task roster")
    for case, assessment in zip(restored["evaluator_cases"], profile.TASK_ASSESSMENTS):
        task = case["task_id"]
        prior = old_cases.get(task)
        if (
            prior is None
            or task != assessment["task_id"]
            or prior["task_digest"] != assessment["source_task_digest"]
            or case["task_digest"] != assessment["current_task_digest"]
        ):
            raise ValueError("task review binding mismatch")
        if task == "followup-flow_conservation-0":
            case["distractor_errors"]["A"] = prior["distractor_errors"]["A"]
        if task == "followup-task_precedence-1":
            current_prompt = case["prompt"]
            case["prompt"], case["task_digest"] = prior["prompt"], prior["task_digest"]
            for request in restored["actor_requests"]:
                if request["task_id"] == task:
                    content = request["messages"][1]["content"]
                    if content.count(current_prompt) != 1:
                        raise ValueError("actor prompt mismatch")
                    request["messages"][1]["content"] = content.replace(
                        current_prompt, prior["prompt"]
                    )
            for review in restored["review_templates"]["task_reviews"]:
                if review["task_id"] == task:
                    review["task_digest"] = prior["task_digest"]
    restored["packet_digest"] = old["packet_digest"]
    if restored != old:
        raise ValueError("unexpected correction scope")


def prepare_acceptance(packet: object, evidence: dict[str, bytes]) -> JsonObject:
    """Bind preserved evidence to all task assessments without approving or freezing it."""
    source = _packet(packet, SUPPORTED_PACKET_DIGEST)
    metadata, predecessor = _evidence(evidence)
    _correction_scope(predecessor, source)
    result = dict(
        schema_version="knowledge-utility-followup-acceptance-candidate-v1",
        source_packet_digest=source["packet_digest"],
        predecessor_packet_digest=predecessor["packet_digest"],
        evidence=metadata,
        tasks=deepcopy(list(profile.TASK_ASSESSMENTS)),
        resolved_findings=deepcopy(list(profile.RESOLVED_FINDINGS)),
        limitations=deepcopy(list(profile.LIMITATIONS)),
        pending_task_count=0,
    )
    result["candidate_digest"] = content_digest(result)
    validate_json_value(result, max_bytes=MAX_RECORD_BYTES - 1)
    return result


def freeze_followup(
    packet: object, evidence: dict[str, bytes], candidate: object, decision: object
) -> JsonObject:
    """Validate a declared human decision; hashes do not authenticate its authorization."""
    validate_json_value(candidate, max_bytes=MAX_RECORD_BYTES)
    validate_json_value(decision, max_bytes=MAX_RECORD_BYTES)
    expected = prepare_acceptance(packet, evidence)
    # Canonical hashing distinguishes JSON false from zero, unlike Python object equality.
    if content_digest(candidate) != content_digest(expected):
        raise ValueError("acceptance candidate mismatch")
    approved = object_fields(
        decision,
        "schema_version candidate_digest source_packet_digest maintainer_id authorization_ref "
        "accepted_limitations accept_task_review authorize_freeze execution_authorized "
        "decision_digest",
    )
    if (
        approved["schema_version"] != "knowledge-utility-followup-freeze-decision-v1"
        or approved["candidate_digest"] != expected["candidate_digest"]
        or approved["source_packet_digest"] != expected["source_packet_digest"]
        or approved["accept_task_review"] is not True
        or approved["authorize_freeze"] is not True
        or approved["execution_authorized"] is not False
        or approved["accepted_limitations"] != [row["id"] for row in expected["limitations"]]
        or approved["decision_digest"] != _digest(approved, "decision_digest")
    ):
        raise ValueError("invalid freeze decision")
    bounded_text(approved["maintainer_id"], 1024)
    bounded_text(approved["authorization_ref"], 1024)
    result = deepcopy(_packet(packet, SUPPORTED_PACKET_DIGEST))
    result.update(
        schema_version="knowledge-utility-followup-frozen-packet-v1",
        status="frozen_with_accepted_task_review",
        packet_frozen=True,
        freeze_record=dict(
            source_packet_digest=expected["source_packet_digest"],
            acceptance=deepcopy(expected),
            decision=deepcopy(approved),
        ),
    )
    result["packet_digest"] = _digest(result, "packet_digest")
    validate_json_value(result, max_bytes=MAX_INPUT_BYTES - 1)
    return result
