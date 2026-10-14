"""Synthetic offline fixtures; these are not model or independent-review evidence."""

# Standard library
from copy import deepcopy
from functools import lru_cache

# Third-party
# Local
from scripts.prepare_knowledge_utility_followup import prepare_followup


def fresh_packet():
    return deepcopy(_synthetic_packet())


@lru_cache(maxsize=1)
def _synthetic_packet():
    return prepare_followup()


def capture_batch(packet, rows=()):
    return dict(
        schema_version="knowledge-utility-followup-captures-v1",
        packet_digest=packet["packet_digest"],
        captures=deepcopy(list(rows)),
    )


def row_for(packet, condition, index, *, status="completed", text=None, reason=None):
    slots = [s for s in packet["schedule"]["slots"] if s["condition"] == condition]
    return dict(slot_id=slots[index]["slot_id"], status=status, response_text=text, reason=reason)


def case_for_slot(packet, slot_id):
    slot = next(s for s in packet["schedule"]["slots"] if s["slot_id"] == slot_id)
    return deepcopy(next(c for c in packet["evaluator_cases"] if c["task_id"] == slot["task_id"]))
