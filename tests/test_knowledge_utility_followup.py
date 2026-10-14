"""Check the offline follow-up roster, preparation boundaries, and publication contract."""

# Standard library
from collections import Counter
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from hashlib import sha256
from itertools import permutations

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts.prepare_knowledge_utility_pilot import prepare_pilot


@pytest.fixture
def task_data():
    from scripts import knowledge_utility_followup_tasks

    return knowledge_utility_followup_tasks


def test_roster_has_48_structural_variants(task_data):
    tasks = task_data.authored_tasks()
    task_data.validate_tasks(tasks)
    assert len(tasks) == len({t.task_id for t in tasks}) == 48
    assert len(task_data.MECHANISM_IDS) == 24
    assert Counter(t.mechanism_id for t in tasks) == dict.fromkeys(task_data.MECHANISM_IDS, 2)
    for mechanism in task_data.MECHANISM_IDS:
        pair = [t for t in tasks if t.mechanism_id == mechanism]
        assert {t.variant for t in pair} == {0, 1}
        assert pair[0].variant_rationale != pair[1].variant_rationale
        assert pair[0].prompt != pair[1].prompt
    for task in tasks:
        assert task.task_id == f"followup-{task.mechanism_id}-{task.variant}"
        assert len(task.derivation) >= 3
        assert all(d.error for d in task.distractors)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "unknown",
        "boolean",
        "repeat",
        "blank",
        "option",
        "space-option",
        "error",
        "long",
        "steps",
        "nul",
        "long-option",
        "long-step",
        "long-error",
        "long-rationale",
        "many-steps",
    ],
)
def test_invalid_task_roster_rejected(task_data, fault):
    tasks = task_data.authored_tasks()
    first = tasks[0]
    if fault == "missing":
        tasks = tasks[:-1]
    elif fault == "extra":
        tasks += (first,)
    elif fault == "duplicate":
        tasks = (first, first) + tasks[2:]
    else:
        changes = {
            "unknown": dict(mechanism_id="unknown"),
            "boolean": dict(variant=True),
            "repeat": dict(variant=1),
            "blank": dict(prompt=" \t"),
            "long": dict(prompt="é" * 3001),
            "steps": dict(derivation=("one", "two")),
            "nul": dict(prompt="bad\x00prompt"),
            "long-option": dict(correct_text="x" * 257),
            "long-step": dict(derivation=("x" * 1025, "two", "three")),
            "long-rationale": dict(variant_rationale="x" * 1025),
            "many-steps": dict(derivation=("step",) * 13),
        }
        if fault in {"option", "space-option", "error", "long-error"}:
            distractor = first.distractors[0]
            if fault in {"option", "space-option"}:
                distractor = replace(
                    distractor, text=first.correct_text + (" " if fault == "space-option" else "")
                )
            else:
                distractor = replace(distractor, error="" if fault == "error" else "x" * 1025)
            changes[fault] = dict(distractors=(distractor,) + first.distractors[1:])
        tasks = (replace(first, **changes[fault]),) + tasks[1:]
    with pytest.raises(ValueError):
        task_data.validate_tasks(tasks)


def test_representative_answers_recomputed_from_facts(task_data):
    """Recalculate worked and boundary examples without copying their stored answer keys."""
    tasks = {(t.mechanism_id, t.variant): t for t in task_data.authored_tasks()}
    assert tasks["inventory_reconciliation", 0].correct_text == str(23 + 8 - 3 - 7 - 9)
    assert tasks["inventory_reconciliation", 1].correct_text == str(23 + 8 - 3 - 9 - (7 - 4))
    routes = {"Direct": (15, 20), "Via": (4 + 6 + 2, 28), "Express": (10 + 4, 22)}
    eligible = {name: cost for name, (cost, time) in routes.items() if time <= 25}
    assert tasks["constrained_route_cost", 0].correct_text == min(eligible, key=eligible.get)
    duration = {"A": 2, "B": 3, "C": 1}
    feasible = [
        order
        for order in permutations(duration)
        if order.index("A") < order.index("C")
        and sum(duration[job] for job in order[: order.index("B") + 1]) <= 4
    ]
    assert len(feasible) == 1
    assert tasks["task_precedence", 0].correct_text == "-".join(feasible[0])
    discounted = Decimal("1.25") * Decimal("0.9")
    per_item = discounted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    aggregate = (3 * discounted).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert tasks["rounding_policy", 0].correct_text == str(3 * per_item + Decimal("0.20"))
    assert tasks["rounding_policy", 1].correct_text == str(aggregate + Decimal("0.20"))
    left, right = [(1, 5), (1, 7), (2, 9)], [1, 1]
    inner = sum(value for key, value in left for other in right if key == other)
    unmatched = sum(value for key, value in left if key not in right)
    assert tasks["table_joins", 0].correct_text == str(inner)
    assert tasks["table_joins", 1].correct_text == str(inner + unmatched)
    states = [10]
    states.append(states[-1] + 5)
    states.append(states[-1] * 2)
    states.append(states[-1] - 4)
    assert tasks["transaction_order", 0].correct_text == str(states[-1])
    states.pop()
    assert tasks["transaction_order", 1].correct_text == str(states[-1])
    elapsed = 750 / 1.5 + 70 + 30
    assert elapsed == 10 * 60
    assert not elapsed < 10 * 60
    assert tasks["unit_conversion", 1].correct_text == f"LATE; {int(elapsed - 599)} second"
    possibilities = [certified for certified in (False, True) if not certified or False]
    assert possibilities == [False]
    assert tasks["conditional_statements", 0].correct_text == "K is not certified"


@pytest.fixture
def preparer():
    from scripts import prepare_knowledge_utility_followup

    return prepare_knowledge_utility_followup


def test_packet_is_deterministic_and_digest_bound(preparer):
    packet = preparer.prepare_followup()
    assert set(packet) == set(
        "schema_version study_id as_of status execution_authorized packet_frozen optimizer_input "
        "confers_authority training_eligibility model limits knowledge rubric schedule "
        "evaluator_cases actor_requests review_templates report_template packet_digest".split()
    )
    assert packet["schema_version"] == "knowledge-utility-followup-packet-v1"
    assert canonical_json(packet) == canonical_json(preparer.prepare_followup())
    original = packet.pop("packet_digest")
    assert original == content_digest(packet)
    packet["evaluator_cases"][0]["expected_option"] = "changed"
    assert content_digest(packet) != original
    assert preparer.prepare_followup()["packet_digest"] == original


def test_schedule_and_options_are_balanced(preparer):
    packet = preparer.prepare_followup()
    cases = packet["evaluator_cases"]
    assert len(cases) == len({c["task_id"] for c in cases}) == 48
    assert Counter(c["expected_option"] for c in cases) == dict.fromkeys("ABCD", 12)
    slots = packet["schedule"]["slots"]
    assert len(slots) == len({s["slot_id"] for s in slots}) == 384
    conditions = sorted(["none", "lesson_only", "bank_only", "bank_plus_lesson"])
    assert Counter(s["condition"] for s in slots) == dict.fromkeys(conditions, 96)
    assert [s["sequence"] for s in slots] == list(range(384))
    key = "knowledge-utility-followup-order-v1"
    ids = [c["task_id"] for c in cases]
    assert ids == sorted(ids, key=lambda t: (sha256(f"{key}|task|{t}".encode()).hexdigest(), t))
    blocks = [(t, repeat) for t in ids for repeat in (0, 1)]
    blocks.sort(key=lambda b: (sha256(f"{key}|block|{b[0]}|{b[1]}".encode()).hexdigest(), *b))
    orders = list(permutations(conditions))
    for index, block in enumerate(blocks):
        group = slots[index * 4 : index * 4 + 4]
        assert [(s["task_id"], s["repeat"]) for s in group] == [block] * 4
        assert tuple(s["condition"] for s in group) == orders[index % 24]
    actual_orders = [tuple(s["condition"] for s in slots[i : i + 4]) for i in range(0, 384, 4)]
    assert Counter(actual_orders) == dict.fromkeys(orders, 4)


def test_option_remap_preserves_key_and_distractors(preparer, task_data):
    authored = {t.task_id: t for t in task_data.authored_tasks()}
    for case in preparer.prepare_followup()["evaluator_cases"]:
        task = authored[case["task_id"]]
        assert case["options"][case["expected_option"]] == task.correct_text
        assert {case["options"][k]: v for k, v in case["distractor_errors"].items()} == {
            d.text: d.error for d in task.distractors
        }
        assert case["task_digest"] == content_digest(
            dict(prompt=case["prompt"], options=case["options"])
        )


def test_actor_messages_only_contain_public_material(preparer, task_data, monkeypatch):
    tasks = task_data.authored_tasks()
    sentinel = "PRIVATE_DERIVATION_SENTINEL"
    altered = (replace(tasks[0], derivation=(sentinel,) * 3, variant_rationale=sentinel),) + tasks[
        1:
    ]
    monkeypatch.setattr(preparer, "authored_tasks", lambda: altered)
    packet = preparer.prepare_followup()
    cases = {c["task_id"]: c for c in packet["evaluator_cases"]}
    knowledge = packet["knowledge"]
    lesson = knowledge["lesson"]["summary"]
    bank = knowledge["bank"]["entries"][0]["lesson"]["summary"]
    contexts = {
        "none": "",
        "lesson_only": lesson,
        "bank_only": bank,
        "bank_plus_lesson": bank + "\n" + lesson,
    }
    base = (
        "Solve the task using its explicit facts and rules. Return only JSON with one key, "
        '"answer", whose value is exactly "A", "B", "C", or "D". '
        "Do not add explanation or other keys."
    )
    for slot, request in zip(packet["schedule"]["slots"], packet["actor_requests"], strict=True):
        assert {k: v for k, v in request.items() if k != "messages"} == slot
        case = cases[request["task_id"]]
        context = contexts[request["condition"]]
        system = base + ("\nKnowledge context:\n" + context if context else "")
        user = (
            case["prompt"]
            + "\nOptions:\n"
            + "\n".join(f"{label}: {text}" for label, text in case["options"].items())
        )
        assert request["messages"] == [
            dict(role="system", content=system),
            dict(role="user", content=user),
        ]
        assert sentinel not in canonical_json(request["messages"])


def test_authority_and_templates_stay_pending(preparer):
    packet = preparer.prepare_followup()
    for field in ("execution_authorized", "packet_frozen", "optimizer_input", "confers_authority"):
        assert packet[field] is False
    assert packet["training_eligibility"] == "DEVELOPMENT"
    assert packet["status"] == "pending_independent_review"
    assert packet["limits"]["approved_generation_budget_usd"] is None
    assert packet["limits"]["generation_requests"] == 384
    assert packet["schedule"]["generation_seed_policy"] == "unknown"
    assert all(s["generation_seed"] is None for s in packet["schedule"]["slots"])
    for case, review in zip(
        packet["evaluator_cases"], packet["review_templates"]["task_reviews"], strict=True
    ):
        assert case["review_status"] == review["status"] == "pending"
        assert case["dependency_group"] is None and case["dependencies_known"] is False
        assert review["task_digest"] == case["task_digest"]
        assert review["reviewer_id"] is None and review["evidence_refs"] == []
        assert all(value is None for value in review["checks"].values())
    response = packet["review_templates"]["response_review"]
    assert response["status"] == "pending" and response["evidence_refs"] == []
    for key in ("review_id", "task_digest", "response_digest", "reviewer_id", "selected_option"):
        assert response[key] is None
    assert response["rubric_digest"] == content_digest(packet["rubric"])
    report = packet["report_template"]
    assert report["observations"] == [] and report["execution_state"] == "not_started"
    assert report["verified"] is report["paired"] is False
    for row in report["conditions"].values():
        assert row["expected"] == 96
        assert all(v is None for k, v in row.items() if k not in ("expected", "rates"))
        assert all(v is None for v in row["rates"].values())
    assert all(v is None for row in report["comparisons"].values() for v in row.values())


def test_knowledge_matches_frozen_pilot(preparer, monkeypatch):
    prior = prepare_pilot()
    assert (
        prior["packet_digest"] == "637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1"
    )
    knowledge = preparer.prepare_followup()["knowledge"]
    assert knowledge["origin_packet_digest"] == prior["packet_digest"]
    for field in ("source_episodes", "lesson", "bank"):
        assert knowledge[field] == prior[field]
    prior["packet_digest"] = "changed"
    monkeypatch.setattr(preparer, "prepare_pilot", lambda: prior)
    with pytest.raises(ValueError):
        preparer.prepare_followup()


def test_packet_size_is_bounded(preparer, monkeypatch):
    rubric = preparer._rubric()
    rubric["oversized_authored_metadata"] = "x" * (4 * 1024 * 1024)
    monkeypatch.setattr(preparer, "_rubric", lambda: rubric)
    with pytest.raises(ValueError, match="exceeds byte limit"):
        preparer.prepare_followup()


def test_origin_content_checked_even_with_unchanged_digest(preparer, monkeypatch):
    prior = prepare_pilot()
    prior["lesson"]["summary"] = "Substituted content"
    monkeypatch.setattr(preparer, "prepare_pilot", lambda: prior)
    with pytest.raises(ValueError, match="original pilot content changed"):
        preparer.prepare_followup()


def test_original_interfaces_reject_followup(preparer):
    from scripts.knowledge_utility_capture_records import build_manifest
    from scripts.score_knowledge_utility_pilot import score_pilot

    packet = preparer.prepare_followup()
    with pytest.raises(ValueError):
        build_manifest(packet)
    with pytest.raises(ValueError):
        score_pilot(packet, {})
