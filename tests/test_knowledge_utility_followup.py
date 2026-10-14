"""Check the offline follow-up roster, preparation boundaries, and publication contract."""

# Standard library
from collections import Counter
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from itertools import permutations

# Third-party
import pytest

# Local


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
