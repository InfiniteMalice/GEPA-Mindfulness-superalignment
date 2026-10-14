"""Authored development tasks; structural validation does not certify their answers."""

# Standard library
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# Third-party
# Local

MECHANISM_IDS = (
    "inventory_reconciliation",
    "tiered_pricing",
    "unit_conversion",
    "flow_conservation",
    "weighted_allocation",
    "rounding_policy",
    "constrained_route_cost",
    "time_window_scheduling",
    "task_precedence",
    "resource_packing",
    "matching_constraints",
    "queue_discipline",
    "access_conjunction",
    "policy_precedence",
    "quorum_with_abstentions",
    "quantified_rules",
    "conditional_statements",
    "exception_scope",
    "state_transitions",
    "transaction_order",
    "table_joins",
    "aggregation_filters",
    "set_operations",
    "interval_composition",
)


@dataclass(frozen=True)
class Distractor:
    """An alternative answer and the specific error it represents."""

    text: str
    error: str


@dataclass(frozen=True)
class AuthoredTask:
    """Immutable public facts with separate evaluator-only derivation and annotations."""

    task_id: str
    mechanism_id: str
    variant: int
    prompt: str
    correct_text: str
    distractors: tuple[Distractor, Distractor, Distractor]
    derivation: tuple[str, ...]
    variant_rationale: str


def _task(
    mechanism: str,
    variant: int,
    prompt: str,
    correct: str,
    distractors: tuple[tuple[str, str], tuple[str, str], tuple[str, str]],
    steps: tuple[str, ...],
    rationale: str,
) -> AuthoredTask:
    """Keep authored option text independent of the later A/B/C/D assignment."""
    return AuthoredTask(
        f"followup-{mechanism}-{variant}",
        mechanism,
        variant,
        prompt,
        correct,
        (Distractor(*distractors[0]), Distractor(*distractors[1]), Distractor(*distractors[2])),
        steps,
        rationale,
    )


_TASKS = (
    _task(
        "inventory_reconciliation",
        0,
        "Start with 23 items. Receive 8, discard 3 damaged items, reserve 7 of the remainder, "
        "then ship 9 from unreserved stock. Reservations remain unavailable. "
        "How many items are available?",
        "12",
        (
            ("19", "Counts reserved stock as available."),
            ("4", "Omits the receipt."),
            ("15", "Fails to discard damaged stock."),
        ),
        (
            "Physical stock before shipping is 23+8-3=28.",
            "Shipping leaves 28-9=19 items.",
            "Seven are reserved, so available stock is 19-7=12.",
        ),
        "A reservation stays in force through the shipment.",
    ),
    _task(
        "inventory_reconciliation",
        1,
        "Start with 23 physical items, including 7 reserved items. Receive 8, discard 3 "
        "unreserved damaged items, release 4 reserved items, then ship 9 from unreserved "
        "stock. How many items are available?",
        "16",
        (
            ("12", "Fails to release four reservations."),
            ("19", "Ignores remaining reservations."),
            ("20", "Adds released items to physical stock as well as availability."),
        ),
        (
            "Physical stock ends at 23+8-3-9=19.",
            "Reservations end at 7-4=3.",
            "Available stock is 19-3=16; release does not create items.",
        ),
        "A partial release changes availability without changing physical stock.",
    ),
    _task(
        "tiered_pricing",
        0,
        "Buy 10 units. The first 10 cost 3 credits each; later units cost 2 each. "
        "A 4-credit delivery fee applies, and a 5-credit coupon is subtracted after "
        "delivery. What is the total?",
        "29",
        (
            ("19", "Applies the later-unit rate to all units."),
            ("34", "Omits the coupon."),
            ("25", "Omits delivery."),
        ),
        (
            "Exactly ten units all use the first tier: 10*3=30.",
            "Delivery gives 30+4=34.",
            "Subtract the coupon: 34-5=29.",
        ),
        "The quantity meets but does not cross the tier.",
    ),
    _task(
        "tiered_pricing",
        1,
        "Buy 11 units. The first 10 cost 3 credits each; later units cost 2 each. "
        "Add 4 delivery credits, subtract a 5-credit coupon, then enforce a 35-credit "
        "minimum on the entire bill. What is the total?",
        "35",
        (
            ("31", "Ignores the minimum after the coupon."),
            ("39", "Enforces the minimum on the discounted item charge, then adds delivery."),
            ("36", "Omits the coupon and does not reconsider the minimum."),
        ),
        (
            "The item charge is 10*3+1*2=32.",
            "Delivery and coupon give 32+4-5=31.",
            "The final minimum raises 31 to 35.",
        ),
        "Crossing the tier is followed by a binding final minimum.",
    ),
    _task(
        "unit_conversion",
        0,
        "A trip covers 750 metres at 1.5 metres per second, followed by a 70-second wait "
        "and a 30-second transfer. One minute is 60 seconds. Arrival at or before "
        "10 minutes is on time. Is this trip ON_TIME or LATE, and how many seconds remain?",
        "ON_TIME; 0 seconds",
        (
            ("LATE; 0 seconds", "Treats an inclusive deadline as exclusive."),
            ("ON_TIME; 100 seconds", "Omits the wait and transfer."),
            ("ON_TIME; 30 seconds", "Omits the transfer."),
        ),
        (
            "Travel takes 750/1.5=500 seconds.",
            "Total is 500+70+30=600 seconds.",
            "Ten minutes is 600 seconds; equality is on time with zero remaining.",
        ),
        "An inclusive deadline accepts an exact converted boundary.",
    ),
    _task(
        "unit_conversion",
        1,
        "A trip covers 750 metres at 1.5 metres per second, followed by a 70-second wait "
        "and a 30-second transfer. One minute is 60 seconds. Arrival strictly before "
        "10 minutes is on time. Is this trip ON_TIME or LATE, and how many seconds "
        "must be saved to arrive a whole second before the deadline?",
        "LATE; 1 second",
        (
            ("ON_TIME; 0 seconds", "Ignores the strict inequality."),
            ("LATE; 0 seconds", "Detects lateness but fails to cross the strict boundary."),
            ("LATE; 101 seconds", "Counts the wait and transfer twice."),
        ),
        (
            "Travel is 500 seconds and waiting plus transfer is 100.",
            "The total equals the 600-second deadline and is not strictly before it.",
            "Saving one second produces arrival at 599 seconds.",
        ),
        "A strict deadline reverses classification at the same boundary.",
    ),
    _task(
        "flow_conservation",
        0,
        "A tank supplies 120 litres. A leak removes 12 litres before a splitter. "
        "The splitter sends one third of the remainder to A and the rest to B. "
        "B then uses 18 litres. How many litres remain at B?",
        "54",
        (
            ("62", "Ignores the upstream leak for B: 120*2/3-18=62."),
            ("72", "Omits B's use."),
            ("18", "Uses A's one-third share for B."),
        ),
        (
            "The splitter receives 120-12=108 litres.",
            "B receives 108*2/3=72 litres.",
            "B retains 72-18=54 litres.",
        ),
        "Loss precedes the split and affects both shares.",
    ),
    _task(
        "flow_conservation",
        1,
        "A tank supplies 120 litres to a splitter. One third goes to A and the rest "
        "to B. A then leaks 12 litres; B uses 18 litres. No fluid moves between branches. "
        "How many litres remain at B?",
        "62",
        (
            ("54", "Moves A's leak upstream of the splitter."),
            ("50", "Subtracts A's leak from B."),
            ("80", "Omits B's use."),
        ),
        (
            "B's share is 120*2/3=80 litres.",
            "The 12-litre leak affects A only.",
            "B retains 80-18=62 litres.",
        ),
        "Loss on one downstream branch does not affect the other.",
    ),
    _task(
        "weighted_allocation",
        0,
        "Divide 100 credits. First pay a 10-credit fixed overhead. Divide the remainder "
        "between A and B in ratio 2:1. A then pays a 7-credit processing fee from its "
        "share. How many credits does A keep?",
        "53",
        (
            ("60", "Omits A's processing fee."),
            ("43", "Charges the overhead to A twice."),
            ("23", "Assigns B's one-third share to A."),
        ),
        ("The distributable pool is 100-10=90.", "A receives two thirds: 60.", "A keeps 60-7=53."),
        "Overhead reduces the pool before weighting.",
    ),
    _task(
        "weighted_allocation",
        1,
        "Divide 100 credits after a 10-credit fixed overhead. Tentative A:B shares "
        "have ratio 2:1. Cap A's gross share at 50 and transfer its excess to B. "
        "A then pays a 7-credit fee. How many credits does A keep?",
        "43",
        (
            ("53", "Ignores A's cap."),
            ("50", "Omits A's fee."),
            ("40", "Reports B's gross share after redistribution instead of A's net share."),
        ),
        (
            "The 90-credit pool gives A a tentative share of 60.",
            "The cap transfers 10 to B and leaves A with 50 gross.",
            "A keeps 50-7=43.",
        ),
        "A binding gross cap interrupts proportional allocation.",
    ),
    _task(
        "rounding_policy",
        0,
        "Buy 3 items priced at 1.25 credits each. Apply a 10% discount to each item, "
        "round each discounted price to two decimals using half-up rounding, then add "
        "the prices and a 0.20-credit fee. What is the total?",
        "3.59",
        (
            ("3.58", "Rounds only the aggregate."),
            ("3.39", "Omits the fee."),
            ("3.56", "Rounds halfway values down."),
        ),
        (
            "Each discounted price is 1.25*0.9=1.125.",
            "Each rounds half-up to 1.13, giving 3.39 for three.",
            "Adding the fee gives 3.59.",
        ),
        "Rounding each item accumulates the rounding increments.",
    ),
    _task(
        "rounding_policy",
        1,
        "Buy 3 items priced at 1.25 credits each. Apply a 10% discount to their total, "
        "then round that discounted aggregate to two decimals using half-up rounding. "
        "Add a 0.20-credit fee. What is the total?",
        "3.58",
        (
            ("3.59", "Rounds discounted items separately."),
            ("3.38", "Omits the fee."),
            ("3.57", "Rounds the aggregate halfway value down."),
        ),
        (
            "Undiscounted aggregate is 3*1.25=3.75.",
            "Discounted aggregate 3.375 rounds half-up to 3.38.",
            "Adding the fee gives 3.58.",
        ),
        "One aggregate rounding differs from per-item rounding.",
    ),
    _task(
        "constrained_route_cost",
        0,
        "Direct costs 15 and takes 20 minutes. Via costs 4+6 plus a toll of 2 and takes "
        "28 minutes. Express costs 10 plus a fee of 4 and takes 22 minutes. Routes taking "
        "more than 25 minutes are ineligible. Choose the cheapest eligible route.",
        "Express",
        (
            ("Direct", "Selects the fastest instead of the cheapest eligible route."),
            ("Via", "Ignores eligibility."),
            ("None", "Rejects eligible routes."),
        ),
        (
            "Via costs 12 but its 28 minutes exceed the limit.",
            "Express costs 14 and Direct costs 15; both are eligible.",
            "Express is the cheapest eligible route.",
        ),
        "Eligibility removes the cheapest route.",
    ),
    _task(
        "constrained_route_cost",
        1,
        "Direct costs 15 and takes 20 minutes. Via costs 4+6 plus a toll of 6 and takes "
        "24 minutes. Express costs 10 plus a fee of 4 and takes 22 minutes. Routes taking "
        "more than 25 minutes are ineligible. Choose the cheapest eligible route.",
        "Express",
        (
            ("Via", "Ignores the toll that reverses the base-price ranking."),
            ("Direct", "Selects the fastest instead of cheapest."),
            ("None", "Rejects routes that meet the time limit."),
        ),
        (
            "All three routes meet the 25-minute limit.",
            "Their total costs are Direct 15, Via 16, Express 14.",
            "Express minimizes total cost even though Via has a lower base cost.",
        ),
        "A fee changes the ranking when every route is eligible.",
    ),
    _task(
        "time_window_scheduling",
        0,
        "A machine is free at time 3. Setup takes 2 units and processing takes 4 units. "
        "Cleanup takes 1 unit. The entire operation must finish at or before time 10. "
        "If it starts immediately, is it feasible and when is cleanup complete?",
        "Feasible; time 10",
        (
            ("Infeasible; time 10", "Makes an inclusive endpoint exclusive."),
            ("Feasible; time 9", "Omits cleanup."),
            ("Feasible; time 8", "Omits setup."),
        ),
        (
            "Setup ends at 3+2=5.",
            "Processing ends at 5+4=9; cleanup ends at 10.",
            "The inclusive finish limit accepts time 10.",
        ),
        "An inclusive endpoint permits exact fit.",
    ),
    _task(
        "time_window_scheduling",
        1,
        "A machine is free at time 3. Setup takes 2 units, processing 4, and cleanup 1. "
        "The entire operation must finish strictly before time 10. Starting times must "
        "be integers. What is the latest start that meets the deadline, and is it available?",
        "Time 2; unavailable",
        (
            ("Time 3; available", "Allows completion exactly at the strict endpoint."),
            ("Time 2; available", "Ignores machine availability."),
            ("Time 1; unavailable", "Adds an unnecessary second unit of slack."),
        ),
        (
            "The operation takes 2+4+1=7 units.",
            "An integer finish below 10 is at most 9, so start is at most 2.",
            "The machine is unavailable before 3, so that start is unavailable.",
        ),
        "An exclusive endpoint conflicts with resource availability.",
    ),
    _task(
        "task_precedence",
        0,
        "One machine runs jobs without gaps from time 0. A takes 2 units, B takes 3, "
        "C takes 1. A precedes C. B must finish no later than time 4. "
        "Which complete order is feasible?",
        "B-A-C",
        (
            ("A-B-C", "B finishes at 5, after its deadline."),
            ("A-C-B", "B finishes at 6, after its deadline."),
            ("C-B-A", "C precedes A, violating precedence."),
        ),
        (
            "B cannot follow A because it would finish at 5 or later.",
            "Starting B first completes it at 3.",
            "A must then precede C, giving B-A-C.",
        ),
        "One machine and a deadline force B first.",
    ),
    _task(
        "task_precedence",
        1,
        "Two machines are available at time 0. A takes 2 units, B takes 3, C takes 1. "
        "C may start at or after A finishes, including at the same instant. "
        "B is independent and must finish by time 4. "
        "Jobs cannot be split. What is the earliest finish time for all jobs?",
        "3",
        (
            ("6", "Serializes all jobs on one machine."),
            ("4", "Waits unnecessarily before starting C."),
            ("2", "Ignores B's duration."),
        ),
        (
            "Run A and B simultaneously from time 0.",
            "At time 2 run C on A's machine until time 3.",
            "B also ends at 3, which is a lower bound from B alone.",
        ),
        "A second machine removes a serial ordering requirement.",
    ),
    _task(
        "resource_packing",
        0,
        "Select whole items for a bag of capacity 10. Item P weighs 6 with value 12; "
        "Q weighs 5 with value 11; R weighs 4 with value 7. Each item is available once. "
        "Which feasible selection has greatest total value?",
        "P and R",
        (
            ("P and Q", "Exceeds capacity with weight 11."),
            ("Q and R", "Value 18 is below the feasible value 19."),
            ("P only", "Leaves room for valuable R."),
        ),
        (
            "P+Q weighs 11 and is infeasible.",
            "P+R weighs 10 and is worth 19; Q+R is worth 18.",
            "No single item exceeds 19; P+R is optimal.",
        ),
        "The highest unconstrained pair is infeasible.",
    ),
    _task(
        "resource_packing",
        1,
        "Select whole items for a bag of capacity 10. P weighs 6 with value 12; "
        "Q weighs 5 with value 11; R weighs 4 with value 1. P and R are incompatible. "
        "Which feasible selection has greatest value, breaking value ties by lower weight?",
        "P only",
        (
            ("Q and R", "Ignores the lower-weight tie breaker at value 12."),
            ("P and R", "Violates incompatibility."),
            ("P and Q", "Exceeds capacity."),
        ),
        (
            "P+Q exceeds capacity and P+R is incompatible.",
            "P alone and Q+R both have value 12.",
            "P weighs 6 versus 9 for Q+R, so the tie breaker selects P.",
        ),
        "A tie breaker and incompatibility make unused capacity optimal.",
    ),
    _task(
        "matching_constraints",
        0,
        "Assign Ana, Bo and Cy to distinct jobs X,Y,Z. Ana can do X or Y; Bo can do "
        "Y or Z; Cy can do X only. Everyone gets one job. Which assignment is feasible?",
        "Ana-Y, Bo-Z, Cy-X",
        (
            ("Ana-X, Bo-Y, Cy-Z", "Cy cannot do Z."),
            ("Ana-Y, Bo-X, Cy-Z", "Bo cannot do X and Cy cannot do Z."),
            ("Ana-Z, Bo-Y, Cy-X", "Ana cannot do Z."),
        ),
        (
            "Cy must take X.",
            "Ana cannot then take X, so takes Y.",
            "The remaining job Z goes to Bo.",
        ),
        "A constrained worker determines the unique matching.",
    ),
    _task(
        "matching_constraints",
        1,
        "Assign Ana, Bo and Cy to distinct jobs X,Y,Z. Ana can do X at cost 1 or Y "
        "at cost 4; Bo can do Y at cost 1 or Z at cost 2; Cy can do X at cost 3 only. "
        "All workers must get a job. What is the minimum feasible total cost?",
        "9",
        (
            ("5", "Uses Ana-X and Cy-X simultaneously."),
            ("6", "Omits Cy's cost from the feasible assignment."),
            ("No feasible assignment", "Mistakes a failed greedy choice for infeasibility."),
        ),
        (
            "Cy must use X, preventing Ana's cheapest choice.",
            "Ana takes Y for 4 and Bo takes Z for 2.",
            "Adding Cy's 3 gives the unique feasible total 9.",
        ),
        "A locally cheap assignment blocks a required worker.",
    ),
    _task(
        "queue_discipline",
        0,
        "A server becomes free at time 5. Waiting jobs P and Q arrived at times 1 and 2. "
        "Their original priorities are P=2 and Q=1. At time 4 Q's priority changes to 3. "
        "Choose larger current priority first, then earlier arrival. Which job runs next?",
        "Q",
        (
            ("P", "Uses arrival order or original priority instead of current priority."),
            ("Both simultaneously", "Ignores the single server."),
            ("Neither", "Ignores eligible waiting jobs."),
        ),
        (
            "Both jobs have arrived before time 5.",
            "Q's current priority is 3 versus P's 2.",
            "The server selects Q; arrival time does not override unequal priorities.",
        ),
        "A waiting job's updated priority changes the order.",
    ),
    _task(
        "queue_discipline",
        1,
        "A server becomes free at time 5. P arrived at time 1 with priority 2; Q arrived "
        "at time 2 with priority 1. At time 4 Q's priority becomes 2. Higher current "
        "priority wins; ties use original arrival time, which updates do not reset. "
        "Which job runs next?",
        "P",
        (
            ("Q", "Treats a priority update as higher priority or tie preference."),
            ("Both simultaneously", "Ignores the single server."),
            ("Neither", "Mistakes a resolvable tie for no eligible job."),
        ),
        (
            "Both jobs are waiting and now have priority 2.",
            "The tie uses original arrivals 1 and 2.",
            "P arrived earlier and runs next.",
        ),
        "A priority update creates a tie resolved stably.",
    ),
    _task(
        "access_conjunction",
        0,
        "Entry requires a valid badge, a current reservation, and completed training. "
        "An expired reservation is not current. Lee has a valid badge and completed "
        "training, but the reservation expired yesterday. Which decision follows?",
        "Deny entry",
        (
            ("Allow entry", "Treats two satisfied conditions as sufficient."),
            ("Allow if badge is shown twice", "Invents an exception not in the rule."),
            ("Cannot decide", "Ignores that every relevant fact is supplied."),
        ),
        (
            "The badge and training conditions hold.",
            "The expired reservation condition fails.",
            "All three are necessary, so entry is denied.",
        ),
        "One missing conjunct blocks access.",
    ),
    _task(
        "access_conjunction",
        1,
        "Entry normally requires a valid badge, a current reservation, and completed "
        "training. A verified emergency pass waives only the reservation requirement. "
        "Lee has a valid badge, completed training, an expired reservation, and a verified "
        "emergency pass. Which decision follows?",
        "Allow entry",
        (
            ("Deny because reservation expired", "Ignores the explicit waiver."),
            ("Deny because training is missing", "Contradicts the stated training fact."),
            ("Cannot decide", "Ignores the exception's specified scope."),
        ),
        (
            "The verified pass waives the reservation condition only.",
            "Badge and training still apply and both hold.",
            "All remaining requirements hold, so entry is allowed.",
        ),
        "A narrow explicit waiver removes the otherwise failing conjunct.",
    ),
    _task(
        "policy_precedence",
        0,
        "General policy permits export of reports signed by a manager. A more specific "
        "rule prohibits export of reports marked SECRET, even when signed. Report R is "
        "SECRET and manager-signed; a colleague says it is useful to export. What follows?",
        "Export prohibited",
        (
            ("Export permitted", "Lets the general permission override the specific prohibition."),
            ("Colleague decides", "Gives authority to an irrelevant opinion."),
            ("Rules give no answer", "Ignores the stated priority."),
        ),
        (
            "The signature meets the general permission.",
            "The SECRET rule applies specifically and has explicit precedence.",
            "Export is prohibited regardless of usefulness.",
        ),
        "A specific prohibition overrides permission.",
    ),
    _task(
        "policy_precedence",
        1,
        "General policy permits export of manager-signed reports. A specific rule "
        "prohibits export of reports marked SECRET. Report R is PUBLIC and manager-signed. "
        "A colleague calls the report sensitive, but only the recorded classification "
        "determines which rule applies. What follows?",
        "Export permitted",
        (
            ("Export prohibited", "Applies a SECRET-only rule to a PUBLIC report."),
            ("Colleague decides", "Substitutes an opinion for the recorded classification."),
            ("Rules give no answer", "Ignores the applicable general permission."),
        ),
        (
            "The recorded PUBLIC classification does not trigger the SECRET rule.",
            "The manager signature meets the general rule.",
            "Export is permitted; the colleague's label does not change applicability.",
        ),
        "The override is inapplicable, leaving the general permission operative.",
    ),
    _task(
        "quorum_with_abstentions",
        0,
        "A meeting has 8 members present, satisfying a quorum of 7. Votes are 4 yes, "
        "2 no, and 2 abstentions. Passing requires strictly more than half of all present "
        "members voting yes. Do quorum and passage hold?",
        "Quorum yes; passage no",
        (
            ("Quorum yes; passage yes", "Excludes abstentions or treats half as sufficient."),
            ("Quorum no; passage no", "Counts only nonabstaining voters for quorum."),
            ("Quorum no; passage yes", "Uses inconsistent reduced denominators."),
        ),
        (
            "Eight present meets the quorum of seven.",
            "Half of eight is four.",
            "Four yes is not strictly more than four, so passage fails.",
        ),
        "Abstentions remain in the passage denominator.",
    ),
    _task(
        "quorum_with_abstentions",
        1,
        "A meeting has 8 members present, satisfying a quorum of 7. Votes are 4 yes, "
        "2 no, and 2 abstentions. Passing requires strictly more than half of yes-plus-no "
        "votes to be yes; abstentions do not count for passage. Do quorum and passage hold?",
        "Quorum yes; passage yes",
        (
            ("Quorum yes; passage no", "Keeps abstentions in the passage denominator."),
            ("Quorum no; passage yes", "Excludes abstentions from quorum too."),
            ("Quorum no; passage no", "Uses the wrong denominator for both tests."),
        ),
        (
            "All eight present count toward quorum, so quorum holds.",
            "The passage denominator is 4+2=6; half is three.",
            "Four yes exceeds three, so passage holds.",
        ),
        "Abstentions leave only the passage denominator, not the quorum denominator.",
    ),
    _task(
        "quantified_rules",
        0,
        "A batch may ship if and only if every fragile item is padded and every item over 5 kg "
        "is strapped. A is fragile, 4 kg, padded, not strapped. B is nonfragile, 7 kg, "
        "not padded, strapped. C is fragile, 6 kg, padded, strapped. May it ship?",
        "Yes",
        (
            ("No: A lacks a strap", "Applies the heavy-item rule to a light item."),
            ("No: B lacks padding", "Applies the fragile-item rule to a nonfragile item."),
            ("Cannot decide", "Ignores complete facts for both quantified requirements."),
        ),
        (
            "Fragile items A and C are both padded.",
            "Heavy items B and C are both strapped.",
            "Every relevant item satisfies its applicable requirements, so the batch may ship.",
        ),
        "Universal requirements are evaluated only over their stated subsets.",
    ),
    _task(
        "quantified_rules",
        1,
        "A batch may ship if and only if every fragile item is padded and every item over 5 kg "
        "is strapped. A is fragile, 4 kg, padded, not strapped. B is nonfragile, 7 kg, "
        "strapped. C is fragile, 6 kg, padded, not strapped. May it ship?",
        "No: C lacks a strap",
        (
            ("Yes", "Treats most compliant items as enough for a universal condition."),
            ("No: A lacks a strap", "Applies the heavy-item rule to a light item."),
            ("Cannot decide", "Ignores the decisive stated counterexample."),
        ),
        (
            "All fragile items are padded.",
            "C is over 5 kg but is not strapped.",
            "One counterexample defeats the heavy-item requirement and blocks shipping.",
        ),
        "A single counterexample defeats a universal claim.",
    ),
    _task(
        "conditional_statements",
        0,
        "Every certified unit passed inspection. Every unit that passed inspection was "
        "tested. Unit K was tested but did not pass inspection. Which conclusion is forced?",
        "K is not certified",
        (
            ("K is certified", "Treats testing as sufficient for certification."),
            ("K passed inspection", "Reverses the inspection-implies-testing rule."),
            ("Certification cannot be determined", "Misses the contrapositive of the first rule."),
        ),
        (
            "Certification requires passing inspection.",
            "K did not pass inspection.",
            "K cannot be certified; being tested supplies no converse implication.",
        ),
        "Failure of a necessary condition rules out certification.",
    ),
    _task(
        "conditional_statements",
        1,
        "Every certified unit passed inspection. Every unit that passed inspection was "
        "tested. K passed inspection and was tested. No other rule is given. "
        "Is K necessarily certified?",
        "Cannot determine whether K is certified",
        (
            (
                "Yes, necessarily certified",
                "Affirms the converse of certification implies inspection.",
            ),
            ("No, necessarily not certified", "Confuses lack of a sufficient rule with negation."),
            ("K was not tested", "Contradicts both a fact and a rule."),
        ),
        (
            "The rules imply inspection from certification, not certification from inspection.",
            "A certified K and an uncertified K both satisfy the supplied facts.",
            "Certification is therefore undetermined.",
        ),
        "Satisfying necessary conditions does not establish the antecedent.",
    ),
    _task(
        "exception_scope",
        0,
        "All crates require inspection before loading. A signed waiver exempts only "
        "crate A. Crates A and B are both red and neither has been inspected. Color "
        "grants no exemption. Which crates may be loaded?",
        "A only",
        (
            ("A and B", "Extends A's individual waiver to another red crate."),
            ("B only", "Applies the waiver to the wrong crate."),
            ("Neither", "Ignores the valid waiver."),
        ),
        (
            "The general rule blocks uninspected crates.",
            "Only A has an applicable exemption.",
            "A may load and B may not.",
        ),
        "An exception is attached to one named object.",
    ),
    _task(
        "exception_scope",
        1,
        "All crates require inspection before loading. A signed waiver exempts every "
        "red crate. A and B are red, C is blue, and none has been inspected. "
        "Which crates may be loaded?",
        "A and B only",
        (
            ("A only", "Treats a category-wide waiver as an individual waiver."),
            ("A, B and C", "Extends the red-category exception to blue."),
            ("None", "Ignores the category waiver."),
        ),
        (
            "Both A and B belong to the exempt red category.",
            "C is blue and still requires inspection.",
            "Only A and B may load.",
        ),
        "The exception applies to a category but not outside it.",
    ),
    _task(
        "state_transitions",
        0,
        "A state has x=4 and alarm=OFF. One tick first tests x>=5; if true it sets alarm "
        "ON. It then adds 2 to x and caps x at 5. Alarm otherwise stays unchanged. "
        "What is the state after one tick?",
        "x=5, alarm=OFF",
        (
            ("x=5, alarm=ON", "Tests the guard after updating x."),
            ("x=6, alarm=OFF", "Omits the cap."),
            ("x=4, alarm=OFF", "Omits the update."),
        ),
        (
            "The guard sees the initial x=4 and is false.",
            "Adding two produces 6.",
            "The cap gives x=5; the false guard left the alarm OFF.",
        ),
        "A guard observes the pre-update state.",
    ),
    _task(
        "state_transitions",
        1,
        "A state has x=4 and alarm=OFF. One tick first adds 2 to x and caps x at 5. "
        "It then tests x>=5; if true it sets alarm ON. What is the state after one tick?",
        "x=5, alarm=ON",
        (
            ("x=5, alarm=OFF", "Tests the guard before the update."),
            ("x=6, alarm=ON", "Omits the cap."),
            ("x=4, alarm=ON", "Sets the alarm without retaining the update."),
        ),
        (
            "Adding two to four gives six.",
            "The cap sets x to five.",
            "The guard now sees five, so alarm becomes ON.",
        ),
        "The guard observes the capped post-update state.",
    ),
    _task(
        "transaction_order",
        0,
        "Start with balance 10. Transaction T1 adds 5. T2 doubles the current balance. "
        "T3 subtracts 4. Apply T1, T2, T3 in that order, with no rollback. "
        "What is the final balance?",
        "26",
        (
            ("21", "Doubles before adding five."),
            ("22", "Subtracts four before doubling."),
            ("30", "Omits the final subtraction."),
        ),
        ("T1 changes 10 to 15.", "T2 changes 15 to 30.", "T3 changes 30 to 26."),
        "Noncommuting arithmetic updates must follow their specified order.",
    ),
    _task(
        "transaction_order",
        1,
        "Start with balance 10. T1 adds 5, T2 doubles the balance, and T3 subtracts 4. "
        "Apply T1,T2,T3, then roll back only T3, restoring the balance immediately "
        "before T3. T1 and T2 remain committed. What is the balance?",
        "30",
        (
            ("26", "Ignores rollback."),
            ("10", "Rolls back all transactions."),
            ("15", "Rolls back T2 as well as T3."),
        ),
        (
            "T1 and T2 produce (10+5)*2=30.",
            "T3 temporarily reduces this to 26.",
            "Rolling back only T3 restores 30 while preserving T1 and T2.",
        ),
        "Rollback scope is limited to the latest operation.",
    ),
    _task(
        "table_joins",
        0,
        "Table L has rows (k=1,v=5), (k=1,v=7), (k=2,v=9). Table R has two rows "
        "with k=1 and none with k=2. An inner join on k emits one row for every "
        "matching pair. Sum L.v across joined rows, counting duplicates. What is the sum?",
        "24",
        (
            ("12", "Counts each matched L row only once."),
            ("21", "Sums all original L rows without applying the join."),
            ("33", "Retains unmatched k=2 as though this were a left join."),
        ),
        (
            "Each of the two k=1 rows matches two R rows.",
            "Their contribution is 2*5+2*7=24.",
            "The unmatched k=2 row contributes nothing to an inner join.",
        ),
        "Duplicate keys multiply rows in a join.",
    ),
    _task(
        "table_joins",
        1,
        "Table L has rows (k=1,v=5), (k=1,v=7), (k=2,v=9). Table R has two rows "
        "with k=1 and none with k=2. A left join emits every matching pair and one "
        "row for each unmatched L row. Sum L.v across joined rows, counting duplicates.",
        "33",
        (
            ("24", "Drops the unmatched row as if this were an inner join."),
            ("21", "Ignores multiplication by duplicate matches."),
            ("42", "Duplicates the unmatched row too."),
        ),
        (
            "Matching k=1 rows contribute 2*5+2*7=24.",
            "The unmatched k=2 row is retained once and contributes 9.",
            "The left-join sum is 24+9=33.",
        ),
        "The join retains unmatched left rows exactly once.",
    ),
    _task(
        "aggregation_filters",
        0,
        "Rows are (A,4),(A,8),(B,7),(B,9),(C,12). First discard rows with value below 6, "
        "then sum by letter. Keep groups whose sum is at least 12. "
        "What is the sum of the retained group totals?",
        "28",
        (
            ("40", "Groups before applying the row filter."),
            ("36", "Keeps filtered group A even though its sum is below 12."),
            ("16", "Uses a strict group threshold and discards C."),
        ),
        (
            "Discarding value 4 leaves group totals A=8, B=16, C=12.",
            "The inclusive group threshold retains B and C.",
            "Their total is 16+12=28.",
        ),
        "A row filter changes aggregates before the group test.",
    ),
    _task(
        "aggregation_filters",
        1,
        "Rows are (A,4),(A,8),(B,7),(B,9),(C,12). Sum all rows by letter, without a "
        "row filter. Keep groups whose sum is at least 12 and whose row count is at "
        "least 2. What is the sum of the retained group totals?",
        "28",
        (
            ("40", "Ignores the group-size condition and retains C."),
            ("16", "Applies a nonexistent row filter before grouping."),
            ("12", "Ignores the qualifying B group."),
        ),
        (
            "The totals are A=12 with 2 rows, B=16 with 2, and C=12 with 1.",
            "Both group predicates hold for A and B but not C.",
            "Retained totals sum to 12+16=28.",
        ),
        "A group-level count condition differs from filtering individual rows.",
    ),
    _task(
        "set_operations",
        0,
        "Universe U={1,2,3,4,5,6}; A={1,2,3}; B={3,4}; C={2,4,6}. "
        "Take A union B, complement that result relative to U, then intersect with C. "
        "What set remains?",
        "{6}",
        (
            ("{5,6}", "Omits the final intersection."),
            ("{4,6}", "Complements only A rather than A union B."),
            ("{2,4}", "Intersects the uncomplemented union with C."),
        ),
        (
            "A union B is {1,2,3,4}.",
            "Its complement in U is {5,6}.",
            "Intersecting {5,6} with C leaves {6}.",
        ),
        "A stated finite universe fixes the complement.",
    ),
    _task(
        "set_operations",
        1,
        "A={1,2,3}; B={3,4}; C={2,4,6}. Evaluate A union (B intersect C), "
        "then remove the element 2. Parentheses determine the order. What set remains?",
        "{1,3,4}",
        (
            ("{4}", "Computes (A union B) intersect C before removing 2."),
            ("{1,2,3,4}", "Omits removal of 2."),
            ("{1,3}", "Omits the contribution of B intersect C."),
        ),
        ("B intersect C is {4}.", "Union with A gives {1,2,3,4}.", "Removing 2 leaves {1,3,4}."),
        "Intersection is nested inside union before a final removal.",
    ),
    _task(
        "interval_composition",
        0,
        "Intervals include their lower bound and exclude their upper bound. "
        "Intersect [1,7) with [3,9), then intersect the result with [7,10). "
        "What remains?",
        "Empty set",
        (
            ("{7}", "Treats excluded upper bound 7 as included."),
            ("[3,7)", "Omits the final intersection."),
            ("[7,9)", "Ignores the first interval's upper bound."),
        ),
        (
            "The first intersection is [3,7).",
            "It excludes 7 and all larger points.",
            "The last interval contains only points at least 7, so the intersection is empty.",
        ),
        "Touching half-open endpoints do not overlap.",
    ),
    _task(
        "interval_composition",
        1,
        "Intervals include their lower bound and exclude their upper bound. "
        "Intersect [1,8) with [3,9), then clip the result to [7,10). "
        "Report the remaining interval and its length.",
        "[7,8); length 1",
        (
            ("[3,8); length 5", "Omits clipping."),
            ("[7,9); length 2", "Ignores the first upper bound."),
            ("Empty set; length 0", "Mistakes a positive overlap for touching endpoints."),
        ),
        (
            "The first intersection is [3,8).",
            "Clipping to [7,10) leaves [7,8).",
            "The interval's length is 8-7=1.",
        ),
        "A changed upper bound creates nonempty overlap after clipping.",
    ),
)


def authored_tasks() -> tuple[AuthoredTask, ...]:
    """Return the immutable authored roster without implying independent review."""
    return _TASKS


def _text(value: str, limit: int) -> None:
    """Bound serialized text while leaving token counting to a future capture review."""
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError("invalid task text")
    if len(value.encode("utf-8")) > limit:
        raise ValueError("task text exceeds byte limit")


def validate_tasks(tasks: tuple[AuthoredTask, ...]) -> None:
    """Reject structural defects; never claim proof of answers or independence."""
    if type(tasks) is not tuple or len(tasks) != 48:
        raise ValueError("expected 48 authored tasks")
    identities = set()
    counts: Counter[str] = Counter()
    for task in tasks:
        if not isinstance(task, AuthoredTask) or task.mechanism_id not in MECHANISM_IDS:
            raise ValueError("unknown task mechanism")
        if type(task.variant) is not int or task.variant not in (0, 1):
            raise ValueError("invalid task variant")
        expected_id = f"followup-{task.mechanism_id}-{task.variant}"
        if task.task_id != expected_id or task.task_id in identities:
            raise ValueError("invalid or duplicate task identity")
        identities.add(task.task_id)
        counts[task.mechanism_id] += 1
        _text(task.prompt, 6000)
        _text(task.correct_text, 256)
        _text(task.variant_rationale, 1024)
        if type(task.distractors) is not tuple or len(task.distractors) != 3:
            raise ValueError("expected three distractors")
        options = [task.correct_text.strip()]
        for alternative in task.distractors:
            if not isinstance(alternative, Distractor):
                raise ValueError("invalid distractor")
            _text(alternative.text, 256)
            _text(alternative.error, 1024)
            options.append(alternative.text.strip())
        if len(set(options)) != 4:
            raise ValueError("options must be distinct")
        if type(task.derivation) is not tuple or not 3 <= len(task.derivation) <= 12:
            raise ValueError("expected three to twelve derivation steps")
        for step in task.derivation:
            _text(step, 1024)
    if counts != dict.fromkeys(MECHANISM_IDS, 2):
        raise ValueError("expected both variants of every mechanism")
