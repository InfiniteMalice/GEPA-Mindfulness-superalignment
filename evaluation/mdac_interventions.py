"""Opt-in matched MDAC replay over public causal pairs and authenticated outcomes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import log, sqrt
from statistics import mean
from typing import Any
from uuid import uuid4

from evaluation.causal_diagnostics import evaluate_causal_suite, protocol_digest
from evaluation.causal_records import (
    CausalCapture,
    CausalPair,
    PairAdjudication,
    canonical_json,
    content_digest,
)
from evaluation.v5_records import SystemIdentity
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from synthetic_data.causal_interventions import render_causal_variant

Messages = tuple[dict[str, str], ...]
Interaction = tuple[str, str, str]


def _text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


@dataclass(frozen=True)
class ReplayIdentity:
    """Host-resolved immutable references shared by every execution in a matched block."""

    task_id: str
    initial_state: str
    checkpoint: str
    configuration: str
    system: SystemIdentity

    def __post_init__(self) -> None:
        if not all(
            _text(x)
            for x in (self.task_id, self.initial_state, self.checkpoint, self.configuration)
        ):
            raise ValueError("replay identity requires nonblank identifiers")
        if type(self.system) is not SystemIdentity:
            raise ValueError("expected a SystemIdentity")


@dataclass(frozen=True)
class ReplayResult:
    """Receipt from the host that actually reset state and executed one arm."""

    identity: ReplayIdentity
    prompt_digest: str
    actions: tuple[str, ...]
    evidence_ref: EvidenceReference
    procedure_id: str | None = None

    @classmethod
    def from_actions(
        cls,
        identity: ReplayIdentity,
        turns: Messages,
        actions: tuple[str, ...],
        *,
        procedure_id: str | None = None,
        execution_id: str | None = None,
    ) -> ReplayResult:
        """Content-address the public receipt for an in-process runner."""
        prompt = content_digest(list(turns))
        if execution_id is None:
            execution_id = uuid4().hex
        elif not _text(execution_id):
            raise ValueError("execution_id must name an actual host execution")
        digest = content_digest([execution_id, asdict(identity), prompt, actions, procedure_id])
        ref = EvidenceReference(f"capture:{digest}", EvidenceSourceKind.OBSERVABLE_ACTION)
        return cls(identity, prompt, actions, ref, procedure_id)


@dataclass(frozen=True)
class Treatment:
    """One declared repair, positive control or negative control with its original prompt."""

    name: str
    role: str
    pair: CausalPair
    procedure_id: str | None = None
    selection_policy: str | None = None


@dataclass(frozen=True)
class ReplayBlock:
    """Predeclared, common-start trial. The first pair defines the shared control."""

    identity: ReplayIdentity
    evaluator: TrustedEvaluatorContract
    treatments: tuple[Treatment, ...]


Execute = Callable[[ReplayIdentity, Messages], ReplayResult]
Adjudicate = Callable[[CausalPair, tuple[CausalCapture, ...], str], PairAdjudication | None]
Authenticate = Callable[[PairAdjudication], bool]


def _validate_block(
    block: ReplayBlock, interactions: tuple[Interaction, ...], procedures: Mapping[str, Execute]
) -> None:
    if type(block) is not ReplayBlock or type(block.identity) is not ReplayIdentity:
        raise ValueError("expected a matched replay block")
    if type(block.evaluator) is not TrustedEvaluatorContract:
        raise ValueError("expected a trusted evaluator contract")
    if type(block.treatments) is not tuple or not block.treatments:
        raise ValueError("a block requires a nonempty treatment roster")
    names = {}
    baseline = None
    for treatment in block.treatments:
        if type(treatment) is not Treatment or not _text(treatment.name):
            raise ValueError("expected a named treatment")
        if treatment.name in names or treatment.role not in ("repair", "positive", "negative"):
            raise ValueError("duplicate treatment or invalid control role")
        if type(treatment.pair) is not CausalPair:
            raise ValueError("expected a causal pair")
        p = CausalPair.from_dict(treatment.pair.to_dict())
        if set(p.before.expected_actions) != set(p.after.expected_actions):
            raise ValueError("MDAC repair arms must preserve the common outcome oracle")
        if treatment.selection_policy is not None and not _text(treatment.selection_policy):
            raise ValueError("treatment selection policy requires a nonempty identity")
        plan = treatment.procedure_id
        declared = [c for c in p.changes if c.factor == "activation_plan"]
        if plan is not None:
            if (
                type(plan) is not str
                or len(plan) != 64
                or any(c not in "0123456789abcdef" for c in plan)
                or plan not in procedures
                or not callable(procedures[plan])
            ):
                raise ValueError("scoped activation intervention requires its registered plan")
            if (
                len(declared) != 1
                or declared[0].before != "null"
                or declared[0].after != canonical_json(plan)
            ):
                raise ValueError("activation procedure must match the predeclared factor")
        elif declared:
            raise ValueError("activation intervention requires a registered procedure")
        if p.seed_policy != "shared" or p.before.system != block.identity.system:
            raise ValueError("replays require a matching shared system identity and seed")
        # The destination case may change. The common starting task and oracle must agree.
        start = (
            p.before.case,
            p.before.robustness,
            p.before.turns,
            p.before.expected_actions,
            dict(p.before.factors),
            p.training_eligibility,
        )
        if baseline is not None and baseline != start:
            raise ValueError("control initial conditions differ between treatments")
        baseline = start
        names[treatment.name] = treatment
    if {t.role for t in block.treatments} != {"repair", "positive", "negative"}:
        raise ValueError("declare a repair, positive control and negative control")
    for a, b, joint in interactions:
        if {a, b, joint} - names.keys() or len({a, b, joint}) != 3:
            raise ValueError("factorial interaction references missing or duplicate arms")
        left, right, combined = (set(names[k].pair.changes) for k in (a, b, joint))
        if {v.factor for v in left} & {v.factor for v in right} or left | right != combined:
            raise ValueError("factorial joint must apply exactly the two disjoint interventions")


def _execute(
    identity: ReplayIdentity, treatment: Treatment, arm: str, execute: Execute
) -> ReplayResult:
    pair = treatment.pair
    variant = getattr(pair, arm)
    turns = render_causal_variant(pair, variant.variant_id, enabled=True)
    receipt = execute(identity, turns)
    if (
        type(receipt) is not ReplayResult
        or receipt.identity != identity
        or receipt.prompt_digest != variant.prompt_digest
        or receipt.procedure_id != (treatment.procedure_id if arm == "after" else None)
    ):
        raise ValueError(
            "returned execution identity, procedure or prompt digest does not match planned arm"
        )
    return receipt


def _capture(treatment: Treatment, arm: str, receipt: ReplayResult) -> CausalCapture:
    p = treatment.pair
    variant = getattr(p, arm)
    return CausalCapture(
        p.digest,
        variant.variant_id,
        receipt.prompt_digest,
        receipt.identity.system,
        "observed" if receipt.actions else "censored",
        receipt.actions,
        (receipt.evidence_ref,),
        "public replay captured" if receipt.actions else "no complete public action",
    )


def _observe(
    block: ReplayBlock,
    treatment: Treatment,
    control: ReplayResult,
    execute: Execute,
    adjudicate: Adjudicate,
    authenticate: Authenticate | None,
    study_digest: str,
) -> dict[str, Any]:
    pair = treatment.pair
    response = _execute(block.identity, treatment, "after", execute)
    captures = (_capture(treatment, "before", control), _capture(treatment, "after", response))
    protocol = content_digest([study_digest, asdict(block.identity), treatment.name, pair.digest])
    judgment = adjudicate(pair, captures, protocol_digest(protocol, (pair,), ()))
    if judgment is not None and judgment.evaluator != block.evaluator:
        raise ValueError("judgment uses a different evaluator identity")
    report = evaluate_causal_suite(
        (pair,),
        captures,
        () if judgment is None else (judgment,),
        protocol_id=protocol,
        opportunities=(),
        authenticate=authenticate,
        enabled=True,
    )
    row = report["pairs"][0]
    return dict(
        identity=asdict(block.identity),
        treatment=treatment.name,
        pair_id=pair.pair_id,
        pair_digest=pair.digest,
        role=treatment.role,
        source_record=row["source_record"],
        procedure_id=treatment.procedure_id,
        control=row["before_correct"],
        intervention=row["after_correct"],
        verification=row["verification_reason"],
        captures=[c.to_dict() for c in captures],
        adjudication=row["adjudication"],
    )


def _interval(values: list[tuple[str, int]], bound: int, independent: bool) -> list[float] | None:
    """95% Hoeffding interval for bounded matched differences clustered by task."""
    groups = Counter(task for task, _ in values)
    if not independent or len(groups) < 2:
        return None
    variance_bound = sum((size / len(values)) ** 2 for size in groups.values())
    radius = 2 * bound * sqrt(log(40) * variance_bound / 2)
    center = mean(v for _, v in values)
    return [max(-bound, center - radius), min(bound, center + radius)]


def _effect(rows: list[dict[str, Any]], independent: bool) -> dict[str, Any]:
    matched = [r for r in rows if r["control"] is not None and r["intervention"] is not None]
    differences = [
        (r["identity"]["task_id"], int(r["intervention"]) - int(r["control"])) for r in matched
    ]
    controls = sum(r["control"] for r in matched)
    passes = sum(r["intervention"] for r in matched)
    recoveries = sum(not r["control"] and r["intervention"] for r in matched)
    n = len(matched)
    return dict(
        matched_trials=n,
        control_passes=controls,
        intervention_passes=passes,
        control_pass_rate=controls / n if n else None,
        intervention_pass_rate=passes / n if n else None,
        recovered=recoveries,
        regressed=sum(r["control"] and not r["intervention"] for r in matched),
        recovery_given_control_failure=recoveries / (n - controls) if n > controls else None,
        drs=(passes - controls) / n if n else None,
        interval=_interval(differences, 1, independent),
    )


def _treatment_roster(block: ReplayBlock) -> list[tuple[Any, ...]]:
    """Bind pooled arms to either an exact pair or a registered per-task selection policy."""
    return [
        (
            t.name,
            t.role,
            t.procedure_id,
            t.selection_policy,
            (
                t.pair.digest
                if t.selection_policy is None
                else tuple(sorted(c.factor for c in t.pair.changes))
            ),
        )
        for t in block.treatments
    ]


def run_interventions(
    blocks: tuple[ReplayBlock, ...],
    *,
    execute: Execute,
    adjudicate: Adjudicate,
    authenticate: Authenticate | None,
    max_calls: int,
    alternatives: tuple[str, ...],
    interactions: tuple[Interaction, ...] = (),
    independent_tasks: bool = False,
    procedures: Mapping[str, Execute] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Execute whole matched blocks within budget and report authenticated paired differences."""
    if enabled is not True:
        raise ValueError("MDAC interventions require enabled=True")
    if type(max_calls) is not int or max_calls < 0 or type(independent_tasks) is not bool:
        raise ValueError("invalid execution budget or independence declaration")
    if type(blocks) is not tuple or not blocks:
        raise ValueError("declare a nonempty block roster")
    if type(alternatives) is not tuple or not all(_text(a) for a in alternatives):
        raise ValueError("alternatives must be a tuple of nonblank explanations")
    if (
        type(interactions) is not tuple
        or any(
            type(t) is not tuple or len(t) != 3 or not all(_text(k) for k in t)
            for t in interactions
        )
        or len({t[2] for t in interactions}) != len(interactions)
    ):
        raise ValueError("declare distinct factorial contrasts")
    if (
        not callable(execute)
        or not callable(adjudicate)
        or (authenticate is not None and not callable(authenticate))
    ):
        raise ValueError("invalid host callbacks")
    procedures = {} if procedures is None else dict(procedures)
    seen = set()
    roster = []
    for block in blocks:
        _validate_block(block, interactions, procedures)
        identity = asdict(block.identity)
        # A repeat label alone does not create a new sampled unit.
        identity["system"].pop("repeat_id")
        key = content_digest(identity)
        if key in seen:
            raise ValueError("duplicate matched trial")
        seen.add(key)
        if (
            _treatment_roster(block) != _treatment_roster(blocks[0])
            or block.evaluator != blocks[0].evaluator
        ):
            raise ValueError("treatment roster and evaluator must match across blocks")
        roster.append(
            dict(
                identity=asdict(block.identity),
                evaluator=asdict(block.evaluator),
                selection_policies={t.name: t.selection_policy for t in block.treatments},
                arms=[(t.name, t.role, t.pair.digest, t.procedure_id) for t in block.treatments],
            )
        )
    digest = content_digest([roster, alternatives, interactions, max_calls, independent_tasks])
    calls = 0
    observations: list[dict[str, Any]] = []
    reasons = []
    contrasts: dict[str, list[tuple[str, int]]] = {joint: [] for _, _, joint in interactions}
    for index, block in enumerate(blocks):
        if calls + 1 + len(block.treatments) > max_calls:
            reasons.append("execution budget exhausted")
            break
        control = _execute(block.identity, block.treatments[0], "before", execute)
        rows = {
            t.name: _observe(
                block,
                t,
                control,
                execute if t.procedure_id is None else procedures[t.procedure_id],
                adjudicate,
                authenticate,
                digest,
            )
            for t in block.treatments
        }
        calls += 1 + len(rows)
        observed_controls = {r["control"] for r in rows.values() if r["control"] is not None}
        if len(observed_controls) > 1:
            raise ValueError("contradictory adjudications of the same control action")
        for a, b, joint in interactions:
            cells = [rows[k] for k in (a, b, joint)]
            if all(r["control"] is not None and r["intervention"] is not None for r in cells):
                value = (
                    int(cells[2]["intervention"])
                    - int(cells[0]["intervention"])
                    - int(cells[1]["intervention"])
                    + int(cells[2]["control"])
                )
                contrasts[joint].append((block.identity.task_id, value))
        observations.extend(dict(block=index, **r) for r in rows.values())
    effects = {}
    for t in blocks[0].treatments:
        result = _effect([r for r in observations if r["treatment"] == t.name], independent_tasks)
        result["selection_policy"] = t.selection_policy
        effects[t.name] = result
        if result["matched_trials"] < len(blocks):
            reasons.append(f"{t.name}: incomplete paired outcomes")
        if t.role == "positive" and (result["drs"] is None or result["drs"] <= 0):
            reasons.append(f"{t.name}: positive control failed to improve")
        if t.role == "negative" and (result["recovered"] or result["regressed"]):
            reasons.append(f"{t.name}: negative control changed correctness")
    effective = [
        t.name
        for t in blocks[0].treatments
        if t.role == "repair" and effects[t.name]["drs"] is not None and effects[t.name]["drs"] > 0
    ]
    if reasons or not effective:
        attribution = "inconclusive"
    else:
        attribution = "ambiguous" if len(effective) > 1 else "observed_recovery"
    source_eligibility = {t.pair.training_eligibility.value for b in blocks for t in b.treatments}
    eligibility = next(
        label
        for label in ("HIDDEN_EVAL", "REGRESSION", "DEVELOPMENT")
        if label in source_eligibility
    )
    result = dict(
        schema_version="mdac-interventions-v1",
        training_eligibility=eligibility,
        attribution=attribution,
        effective_repairs=effective,
        mechanism_recovery_established=False,
        protocol_digest=digest,
        roster=roster,
        status="inconclusive" if reasons else "observed",
        reasons=reasons,
        budget_exhausted="execution budget exhausted" in reasons,
        max_calls=max_calls,
        calls_used=calls,
        planned_blocks=len(blocks),
        observations=observations,
        effects=effects,
        interactions={
            joint: dict(
                estimate=mean(v for _, v in values) if values else None,
                matched_trials=len(values),
                interval=_interval(values, 2, independent_tasks),
            )
            for joint, values in contrasts.items()
        },
        alternatives=list(alternatives),
        independent_tasks=independent_tasks,
        interval_method="95% task-clustered Hoeffding bound",
        confers_authority=False,
    )
    result["result_digest"] = content_digest(result)
    return result
