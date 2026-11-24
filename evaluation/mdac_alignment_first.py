"""Staged alignment-first experiments with diagnostic allocation and private checkpoint review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.mdac_admission import (
    AdmissionPanel,
    AdmissionPolicy,
    Proposal,
    review_candidate,
)
from gepa_mindfulness.private_promotion import PrivatePromotionStore
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.mdac_repair import Continuation, RepairResult
from gepa_mindfulness.training.trajectory import RolloutRequest
from synthetic_data.lesson_curation import HostCheck, authenticate_payload, require_enabled
from synthetic_data.mdac_curriculum import AllocationPolicy, CurriculumCriterion, plan_curriculum

from .causal_records import content_digest
from .mdac_comparisons import (
    Completion,
    Predict,
    Study,
    TrainingRequest,
    TrainingRun,
    heldout_report,
    intervention_effect,
)
from .mdac_reporting import SplitManifest

ORDERS = ("alignment_first", "capability_first", "mixed")
_ROUNDS = {
    "alignment_first": (("alignment",) * 4, ("capability",) * 3 + ("alignment",)),
    "capability_first": (("capability",) * 3 + ("alignment",), ("alignment",) * 4),
    "mixed": (
        ("alignment", "capability") * 2,
        ("alignment", "capability", "alignment", "alignment"),
    ),
}


@dataclass(frozen=True)
class Example:
    family: str
    kind: str
    request: RolloutRequest
    mechanism: str | None = None
    channels: tuple[str, ...] = ()
    target: str | None = None


@dataclass(frozen=True)
class Probe:
    family: str
    partition: str
    prompt: str
    dimension: str


@dataclass(frozen=True)
class ReviewContext:
    store: PrivatePromotionStore
    proposal: Proposal
    panel: AdmissionPanel
    policy: AdmissionPolicy
    reviewer: str
    authenticate: HostCheck


@dataclass(frozen=True)
class CaseReplay:
    """Preregister one development family, repair factor, mechanism, and independent judge."""

    family: str
    mechanism: str
    treatment: str
    factor: str
    evaluator: tuple[str, str, str]
    run: Callable[[str, Predict], dict[str, Any]]


def repair_receipt(example: Example, checkpoint: str, result: RepairResult) -> dict[str, Any]:
    """Bind the independently replayed decisions to the eligible public supervision."""
    verified = result.training_example
    if (
        type(result) is not RepairResult
        or not verified
        or not result.verified
        or not result.trials
        or any(type(t) is not Continuation for t in result.trials)
    ):
        raise ValueError("repair requires complete continuation-verified trials")
    metadata = verified.request.metadata
    anchor = metadata.get("anchor")
    roster = metadata.get("replay_evidence")
    refs = [trial.evidence.to_dict() for trial in result.trials]
    if (
        verified.request.policy_version != checkpoint
        or metadata.get("training_eligibility") != "TRAIN"
        or not isinstance(anchor, (list, tuple))
        or len(anchor) != 3
        or anchor[1] != example.family
        or verified.request.prompt != example.request.prompt
        or verified.target != example.target
        or metadata.get("actor_input_digest") != content_digest(example.request.prompt)
        or not isinstance(roster, (list, tuple))
        or list(roster) != refs
        or len(refs) < 4
        or len({r["reference_id"] for r in refs}) != len(refs)
    ):
        raise ValueError(
            "repair receipt must bind this checkpoint, actor, target and complete trial roster"
        )
    require_training_eligible(metadata)
    return dict(
        family=example.family,
        checkpoint=checkpoint,
        actor_input_digest=content_digest(verified.request.prompt),
        target=verified.target,
        evaluator=metadata.get("evaluator"),
        anchor=list(anchor),
        trials=[
            dict(
                request=t.request_digest,
                state=t.restored_state_digest,
                checks=dict(t.checks),
                steps=list(t.verified_steps),
                evidence=t.evidence.to_dict(),
                actor_input=t.actor_input_digest,
            )
            for t in result.trials
        ],
    )


def _paired_effect(report: dict[str, Any], treatment: str) -> None:
    """Reconcile the claimed repair effect with authenticated matched-capture judgments."""
    measured = [
        r
        for r in report["observations"]
        if r["treatment"] == treatment
        and type(r["control"]) is bool
        and type(r["intervention"]) is bool
    ]
    effect = report["effects"].get(treatment, {})
    if (
        not measured
        or any(r["role"] != "repair" or r["verification"] != "host authenticated" for r in measured)
        or len(measured) != effect.get("matched_trials")
        or sum(int(r["intervention"]) - int(r["control"]) for r in measured) / len(measured)
        != effect.get("drs")
        or not 0 < effect["drs"] <= 1
    ):
        raise ValueError("positive paired repair evidence does not match the measured treatment")


def _case_support(
    spec: CaseReplay, report: dict[str, Any], row: dict[str, Any], checkpoint: str
) -> str:
    _paired_effect(report, spec.treatment)
    digest, _ = intervention_effect(report, spec.treatment)
    rows = [r for r in report["observations"] if r["treatment"] == spec.treatment]
    expected_judge = dict(zip(("evaluator_id", "evaluator_version", "contract_id"), spec.evaluator))
    if not rows or any(r["evaluator"] != expected_judge for r in report["roster"]):
        raise ValueError("case-specific intervention evaluator mismatch")
    for measured in rows:
        source = measured["source_record"]
        before = source["before"]["turns"]
        prompt = "\n".join(turn["content"] for turn in before)
        if (
            measured["identity"]["checkpoint"] != checkpoint
            or measured["identity"]["task_id"] != spec.family
            or source["family_id"] != spec.family
            or measured["role"] != "repair"
            or {change["factor"] for change in source["changes"]} != {spec.factor}
            or measured["verification"] != "host authenticated"
            or measured["control"] != row["correct"]
            or not measured["captures"][0]["actions"]
            or measured["captures"][0]["actions"][-1] != row["response"]
            or content_digest(prompt) != row["prompt_digest"]
            or measured["adjudication"]["evaluator"] != expected_judge
        ):
            raise ValueError(
                "case-specific intervention does not match the developmental observation"
            )
    return digest


def run_lifecycle(
    *,
    order: str,
    study: Study,
    manifest: SplitManifest,
    catalog: tuple[Example, ...],
    probes: tuple[Probe, ...],
    baseline: Predict,
    train: Callable[[TrainingRequest, tuple[str | None, ...]], TrainingRun],
    grade: Callable[[Probe, str], bool],
    diagnose: Callable[
        [dict[str, Any], Predict, dict[tuple[str, str], dict[str, Any]]],
        tuple[tuple[CurriculumCriterion, ...], str, dict[str, Any]],
    ],
    verify_repair: Callable[[Example, str], RepairResult],
    calibration_evaluator: TrustedEvaluatorContract,
    authenticate_repair: HostCheck,
    repair_evidence_kind: str,
    case_tests: tuple[CaseReplay, ...] = (),
    prepare_admission: Callable[[str, dict[str, Any]], ReviewContext],
    enabled: bool = False,
) -> dict[str, Any]:
    """Run matched four-example stages, preserving the incumbent through advisory review."""
    require_enabled(enabled)
    if order not in ORDERS or len(study.methods) != 1 or len(study.seeds) != 1 or study.samples % 4:
        raise ValueError("choose one order, method, seed and a stage budget divisible by four")
    if repair_evidence_kind not in ("cpu_fixture", "model_backed"):
        raise ValueError("declare repair continuation provenance")
    if len({p.prompt for p in probes}) != len(probes):
        raise ValueError("independent probe prompts must be distinct")
    versions = dict(
        dataset_version=study.dataset_version, evaluator_version=study.evaluator_version
    )
    if type(manifest) is not SplitManifest:
        raise ValueError("expected a frozen family manifest")
    if not catalog or not probes:
        raise ValueError("training catalog and evaluation probes are required")
    pools: dict[str, list[Example]] = {"alignment": [], "capability": [], "remediation": []}
    for example in catalog:
        manifest.require(example.family, "training", **versions)
        if type(example.request) is not RolloutRequest or example.kind not in pools:
            raise ValueError("expected an alignment or capability training request")
        if example.request.metadata.get("training_eligibility") != "TRAIN":
            raise ValueError("lifecycle requires explicit TRAIN eligibility")
        require_training_eligible(example.request.metadata)
        pools[example.kind].append(example)
    if not pools["alignment"] or not pools["capability"]:
        raise ValueError("both training pools are required")
    partitions = {
        "diagnostic_development",
        "curriculum_validation",
        "governance_admission",
        "final_held_out",
        "ood",
    }
    if {p.partition for p in probes} != partitions or len({p.family for p in probes}) != len(
        probes
    ):
        raise ValueError("use distinct development, validation and governance probe families")
    if {p.prompt for p in probes} & {e.request.prompt for e in catalog}:
        raise ValueError("evaluation prompts overlap the training catalog")
    for probe in probes:
        manifest.require(probe.family, probe.partition, **versions)
    if not {"alignment", "ethical_generalization", "capability"}.issubset(
        {p.dimension for p in probes if p.partition == "curriculum_validation"}
    ):
        raise ValueError(
            "checkpoint profiles require alignment, ethical generalization and capability"
        )
    seed = study.seeds[0]

    def profile(partition: str, checkpoint: str, predict: Predict) -> dict[str, Any]:
        rows = []
        for probe in probes:
            if probe.partition != partition:
                continue
            result = predict(probe.prompt, seed, study.max_eval_tokens_per_case)
            if (
                type(result) is not Completion
                or type(result.tokens) is not int
                or not 0 <= result.tokens <= study.max_eval_tokens_per_case
            ):
                raise ValueError("independent evaluation token budget exceeded")
            score = grade(probe, result.text)
            if type(score) is not bool:
                raise ValueError("checkpoint grader requires a measured boolean")
            rows.append(
                dict(
                    family=probe.family,
                    dimension=probe.dimension,
                    correct=score,
                    response=result.text,
                    prompt_digest=content_digest(probe.prompt),
                    tokens=result.tokens,
                )
            )
        return dict(
            checkpoint=checkpoint,
            partition=partition,
            split_digest=manifest.digest,
            evaluator=study.evaluator_version,
            rows=rows,
        )

    checkpoint = study.checkpoint
    predict = baseline
    initial = profile("curriculum_validation", checkpoint, predict)
    rounds = []
    uses: Counter[str] = Counter()

    def take(kind: str) -> Example:
        pool = pools[kind]
        example = pool[uses[kind] % len(pool)]
        uses[kind] += 1
        return example

    def run_stage(stage: str, examples: list[Example], retained: int) -> None:
        nonlocal checkpoint, predict
        prompts = tuple(e.request.prompt for e in examples)
        if len(prompts) != study.samples:
            raise ValueError("incomplete stage allocation")
        request = TrainingRequest(
            study.methods[0],
            seed,
            checkpoint,
            study.conditions,
            study.governance_version,
            prompts,
            study.compute_units,
            study.training_tokens,
            study.model_config_digest,
            study.runtime_options,
        )
        run = train(request, tuple(e.target for e in examples))
        if (
            type(run) is not TrainingRun
            or run.start != checkpoint
            or run.request_digest != request.digest
            or type(run.samples) is not int
            or run.samples != study.samples
            or type(run.training_tokens) is not int
            or run.training_tokens != study.training_tokens
            or type(run.compute_units) is not int
            or run.compute_units != study.compute_units
            or run.model_config_digest != study.model_config_digest
            or not run.end
        ):
            raise ValueError("stage checkpoint, model identity or metered budget mismatch")
        checkpoint, predict = run.end, run.predict
        rounds.append(
            dict(
                stage=stage,
                source=run.start,
                checkpoint=checkpoint,
                request_digest=request.digest,
                training_families=[e.family for e in examples],
                samples=run.samples,
                tokens=run.training_tokens,
                compute_units=run.compute_units,
                retention_samples=retained,
                validation=profile("curriculum_validation", checkpoint, predict),
            )
        )

    for stage, kinds in enumerate(_ROUNDS[order]):
        run_stage(f"stage-{stage+1}", [take(k) for k in kinds * (study.samples // 4)], 0)
    development = profile("diagnostic_development", checkpoint, predict)
    case_reports = {}
    devrows = {r["family"]: r for r in development["rows"]}
    for test in case_tests:
        key = test.family, test.mechanism
        if key in case_reports or test.family not in devrows:
            raise ValueError(
                "case-specific replay requires a unique development family and mechanism"
            )
        report = test.run(checkpoint, predict)
        check = content_digest({k: v for k, v in report.items() if k != "result_digest"})
        if (
            report.get("result_digest") != check
            or report.get("schema_version") != "mdac-interventions-v1"
            or report.get("training_eligibility") != "DEVELOPMENT"
            or not report.get("roster")
            or any(b["identity"]["checkpoint"] != checkpoint for b in report["roster"])
        ):
            raise ValueError("case-specific report does not bind this checkpoint")
        case_reports[key] = (test, report)
    criteria, family, intervention = diagnose(
        deepcopy(development),
        predict,
        {key: deepcopy(binding[1]) for key, binding in case_reports.items()},
    )
    manifest.require(family, "intervention_calibration", **versions)
    digest = content_digest({k: v for k, v in intervention.items() if k != "result_digest"})
    if (
        intervention.get("result_digest") != digest
        or intervention.get("schema_version") != "mdac-interventions-v1"
        or intervention.get("training_eligibility") != "DEVELOPMENT"
        or not intervention.get("roster")
        or any(b["identity"]["checkpoint"] != checkpoint for b in intervention["roster"])
    ):
        raise ValueError("intervention must bind the current development checkpoint")

    expected = asdict(calibration_evaluator)
    if any(b["evaluator"] != expected for b in intervention["roster"]) or any(
        r["source_record"]["family_id"] != family for r in intervention["observations"]
    ):
        raise ValueError("calibration family or evaluator substitution")
    for name in intervention["effective_repairs"]:
        _paired_effect(intervention, name)
    for criterion in criteria:
        if (
            criterion.partition != "diagnostic_development"
            or criterion.record.checkpoint_id != checkpoint
            or criterion.record.dataset_version != study.dataset_version
            or criterion.record.evaluator_version != study.evaluator_version
        ):
            raise ValueError("diagnosis must bind the current independent development checkpoint")
        assessment = next(
            (a for a in criterion.record.assessments if a.mechanism == criterion.mechanism), None
        )
        if assessment and assessment.state in ("supported", "strongly_supported", "regressed"):
            family_id, mechanism = criterion.record.task_id, criterion.mechanism
            if not family_id or not mechanism:
                raise ValueError("supported hypothesis requires a named case and mechanism")
            bound = case_reports.get((family_id, mechanism))
            if (
                not bound
                or _case_support(bound[0], bound[1], devrows[bound[0].family], checkpoint)
                not in assessment.intervention_results
                or content_digest(development)
                not in {r.reference_id for r in assessment.supporting}
            ):
                raise ValueError(
                    "supported hypothesis requires case-specific replay and development evidence"
                )
    anchors = study.samples // 4
    allocation = plan_curriculum(
        criteria, policy=AllocationPolicy(budget=study.samples - anchors), enabled=True
    )
    correction = [take("alignment") for _ in range(anchors)]
    verification = []
    for slot in allocation["slots"]:
        if slot["channel"] == "baseline_capability":
            correction.append(take("capability"))
        else:
            pool = [
                e
                for e in catalog
                if e.mechanism == slot["mechanism"] and slot["channel"] in e.channels
            ]
            if not pool:
                raise ValueError("no reviewed example for requested mechanism")
            selected = pool[slot["index"] % len(pool)]
            result = verify_repair(selected, checkpoint)
            evidence = repair_receipt(selected, checkpoint, result)
            if not authenticate_payload(evidence, authenticate_repair):
                raise ValueError("independent repair verifier did not authenticate trial receipts")
            verified = result.training_example
            if (
                not result.verified
                or not verified
                or verified.request.prompt != selected.request.prompt
            ):
                raise ValueError("corrective slot lacks a continuation-verified example")
            require_training_eligible(verified.request.metadata)
            if not selected.target or verified.target != selected.target:
                raise ValueError("verified training target does not match reviewed example")
            correction.append(replace(selected, request=verified.request, target=verified.target))
            verification.append(
                dict(
                    family=selected.family,
                    target=verified.target,
                    trial_refs=[t.evidence.reference_id for t in result.trials],
                )
            )
    run_stage(
        "diagnostic_correction_and_retention",
        correction,
        anchors + allocation["channel_counts"]["retention_replay"],
    )
    admission = profile("governance_admission", checkpoint, predict)
    context = prepare_admission(checkpoint, deepcopy(admission))
    if (
        context.proposal.artifact_digest != checkpoint
        or context.panel.suite_ref != content_digest(admission)
        or context.policy.version != study.governance_version
    ):
        raise ValueError("admission must bind the evaluated checkpoint, suite and frozen policy")
    decision = review_candidate(
        context.store,
        context.proposal,
        context.panel,
        context.policy,
        reviewer=context.reviewer,
        active_epoch=context.proposal.source_epoch,
        authenticate=context.authenticate,
        enabled=True,
    )
    finals = {p: profile(p, checkpoint, predict) for p in ("final_held_out", "ood")}
    return heldout_report(
        "mdac-alignment-lifecycle-v1",
        dict(
            order=order,
            study=asdict(study),
            split_digest=manifest.digest,
            case_tests=[
                dict(
                    family=t.family,
                    mechanism=t.mechanism,
                    treatment=t.treatment,
                    factor=t.factor,
                    evaluator=list(t.evaluator),
                )
                for t in case_tests
            ],
        ),
        initial=initial,
        rounds=rounds,
        diagnostic_profile=development,
        allocation=allocation,
        admission_profile=admission,
        admission=decision,
        final_profiles=finals,
        intervention=intervention,
        intervention_family=family,
        case_interventions=[item[1] for item in case_reports.values()],
        phase_evidence=dict(
            training=study.evidence_kind,
            calibration_replay=study.evidence_kind,
            repair_continuation=repair_evidence_kind,
            admission="private_advisory",
        ),
        verification=verification,
        evidence_kind=study.evidence_kind,
    )
