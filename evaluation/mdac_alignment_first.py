"""Staged alignment-first experiments with diagnostic allocation and private checkpoint review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from gepa_mindfulness.mdac_admission import (
    AdmissionPanel,
    AdmissionPolicy,
    Proposal,
    review_candidate,
)
from gepa_mindfulness.private_promotion import PrivatePromotionStore
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.mdac_repair import RepairResult
from gepa_mindfulness.training.trajectory import RolloutRequest
from synthetic_data.lesson_curation import HostCheck, require_enabled
from synthetic_data.mdac_curriculum import AllocationPolicy, CurriculumCriterion, plan_curriculum

from .causal_records import content_digest
from .mdac_comparisons import (
    Completion,
    Predict,
    Study,
    TrainingRequest,
    TrainingRun,
    heldout_report,
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
        [dict[str, Any], Predict], tuple[tuple[CurriculumCriterion, ...], str, dict[str, Any]]
    ],
    verify_repair: Callable[[Example, str], RepairResult],
    prepare_admission: Callable[[str, dict[str, Any]], ReviewContext],
    enabled: bool = False,
) -> dict[str, Any]:
    """Run matched four-example stages, preserving the incumbent through advisory review."""
    require_enabled(enabled)
    if order not in ORDERS or len(study.methods) != 1 or len(study.seeds) != 1 or study.samples % 4:
        raise ValueError("choose one order, method, seed and a stage budget divisible by four")
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
    criteria, family, intervention = diagnose(development, predict)
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
        if (
            assessment
            and assessment.state in ("supported", "strongly_supported")
            and (
                digest not in assessment.intervention_results
                or intervention.get("status") != "observed"
                or not intervention.get("effective_repairs")
            )
        ):
            raise ValueError("supported hypothesis needs observed intervention evidence")
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
    context = prepare_admission(checkpoint, admission)
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
        dict(order=order, study=asdict(study), split_digest=manifest.digest),
        initial=initial,
        rounds=rounds,
        diagnostic_profile=development,
        allocation=allocation,
        admission_profile=admission,
        admission=decision,
        final_profiles=finals,
        intervention=intervention,
        intervention_family=family,
        verification=verification,
        evidence_kind=study.evidence_kind,
    )
