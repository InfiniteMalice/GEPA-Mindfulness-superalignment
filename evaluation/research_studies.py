"""Pure study preparation over existing audit contracts; no training or model execution."""

# Standard library
from __future__ import annotations

# Third-party
# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_audit import validate_manifest
from evaluation.improvement_records import (
    CandidateSpec,
    EvaluationSlot,
    ImprovementProtocol,
    record_digest,
)
from evaluation.research_study_records import StudyConfiguration
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope


def prepare_research_study(config: StudyConfiguration, *, enabled: bool = False) -> dict:
    """Compile declared comparisons, preserving unsupported arms as explicit unrun entries.

    This creates metadata only. Runtime declarations, hashes and prepared slots do not authorize
    model training, paid API calls, protected evaluation, deployment or a claim of qualification.
    Source authentication, actual metering and final-test permission remain existing host duties.
    """
    if enabled is not True:
        raise ValueError("study preparation requires enabled=True")
    if type(config) is not StudyConfiguration:
        raise ValueError("exact study configuration required")
    c = StudyConfiguration.from_dict(StudyConfiguration.to_dict(config))
    validate_input_envelope(c.to_dict())
    validate_manifest(c.manifest)
    active = [r for r in c.runtimes[1:] if r.availability == "configured"]
    cases = [v for v in c.manifest.cases if v.evaluated and v.purpose != "synthetic_training"]
    count = len(active) * len(cases) * len(c.seeds) * c.repeats * len(c.metrics) * 2
    if count > 4096:
        raise ValueError("study slot capacity exceeded; split preregistered metric batches")
    if active and not cases:
        raise ValueError("study requires evaluated non-training tasks")
    config_digest = record_digest(c)
    candidates, slots = [], []
    for runtime in active:
        cid = f"{c.study_id}:{runtime.condition_id}"
        candidates.append(
            CandidateSpec(
                cid,
                c.runtimes[0].system,
                runtime.system,
                content_digest(dict(configuration_digest=config_digest, runtime=runtime.to_dict())),
                0,
            )
        )
        for case in cases:
            for seed in c.seeds:
                for repeat in range(c.repeats):
                    for metric in c.metrics:
                        for arm in ("baseline", "candidate"):
                            identity = dict(
                                candidate=cid,
                                case=case.case_id,
                                seed=seed,
                                repeat=repeat,
                                metric=metric.metric_id,
                                arm=arm,
                            )
                            slots.append(
                                EvaluationSlot(
                                    content_digest(identity),
                                    cid,
                                    case.case_id,
                                    metric.metric_id,
                                    arm,
                                    repeat,
                                    c.inference_budget_digest,
                                    f"{c.study_id}:{runtime.condition_id}:seed:{seed}",
                                    seed,
                                    "shared",
                                )
                            )
    protocol = (
        ImprovementProtocol(
            "research-study:" + config_digest,
            record_digest(c.manifest),
            tuple(candidates),
            c.metrics,
            tuple(slots),
            c.evaluator,
            c.evidence_refs,
            c.policy,
        )
        if active
        else None
    )
    result = dict(
        schema_version="research-study-plan-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        execute_runtime=False,
        runtime_qualification="not_assessed",
        status="planned" if protocol else "not_run",
        configuration=c.to_dict(),
        protocol=protocol.to_dict() if protocol else None,
        availability=[
            dict(condition_id=r.condition_id, status=r.availability, reason=r.reason)
            for r in c.runtimes
        ],
    )
    result["result_digest"] = content_digest(result)
    validate_input_envelope(result)
    return result
