"""Opt-in improvement dispatch and JSON-only harness trials under existing admission."""

# Standard library
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from threading import Lock
from time import monotonic

# Local
from evaluation.causal_records import canonical_json
from gepa_mindfulness.coevolution import (
    AcceptanceDecision,
    CandidateComponent,
    CoevolutionStore,
    CorrectionProposal,
    ValidationBundle,
    decide_candidate_acceptance,
)
from gepa_mindfulness.controlled_improvement import TriageDiagnostics
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.improvement_workflows import (
    PROCEDURES,
    WorkflowDiagnosis,
    WorkflowReview,
    select_improvement_workflow,
)
from gepa_mindfulness.training.causal_reward_records import _digest, _text, validate_input_envelope
from gepa_mindfulness.verification.epistemic_state import EpistemicStateEstimate
from synthetic_data.curriculum_records import decode_curriculum_json
from synthetic_data.lesson_curation import authenticate_payload

_DESTINATIONS = {
    "model_correction",
    "data_repair",
    "memory_repair",
    "skill_refinement",
    "harness_procedure",
}
_KEYS = {
    "context_compaction": {"max_tokens", "preserve_sources", "preserve_action_history"},
    "retrieval_sequencing": {"order"},
    "tool_selection": {"preferred_tools"},
    "review_routing": {"preferred_reviewers"},
    "task_decomposition": {"max_subtasks"},
    "reasoning_budget": {"max_tokens"},
    "recovery": {"max_retries"},
    "non_authoritative_cache": {"max_entries", "provenance_required"},
    "diagnostic_instrumentation": {"extra_events"},
}
_BOOTSTRAP = (
    "import hashlib,resource,sys\n"
    "resource.setrlimit(resource.RLIMIT_FSIZE,(int(sys.argv[2]),int(sys.argv[2])))\n"
    "resource.setrlimit(resource.RLIMIT_CPU,(int(sys.argv[3]),int(sys.argv[3])))\n"
    "with open(sys.argv[1],'rb') as source: code=source.read(4194305)\n"
    "if len(code)>4194304 or hashlib.sha256(code).hexdigest()!=sys.argv[4]: sys.exit(2)\n"
    "exec(compile(code,sys.argv[1],'exec'),{'__name__':'__main__','__file__':sys.argv[1]})\n"
)


def _parameters(procedure: str, encoded: str) -> dict:
    validate_input_envelope(encoded)
    params = json.loads(encoded)
    if (
        type(params) is not dict
        or canonical_json(params) != encoded
        or set(params) != _KEYS[procedure]
    ):
        raise ValueError("closed procedure parameters required; policy edits are prohibited")
    for name, value in params.items():
        if name in {"preserve_sources", "preserve_action_history", "provenance_required"}:
            if value is not True:
                raise ValueError("evidence and action history preservation cannot be disabled")
        elif name in {"order", "preferred_tools", "preferred_reviewers", "extra_events"}:
            if type(value) is not list or not 1 <= len(value) <= 128:
                raise ValueError("nonempty bounded preference IDs required")
            for item in value:
                _text(item, name)
            if len(set(value)) != len(value):
                raise ValueError("unique preference IDs required")
        else:
            maximum = {
                "max_tokens": 65536,
                "max_subtasks": 128,
                "max_retries": 3,
                "max_entries": 1024,
            }[name]
            minimum = 1 if name in {"max_tokens", "max_subtasks"} else 0
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("bounded integer procedure setting required")
    return params


@dataclass(frozen=True)
class HarnessProcedureConfig:
    """One advisory procedure setting, excluding all protected policy and authority fields."""

    harness_version: str
    procedure: str
    parameters_json: str

    def __post_init__(self) -> None:
        _text(self.harness_version, "harness_version")
        if self.procedure not in PROCEDURES:
            raise ValueError("eligible procedure required")
        _parameters(self.procedure, self.parameters_json)

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": "harness-procedure-config-v1",
            "harness_version": self.harness_version,
            "procedure": self.procedure,
            "parameters": _parameters(self.procedure, self.parameters_json),
        }

    @classmethod
    def from_dict(cls, value: object) -> HarnessProcedureConfig:
        validate_input_envelope(value)
        if (
            type(value) is not dict
            or set(value) != {"schema_version", "harness_version", "procedure", "parameters"}
            or value["schema_version"] != "harness-procedure-config-v1"
        ):
            raise ValueError("invalid harness procedure config")
        return cls(
            value["harness_version"], value["procedure"], canonical_json(value["parameters"])
        )


@dataclass(frozen=True)
class HarnessTrialConfig:
    """Explicit installed worker identity and local resource caps, not model permission."""

    enabled: bool = False
    trial_id: str = ""
    worker_path: str = ""
    worker_sha256: str = ""
    seed: int = 0
    max_calls: int = 128
    timeout_seconds: float = 30.0
    max_output_bytes: int = 65536

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("explicit enabled flag and bounded seed required")
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 512:
            raise ValueError("invalid worker call budget")
        if type(self.max_output_bytes) is not int or not 128 <= self.max_output_bytes <= 1048576:
            raise ValueError("invalid output byte budget")
        if type(self.timeout_seconds) not in (float, int) or not math.isfinite(
            self.timeout_seconds
        ):
            raise ValueError("finite timeout required")
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("bounded positive timeout required")
        for name in ("trial_id", "worker_path", "worker_sha256"):
            if type(getattr(self, name)) is not str:
                raise ValueError("worker identities must be strings")
        if self.enabled:
            _text(self.trial_id, "trial_id")
            _digest(self.worker_sha256)
            if not Path(self.worker_path).is_absolute():
                raise ValueError("explicit absolute installed worker path required")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": "harness-trial-config-v1",
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    @classmethod
    def from_dict(cls, value: object) -> HarnessTrialConfig:
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or (value["schema_version"] != "harness-trial-config-v1")
        ):
            raise ValueError("invalid harness trial config")
        return cls(**{name: value[name] for name in names})


@dataclass(frozen=True)
class WorkflowDispatchResult:
    invoked: bool
    _report_json: str
    result: object = None

    def to_dict(self) -> dict:
        return json.loads(self._report_json)


@dataclass(frozen=True)
class HarnessTrialResult:
    configuration: HarnessProcedureConfig | None
    decision: AcceptanceDecision | None
    _report_json: str

    def to_dict(self) -> dict:
        return json.loads(self._report_json)


def _select(store, correction, estimate, triage, diagnosis, evaluator, review):
    return select_improvement_workflow(
        store,
        correction,
        estimate,
        triage,
        diagnosis,
        evaluator=evaluator,
        review=review,
        enabled=True,
    )


def dispatch_improvement_workflow(
    store: CoevolutionStore,
    correction: CorrectionProposal,
    estimate: EpistemicStateEstimate,
    triage: TriageDiagnostics,
    diagnosis: WorkflowDiagnosis,
    *,
    enabled: bool = False,
    evaluator: TrustedEvaluatorContract | None = None,
    review: Callable[[WorkflowDiagnosis], WorkflowReview] | None = None,
    authenticate: Callable[[dict], bool] | None = None,
    handlers: Mapping[str, Callable[[dict], object]] | None = None,
) -> WorkflowDispatchResult | None:
    """Dispatch eligible work, never infer improvement or authorize its returned candidate."""
    if type(enabled) is not bool:
        raise ValueError("explicit boolean enablement required")
    if not enabled:
        return None
    selected = _select(store, correction, estimate, triage, diagnosis, evaluator, review)
    report = {
        "schema_version": "improvement-dispatch-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "optimizer_input": False,
        "selection": selected,
        "workflow": selected["workflow"],
        "status": "withheld",
    }
    if selected["workflow"] not in _DESTINATIONS:
        report["status"] = selected["workflow"]
        return WorkflowDispatchResult(False, canonical_json(report))
    if not isinstance(handlers, Mapping) or any(
        name not in _DESTINATIONS or not callable(callback) for name, callback in handlers.items()
    ):
        raise ValueError("eligible configured workflow handlers required")
    handler = dict(handlers).get(selected["workflow"])
    if handler is None:
        report["status"] = "handler_unavailable"
        return WorkflowDispatchResult(False, canonical_json(report))
    if not callable(authenticate):
        raise ValueError("fresh host workflow admission required")
    if not authenticate_payload(report, authenticate):
        return WorkflowDispatchResult(False, canonical_json(report))
    report["status"] = "handler_invoked"
    validate_input_envelope(report)
    return WorkflowDispatchResult(
        True, canonical_json(report), handler(json.loads(canonical_json(selected)))
    )


def _worker_digest(config: HarnessTrialConfig) -> str:
    with Path(config.worker_path).open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != config.worker_sha256:
        raise ValueError("installed worker bytes changed or exceed capacity")
    return config.worker_sha256


def _run_worker(config: HarnessTrialConfig, directory: Path, job: dict) -> dict:
    """Run fixed trusted code with bounded files/CPU/output; candidate data cannot select code."""
    if os.name != "posix":
        raise ValueError("isolated worker requires supported POSIX resource limits")
    _worker_digest(config)
    validate_input_envelope(job)
    payload = canonical_json(job).encode("utf-8")
    started = monotonic()
    status, output = "failed", None
    with tempfile.TemporaryFile() as captured:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                _BOOTSTRAP,
                config.worker_path,
                str(config.max_output_bytes),
                str(max(1, math.ceil(config.timeout_seconds))),
                config.worker_sha256,
            ],
            stdin=subprocess.PIPE,
            stdout=captured,
            stderr=subprocess.DEVNULL,
            cwd=directory,
            env={"PYTHONHASHSEED": str(config.seed), "OMP_NUM_THREADS": "1"},
            start_new_session=True,
        )
        try:
            process.communicate(payload, timeout=config.timeout_seconds)
            if process.returncode == 0:
                captured.seek(0)
                raw = captured.read(config.max_output_bytes + 1)
                if len(raw) <= config.max_output_bytes:
                    try:
                        value = decode_curriculum_json(raw.decode("utf-8"))
                        validate_input_envelope(value)
                        if type(value) is dict:
                            output, status = value, "completed"
                    except (ValueError, RecursionError, UnicodeError):
                        pass
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {
        "status": status,
        "output": output,
        "wall_time_ms": math.ceil((monotonic() - started) * 1000),
    }


def _write_report(directory: Path, report: dict) -> None:
    validate_input_envelope(report)
    pending = directory / "trial.pending.json"
    pending.write_text(canonical_json(report) + "\n", encoding="utf-8")
    pending.replace(directory / "trial.json")


def run_harness_improvement(
    store: CoevolutionStore,
    correction: CorrectionProposal,
    estimate: EpistemicStateEstimate,
    triage: TriageDiagnostics,
    diagnosis: WorkflowDiagnosis,
    original: HarnessProcedureConfig,
    proposed: HarnessProcedureConfig,
    *,
    config: HarnessTrialConfig = HarnessTrialConfig(),
    evaluator: TrustedEvaluatorContract | None = None,
    review: Callable[[WorkflowDiagnosis], WorkflowReview] | None = None,
    authenticate: Callable[[dict], bool] | None = None,
    evaluate: Callable | None = None,
    candidate_id: str = "",
    artifact_root: Path | None = None,
) -> HarnessTrialResult | None:
    """Run a host-admitted isolated procedural trial; only existing protected admission accepts it.

    Worker code, evaluator and catalog are trusted host integrations. This process
    boundary is not a security sandbox for hostile executable code. Only closed
    procedure JSON reaches the installed worker; deployment and paid APIs are absent.
    """
    if type(config) is not HarnessTrialConfig:
        raise ValueError("exact HarnessTrialConfig required")
    config = HarnessTrialConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    selection = _select(store, correction, estimate, triage, diagnosis, evaluator, review)
    report = {
        "schema_version": "harness-trial-report-v1",
        "training_eligibility": "HIDDEN_EVAL",
        "authority_granted": False,
        "optimizer_input": False,
        "selection": selection,
        "config": config.to_dict(),
        "status": selection["workflow"],
        "observations": [],
    }
    if selection["workflow"] != "harness_procedure":
        return HarnessTrialResult(None, None, canonical_json(report))
    if type(original) is not HarnessProcedureConfig or type(proposed) is not HarnessProcedureConfig:
        raise ValueError("exact original/candidate procedure configurations required")
    original = HarnessProcedureConfig.from_dict(original.to_dict())
    proposed = HarnessProcedureConfig.from_dict(proposed.to_dict())
    if (
        correction.changed_components != (CandidateComponent.HARNESS,)
        or original.procedure != proposed.procedure
        or proposed.procedure != diagnosis.procedure
    ):
        raise ValueError("one registered harness-only procedural correction required")
    epochs = store.evaluation_epochs()
    if len(epochs) < 2:
        raise ValueError("separate source/candidate evaluation epochs required")
    source, target = epochs[-2:]
    if (
        source.epoch_id != correction.source_epoch_id
        or not source.closed
        or target.closed
        or target.record_ids
        or source.model_version != target.model_version
        or source.harness_version != original.harness_version
        or target.harness_version != proposed.harness_version
        or source.harness_version == target.harness_version
    ):
        raise ValueError("fixed model/source/candidate version or empty epoch mismatch")
    old_params = _parameters(original.procedure, original.parameters_json)
    new_params = _parameters(proposed.procedure, proposed.parameters_json)
    for name in {"order", "preferred_tools", "preferred_reviewers"} & set(old_params):
        if set(old_params[name]) != set(new_params[name]):
            raise ValueError("candidate cannot change the fixed eligible preference pool")
    if original.parameters_json == proposed.parameters_json:
        raise ValueError("no justified procedure change")
    if not callable(authenticate) or not callable(evaluate) or not isinstance(artifact_root, Path):
        raise ValueError("fresh host admission, independent evaluator and artifact root required")
    _text(candidate_id, "candidate_id")
    _worker_digest(config)
    report.update(
        original=original.to_dict(),
        proposed=proposed.to_dict(),
        model_version=source.model_version,
        candidate_id=candidate_id,
        status="host_admission_pending",
        runtime={"python_version": sys.version.split()[0], "platform": sys.platform},
    )
    validate_input_envelope(
        report
        | {
            "candidate_projections": [correction.to_dict(), correction.to_dict()],
            "authority": store.authority().to_dict(),
            "final_metadata_capacity": "x" * 65536,
        }
    )
    if not authenticate_payload(report, authenticate):
        report["status"] = "host_admission_withheld"
        return HarnessTrialResult(None, None, canonical_json(report))
    directory = (artifact_root / hashlib.sha256(candidate_id.encode()).hexdigest()).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    artifact = directory / "harness.json"
    raw = (canonical_json(proposed.to_dict()) + "\n").encode("utf-8")
    artifact.write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    candidate = store.register_candidate(
        candidate_id=candidate_id, correction=correction, artifact_digest=digest
    )
    report.update(
        candidate=candidate.to_dict(),
        artifact_path=str(artifact),
        artifact_digest=digest,
        status="pending_independent_validation",
    )
    _write_report(directory, report)

    evaluation_open = True
    run_lock = Lock()

    def run(task: dict) -> dict:
        if not run_lock.acquire(blocking=False):
            raise ValueError("concurrent evaluation calls are unsupported")
        try:
            return run_public(task)
        finally:
            run_lock.release()

    def run_public(task: dict) -> dict:
        if not evaluation_open:
            raise ValueError("independent evaluation is closed")
        if type(task) is not dict or set(task) != {"task_id", "prompt"}:
            raise ValueError("public task identity/prompt only; no private labels or policy")
        _text(task["task_id"], "task_id")
        _text(task["prompt"], "prompt", limit=65536)
        task = {"task_id": task["task_id"], "prompt": task["prompt"]}
        if len(report["observations"]) >= config.max_calls:
            raise ValueError("worker evaluation call budget exhausted")
        if artifact.read_bytes() != raw:
            raise ValueError("candidate artifact changed")
        index = len(report["observations"])
        # Reserve all allowed output bytes and duplicated public metadata before launch.
        validate_input_envelope(
            report
            | {
                "next_output": "x" * (config.max_output_bytes * 6),
                "next_task": task,
                "decision_projection": candidate.to_dict(),
                "final_capacity": "x" * 65536,
            }
        )
        run_dir = directory / ("run-" + str(index))
        run_dir.mkdir(exist_ok=False)
        job = {
            "harness": proposed.to_dict(),
            "model_version": candidate.model_version,
            "task": task,
            "seed": config.seed,
        }
        captured = {
            "run_id": index,
            "task": task,
            "job_sha256": hashlib.sha256(canonical_json(job).encode("utf-8")).hexdigest(),
            "status": "pending",
            "output": None,
            "wall_time_ms": 0,
        }
        report["observations"].append(captured)
        _write_report(directory, report)
        try:
            result = _run_worker(config, run_dir, job)
        except OSError:
            result = {"status": "failed", "output": None, "wall_time_ms": 0}
        captured.update(result)
        _write_report(directory, report)
        return json.loads(canonical_json(captured))

    try:
        try:
            bundle = evaluate(candidate, run)
        finally:
            with run_lock:
                evaluation_open = False
        if artifact.read_bytes() != raw:
            raise ValueError("candidate artifact changed during evaluation")
        _worker_digest(config)
        if not report["observations"] or any(
            row["status"] != "completed" for row in report["observations"]
        ):
            report["status"] = "worker_failed"
            _write_report(directory, report)
            return HarnessTrialResult(None, None, canonical_json(report))
        if type(bundle) is not ValidationBundle or bundle.candidate != candidate:
            raise ValueError("independent validation bundle must name exact candidate")
        decision = decide_candidate_acceptance(bundle, authority_store=store)
    except Exception:
        report["status"] = "validation_failed"
        _write_report(directory, report)
        raise
    report.update(
        status="catalog_accepted" if decision.accepted else "catalog_rejected",
        decision=decision.to_dict(),
    )
    _write_report(directory, report)
    return HarnessTrialResult(
        proposed if decision.accepted else None, decision, canonical_json(report)
    )
