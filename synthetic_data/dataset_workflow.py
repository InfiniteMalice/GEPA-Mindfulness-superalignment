"""Bounded dataset-role execution and fresh host admission over existing curation contracts."""

# Standard library
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

# Local
from evaluation.cases.registry import load_case_manifest
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import record_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.causal_reward_records import _digest, _text
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.trajectory import RolloutRequest

from .curriculum_records import (
    CurationPolicy,
    GenerationReview,
    curriculum_digest,
    read_curriculum_object,
    validate_curriculum_envelope,
)
from .dataset_research import (
    DatasetPool,
    DatasetRecipe,
    DatasetRevision,
    DatasetRole,
    DatasetRow,
    ResearchRound,
    prepare_dataset_revision,
)
from .lesson_curation import HostCheck, authenticate_payload, snapshot

_ANALYSES = ("analysis_data", "analysis_failures", "analysis_dynamics", "analysis_history")
_ROLES = {*_ANALYSES, "meta", "synthesis", "execution"}
RoleCall = Callable[[dict[str, object]], object]


@dataclass(frozen=True)
class DatasetWorkflowConfig:
    """Closed operator settings; declarations do not authenticate physical role independence."""

    enabled: bool = False
    workflow_id: str = ""
    roles: tuple[DatasetRole, ...] = ()
    policy_digest: str = ""
    trainer_digest: str = ""
    admission_authority_id: str = ""
    seed: int = 0
    max_examples: int = 4096
    max_output_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("explicit boolean enablement and bounded integer seed required")
        if type(self.roles) is not tuple or any(
            type(role) is not DatasetRole for role in self.roles
        ):
            raise ValueError("exact role tuple required")
        if type(self.max_examples) is not int or not 1 <= self.max_examples <= 65536:
            raise ValueError("invalid example budget")
        if (
            type(self.max_output_bytes) is not int
            or not 1 <= self.max_output_bytes <= 4 * 1024 * 1024
        ):
            raise ValueError("invalid output byte budget")
        if self.enabled:
            for name in ("workflow_id", "admission_authority_id"):
                _text(getattr(self, name), name)
            _digest(self.policy_digest)
            _digest(self.trainer_digest)
            if len(self.roles) != 7 or {role.role for role in self.roles} != _ROLES:
                raise ValueError("all seven distinct dataset roles required")
            actors = {role.actor_id for role in self.roles}
            if len(actors) != 7 or self.admission_authority_id in actors:
                raise ValueError("dataset proposers cannot be the admission authority")
        elif any(
            type(getattr(self, name)) is not str
            for name in ("workflow_id", "policy_digest", "trainer_digest", "admission_authority_id")
        ):
            raise ValueError("scope identifiers must be strings")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["roles"] = [role.to_dict() for role in self.roles]
        value["schema_version"] = "dataset-workflow-config-v1"
        validate_curriculum_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> DatasetWorkflowConfig:
        validate_curriculum_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or (
                value["schema_version"] != "dataset-workflow-config-v1"
                or type(value["roles"]) is not list
            )
        ):
            raise ValueError("invalid dataset workflow config schema")
        args = {name: value[name] for name in names}
        args["roles"] = tuple(DatasetRole.from_dict(role) for role in value["roles"])
        return cls(**args)


@dataclass(frozen=True, init=False)
class AdmittedDataset:
    """A live host-admitted provider; JSON/pickle cannot restore training permission."""

    path: Path
    _sha256: str
    _reference_json: str

    def __init__(self) -> None:
        raise TypeError("datasets are admitted only by the configured host workflow")

    def materialize(self, mode: str) -> tuple[RolloutRequest, ...]:
        """Read one verified snapshot; every request remains explicitly training-purpose data."""
        if mode not in {"train", "resume", "collect", "evaluate", "inspect"}:
            raise ValueError("unsupported dataset consumer mode")
        payload = self.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self._sha256:
            raise ValueError("admitted dataset bytes changed; fresh admission required")
        requests = []
        for index, line in enumerate(payload.decode("utf-8").rstrip("\n").split("\n"), 1):
            record = read_curriculum_object(line)
            require_training_eligible(record)
            if record.get("training_eligibility") != "TRAIN":
                raise ValueError("admitted training declaration missing")
            requests.append(
                RolloutRequest(
                    record["prompt"],
                    case_id=record["source_case"]["case_id"],
                    metadata={
                        "training_eligibility": "TRAIN",
                        "purpose": "synthetic_training",
                        "supervised_response": record["response"],
                        "source_record": record,
                        "source_path": str(self.path),
                        "source_line": index,
                        "source_sha256": self._sha256,
                        "admission_reference": json.loads(self._reference_json),
                    },
                )
            )
        return tuple(requests)

    def __reduce__(self) -> object:
        raise TypeError("admitted dataset capabilities require fresh host admission")


@dataclass(frozen=True)
class DatasetWorkflowResult:
    """DEVELOPMENT audit plus an optional live data provider, never model deployment."""

    dataset: AdmittedDataset | None
    _report_json: str

    def to_dict(self) -> dict[str, object]:
        return read_curriculum_object(self._report_json)


def _invoke(role: str, callback: RoleCall, packet: dict[str, object], trace: list) -> object:
    validate_curriculum_envelope(packet)
    serialized = canonical_json(packet)
    detached = json.loads(serialized)
    try:
        result = callback(detached)
        validate_curriculum_envelope(detached)
        if canonical_json(detached) != serialized:
            raise ValueError("changed input")
    except Exception:
        raise ValueError("dataset role failed or mutated its input: " + role) from None
    trace.append({"role": role, "input_digest": content_digest(packet)})
    return result


def run_dataset_workflow(
    pool: DatasetPool,
    previous: DatasetRecipe,
    contract: ResearchRound,
    policy: CurationPolicy,
    *,
    config: DatasetWorkflowConfig = DatasetWorkflowConfig(),
    analyses: Mapping[str, RoleCall] | None = None,
    plan: RoleCall | None = None,
    synthesize: RoleCall | None = None,
    execute: RoleCall | None = None,
    verify: RoleCall | None = None,
    render: Callable[[dict], tuple[str, str]] | None = None,
    count_tokens: Callable[[str, str], int] | None = None,
    authorize_sources: HostCheck | None = None,
    authenticate: HostCheck | None = None,
    authorize_training: HostCheck | None = None,
    admission_reference: EvidenceReference | None = None,
    output_root: Path | None = None,
) -> DatasetWorkflowResult | None:
    """Execute one bounded data revision; only the independent host admits its artifact.

    Callbacks are operator-installed synchronous roles, not a Python sandbox or an
    external model service. Their actual identity, evidence and spending authority
    remain host responsibilities. No model training or promotion occurs here.
    """
    if type(config) is not DatasetWorkflowConfig:
        raise ValueError("exact DatasetWorkflowConfig required")
    config = DatasetWorkflowConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    pool, previous = snapshot(pool, DatasetPool), snapshot(previous, DatasetRecipe)
    contract, policy = snapshot(contract, ResearchRound), snapshot(policy, CurationPolicy)
    if config.policy_digest != curriculum_digest(policy) or (
        config.trainer_digest != contract.trainer_digest or config.seed != contract.seed
    ):
        raise ValueError("fixed policy/trainer/seed scope mismatch")
    if policy.evaluator.evaluator_id in {role.actor_id for role in config.roles}:
        raise ValueError("dataset generator cannot certify its own examples")
    if (
        analyses is None
        or set(analyses) != set(_ANALYSES)
        or any(
            not callable(callback)
            for callback in (
                *analyses.values(),
                plan,
                synthesize,
                execute,
                verify,
                render,
                count_tokens,
                authorize_sources,
                authenticate,
                authorize_training,
            )
        )
    ):
        raise ValueError("all specialized roles, trusted rendering and host authorities required")
    if (
        type(admission_reference) is not EvidenceReference
        or (admission_reference.source_kind is not EvidenceSourceKind.EXTERNAL_RECORD)
        or not isinstance(output_root, Path)
    ):
        raise ValueError("external host admission reference and output root required")
    reference = EvidenceReference.from_dict(admission_reference.to_dict())
    records = {row.case.case_id: read_curriculum_object(row.record_json) for row in pool.rows}
    for record in records.values():
        if record.get("metadata", {}).get("training_eligibility") != "TRAIN":
            raise ValueError("original explicit TRAIN sources required")
        require_training_eligible(record)
    cases = load_case_manifest().cases
    counts = {case.id: 0 for case in cases}
    unassigned = 0
    for record in records.values():
        case_id = record.get("metadata", {}).get("canonical_case_id")
        if type(case_id) is int and case_id in counts:
            counts[case_id] += 1
        else:
            unassigned += 1
    coverage = {
        "canonical_cases": [
            {"case_id": case.id, "case_key": case.key, "rows": counts[case.id]} for case in cases
        ],
        "unassigned_rows": unassigned,
    }
    source = {
        "coverage": coverage,
        "config": config.to_dict(),
        "pool": pool.to_dict(),
        "previous": previous.to_dict(),
        "contract": contract.to_dict(),
        "policy": policy.to_dict(),
    }
    validate_curriculum_envelope(source)
    if not authenticate_payload(source, authorize_sources):
        raise ValueError("source admission/authentication withheld")
    trace = []
    notes = {}
    for name in _ANALYSES:
        note = _invoke(name, analyses[name], source, trace)
        if type(note) is not dict:
            raise ValueError("bounded analysis object required")
        validate_curriculum_envelope(note)
        notes[name] = note
        trace[-1]["output"] = note
    planned = snapshot(_invoke("meta", plan, source | {"analyses": notes}, trace), DatasetRevision)
    if (
        planned.additions
        or planned.roles != config.roles
        or (
            planned.parent_pool_digest != record_digest(pool)
            or planned.parent_recipe_digest != record_digest(previous)
            or planned.round_contract_digest != record_digest(contract)
        )
    ):
        raise ValueError("meta plan changed fixed source/roles or bypassed synthesis")
    trace[-1]["output"] = planned.to_dict()
    generated = _invoke("synthesis", synthesize, source | {"plan": planned.to_dict()}, trace)
    if type(generated) is not tuple or len(generated) > 128:
        raise ValueError("bounded generated-row tuple required")
    additions = tuple(snapshot(row, DatasetRow) for row in generated)
    synthesizer = next(role for role in config.roles if role.role == "synthesis")
    actors = {role.actor_id for role in config.roles}
    for row in additions:
        record = read_curriculum_object(row.record_json)
        provenance = record.get("metadata", {})
        if (
            provenance.get("training_eligibility") != "TRAIN"
            or provenance.get("generator") != synthesizer.actor_id
            or not provenance.get("source_template")
            or provenance.get("reviewed_by") in actors
            or row.producer.model_version != synthesizer.model_version
            or row.producer.config_digest != synthesizer.config_digest
        ):
            raise ValueError("synthesis provenance and independent completed review required")
        require_training_eligible(record)
    trace[-1]["output"] = [row.to_dict() for row in additions]
    all_rows = {row.case.case_id: row for row in (*pool.rows, *additions)}
    rendered = {}
    for row_id, row in all_rows.items():
        record = read_curriculum_object(row.record_json)
        supplied = render(json.loads(canonical_json(record)))
        if (
            type(supplied) is not tuple
            or len(supplied) != 2
            or any(type(value) is not str or not value.strip() for value in supplied)
        ):
            raise ValueError("trusted rendering must return public prompt and supervised response")
        tokens = count_tokens(*supplied)
        if type(tokens) is not int or not 0 < tokens <= contract.token_budget:
            raise ValueError("invalid actual token count")
        rendered[row_id] = {"prompt": supplied[0], "response": supplied[1], "token_count": tokens}
    if any(rendered[item.case_id]["token_count"] != item.token_count for item in previous.items):
        raise ValueError("immutable source token declaration mismatch")
    candidate_packet = source | {
        "analyses": notes,
        "plan": planned.to_dict(),
        "additions": [row.to_dict() for row in additions],
        "rendered": rendered,
    }
    proposal = snapshot(_invoke("execution", execute, candidate_packet, trace), DatasetRevision)
    expected = planned.to_dict()
    expected["additions"] = [row.to_dict() for row in additions]
    # Only new-row tokenizer counts can be finalized by the executor.
    new_ids = {row.case.case_id for row in additions}
    for item in expected["recipe"]["items"]:
        if item["case_id"] in new_ids:
            item["token_count"] = rendered[item["case_id"]]["token_count"]
    if proposal.to_dict() != expected or any(
        item.token_count != rendered[item.case_id]["token_count"] for item in proposal.recipe.items
    ):
        raise ValueError("execution changed the approved operation, recipe, roles or token counts")
    trace[-1]["output"] = proposal.to_dict()
    verification_packet = candidate_packet | {"proposal": proposal.to_dict()}
    review = snapshot(_invoke("verification", verify, verification_packet, trace), GenerationReview)
    trace[-1]["output"] = review.to_dict()
    preparation = prepare_dataset_revision(
        pool=pool,
        previous=previous,
        proposal=proposal,
        contract=contract,
        review=review,
        policy=policy,
        authenticate=lambda packet: authenticate_payload(
            {"preparation": packet, "workflow": verification_packet}, authenticate
        ),
        enabled=True,
    )
    report = {
        "schema_version": "dataset-workflow-report-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "optimizer_input": False,
        "source_reference": "REF-AUTOSFT",
        "config": config.to_dict(),
        "roles": trace,
        "preparation": preparation,
        "status": "review_withheld",
        "admission_reference": reference.to_dict(),
    }
    validate_curriculum_envelope(report)
    if preparation["disposition"] != "reviewed_candidate":
        return DatasetWorkflowResult(None, canonical_json(report))
    chunks = []
    example_count, byte_count = 0, 0
    for item in proposal.recipe.items:
        raw = read_curriculum_object(all_rows[item.case_id].record_json)
        require_training_eligible(raw)
        if raw.get("training_labels", {}).get("use_for_sft") is not True:
            raise ValueError("selected source is not approved for SFT")
        data = {
            "schema_version": "admitted-supervised-example-v1",
            "training_eligibility": "TRAIN",
            "prompt": rendered[item.case_id]["prompt"],
            "response": rendered[item.case_id]["response"],
            "source_record": raw,
            "source_case": all_rows[item.case_id].case.to_dict(),
            "source_evidence_refs": [ref.to_dict() for ref in all_rows[item.case_id].evidence_refs],
            "dataset_version": proposal.recipe.version,
            "admission_reference": reference.to_dict(),
        }
        require_training_eligible(data)
        if example_count + item.repetitions > config.max_examples:
            raise ValueError("materialized example budget exceeded")
        line = (canonical_json(data) + "\n").encode("utf-8")
        reserved = len(line) * item.repetitions
        if byte_count + reserved > config.max_output_bytes:
            raise ValueError("materialized dataset byte budget exceeded")
        chunks.append(line * item.repetitions)
        byte_count += reserved
        example_count += item.repetitions
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    directory = (output_root / hashlib.sha256(config.workflow_id.encode()).hexdigest()).resolve()
    data_path = (directory / "train.jsonl").resolve()
    report.update(
        status="admission_pending",
        training_sha256=digest,
        example_count=example_count,
        dataset_path=str(data_path.resolve()),
    )
    validate_curriculum_envelope(report)
    if directory.exists():
        raise FileExistsError("dataset workflow artifact already exists")
    if not authenticate_payload(report, authorize_training):
        report["status"] = "training_admission_withheld"
        return DatasetWorkflowResult(None, canonical_json(report))
    directory.mkdir(parents=True, exist_ok=False)
    data_path.write_bytes(payload)
    (directory / "pool.json").write_text(
        canonical_json(preparation["candidate_pool"]) + "\n", encoding="utf-8"
    )
    (directory / "recipe.json").write_text(
        canonical_json(proposal.recipe.to_dict()) + "\n", encoding="utf-8"
    )
    report.update(status="dataset_admitted", dataset_path=str(data_path.resolve()))
    validate_curriculum_envelope(report)
    (directory / "workflow.json").write_text(canonical_json(report) + "\n", encoding="utf-8")
    admitted = object.__new__(AdmittedDataset)
    for name, value in (
        ("path", data_path),
        ("_sha256", digest),
        ("_reference_json", canonical_json(reference.to_dict())),
    ):
        object.__setattr__(admitted, name, value)
    return DatasetWorkflowResult(admitted, canonical_json(report))
