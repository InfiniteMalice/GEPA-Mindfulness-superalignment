"""Bounded black-box teacher sampling through existing live verified-credit admission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from math import fsum, isfinite, sqrt
from typing import TYPE_CHECKING

from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.verification.interfaces import RelationalVerificationResult

from .actionopd import (
    SemanticAction,
    TeacherActionSample,
    actionopd_advantages,
    evaluate_action_group,
    public_state_digest,
)
from .causal_reward import _accepted
from .causal_reward_records import (
    _json_value,
    _number,
    _read_request,
    _request_wire,
    _text,
    request_digest,
    validate_input_envelope,
)
from .contracts import RewardRequest, RLAlgorithm
from .eligibility import require_training_eligible
from .research_credit import ResearchCreditClaim
from .research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from .research_credit_runtime import (
    ResearchRewardPolicy,
    ResearchRewardProvider,
    ResearchRewardSubmission,
)
from .reward_pipeline import RewardPipeline
from .trajectory import RolloutRequest, Trajectory

if TYPE_CHECKING:
    from .engine import PreparedBatch
    from .runtime_config import RLRunConfig

HostReview = Callable[[dict[str, object]], bool]
TeacherSampler = Callable[[dict[str, object], int], TeacherActionSample]
ActionParser = Callable[[RewardRequest], SemanticAction]
ActionVerifier = Callable[[RewardRequest, SemanticAction], RelationalVerificationResult]
EventAccountant = Callable[[RewardRequest], tuple[str, tuple[tuple[str, str], ...]]]
SourceCapture = Callable[[Trajectory], RewardRequest]


@dataclass(frozen=True)
class ActionOPDConfig:
    """Operator-selected generation/review identity; no external client is installed."""

    enabled: bool = False
    teacher_id: str = ""
    teacher_version: str = ""
    teacher_contract_id: str = ""
    verifier_id: str = ""
    verifier_version: str = ""
    verifier_contract_id: str = ""
    source_review_policy_id: str = ""
    teacher_samples: int = 0
    sampling_config_json: str = "{}"
    frequency_bonus: float = 0.5
    whitening_epsilon: float = 1e-8
    max_states: int = 256
    max_candidates: int = 256
    max_cache_bytes: int = 4_194_304

    def __post_init__(self) -> None:
        """Reject implicit runtime enablement, unbounded budgets and ambiguous identities."""
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        for field in fields(self):
            if field.name.endswith("_id") or field.name.endswith("_version"):
                value = getattr(self, field.name)
                if self.enabled:
                    _text(value, field.name)
                elif type(value) is not str:
                    raise ValueError("identity fields must be strings")
        if type(self.teacher_samples) is not int or not 0 <= self.teacher_samples <= 256:
            raise ValueError("teacher sample budget must be within 0..256")
        if self.enabled and (not self.teacher_samples or self.teacher_id == self.verifier_id):
            raise ValueError("explicit teacher samples and independent verifier required")
        for name, upper in (
            ("max_states", 1024),
            ("max_candidates", 256),
            ("max_cache_bytes", 16_777_216),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"invalid {name} budget")
        _number(self.frequency_bonus, nonnegative=True)
        _number(self.whitening_epsilon, nonnegative=True)
        if self.whitening_epsilon <= 0:
            raise ValueError("positive whitening epsilon required")
        if self.enabled and (
            not isfinite(self.credit_weight * self.whitening_epsilon)
            or self.credit_weight * self.whitening_epsilon <= 0
        ):
            raise ValueError("scaled whitening epsilon must remain finite and positive")
        sampling = _json_value(self.sampling_config_json)
        if type(sampling) is not dict:
            raise ValueError("teacher sampling configuration must be an object")
        object.__setattr__(self, "sampling_config_json", canonical_json(sampling))

    @property
    def teacher(self) -> TrustedEvaluatorContract:
        """Return generation provenance, not a verifier contract."""
        return TrustedEvaluatorContract(
            self.teacher_id, self.teacher_version, self.teacher_contract_id
        )

    @property
    def verifier(self) -> TrustedEvaluatorContract:
        """Name the independent host verifier whose findings must be authenticated."""
        return TrustedEvaluatorContract(
            self.verifier_id, self.verifier_version, self.verifier_contract_id
        )

    @property
    def credit_weight(self) -> float:
        """Bound scalar credit while preserving Eq. 7 with a matching epsilon scale."""
        return 1.0 / (1.0 + self.frequency_bonus)

    def to_dict(self) -> dict[str, object]:
        """Export all settings without executable callbacks or credentials."""
        self.__post_init__()
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["schema_version"] = "actionopd-runtime-config-v1"
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> ActionOPDConfig:
        """Restore only an exact versioned configuration."""
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or value["schema_version"] != "actionopd-runtime-config-v1"
        ):
            raise ValueError("invalid ActionOPD runtime config")
        return cls(**{name: value[name] for name in names})


class ActionOPDProposer:
    """Untrusted credit proposals: existing host reviews and accounting still decide."""

    def __init__(
        self,
        config: ActionOPDConfig,
        policy: ResearchRewardPolicy,
        *,
        authority_domain: str,
        authorize_source: Callable[[dict[str, object]], bool],
        sample_teacher: Callable[[dict[str, object], int], TeacherActionSample],
        parse_action: Callable[[RewardRequest], SemanticAction],
        verify_action: Callable[[RewardRequest, SemanticAction], RelationalVerificationResult],
        account_event: Callable[[RewardRequest], tuple[str, tuple[tuple[str, str], ...]]],
        capture_source: Callable[[Trajectory], RewardRequest] | None = None,
    ) -> None:
        """Bind operator callbacks and bounded private generation caches."""
        self.config = ActionOPDConfig.from_dict(config.to_dict())
        self.policy = ResearchRewardPolicy.from_dict(policy.to_dict())
        if (
            not self.config.enabled
            or not self.policy.enabled
            or self.policy.coefficients != (("verified_decision", config.credit_weight),)
        ):
            raise ValueError("ActionOPD requires its explicit verified-decision credit policy")
        _text(authority_domain, "authority_domain")
        for callback in (
            authorize_source,
            sample_teacher,
            parse_action,
            verify_action,
            account_event,
        ):
            if not callable(callback):
                raise ValueError("all independent host bindings are required")
        self.authority_domain = authority_domain
        self.authorize_source = authorize_source
        self.sample_teacher = sample_teacher
        self.parse_action = parse_action
        self.verify_action = verify_action
        self.account_event = account_event
        if capture_source is not None and not callable(capture_source):
            raise ValueError("source capture must be a host callable")
        self.capture_source = capture_source
        self._cache: dict[str, tuple[TeacherActionSample, ...]] = {}
        self._cache_bytes = 0
        self._cache_scope: tuple[str, str | None] | None = None

    def __call__(self, request: RewardRequest) -> ResearchRewardSubmission:
        """Sample at a student state, retain separate verification, and propose credit."""
        source = _read_request(_request_wire(request))
        original = canonical_json(_request_wire(source))
        if source.trajectory.sampling_parameters.get("training_eligibility") != "TRAIN":
            raise ValueError("explicit source TRAIN declaration required")
        require_training_eligible(source.trajectory.to_dict())
        if self.config.verifier_id in {
            source.trajectory.model_identifier,
            source.trajectory.policy_version,
        }:
            raise ValueError("student cannot verify its own decision")
        state = public_state_digest(source)
        envelope = {
            "schema_version": "actionopd-source-review-v1",
            "training_eligibility": "DEVELOPMENT",
            "authority_granted": False,
            "source_request": _request_wire(source),
            "config": self.config.to_dict(),
            "policy": self.policy.to_dict(),
            "state_digest": state,
            "teacher_calls": 0 if state in self._cache else self.config.teacher_samples,
        }
        if not _accepted(self.authorize_source, envelope):
            raise ValueError("independent ActionOPD source review withheld")
        scope = (source.trajectory.model_identifier, source.trajectory.policy_version)
        if self._cache_scope != scope:
            # Old policy states cannot be reused by the current student. Keep the
            # cache bounded by live scope, not total run length. Durable source
            # evidence and existing admission/accounting records are untouched.
            self._cache.clear()
            self._cache_bytes = 0
            self._cache_scope = scope
        if state not in self._cache:
            self._sample_state(source, state)
        teachers = self._cache[state]
        try:
            action = self.parse_action(source)
            if type(action) is not SemanticAction:
                raise ValueError("typed student semantic action required")
            action = SemanticAction.from_dict(action.to_dict())
            verification = self.verify_action(source, action)
            if type(verification) is not RelationalVerificationResult:
                raise ValueError("independent outcome record required")
            verification = RelationalVerificationResult.from_dict(verification.to_dict())
            event, layers = self.account_event(source)
            _text(event, "host-normalized semantic event")
        except Exception:
            raise ValueError("ActionOPD host parsing/verification/accounting failed") from None
        if canonical_json(_request_wire(source)) != original:
            raise ValueError("ActionOPD host callback mutated its detached source")
        comparison = evaluate_action_group(
            (source,),
            (action,),
            teachers,
            (verification,),
            frequency_bonus=self.config.frequency_bonus,
            nominal_teacher_count=self.config.teacher_samples,
            enabled=True,
        )
        score = comparison["candidates"][0]["agreement_reward"]
        identity = content_digest(
            {"target": request_digest(source), "policy": self.policy.policy_id}
        )
        claim_id = "actionopd:" + identity
        metadata = {
            "training_eligibility": "TRAIN",
            "target_request_digest": request_digest(source),
            "decision_source": _request_wire(source),
            "verified_decision": {
                "schema_version": "actionopd-verified-decision-v1",
                "action": action.to_dict(),
                "teacher_samples": [item.to_dict() for item in teachers],
                "verification": verification.to_dict(),
                "frequency_bonus": self.config.frequency_bonus,
                "nominal_teacher_count": self.config.teacher_samples,
            },
        }
        claim = ResearchCreditClaim(
            claim_id,
            "ActionOPD",
            "verified_decision",
            "verified_decision",
            "REF-ACTIONOPD",
            self.authority_domain,
            self.policy.run_id,
            self.policy.task_id,
            event,
            source.trajectory.trajectory_id,
            source.trajectory.policy_version,
            self.policy.harness_version,
            source.trajectory.model_identifier or "student",
            self.config.verifier,
            self.policy.dataset_digest,
            state,
            verification.evidence_refs,
            0.0 if score is None else score,
            "admitted",
            RewardProvenance(
                "verified_decision",
                "independently verified semantic decision",
                VerificationRoute.TRUSTED_EVALUATOR,
                evaluator=self.config.verifier,
            ),
            canonical_json(metadata),
        )
        use = ResearchCreditUse(
            "actionopd-use:" + identity,
            self.policy.policy_id,
            request_digest(source),
            (claim_id,),
            ((claim_id, layers),),
        )
        return ResearchRewardSubmission((claim,), use)

    def _sample_state(self, source: RewardRequest, state: str) -> None:
        """Expose only public inputs, never the student response or retained source labels."""
        if len(self._cache) >= self.config.max_states:
            raise ValueError("ActionOPD state budget exceeded")
        packet = {
            "schema_version": "actionopd-public-query-v1",
            "state_digest": state,
            "prompt": source.trajectory.prompt,
            "prompt_token_ids": (
                None
                if source.trajectory.prompt_token_ids is None
                else list(source.trajectory.prompt_token_ids)
            ),
            "model_identifier": source.trajectory.model_identifier,
            "policy_version": source.trajectory.policy_version,
            "student_sampling": {
                key: value
                for key, value in source.trajectory.sampling_parameters.items()
                if key in {"temperature", "top_p", "do_sample", "max_new_tokens"}
            },
            "teacher": {
                "evaluator_id": self.config.teacher_id,
                "evaluator_version": self.config.teacher_version,
                "contract_id": self.config.teacher_contract_id,
            },
            "sampling_config": _json_value(self.config.sampling_config_json),
        }
        encoded = canonical_json(packet)
        validate_input_envelope(packet)
        samples = []
        size = self._cache_bytes
        for index in range(self.config.teacher_samples):
            public_packet = _json_value(encoded)
            try:
                sample = self.sample_teacher(public_packet, index)
            except Exception:
                raise ValueError("configured teacher sampling failed") from None
            if canonical_json(public_packet) != encoded:
                raise ValueError("teacher callback mutated its public query")
            if type(sample) is not TeacherActionSample:
                raise ValueError("typed teacher generation record required")
            sample = TeacherActionSample.from_dict(sample.to_dict())
            if (
                sample.state_digest != state
                or sample.teacher != self.config.teacher
                or sample.sampling_config_json != self.config.sampling_config_json
            ):
                raise ValueError("teacher generation identity/state/config mismatch")
            size += len(canonical_json(sample.to_dict()).encode("utf-8"))
            if size > self.config.max_cache_bytes:
                raise ValueError("teacher cache byte budget exceeded")
            samples.append(sample)
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("duplicate teacher sample identity")
        self._cache[state] = tuple(samples)
        self._cache_bytes = size


class _ZeroBaseline:
    """Keep this isolated policy objective separate from task/process reward mixtures."""

    def score(self, request: RewardRequest) -> float:
        return 0.0


def build_actionopd_provider(
    config: ActionOPDConfig = ActionOPDConfig(),
    *,
    policy: ResearchRewardPolicy | None = None,
    ledger: ResearchCreditLedger | None = None,
    authorize_source: HostReview | None = None,
    sample_teacher: TeacherSampler | None = None,
    parse_action: ActionParser | None = None,
    verify_action: ActionVerifier | None = None,
    account_event: EventAccountant | None = None,
    authenticate_claims: HostReview | None = None,
    authorize_optimizer_use: HostReview | None = None,
    authorize_accounting: HostReview | None = None,
    capture_source: SourceCapture | None = None,
) -> ResearchRewardProvider | None:
    """Install no teacher/API by default; live credit uses the existing host owner."""
    if type(config) is not ActionOPDConfig:
        raise ValueError("expected exact ActionOPDConfig")
    config = ActionOPDConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    if type(policy) is not ResearchRewardPolicy or type(ledger) is not ResearchCreditLedger:
        raise ValueError("explicit host credit policy and durable journal required")
    proposer = ActionOPDProposer(
        config,
        policy,
        authority_domain=ledger.authority_domain,
        authorize_source=authorize_source,
        sample_teacher=sample_teacher,
        parse_action=parse_action,
        verify_action=verify_action,
        account_event=account_event,
        capture_source=capture_source,
    )
    return ResearchRewardProvider(
        RewardPipeline(_ZeroBaseline()),
        ledger=ledger,
        policy=policy,
        propose=proposer,
        authenticate_claims=authenticate_claims,
        authorize_optimizer_use=authorize_optimizer_use,
        authorize_accounting=authorize_accounting,
    )


def validate_actionopd_binding(config: ActionOPDConfig | None, provider: object) -> None:
    """Reject undeclared or mismatched native teacher services before any queries."""
    if config is not None and type(config) is not ActionOPDConfig:
        raise ValueError("expected exact ActionOPDConfig")
    active = config is not None and config.enabled
    live = type(provider) is ResearchRewardProvider and type(provider.propose) is ActionOPDProposer
    if active or live:
        if (
            not active
            or not live
            or provider.propose.config != config
            or provider.propose.policy != provider.policy
            or type(provider.pipeline) is not RewardPipeline
            or type(provider.pipeline.provider) is not _ZeroBaseline
            or provider.pipeline.overlay_enabled
        ):
            raise ValueError("ActionOPD requires the matching configured host teacher provider")


def capture_actionopd_sources(
    config: ActionOPDConfig,
    provider: ResearchRewardProvider,
    selected: tuple[RolloutRequest, ...],
    trajectories: tuple[Trajectory, ...],
) -> tuple[Trajectory, ...]:
    """Bind real native outputs to retained admitted rows before independent verification."""
    from .engine import _canonical_config_value

    validate_actionopd_binding(config, provider)
    capture = provider.propose.capture_source
    if not callable(capture):
        raise ValueError("native ActionOPD requires independent source capture")
    captured = []
    for trajectory in trajectories:
        matches = [
            request
            for request in selected
            if (request.prompt, request.case_id, request.policy_version)
            == (trajectory.prompt, trajectory.case_id, trajectory.policy_version)
        ]
        if len(matches) != 1:
            raise ValueError("ActionOPD source row association is ambiguous")
        metadata = _canonical_config_value(matches[0].metadata)
        if type(metadata) is not dict:
            raise ValueError("retained source metadata must be a JSON object")
        declared = metadata.get("training_eligibility")
        if declared is None and type(metadata.get("metadata")) is dict:
            declared = metadata["metadata"].get("training_eligibility")
        if declared != "TRAIN":
            raise ValueError("native ActionOPD requires explicit admitted TRAIN source rows")
        require_training_eligible(metadata)
        require_training_eligible(trajectory.to_dict())
        parameters = dict(trajectory.sampling_parameters)
        additions = {"training_eligibility": "TRAIN", "source_record": metadata}
        if any(key in parameters and parameters[key] != value for key, value in additions.items()):
            raise ValueError("source capture cannot replace retained provenance")
        base = replace(trajectory, sampling_parameters={**parameters, **additions})
        expected_json = canonical_json(base.to_dict())
        try:
            result = capture(Trajectory.from_dict(_json_value(expected_json)))
        except Exception:
            raise ValueError("independent source capture failed") from None
        if type(result) is not RewardRequest:
            raise ValueError("source capture must return an exact observed RewardRequest")
        source = _read_request(_request_wire(result))
        actual = source.trajectory.to_dict()
        expected = _json_value(expected_json)
        expected.pop("evidence_references", None)
        actual.pop("evidence_references", None)
        if actual != expected or not set(base.evidence_references).issubset(
            source.trajectory.evidence_references
        ):
            raise ValueError("source capture changed the actual student trajectory")
        # The engine later selects all observable references. Require the capture's
        # boundary to agree rather than silently broadening a restricted selection.
        if source.observable_references != tuple(
            ref for ref in source.trajectory.evidence_references if ref.is_observable
        ):
            raise ValueError("source capture must declare the complete observable boundary")
        captured.append(source.trajectory)
    return tuple(captured)


class ActionOPDBatchPreparer:
    """Eq. 7 over live authorized verified decisions, using the native GRPO loss."""

    def __init__(self, config: RLRunConfig) -> None:
        """Require the same enabled settings and credit scope used by the runtime."""
        from .runtime_config import RLRunConfig

        if (
            type(config) is not RLRunConfig
            or config.actionopd is None
            or not config.actionopd.enabled
        ):
            raise ValueError("explicit canonical ActionOPD configuration required")
        self.config = config

    def prepare(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[object, ...],
        algorithm: RLAlgorithm | None,
    ) -> PreparedBatch:
        """Preserve source order within groups, skipping unsupervised/constant groups."""
        from .engine import PreparedBatch, _response_width, _trajectory_batch
        from .research_credit_runtime import ResearchRewardDecision

        config = self.config.actionopd
        if not 1 <= len(trajectories) <= config.max_candidates or len(trajectories) != len(rewards):
            raise ValueError("ActionOPD candidate budget or alignment mismatch")
        groups = {}
        size = 0
        for trajectory, reward in zip(trajectories, rewards):
            if type(reward) is not ResearchRewardDecision:
                raise ValueError("live authorized research decisions required")
            value = reward.optimizer_reward()
            report = reward.to_dict()
            size += len(canonical_json(report).encode("utf-8"))
            if size > 4 * 1024 * 1024:
                raise ValueError("ActionOPD aggregate batch envelope exceeded")
            review = report["review"]
            if (
                review["policy"] != self.config.research_credit.to_dict()
                or review["proposed_trajectory"] != trajectory.to_dict()
                or report["baseline_total"] != 0.0
            ):
                raise ValueError("ActionOPD target/policy/isolated-objective mismatch")
            claims = review["audit"]["claims"]
            if len(claims) != 1 or claims[0]["credit_kind"] != "verified_decision":
                raise ValueError("ActionOPD requires one eligible verified decision per action")
            claim = ResearchCreditClaim.from_dict(claims[0])
            proof = _json_value(claim.source_metadata_json)["verified_decision"]
            if (
                proof["frequency_bonus"] != config.frequency_bonus
                or proof["nominal_teacher_count"] != config.teacher_samples
                or claim.evaluator != config.verifier
            ):
                raise ValueError("ActionOPD evidence differs from configured method")
            for sample in proof["teacher_samples"]:
                record = TeacherActionSample.from_dict(sample)
                if (
                    record.teacher != config.teacher
                    or record.sampling_config_json != config.sampling_config_json
                ):
                    raise ValueError("ActionOPD teacher identity/config mismatch")
            supervised = any(sample["action"] is not None for sample in proof["teacher_samples"])
            groups.setdefault(claim.state_digest, []).append(
                (trajectory, value if supervised else None)
            )
        if algorithm is None:
            return PreparedBatch(_trajectory_batch(trajectories), trajectories)
        values = tuple(tuple(value for _, value in group) for group in groups.values())
        advantages = actionopd_advantages(
            values, epsilon=config.whitening_epsilon * config.credit_weight
        )
        prepared = []
        skipped = []
        metrics = []
        for group, group_advantages in zip(groups.values(), advantages):
            numeric = tuple(0.0 if value is None else value for _, value in group)
            mean = fsum(value / len(numeric) for value in numeric)
            std = sqrt(fsum((value - mean) ** 2 / len(numeric) for value in numeric))
            metrics.append(
                {
                    "group_reward_mean": mean,
                    "group_reward_std": std,
                    "group_size": float(len(group)),
                    "group_skipped": float(group_advantages is None),
                    "group_advantage_mean": (
                        0.0
                        if group_advantages is None
                        else fsum(group_advantages) / len(group_advantages)
                    ),
                }
            )
            if group_advantages is None:
                skipped.extend(trajectory for trajectory, _ in group)
            else:
                prepared.extend(
                    replace(
                        trajectory,
                        advantage=(advantage,) * _response_width(trajectory),
                        returns=None,
                    )
                    for (trajectory, _), advantage in zip(group, group_advantages)
                )
        rows = tuple(prepared)
        return PreparedBatch(
            _trajectory_batch(rows) if rows else None, rows, tuple(skipped), tuple(metrics)
        )
