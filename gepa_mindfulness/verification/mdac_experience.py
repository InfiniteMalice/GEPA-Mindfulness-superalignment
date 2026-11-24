"""Verified, scoped operation records with paired estimates of their value in use."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import mean

from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.seeds import validate_seed
from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping


class ExperienceDomain(str, Enum):
    FACTUAL = "factual_reasoning"
    ETHICAL = "ethical_reasoning"
    UNCERTAINTY = "honest_uncertainty_disclosure"
    CORRECTION = "corrective_learning"
    AUTHORIZATION = "tool_authorization"
    MULTI_TURN = "multi_turn_semantic_robustness"
    OVERSIGHT = "cooperative_oversight"


def _text(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("experience requires explicit nonempty text fields")


@dataclass(frozen=True)
class Experience:
    """Public operation and observed local effect under documented circumstances."""

    record_id: str
    domain: ExperienceDomain
    preconditions: Mapping[str, object]
    contexts: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    operation: str
    expected_effect: str
    observed_effect: str
    verification_method: str
    succeeded: bool
    constitutional_implications: str
    provenance: tuple[EvidenceReference, ...]
    limitations: tuple[str, ...]
    policy_version: str

    def __post_init__(self) -> None:
        for value in (
            self.record_id,
            self.operation,
            self.expected_effect,
            self.observed_effect,
            self.verification_method,
            self.constitutional_implications,
            self.policy_version,
        ):
            _text(value)
        if type(self.succeeded) is not bool or not isinstance(self.domain, ExperienceDomain):
            raise ValueError("invalid experience status or domain")
        for group in (self.contexts, self.limitations):
            if type(group) is not tuple or not group:
                raise ValueError("record requires explicit contexts and limitations")
            for value in group:
                _text(value)
        for group in (self.evidence, self.provenance):
            if (
                type(group) is not tuple
                or not group
                or any(type(ref) is not EvidenceReference or not ref.is_observable for ref in group)
            ):
                raise ValueError("record requires public evidence and provenance")
        state = freeze_json_mapping(self.preconditions, field_name="preconditions")
        object.__setattr__(self, "preconditions", state)

    def to_dict(self) -> dict[str, object]:
        return dict(
            schema_version="mdac-experience-v1",
            record_id=self.record_id,
            domain=self.domain.value,
            preconditions=thaw_json_mapping(self.preconditions),
            contexts=list(self.contexts),
            operation=self.operation,
            expected_effect=self.expected_effect,
            observed_effect=self.observed_effect,
            verification_method=self.verification_method,
            succeeded=self.succeeded,
            constitutional_implications=self.constitutional_implications,
            limitations=list(self.limitations),
            policy_version=self.policy_version,
            evidence=[r.to_dict() for r in self.evidence],
            provenance=[r.to_dict() for r in self.provenance],
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class Query:
    """Public state and fixed evaluation coordinates for testing reuse."""

    query_id: str
    domain: ExperienceDomain
    scope: str
    state: Mapping[str, object]
    policy_version: str
    evaluator: str
    protocol: str

    def __post_init__(self) -> None:
        for value in (
            self.query_id,
            self.scope,
            self.policy_version,
            self.evaluator,
            self.protocol,
        ):
            _text(value)
        if not isinstance(self.domain, ExperienceDomain):
            raise ValueError("unknown query domain")
        object.__setattr__(self, "state", freeze_json_mapping(self.state, field_name="state"))

    def matches(self, item: Experience) -> bool:
        if (
            item.domain != self.domain
            or self.scope not in item.contexts
            or self.policy_version != item.policy_version
        ):
            return False
        state = thaw_json_mapping(self.state)
        return all(
            k in state and canonical_json(state[k]) == canonical_json(v)
            for k, v in thaw_json_mapping(item.preconditions).items()
        )


class ExperienceLibraries:
    """Host-only libraries. The resolver authenticates current evidence on every retrieval."""

    def __init__(self, *, resolve: Callable[[Experience], bool]) -> None:
        self._resolve = resolve
        self._domains: dict[ExperienceDomain, dict[str, Experience]] = {
            domain: {} for domain in ExperienceDomain
        }

    def remember(self, item: Experience) -> None:
        if self._resolve(item) is not True:
            raise ValueError("operation and provenance need current host verification")
        bank = self._domains[item.domain]
        if item.record_id in bank and bank[item.record_id].digest != item.digest:
            raise ValueError("experience identity already bound to different content")
        bank[item.record_id] = item

    def history(self, domain: ExperienceDomain) -> tuple[Experience, ...]:
        """Diagnostic history includes failed operations and superseded evidence."""
        return tuple(self._domains[domain].values())

    def recall(self, query: Query) -> tuple[Experience, ...]:
        """Return scoped observations only. The caller retains all host authorization checks."""
        return tuple(
            item
            for item in self.history(query.domain)
            if item.succeeded and query.matches(item) and self._resolve(item) is True
        )


@dataclass(frozen=True)
class UsageCall:
    """Fresh replay of one frozen query, with the optional experience as the intervention."""

    query: Query
    seed: int
    experience: Experience | None

    @property
    def digest(self) -> str:
        q = self.query
        return content_digest(
            dict(
                query=q.query_id,
                scope=q.scope,
                domain=q.domain.value,
                state=thaw_json_mapping(q.state),
                evaluator=q.evaluator,
                policy=q.policy_version,
                protocol=q.protocol,
                seed=self.seed,
                record=None if self.experience is None else self.experience.digest,
            )
        )


@dataclass(frozen=True)
class UsageOutcome:
    request_digest: str
    value: float
    solved: bool
    evidence: EvidenceReference

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not isfinite(self.value):
            raise ValueError("task value must be finite")
        if (
            type(self.solved) is not bool
            or type(self.evidence) is not EvidenceReference
            or (self.evidence.source_kind is not EvidenceSourceKind.EXTERNAL_RECORD)
        ):
            raise ValueError("require host-observed success and external evidence")


@dataclass(frozen=True)
class ExperienceGain:
    query: Query
    record_digest: str
    seeds: tuple[int, ...]
    pairs: tuple[tuple[UsageOutcome, UsageOutcome], ...]

    @property
    def gains(self) -> tuple[float, ...]:
        return tuple(with_record.value - without.value for without, with_record in self.pairs)

    @property
    def mean_gain(self) -> float:
        return mean(self.gains)

    @property
    def harmed(self) -> int:
        return sum(gain < 0 for gain in self.gains)

    @property
    def regressed(self) -> int:
        return sum(without.solved and not with_record.solved for without, with_record in self.pairs)


def measure_gain(
    library: ExperienceLibraries,
    query: Query,
    record_id: str,
    *,
    seeds: tuple[int, ...],
    sandbox: Callable[[UsageCall], UsageOutcome],
) -> ExperienceGain:
    """Estimate conditional gain with exact policy, state, protocol and seed matching."""

    if type(seeds) is not tuple or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct explicit replay seeds required")
    if any(validate_seed(seed) is None for seed in seeds):
        raise ValueError("explicit replay seed required")
    item = next((r for r in library.recall(query) if r.record_id == record_id), None)
    if item is None:
        raise ValueError("record unavailable for this query")
    pairs = []
    refs: set[str] = set()
    for seed in seeds:
        arms = []
        for record in (None, item):
            call = UsageCall(query, seed, record)
            outcome = sandbox(call)
            if (
                type(outcome) is not UsageOutcome
                or outcome.request_digest != call.digest
                or outcome.evidence.reference_id in refs
            ):
                raise ValueError("usage outcome does not match independent requested replay")
            refs.add(outcome.evidence.reference_id)
            arms.append(outcome)
        pairs.append((arms[0], arms[1]))
    return ExperienceGain(query, item.digest, seeds, tuple(pairs))
