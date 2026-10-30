"""Default-off immutable AutoSFT-inspired research preparation, without admission authority."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass

# Third-party
# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_audit import validate_manifest
from evaluation.improvement_records import (
    DatasetCase,
    DatasetManifest,
    ImprovementRecord,
    SystemConfig,
    _encode,
    record_digest,
)
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.synthetic_dataset_validation import validate_rich_record

from .curriculum_records import (
    CurationPolicy,
    GenerationReview,
    curriculum_digest,
    read_curriculum_object,
    validate_curriculum_envelope,
)
from .lesson_curation import (
    HostCheck,
    authenticate_payload,
    report_document,
    require_enabled,
    restricted_origin,
    snapshot,
)

DATASET_CHECKS = (
    "schema",
    "label_fidelity",
    "constitutional_admissibility",
    "source_lineage",
    "decontamination",
    "token_counts",
    "role_independence",
)
_OPERATIONS = (
    "rebalancing",
    "pruning",
    "repair",
    "replacement",
    "targeted_synthesis",
    "curriculum_adjustment",
    "robustness_augmentation",
)
_ROLES = (
    "meta",
    "execution",
    "analysis_data",
    "analysis_failures",
    "analysis_dynamics",
    "analysis_history",
    "synthesis",
)


def _refs(refs: tuple[EvidenceReference, ...]) -> None:
    if not refs or len(refs) > 32 or len({r.reference_id for r in refs}) != len(refs):
        raise ValueError("nonempty unique bounded evidence references required")


class _DatasetRecord(ImprovementRecord):
    def __post_init__(self) -> None:
        super().__post_init__()
        validate_curriculum_envelope(_encode(self))

    def _schema(self, expected: str) -> None:
        if self.schema_version != expected:
            raise ValueError("unsupported dataset schema")


@dataclass(frozen=True)
class DatasetRow(_DatasetRecord):
    """An immutable rich source record using the existing dataset identity contract."""

    case: DatasetCase
    record_json: str
    producer: SystemConfig
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "dataset-row-v1"

    def _validate(self) -> None:
        self._schema("dataset-row-v1")
        record = read_curriculum_object(self.record_json)
        if self.case.content_digest != content_digest(record):
            raise ValueError("source content digest mismatch")
        if self.case.case_id != record.get("id"):
            raise ValueError("source record identity mismatch")
        errors = validate_rich_record(record)
        if errors:
            raise ValueError("rich schema validation failed: " + "; ".join(errors[:3]))
        if restricted_origin(record):
            raise ValueError("protected or held-out source record")
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class DatasetPool(_DatasetRecord):
    version: str
    manifest: DatasetManifest
    rows: tuple[DatasetRow, ...]
    prior_versions: tuple[str, ...] = ()
    schema_version: str = "dataset-pool-v1"

    def _validate(self) -> None:
        self._schema("dataset-pool-v1")
        if not self.rows or len(self.rows) > 128:
            raise ValueError("pool requires 1..128 rows")
        if (
            len(self.prior_versions) > 128
            or len(set(self.prior_versions)) != len(self.prior_versions)
            or self.version in self.prior_versions
        ):
            raise ValueError("invalid immutable version ancestry")
        validate_manifest(self.manifest)
        if self.manifest.dependencies_known is not True:
            raise ValueError("complete source dependencies required")
        cases = {c.case_id: c for c in self.manifest.cases}
        row_ids = [r.case.case_id for r in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("duplicate pool row identity")
        if any(c.purpose != "synthetic_training" for c in self.manifest.cases):
            raise ValueError("protected or non-training partition in pool")
        if any(cases.get(r.case.case_id) != r.case for r in self.rows):
            raise ValueError("row manifest binding mismatch")


@dataclass(frozen=True)
class RecipeItem(_DatasetRecord):
    case_id: str
    repetitions: int
    token_count: int
    schema_version: str = "dataset-recipe-item-v1"

    def _validate(self) -> None:
        self._schema("dataset-recipe-item-v1")
        if min(self.repetitions, self.token_count) <= 0:
            raise ValueError("recipe counts must be positive exact integers")


@dataclass(frozen=True)
class DatasetRecipe(_DatasetRecord):
    """Ordered row blocks; counts are explicit, not a hidden stochastic sampler."""

    version: str
    round_contract_digest: str
    items: tuple[RecipeItem, ...]
    schema_version: str = "dataset-recipe-v1"

    def _validate(self) -> None:
        self._schema("dataset-recipe-v1")
        if not self.items or len(self.items) > 128:
            raise ValueError("recipe requires 1..128 items")
        if len({i.case_id for i in self.items}) != len(self.items):
            raise ValueError("duplicate recipe row identity")


@dataclass(frozen=True)
class DatasetRole(_DatasetRecord):
    role: str
    actor_id: str
    model_version: str
    config_digest: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "dataset-role-v1"

    def _validate(self) -> None:
        self._schema("dataset-role-v1")
        if self.role not in _ROLES:
            raise ValueError("unsupported dataset role")
        _refs(self.evidence_refs)


@dataclass(frozen=True)
class ResearchRound(_DatasetRecord):
    starting_system: SystemConfig
    trainer_digest: str
    evaluation_protocol_digest: str
    token_budget: int
    example_budget: int
    seed: int
    schema_version: str = "dataset-research-round-v1"

    def _validate(self) -> None:
        self._schema("dataset-research-round-v1")
        if min(self.token_budget, self.example_budget) <= 0 or self.seed < 0:
            raise ValueError("invalid fixed research budget or seed")


@dataclass(frozen=True)
class DatasetRevision(_DatasetRecord):
    proposal_id: str
    operation: str
    parent_pool_digest: str
    parent_recipe_digest: str
    round_contract_digest: str
    additions: tuple[DatasetRow, ...]
    recipe: DatasetRecipe
    roles: tuple[DatasetRole, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "dataset-revision-v1"

    def _validate(self) -> None:
        self._schema("dataset-revision-v1")
        if self.operation not in _OPERATIONS or len(self.additions) > 128:
            raise ValueError("unsupported dataset operation or too many additions")
        if len(self.roles) not in (6, 7) or len({r.role for r in self.roles}) != len(self.roles):
            raise ValueError("distinct bounded research roles required")
        if not set(_ROLES[:6]).issubset({r.role for r in self.roles}):
            raise ValueError("missing research role")
        if len({r.actor_id for r in self.roles}) != len(self.roles):
            raise ValueError("research roles require separate actors")
        if self.additions and "synthesis" not in {r.role for r in self.roles}:
            raise ValueError("new rows require synthesis provenance")
        _refs(self.evidence_refs)


def _recipe_totals(recipe: DatasetRecipe, pool: DatasetPool, contract: ResearchRound) -> dict:
    known = {r.case.case_id for r in pool.rows}
    if any(item.case_id not in known for item in recipe.items):
        raise ValueError("recipe references unknown source row")
    totals = dict(
        examples=sum(i.repetitions for i in recipe.items),
        tokens=sum(i.repetitions * i.token_count for i in recipe.items),
    )
    if totals["examples"] > contract.example_budget or totals["tokens"] > contract.token_budget:
        raise ValueError("fixed research budget exceeded")
    return totals


def _derive(pool: DatasetPool, before: DatasetRecipe, proposal: DatasetRevision) -> DatasetPool:
    old = {r.case.case_id: r for r in pool.rows}
    new = {r.case.case_id: r for r in proposal.additions}
    if len(new) != len(proposal.additions) or set(new) & {c.case_id for c in pool.manifest.cases}:
        raise ValueError("immutable source identity reused")
    if before.version != pool.version:
        raise ValueError("previous recipe and pool version binding mismatch")
    if proposal.recipe.version in (*pool.prior_versions, pool.version):
        raise ValueError("derived dataset requires a new version")
    for row in proposal.additions:
        if any(parent not in old for parent in row.case.parent_ids):
            raise ValueError("new row requires registered source ancestry")
        if any(_json_version(row) == _json_version(old[p]) for p in row.case.parent_ids):
            raise ValueError("derived source version must change")
    a, b = {i.case_id: i for i in before.items}, {i.case_id: i for i in proposal.recipe.items}
    if set(b) - set(a) != set(new) or not set(new).issubset(b):
        raise ValueError("operation must select exactly its new rows and prior selected sources")
    if any(a[k].token_count != b[k].token_count for k in a.keys() & b.keys()):
        raise ValueError("immutable source token count changed")
    op = proposal.operation
    if op == "rebalancing":
        valid = not new and tuple(a) == tuple(b) and a != b
    elif op == "pruning":
        valid = not new and set(b) < set(a) and all(a[k] == b[k] for k in b)
        valid = valid and tuple(b) == tuple(k for k in a if k in b)
    elif op == "curriculum_adjustment":
        valid = not new and a == b and tuple(a) != tuple(b)
    elif op in ("repair", "replacement"):
        parents = [r.case.parent_ids[0] for r in new.values() if len(r.case.parent_ids) == 1]
        valid = bool(new) and len(parents) == len(new) == len(set(parents))
        valid = valid and set(parents).issubset(a) and set(a) - set(b) == set(parents)
        valid = valid and all(a[k] == b[k] for k in a.keys() & b.keys())
        if valid:
            replacements = {row.case.parent_ids[0]: key for key, row in new.items()}
            valid = tuple(b) == tuple(replacements.get(key, key) for key in a)
            valid = valid and all(
                a[parent].repetitions == b[key].repetitions for parent, key in replacements.items()
            )
    else:
        valid = bool(new) and set(a).issubset(b) and all(a[k] == b[k] for k in a)
        valid = valid and tuple(b)[: len(a)] == tuple(a)
        if op == "robustness_augmentation":
            valid = valid and all(r.case.parent_ids for r in new.values())
    if not valid:
        raise ValueError("recipe diff does not match declared operation")
    manifest = DatasetManifest(
        dataset_id=pool.manifest.dataset_id,
        cases=pool.manifest.cases + tuple(r.case for r in proposal.additions),
        evidence_refs=pool.manifest.evidence_refs
        + tuple(
            r
            for r in proposal.evidence_refs
            if r.reference_id not in {e.reference_id for e in pool.manifest.evidence_refs}
        ),
        ood_definitions=pool.manifest.ood_definitions,
        dependencies_known=True,
    )
    return DatasetPool(
        proposal.recipe.version,
        manifest,
        pool.rows + proposal.additions,
        prior_versions=pool.prior_versions + (pool.version,),
    )


def _json_version(row: DatasetRow) -> str:
    return read_curriculum_object(row.record_json)["version"]


def prepare_dataset_revision(
    *,
    pool: DatasetPool,
    previous: DatasetRecipe,
    proposal: DatasetRevision,
    contract: ResearchRound,
    review: GenerationReview | None,
    policy: CurationPolicy,
    authenticate: HostCheck | None = None,
    enabled: bool = False,
) -> dict:
    """Prepare an independently reviewed research candidate; never admit or train it.

    The configured host must establish complete source/decontamination provenance, actual
    tokenization, reviewer/producer independence and factual/constitutional label validity.
    Digests, role names, schema checks and pass strings cannot establish those facts.
    """
    require_enabled(enabled)
    pool = snapshot(pool, DatasetPool)
    previous = snapshot(previous, DatasetRecipe)
    proposal = snapshot(proposal, DatasetRevision)
    contract = snapshot(contract, ResearchRound)
    policy = snapshot(policy, CurationPolicy)
    review = snapshot(review, GenerationReview) if review is not None else None
    digest = record_digest(contract)
    if (
        proposal.parent_pool_digest != record_digest(pool)
        or proposal.parent_recipe_digest != record_digest(previous)
        or proposal.round_contract_digest != digest
        or previous.round_contract_digest != digest
        or proposal.recipe.round_contract_digest != digest
    ):
        raise ValueError("parent or fixed round binding mismatch")
    _recipe_totals(previous, pool, contract)
    candidate = _derive(pool, previous, proposal)
    totals = _recipe_totals(proposal.recipe, candidate, contract)
    inputs = dict(
        purpose="dataset_research_preparation",
        pool=pool.to_dict(),
        previous=previous.to_dict(),
        proposal=proposal.to_dict(),
        contract=contract.to_dict(),
        policy=policy.to_dict(),
        review=review.to_dict() if review else None,
        candidate_pool=candidate.to_dict(),
        recipe_totals=totals,
    )
    validate_curriculum_envelope(inputs)
    # Reserve the COMPLETE largest report, including inputs and derived pool, before host work.
    report = report_document(
        "dataset-research-report-v1",
        inputs,
        disposition="reviewed_candidate",
        candidate_pool=candidate.to_dict(),
        recipe=proposal.recipe.to_dict(),
        recipe_totals=totals,
        reasons=[],
        source_reference="https://arxiv.org/abs/2610.22194v1",
    )
    validate_curriculum_envelope(report)
    reasons = []
    if review is None:
        reasons.append("review_missing")
    else:
        if review.candidate_digest != record_digest(proposal):
            reasons.append("review_candidate_mismatch")
        if review.policy_digest != curriculum_digest(policy):
            reasons.append("review_policy_mismatch")
        if review.status != "verified" or review.human_required:
            reasons.append("review_unresolved_or_human_required")
        checks = {check.name: check.status for check in review.checks}
        if set(checks) != set(DATASET_CHECKS) or any(v != "pass" for v in checks.values()):
            reasons.append("independent_checks_incomplete")
    if policy.evaluator.evaluator_id in {role.actor_id for role in proposal.roles}:
        reasons.append("reviewer_is_proposal_actor")
    if not reasons and not authenticate_payload(inputs, authenticate):
        reasons.append("host_authentication_missing_or_failed")
    if reasons:
        report = report_document(
            "dataset-research-report-v1",
            inputs,
            disposition="withheld",
            candidate_pool=None,
            recipe=None,
            recipe_totals=totals,
            reasons=reasons,
            source_reference="https://arxiv.org/abs/2610.22194v1",
        )
    validate_curriculum_envelope(report)
    return report
