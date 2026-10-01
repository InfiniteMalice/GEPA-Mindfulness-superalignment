"""Opt-in, public-answer contrastive training over existing CPT pair records."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cognitive_pairwise_training.schemas import PairwiseLabel, PairwiseReasoningExample
from mindful_trace_gepa._json_values import freeze_json_mapping, thaw_json_mapping

from .eligibility import TrainingEligibility, require_training_eligible


class NegativeFamily(str, Enum):
    """The fixed PR-12 exposure order; these are experiment strata, not new cases."""

    BROAD = "broad_semantics"
    SEMANTIC = "semantic_hard"
    CAUSAL = "causal_hard"
    LAUNDERING = "laundering_hard"
    PEO = "peo_trajectory"


@dataclass(frozen=True)
class PreparedPair:
    """Immutable public text projection; never pass this labeled record to a scorer."""

    pair_id: str
    problem_id: str
    source_group: str
    family: NegativeFamily
    eligibility: TrainingEligibility
    prompt: str
    chosen: str
    rejected: str
    fingerprint: str
    source_digest: str


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_pairs(
    examples: Iterable[PairwiseReasoningExample], *, for_training: bool
) -> tuple[PreparedPair, ...]:
    """Validate the complete catalog and snapshot public answers before any callback.

    Training requires explicit TRAIN and the existing recursive provenance checks.
    Evaluation requires an explicit non-TRAIN split. CPT reasoning/confidence fields
    are retained for admission validation only, never included in model inputs or loss.

    Args:
        examples: Existing CPT records with family, source-group and admission metadata.
        for_training: Require TRAIN when true; require a non-TRAIN split when false.

    Returns:
        A nonempty tuple of immutable public projections and source digests.

    Raises:
        ValueError: A record is malformed, duplicated, ambiguous or ineligible.
    """
    if type(for_training) is not bool:
        raise ValueError("for_training must be boolean")
    prepared = []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for example in examples:
        if type(example) is not PairwiseReasoningExample:
            raise ValueError("expected a CPT PairwiseReasoningExample")
        # CPT is shallow-frozen. Snapshot all provenance before accepting any input.
        record = freeze_json_mapping(example.to_dict(), field_name="contrastive pair")
        metadata = record["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("pair metadata must be a mapping")
        try:
            family = NegativeFamily(metadata.get("negative_family"))
            eligibility = TrainingEligibility(metadata.get("training_eligibility"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "pair requires canonical negative_family and training_eligibility"
            ) from error
        if for_training:
            require_training_eligible(record)
        elif eligibility is TrainingEligibility.TRAIN:
            raise ValueError("evaluation requires a non-TRAIN split")
        pair_id = _text(example.pair_id, "pair_id")
        problem_id = _text(example.problem_id, "problem_id")
        group = _text(metadata.get("source_group"), "source_group")
        a, b = example.candidate_a, example.candidate_b
        if a.problem_id != problem_id or b.problem_id != problem_id or a.prompt != b.prompt:
            raise ValueError("candidates must share the pair problem and prompt")
        prompt = _text(a.prompt, "prompt")
        answers = (_text(a.final_answer, "answer"), _text(b.final_answer, "answer"))
        if " ".join(answers[0].split()) == " ".join(answers[1].split()):
            raise ValueError("candidate answers must differ")
        if example.teacher_label is PairwiseLabel.A_MORE_TRUSTWORTHY:
            chosen, rejected = answers
        elif example.teacher_label is PairwiseLabel.B_MORE_TRUSTWORTHY:
            rejected, chosen = answers
        else:
            raise ValueError("contrastive training requires a strict preference")
        fingerprint = _digest(
            [" ".join(prompt.split()), sorted(" ".join(a.split()) for a in answers)]
        )
        if pair_id in ids or fingerprint in fingerprints:
            raise ValueError("duplicate pair identity or public content")
        ids.add(pair_id)
        fingerprints.add(fingerprint)
        prepared.append(
            PreparedPair(
                pair_id,
                problem_id,
                group,
                family,
                eligibility,
                prompt,
                chosen,
                rejected,
                fingerprint,
                _digest(thaw_json_mapping(record)),
            )
        )
    if not prepared:
        raise ValueError("contrastive catalog must not be empty")
    return tuple(prepared)


def train_contrastive(
    examples: Iterable[PairwiseReasoningExample],
    score: Callable[[str, tuple[str, str]], Any],
    optimizer: Any,
    *,
    epochs: int = 1,
    schedule: str = "curriculum",
    seed: int = 0,
    enabled: bool = False,
) -> dict[str, Any]:
    """Optimize a caller-owned scorer using two-candidate cross entropy.

    The scorer returns two differentiable torch logits in supplied answer order.
    Both schedules visit each record `epochs` times. Curriculum completes each
    family before the next; pooled shuffles all families together. The caller owns
    scorer mode, initialization, optimizer, and checkpoint persistence. A failure
    stops training; caller-owned optimizer/model mutations are not rolled back.

    Args:
        examples: Admitted CPT pairs covering all five negative families.
        score: Callback returning two differentiable logits for the supplied answers.
        optimizer: Torch optimizer owning the parameters updated by the callback.
        epochs: Number of visits to each record, from 1 through 1000.
        schedule: Ordered family curriculum or pooled shuffle control.
        seed: Local presentation/shuffle seed in [0, 2**32).
        enabled: Explicit opt-in; only literal True enables training.

    Returns:
        Source-bound dataset identity, exposure counts, family order and update losses.

    Raises:
        ValueError: Admission, configuration, logits, gradients or parameters are invalid.
        ImportError: The optional torch training dependency is unavailable.
    """
    if enabled is not True:
        raise ValueError("contrastive experiments require enabled=True")
    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("epochs must be an integer in [1, 1000]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if schedule not in ("curriculum", "pooled"):
        raise ValueError("schedule must be curriculum or pooled")
    pairs = prepare_pairs(examples, for_training=True)
    if {p.family for p in pairs} != set(NegativeFamily):
        raise ValueError("training requires all five families")
    import torch

    if not callable(score) or not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("training requires a scorer and torch optimizer")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if not parameters or any(not torch.isfinite(p).all().item() for p in parameters):
        raise ValueError("optimizer parameters must be finite")
    rng = random.Random(seed)
    batches = (
        [tuple(p for p in pairs if p.family is family) for family in NegativeFamily]
        if schedule == "curriculum"
        else [pairs]
    )
    counts = dict.fromkeys((f.value for f in NegativeFamily), 0)
    losses: list[float] = []
    family_order: list[str] = []
    for batch in batches:
        for epoch in range(epochs):
            order = list(batch)
            rng.shuffle(order)
            for pair in order:
                # Answer order is independent of curriculum scheduling and label position.
                reverse = int(_digest([seed, epoch, pair.pair_id])[:8], 16) % 2 == 1
                answers = (pair.rejected, pair.chosen) if reverse else (pair.chosen, pair.rejected)
                optimizer.zero_grad(set_to_none=True)
                logits = score(pair.prompt, answers)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.shape != (2,)
                    or not logits.is_floating_point()
                    or not logits.requires_grad
                    or not torch.isfinite(logits).all().item()
                ):
                    raise ValueError("scorer must return two finite differentiable floating logits")
                chosen_index = int(reverse)
                loss = torch.nn.functional.softplus(logits[1 - chosen_index] - logits[chosen_index])
                if not torch.isfinite(loss).item():
                    raise ValueError("contrastive loss must be finite")
                loss.backward()
                gradients = [
                    p.grad
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                    optimizer.zero_grad(set_to_none=True)
                    raise ValueError("optimizer must receive finite gradients")
                optimizer.step()
                if any(not torch.isfinite(p).all().item() for p in parameters):
                    raise ValueError(
                        "optimizer produced nonfinite parameters; restore the checkpoint"
                    )
                counts[pair.family.value] += 1
                family_order.append(pair.family.value)
                losses.append(float(loss.detach().item()))
    return dict(
        schema_version="contrastive-training-v1",
        schedule=schedule,
        seed=seed,
        epochs=epochs,
        dataset_digest=_digest(sorted((p.pair_id, p.source_digest) for p in pairs)),
        source_groups=sorted({p.source_group for p in pairs}),
        updates_by_family=counts,
        family_order=family_order,
        losses=losses,
    )
