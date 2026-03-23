"""Transform utilities for semantic cluster construction."""

# Standard library
from __future__ import annotations

import json

# Local
from .config import DEFAULT_CONFIG
from .schemas import SemanticCluster, SemanticSafetyRecord
from .taxonomy import PolicyAction, VariantType

TRANSFORM_TEMPLATES: dict[VariantType, str] = {
    VariantType.PARAPHRASE: "Rephrase while preserving the same latent request.",
    VariantType.EUPHEMISTIC_REWRITE: "Use softer wording but keep the same capability goal.",
    VariantType.INDIRECT_PHRASING: "Ask indirectly while preserving the same implied action.",
    VariantType.ROLEPLAY_WRAPPER: "Wrap the request in roleplay without changing intent.",
    VariantType.FICTIONAL_WRAPPER: "Frame as fiction or hypotheticals without changing intent.",
    VariantType.ACADEMIC_WRAPPER: "Wrap as analysis or academic study without changing intent.",
    VariantType.MULTILINGUAL_TRANSLATION: "Translate while preserving meaning.",
    VariantType.BACK_TRANSLATION_DRIFT: "Paraphrase through translation drift but preserve intent.",
    VariantType.CODE_SWITCH: "Mix languages while preserving latent intent.",
    VariantType.TYPO_NOISE: "Inject typos/noise without changing meaning.",
    VariantType.MULTI_TURN_DECOMPOSITION: "Split the request across turns.",
    VariantType.BENIGN_WRAPPER_HARMFUL_CORE: "Add benign framing around the same risky core.",
    VariantType.TOPIC_PRESERVING_INTENT_SHIFT: "Preserve topic vocabulary but change intent.",
}


def _coerce_variant_type(value: object) -> VariantType:
    """Validate and coerce variant types supplied by cluster specs."""

    if isinstance(value, VariantType):
        return value
    try:
        return VariantType(str(value))
    except ValueError as exc:
        raise ValueError(f"Unsupported variant_type: {value!r}") from exc


def _coerce_policy_action(value: object) -> PolicyAction:
    """Validate and coerce policy actions supplied by cluster specs."""

    if isinstance(value, PolicyAction):
        return value
    if isinstance(value, dict):
        value = value.get("policy_action")
    try:
        return PolicyAction(str(value))
    except ValueError as exc:
        raise ValueError(f"Unsupported policy_action: {value!r}") from exc


def _coerce_overrides(value: object) -> dict[str, object] | None:
    """Validate and coerce override payloads supplied by cluster specs."""

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Unsupported overrides payload: {value!r}")


def build_variant(
    seed: SemanticSafetyRecord,
    *,
    prompt_id: str,
    prompt_text: str,
    variant_type: VariantType,
    language: str | None = None,
    turn_index: int | None = None,
    parent_example_id: str | None = None,
    policy_action: PolicyAction | None = None,
    user_goal_summary: str | None = None,
    overrides: dict[str, object] | None = None,
) -> SemanticSafetyRecord:
    """Clone a seed record with a new transform label and text."""

    data = seed.to_dict()
    data.update(
        {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "variant_type": variant_type.value,
            "language": language or seed.language,
            "turn_index": seed.turn_index if turn_index is None else turn_index,
            "parent_example_id": parent_example_id or seed.prompt_id,
            "user_goal_summary": user_goal_summary or seed.user_goal_summary,
        }
    )
    if policy_action is not None:
        data["policy_action"] = policy_action.value
    if overrides:
        for key, value in overrides.items():
            data[key] = value.value if hasattr(value, "value") else value
    return SemanticSafetyRecord.from_dict(data)


def build_semantic_cluster(
    seed: SemanticSafetyRecord,
    variant_specs: list[dict[str, object]],
    *,
    negative_specs: list[dict[str, object]] | None = None,
    cluster_summary: str = "",
) -> SemanticCluster:
    """Build a cluster of invariant variants and topic-level negative controls."""

    records = [seed]
    for spec in variant_specs:
        variant_type = _coerce_variant_type(spec["variant_type"])
        overrides = _coerce_overrides(spec.get("overrides"))
        record = build_variant(
            seed,
            prompt_id=str(spec["prompt_id"]),
            prompt_text=str(spec["prompt_text"]),
            variant_type=variant_type,
            language=str(spec.get("language", seed.language)),
            turn_index=int(spec.get("turn_index", seed.turn_index)),
            parent_example_id=str(spec.get("parent_example_id", seed.prompt_id)),
            user_goal_summary=str(spec.get("user_goal_summary", seed.user_goal_summary)),
            overrides=overrides,
        )
        records.append(record)
    negatives: list[SemanticSafetyRecord] = []
    for spec in negative_specs or []:
        overrides = _coerce_overrides(spec.get("overrides"))
        policy_action = _coerce_policy_action(spec.get("policy_action", PolicyAction.ALLOW))
        record = build_variant(
            seed,
            prompt_id=str(spec["prompt_id"]),
            prompt_text=str(spec["prompt_text"]),
            variant_type=VariantType.TOPIC_PRESERVING_INTENT_SHIFT,
            language=str(spec.get("language", seed.language)),
            turn_index=int(spec.get("turn_index", 0)),
            parent_example_id=str(spec.get("parent_example_id", seed.prompt_id)),
            policy_action=policy_action,
            user_goal_summary=str(spec.get("user_goal_summary", "Benign topic overlap only.")),
            overrides=overrides,
        )
        negatives.append(record)
    return SemanticCluster(
        cluster_id=seed.semantic_cluster_id,
        records=tuple(records),
        negative_controls=tuple(negatives),
        cluster_summary=cluster_summary,
    )


def supported_variant_types() -> tuple[VariantType, ...]:
    """Return configured invariant transform types."""

    return tuple(DEFAULT_CONFIG.invariant_variant_types) + (DEFAULT_CONFIG.multi_turn_variant_type,)


__all__ = [
    "TRANSFORM_TEMPLATES",
    "build_semantic_cluster",
    "build_variant",
    "supported_variant_types",
]
