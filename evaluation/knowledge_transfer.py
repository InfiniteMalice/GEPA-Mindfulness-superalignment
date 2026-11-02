"""Default-off text-proxy transfer evaluation over independently supplied saved answers."""

# Standard library
from __future__ import annotations

from collections import defaultdict
from math import fsum

# Third-party
# Local
from gepa_mindfulness.verification.debate_records import _record
from gepa_mindfulness.verification.diagnostic_records import choice

from .causal_records import canonical_json, content_digest
from .improvement_records import MetricSpec
from .improvement_statistics import PairedObservation, estimate_paired
from .knowledge_transfer_records import (
    CHANNELS,
    MAX_CELLS,
    MAX_UNITS,
    NORMALIZATION,
    TransferCondition,
    TransferObservation,
    TransferProbeConfig,
    TransferUnit,
    bounded,
    identifiers,
    identity_text,
    normalize_answer,
)


def _records(value: object, cls: type, maximum: int, *, required: bool = True) -> tuple:
    """Bound exact record arrays before evaluating or joining any source data."""
    if type(value) is not tuple or len(value) > maximum or (required and not value):
        raise ValueError("bounded exact record tuple required")
    return tuple(_record(v, cls) for v in value)


def public_transfer_prompt(
    unit: TransferUnit, channel: str, *, enabled: bool = False
) -> str | None:
    """Project only the authored public question; do not expose evaluator metadata or truth."""
    if type(enabled) is not bool:
        raise ValueError("explicit boolean enabled required")
    if not enabled:
        return None
    unit = _record(unit, TransferUnit)
    choice(channel, "channel", CHANNELS)
    return getattr(unit, f"{channel}_question")


def _summary(rows: list[dict]) -> dict:
    """Keep observed answer accuracy separate from known correctness over planned cells."""
    answered = sum(r["status"] == "answered" for r in rows)
    correct = sum(r["correct"] is True for r in rows)
    return dict(
        planned=len(rows),
        answered=answered,
        correct=correct,
        wrong=answered - correct,
        abstained=sum(r["status"] == "abstained" for r in rows),
        errors=sum(r["status"] == "error" for r in rows),
        missing=sum(r["status"] == "missing" for r in rows),
        correct_attainment=correct / len(rows),
        accuracy_when_answered=None if answered == 0 else correct / answered,
    )


def _access_pairs(rows: list[dict], units: tuple, repeats: tuple) -> dict:
    """Compute answered-pair contingencies without treating missing answers as false beliefs."""
    indexed = {(r["unit_id"], r["repeat_id"], r["channel"]): r for r in rows}
    table = dict(
        source_correct_target_correct=0,
        source_correct_target_wrong=0,
        source_wrong_target_correct=0,
        source_wrong_target_wrong=0,
    )
    answered = source_correct = target_correct = 0
    for unit in units:
        for repeat in repeats:
            source = indexed[(unit.unit_id, repeat, "source")]
            target = indexed[(unit.unit_id, repeat, "target")]
            source_correct += int(source["correct"] is True)
            target_correct += int(target["correct"] is True)
            if source["status"] != "answered" or target["status"] != "answered":
                continue
            answered += 1
            left = "correct" if source["correct"] else "wrong"
            right = "correct" if target["correct"] else "wrong"
            table[f"source_{left}_target_{right}"] += 1
    correct_source = table["source_correct_target_correct"] + table["source_correct_target_wrong"]
    wrong_source = table["source_wrong_target_correct"] + table["source_wrong_target_wrong"]
    planned = len(units) * len(repeats)
    return dict(
        **table,
        planned=planned,
        both_answered=answered,
        unpaired=planned - answered,
        source_target_attainment_gap=(source_correct - target_correct) / planned,
        target_accuracy_given_source_correct=(
            None if not correct_source else table["source_correct_target_correct"] / correct_source
        ),
        target_accuracy_given_source_wrong=(
            None if not wrong_source else table["source_wrong_target_correct"] / wrong_source
        ),
    )


def _costs(observations: tuple) -> list[dict]:
    """Aggregate only identical receipt units/bases/currencies; unknown costs remain unknown."""
    groups = defaultdict(list)
    for row in observations:
        for cost in row.costs:
            groups[(row.condition_id, cost.unit, cost.basis, cost.currency)].append(cost.value)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: repr(item[0])):
        known = [v for v in values if v is not None]
        subtotal = fsum(known) if known else None
        result.append(
            dict(
                condition_id=key[0],
                unit=key[1],
                basis=key[2],
                currency=key[3],
                scope="provided_entries_only",
                total=subtotal if len(known) == len(values) else None,
                known_subtotal=subtotal,
                known_values=len(known),
                unknown_values=values.count(None),
            )
        )
    return result


def evaluate_knowledge_transfer(
    units: tuple[TransferUnit, ...],
    conditions: tuple[TransferCondition, ...],
    observations: tuple[TransferObservation, ...],
    *,
    repeat_ids: tuple[str, ...],
    baseline_condition_id: str,
    evaluator_version: str,
    config: TransferProbeConfig = TransferProbeConfig(),
) -> dict | None:
    """Evaluate a sealed text-proxy roster and saved public answers, with no model execution.

    Unit truth, exposure histories, final-answer extraction and artifact authenticity are host
    responsibilities. A correct answer cannot prove acquisition or authorize optimizer credit.
    Internal test units are held out of target-QA alignment; separately admitted source facts
    may intentionally concern them. This evaluator never admits or trains on either packet.
    """
    config = _record(config, TransferProbeConfig)
    if not config.enabled:
        return None
    units = _records(units, TransferUnit, MAX_UNITS)
    conditions = _records(conditions, TransferCondition, 4)
    observations = _records(observations, TransferObservation, MAX_CELLS, required=False)
    repeats = identifiers(repeat_ids, "repeat_ids", 8)
    if not repeats or len(units) * len(conditions) * len(repeats) * len(CHANNELS) > MAX_CELLS:
        raise ValueError("nonempty bounded planned cell roster required")
    bounded(baseline_condition_id, "baseline_condition_id", 128)
    bounded(evaluator_version, "evaluator_version", 128)
    unit_map = {u.unit_id: u for u in units}
    condition_map = {c.condition_id: c for c in conditions}
    if len(unit_map) != len(units) or len(condition_map) != len(conditions):
        raise ValueError("duplicate unit or condition ID")
    if len({(identity_text(u.entity_id), identity_text(u.attribute)) for u in units}) != len(units):
        raise ValueError("duplicate entity/attribute fact")
    spellings = defaultdict(set)
    for unit in units:
        spellings[identity_text(unit.entity_id)].add(unit.entity_id)
    if any(len(values) != 1 for values in spellings.values()):
        raise ValueError("ambiguous entity spelling aliases")
    cohorts = defaultdict(set)
    for unit in units:
        cohorts[unit.entity_id].add(unit.cohort)
    if any(len(values) != 1 for values in cohorts.values()):
        raise ValueError("entity cannot cross alignment/test cohorts")
    baseline = condition_map.get(baseline_condition_id)
    if (
        baseline is None
        or baseline.checkpoint_digest != baseline.base_checkpoint_digest
        or (baseline.acquired_unit_ids or baseline.alignment_entities)
    ):
        raise ValueError("unexposed declared base checkpoint required for baseline")
    for condition in conditions:
        if (
            condition.base_checkpoint_digest,
            condition.source_representation,
            condition.target_representation,
        ) != (
            baseline.base_checkpoint_digest,
            baseline.source_representation,
            baseline.target_representation,
        ):
            raise ValueError("common declared base and representation pair required")
        if set(condition.acquired_unit_ids) - set(unit_map):
            raise ValueError("unknown acquired unit")
        if any(cohorts.get(entity) != {"alignment"} for entity in condition.alignment_entities):
            raise ValueError("QA alignment requires known alignment-only entities")
    units = tuple(sorted(units, key=lambda u: u.unit_id))
    conditions = tuple(sorted(conditions, key=lambda c: c.condition_id))
    indexed = {}
    for observation in observations:
        key = (
            observation.condition_id,
            observation.unit_id,
            observation.repeat_id,
            observation.channel,
        )
        if (
            observation.condition_id not in condition_map
            or observation.unit_id not in unit_map
            or observation.repeat_id not in repeats
            or key in indexed
        ):
            raise ValueError("duplicate or unplanned observation cell")
        if observation.system != condition_map[observation.condition_id].system:
            raise ValueError("saved observation system does not match condition")
        indexed[key] = observation
    inputs = dict(
        units=[u.to_dict() for u in units],
        conditions=[c.to_dict() for c in conditions],
        observations=[indexed[k].to_dict() for k in sorted(indexed)],
        repeat_ids=list(repeats),
        baseline_condition_id=baseline_condition_id,
        evaluator_version=evaluator_version,
        config=config.to_dict(),
    )
    if len(canonical_json(inputs).encode("utf-8")) > 2 * 1048576:
        raise ValueError("complete input capacity exceeded")
    rows = []
    for condition in conditions:
        for unit in units:
            for repeat in repeats:
                for channel in CHANNELS:
                    key = (condition.condition_id, unit.unit_id, repeat, channel)
                    capture = indexed.get(key)
                    expected = unit.application_answer if channel == "application" else unit.value
                    correct = None
                    if capture is not None and capture.status == "answered":
                        correct = normalize_answer(capture.final_answer) == normalize_answer(
                            expected
                        )
                    rows.append(
                        dict(
                            condition_id=condition.condition_id,
                            unit_id=unit.unit_id,
                            entity_id=unit.entity_id,
                            attribute=unit.attribute,
                            cohort=unit.cohort,
                            repeat_id=repeat,
                            channel=channel,
                            correct=correct,
                            status="missing" if capture is None else capture.status,
                            final_answer=None if capture is None else capture.final_answer,
                        )
                    )
    summaries, by_attribute, access, comparisons = [], [], [], []
    row_map = {(r["condition_id"], r["unit_id"], r["repeat_id"], r["channel"]): r for r in rows}
    study_digest = content_digest(inputs)
    for condition in conditions:
        for cohort in sorted({u.cohort for u in units}):
            subset = tuple(u for u in units if u.cohort == cohort)
            cohort_rows = [
                r
                for r in rows
                if r["condition_id"] == condition.condition_id and r["cohort"] == cohort
            ]
            access.append(
                dict(
                    condition_id=condition.condition_id,
                    cohort=cohort,
                    **_access_pairs(cohort_rows, subset, repeats),
                )
            )
            for channel in CHANNELS:
                stratum = [r for r in cohort_rows if r["channel"] == channel]
                summaries.append(
                    dict(
                        condition_id=condition.condition_id,
                        cohort=cohort,
                        channel=channel,
                        **_summary(stratum),
                    )
                )
                for attribute in sorted({u.attribute for u in subset}):
                    by_attribute.append(
                        dict(
                            condition_id=condition.condition_id,
                            cohort=cohort,
                            channel=channel,
                            attribute=attribute,
                            **_summary([r for r in stratum if r["attribute"] == attribute]),
                        )
                    )
                if condition.condition_id == baseline_condition_id:
                    continue
                metric = MetricSpec(
                    "recorded_correct_attainment",
                    "knowledge-transfer-report-v1",
                    channel,
                    content_digest(NORMALIZATION),
                )
                expected_ids, matched = [], []
                stratum_digest = content_digest(
                    (study_digest, condition.condition_id, cohort, channel)
                )
                for unit in subset:
                    for repeat_index, repeat in enumerate(repeats):
                        pair_id = content_digest(
                            (condition.condition_id, unit.unit_id, repeat, channel)
                        )
                        expected_ids.append(pair_id)
                        left = row_map[(baseline_condition_id, unit.unit_id, repeat, channel)]
                        right = row_map[(condition.condition_id, unit.unit_id, repeat, channel)]
                        if left["status"] != "missing" and right["status"] != "missing":
                            matched.append(
                                PairedObservation(
                                    pair_id,
                                    unit.unit_id,
                                    condition.condition_id,
                                    repeat_index,
                                    unit.entity_id,
                                    stratum_digest,
                                    metric,
                                    float(left["correct"] is True),
                                    float(right["correct"] is True),
                                )
                            )
                estimate = estimate_paired(
                    tuple(matched),
                    expected_pair_ids=tuple(expected_ids),
                    policy=config.resampling_policy,
                    dependencies_known=config.dependencies_known,
                )
                comparisons.append(
                    dict(
                        condition_id=condition.condition_id,
                        cohort=cohort,
                        channel=channel,
                        estimate=estimate.to_dict(),
                    )
                )
    result = dict(
        schema_version="knowledge-transfer-report-v1",
        mode="text_proxy",
        training_eligibility="DEVELOPMENT",
        optimizer_eligible=False,
        authority_granted=False,
        knowledge_acquisition_verified=False,
        model_effectiveness_established=False,
        source_reference="REF-KTP",
        source_binding="host_declared_not_authenticated",
        normalization=NORMALIZATION,
        inputs=inputs,
        study_digest=study_digest,
        rows=rows,
        summaries=summaries,
        by_attribute=by_attribute,
        access_pairs=access,
        comparisons=comparisons,
        costs=_costs(observations),
    )
    if len(canonical_json(result).encode("utf-8")) > 4 * 1048576:
        raise ValueError("complete output capacity exceeded")
    return result
