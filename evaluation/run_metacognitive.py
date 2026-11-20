"""Explicit opt-in CPU demonstrations and structural battery; never a control executor."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import TextIO

from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.verification.metacognitive_control import ControlBudget, propose_control
from gepa_mindfulness.verification.metacognitive_peo import monitor_recorded_episode

from .metacognitive import evaluate_metacognitive
from .metacognitive_fixtures import CHALLENGES, make_challenge
from .v5_runner import V5EvaluationCell, plan_v5_cells

_DISABLED = {"enabled": False, "status": "DISABLED"}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate JSON fields are not allowed")
    return dict(pairs)


def run_demo(spec: object = None, *, enabled: bool = False) -> dict:
    """Load a strictly declared synthetic challenge; default off before inspecting input."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    if not enabled:
        return dict(_DISABLED)
    required = {
        "schema_version",
        "challenge",
        "case_id",
        "stripe_id",
        "repeat_id",
        "base_seed",
        "model_version",
        "harness_version",
    }
    if type(spec) is not dict or set(spec) != required:
        raise ValueError("demo specification must contain exactly the declared schema fields")
    if spec["schema_version"] != "metacognitive-demo-v1":
        raise ValueError("unsupported demo schema")
    if type(spec["repeat_id"]) is not int or not 0 <= spec["repeat_id"] < 20:
        raise ValueError("demo repeat_id must be an exact integer in 0..19")
    cells = plan_v5_cells(
        case_ids=[spec["case_id"]],
        stripe_ids=[spec["stripe_id"]],
        repeats=spec["repeat_id"] + 1,
        base_seed=spec["base_seed"],
        model_version=spec["model_version"],
        harness_version=spec["harness_version"],
    )
    fixture = make_challenge(spec["challenge"], cells[-1])
    episode = fixture.episode
    before = content_digest(episode.inputs.sources.to_dict())
    report = monitor_recorded_episode(episode, enabled=True, confidence_threshold=0.8)
    assert report is not None
    proposal = propose_control(
        report.assessment,
        budget=ControlBudget("preserve-root-objective", 1),
        enabled=True,
    )
    assert proposal is not None
    after = content_digest(episode.inputs.sources.to_dict())
    return dict(
        schema_version="metacognitive-demo-result-v1",
        enabled=True,
        training_eligibility="REGRESSION",
        challenge=fixture.name,
        cell=asdict(cells[-1]),
        report=report.to_dict(),
        proposal=proposal.to_dict(),
        source_digest_before=before,
        source_digest_after=after,
        reward_before=None,
        reward_after=None,
        invariants=dict(
            accepted_evidence_unchanged=before == after,
            reward_unchanged=True,
            authority_granted=False,
            execution_performed=False,
        ),
        final_behavior_status="NOT_RUN",
    )


def run_battery(
    cells: tuple[V5EvaluationCell, ...] | None = None,
    *,
    enabled: bool = False,
    output: TextIO | None = None,
) -> dict:
    """Stream per-unit structural observations and summarize them by canonical coordinates."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    if not enabled:
        return dict(_DISABLED)
    if type(cells) is not tuple or not 1 <= len(cells) <= 900:
        raise ValueError("provide 1..900 V5 cells (at most 9900 challenge/control units)")
    if any(type(cell) is not V5EvaluationCell for cell in cells) or len(set(cells)) != len(cells):
        raise ValueError("expected unique exact V5 cells")
    counts: Counter = Counter()
    groups: dict[str, Counter] = {"challenge": Counter(), "case": Counter(), "stripe": Counter()}
    matches: dict[str, Counter] = {key: Counter() for key in groups}
    digest = sha256()
    for name in (*CHALLENGES, "healthy"):
        for cell in cells:
            fixture = make_challenge(
                name, cell, split="HELD_OUT" if cell.repeat_id else "DEVELOPMENT"
            )
            evaluation = evaluate_metacognitive((fixture,), enabled=True)
            assert evaluation is not None
            result = evaluation.to_dict()
            row = result["rows"][2]
            matched = bool(row["monitoring_match"] and row["strategy_match"])
            counts["units"] += 1
            counts["monitoring_matches"] += int(row["monitoring_match"])
            counts["strategy_matches"] += int(row["strategy_match"])
            for group, key in (
                ("challenge", name),
                ("case", str(cell.case_id)),
                ("stripe", cell.stripe_id),
            ):
                groups[group][key] += 1
                matches[group][key] += int(matched)
            raw = dict(
                schema_version="metacognitive-structural-unit-v1",
                challenge=name,
                cell=asdict(cell),
                split=fixture.split,
                training_eligibility="REGRESSION",
                source_digest=row["original_source_digest"],
                findings=sorted({f["kind"] for f in row["assessment"]["findings"]}),
                proposal=row["proposal"]["kind"],
                conditions=[
                    {
                        k: arm[k]
                        for k in (
                            "condition",
                            "monitoring_match",
                            "strategy_match",
                            "execution_status",
                            "final_behavior_status",
                            "resource_limit_steps",
                        )
                    }
                    for arm in result["rows"]
                ],
                metrics=result["metrics"],
                ablations=result["ablations"],
                authority_granted=False,
                reward_changed=False,
                accepted_evidence_changed=False,
            )
            line = canonical_json(raw) + "\n"
            digest.update(line.encode("utf-8"))
            if output is not None:
                output.write(line)
    return dict(
        schema_version="metacognitive-battery-summary-v1",
        enabled=True,
        training_eligibility="REGRESSION",
        unit_count=counts["units"],
        arm_count=counts["units"] * 4,
        cells=len(cells),
        monitoring_matches=counts["monitoring_matches"],
        strategy_matches=counts["strategy_matches"],
        by_group={
            g: {
                key: dict(units=n, both_matches=matches[g][key])
                for key, n in sorted(values.items())
            }
            for g, values in groups.items()
        },
        raw_sha256=digest.hexdigest(),
        raw_written=output is not None,
        final_behavior_status="NOT_RUN",
        verified_task_success=None,
        condition_d="SIMULATION_ONLY",
        authority_granted=False,
        reward_changed=False,
        limitation="Synthetic structural checks only; no trained-model improvement measured",
    )


def main(argv: list[str] | None = None) -> int:
    """Print deterministic JSON; input files are never read unless --enable is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", choices=("unsupported", "healthy"), default=None)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--battery", action="store_true")
    parser.add_argument("--case-id", type=int, action="append")
    parser.add_argument("--stripe-id", action="append")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, help="New JSONL output path for --battery")
    args = parser.parse_args(argv)
    if not args.enable:
        print(canonical_json(_DISABLED))
        return 0
    try:
        if args.battery:
            cells = plan_v5_cells(
                case_ids=args.case_id,
                stripe_ids=args.stripe_id,
                repeats=args.repeats,
                model_version="synthetic-model-v1",
                harness_version="synthetic-host-v1",
            )
            if args.output:
                with args.output.open("x", encoding="utf-8") as stream:
                    result = run_battery(cells, enabled=True, output=stream)
            else:
                result = run_battery(cells, enabled=True)
        else:
            if args.output or args.case_id or args.stripe_id or args.repeats != 2:
                raise ValueError("battery output/selectors require --battery")
            if args.input:
                if args.input.stat().st_size > 16_384:
                    raise ValueError("demo specification exceeds 16 KiB")
                spec = json.loads(
                    args.input.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
                )
            else:
                name = args.demo or "unsupported"
                spec = json.loads(
                    files("gepa_mindfulness")
                    .joinpath(f"data/metacognitive_{name}_demo.json")
                    .read_text()
                )
            result = run_demo(spec, enabled=True)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
