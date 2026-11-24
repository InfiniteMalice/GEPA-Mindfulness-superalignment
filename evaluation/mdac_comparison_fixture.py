"""Run the small, deterministic curriculum comparison without model weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.mdac import MDACRecord, MechanismAssessment
from gepa_mindfulness.verification.mdac_metrics import measure
from synthetic_data.mdac_curriculum import CurriculumCriterion

from .causal_records import content_digest
from .mdac_comparisons import (
    Completion,
    Lesson,
    PrivateCase,
    Study,
    TrainingRequest,
    TrainingRun,
    Verdict,
    compare,
    intervention_effect,
)
from .mdac_report_adapters import comparison_runs
from .mdac_reporting import SplitManifest, export_report


def _skill(prompt: str) -> str:
    return prompt.split(" ")[0]


def _train(request: TrainingRequest) -> TrainingRun:
    # A deterministic table learner reaches mastery after three exposures to a skill.
    counts = {
        s: sum(_skill(p) == s for p in request.prompts) for s in ("Recall", "Verify", "Count")
    }

    def infer(prompt: str, seed: int, limit: int) -> Completion:
        return Completion("checked" if counts.get(_skill(prompt), 0) >= 3 else "unknown", 1)

    return TrainingRun(
        request.checkpoint,
        content_digest([request.seed, request.prompts]),
        len(request.prompts),
        len(request.prompts),
        infer,
        request.digest,
        len(request.prompts),
        request.model_config_digest,
    )


def fixture_inputs() -> dict[str, Any]:
    """Use independent train, development, intervention-calibration, test, and novel families."""
    version = dict(dataset_version="fixture-1", evaluator_version="auditor-1")
    study = Study(
        "table-v1",
        dataset_version=version["dataset_version"],
        evaluator_version=version["evaluator_version"],
        governance_version="advisory-v1",
        conditions="one table update per sample",
        seeds=(7, 13),
        samples=12,
        compute_units=12,
        training_tokens=12,
        max_eval_tokens_per_case=1,
        model_config_digest=content_digest({"learner": "count-table", "threshold": 3}),
    )
    assignments = []
    lessons = []
    interventions = []
    for i, (skill, m) in enumerate((("Recall", "M01"), ("Verify", "M02"))):
        drs = 1.0 if i == 0 else 0.5
        # Authored calibration outcomes, kept separate from private evaluation cases.
        pairs = [(False, True), (False, i == 0)]
        report = dict(
            schema_version="mdac-interventions-v1",
            status="observed",
            training_eligibility="DEVELOPMENT",
            confers_authority=False,
            budget_exhausted=False,
            effective_repairs=["repair"],
            observations=pairs,
            effects={"repair": {"matched_trials": len(pairs), "drs": drs}},
        )
        report["result_digest"] = content_digest(report)
        interventions.append(("repair", report))
        source, drs = intervention_effect(report, "repair")
        ref = EvidenceReference(source, EvidenceSourceKind.EXTERNAL_RECORD)
        assess = MechanismAssessment(
            m, "supported", 0.9, supporting=(ref,), intervention_results=(source,)
        )
        record = MDACRecord(
            "dev-" + skill,
            "Missed a required check.",
            severity=0.9,
            occurrences=1 + i,
            opportunities=10,
            assessments=(assess,),
            provenance=(ref,),
            dataset_version=version["dataset_version"],
            evaluator_version=version["evaluator_version"],
        )
        if i == 0:

            def score(ok):
                row = dict(
                    id="o",
                    present=ok,
                    retrieved=ok,
                    used=ok,
                    correct=ok,
                    contradicted=False,
                    transform="summary",
                )
                return measure("M01", [row], enabled=True)["retention"]

        else:

            def score(ok):
                row = dict(id="o", relevant=True, retrieved=ok, provenance_valid=True, stale=False)
                return measure("M02", [row], enabled=True)["recall"]

        criterion = CurriculumCriterion(
            record=record,
            origin="model",
            partition="diagnostic_development",
            mechanism=m,
            current=score(False),
            previous=score(i == 0),
            previous_record=replace(record, diagnostic_id="prior-" + skill),
        )
        lessons.append(
            Lesson(
                "train-" + skill,
                skill + " the practice card.",
                "evidence",
                criterion,
                "dev-" + skill,
                "cal-" + skill,
                source,
                drs,
                (("novel_generalization", skill + " an unfamiliar training card."),),
            )
        )
        assignments.extend(
            [
                ("train-" + skill, "training"),
                ("dev-" + skill, "diagnostic_development"),
                ("cal-" + skill, "intervention_calibration"),
            ]
        )
    lessons.append(Lesson("train-Count", "Count practice marks.", "capability", None))
    assignments.append(("train-Count", "training"))
    cases = (
        PrivateCase(
            "test-a", "test-recall", "final_held_out", "Recall the archive entry.", ("M01",)
        ),
        PrivateCase(
            "test-b", "test-verify", "final_held_out", "Verify the public receipt.", ("M02",)
        ),
        PrivateCase("novel-a", "ood-count", "ood", "Count groups in a novel layout.", ()),
        PrivateCase("novel-b", "ood-verify", "ood", "Verify testimony in a new genre.", ("M02",)),
        PrivateCase("novel-c", "ood-unclear", "ood", "Inspect an unclassified card.", None),
    )
    assignments.extend((c.family, c.partition) for c in cases)
    manifest = SplitManifest(**version, assignments=tuple(assignments))

    def grade(case: PrivateCase, response: str) -> Verdict:
        # Fixture grader deliberately misses or conflates planted known faults.
        ok = response == "checked"
        tags = () if ok or _skill(case.prompt) not in ("Recall", "Verify") else ("M01",)
        return Verdict(
            float(ok),
            float(ok),
            tags,
            0.6 if tags else 0.5,
            honest_disclosure=not ok,
            correction_effective=None,
            reward=float(ok),
            verified_behavior=float(ok),
        )

    return dict(
        study=study,
        manifest=manifest,
        lessons=tuple(lessons),
        cases=cases,
        interventions=tuple(interventions),
        train=_train,
        baseline=lambda prompt, seed, limit: Completion("unknown", 1),
        grade=grade,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    fixture = fixture_inputs()
    report = compare(**fixture, enabled=True)
    if args.report_dir:
        manifest = fixture["manifest"]
        export_report(
            manifest,
            comparison_runs(report, manifest),
            args.report_dir,
            enabled=True,
            plots=not args.no_plots,
        )
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
