"""Opt-in scripted behavior and actual frozen toy-model research bridge, not alignment evidence."""

# Standard library
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Local
from docs.experiments.phase2_pr6_causal_probe import build, plan_for
from evaluation.causal_diagnostics import protocol_digest
from evaluation.causal_records import PairAdjudication, capture_digest
from evaluation.cva_benchmark import BenchmarkCapture, BenchmarkResponse, ResponseClaim
from evaluation.mechanistic_studies import (
    bind_alignment_studies,
    capture_study_behavior,
    cva_study_slots,
    instrument_study_pair,
    report_alignment_studies,
)
from evaluation.mechanistic_study_records import PublicBehavior, StudyRoles
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.interpret.causal_interventions import freeze_causal_plan
from gepa_mindfulness.interpret.mechanistic_evidence import (
    MechanisticVerifierContext,
    verify_causal_runs,
)
from gepa_mindfulness.interpret.native_causal_records import InterventionMode
from gepa_mindfulness.interpret.native_neurons import _state_digest
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit
from synthetic_data.alignment_studies import alignment_study_templates
from synthetic_data.cva_fixtures import make_security_fixtures, qualify_security_fixture


class Clock:
    def now(self) -> datetime:
        return datetime(2026, 11, 5, tzinfo=timezone.utc)


def roles() -> StudyRoles:
    return StudyRoles(
        "synthetic-discovery",
        "synthetic-prediction",
        "frozen-executor",
        "synthetic-verifier",
        "numeric-controls-skeptic",
        "registrar-not-run",
    )


def scripted_study(templates):
    cells = plan_v5_cells(
        case_ids=(1,),
        stripe_ids=("NONE",),
        repeats=len(templates),
        base_seed=71,
        model_version="constant-script-no-language-model",
        harness_version="uniform-uncertainty-v1",
    )
    plan = bind_alignment_studies(
        templates,
        tuple((c, c) for c in cells),
        plan_id="scripted-study",
        roles=roles(),
        enabled=True,
    )
    _, _, joins = cva_study_slots(plan, enabled=True)
    slots = {variant: slot for _, variant, slot in joins}

    def actor(view):
        refs = (
            EvidenceReference(
                "scripted-capture:" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION
            ),
        )
        response = BenchmarkResponse(
            ResponseClaim.UNRESOLVED,
            None,
            (),
            (),
            None,
            "No result established by this constant script.",
        )
        structured = None
        if view.variant_id in slots:
            structured = BenchmarkCapture(slots[view.variant_id], response, (), refs)
        # The script never receives or reads the private authored expectations.
        return PublicBehavior(
            ("report_uncertainty",),
            json.dumps(response.to_dict()) if structured else response.message,
            refs,
            cva_capture=structured,
        )

    batch = capture_study_behavior(plan, actor, clock=Clock(), enabled=True)
    roster = {c.variant_id: c for c in batch.captures}
    judgments = []
    for item in plan.items:
        pair = item.pair
        before, after = roster[pair.before.variant_id], roster[pair.after.variant_id]
        required = pair.before.expected_actions != pair.after.expected_actions
        own = (before, after)
        judgments.append(
            PairAdjudication(
                pair.digest,
                capture_digest(own),
                protocol_digest(plan.plan_id, tuple(i.pair for i in plan.items), ()),
                TrustedEvaluatorContract(
                    "synthetic-label-checker", "1", "exact-authored-action-labels"
                ),
                "verified",
                "relevant" if required else "irrelevant",
                before.actions == pair.before.expected_actions,
                after.actions == pair.after.expected_actions,
                before.actions != after.actions,
                required,
                after.actions == pair.after.expected_actions if required else None,
                None,
                False,
                "Software fixture: exact normalized label comparison, not semantic review.",
                (
                    EvidenceReference(
                        "scripted-judgment:" + pair.pair_id, EvidenceSourceKind.EXTERNAL_RECORD
                    ),
                ),
                (),
            )
        )
    accepted = [j.to_dict() for j in judgments]
    batch_digest = batch.digest
    return report_alignment_studies(
        plan,
        batch,
        tuple(judgments),
        opportunities=(),
        authenticate=lambda j: j.to_dict() in accepted,
        authenticate_cva=lambda packet: packet.behavior_digest == batch_digest,
        enabled=True,
    )


def native_study(template):
    import torch

    source, native, kind = build("gpt2", constructed=True)
    original = _state_digest(source)
    causal = plan_for(native, kind, InterventionMode.ACTIVATION_PATCH, constructed=True)
    cell = plan_v5_cells(
        case_ids=(1,),
        stripe_ids=("NONE",),
        repeats=1,
        base_seed=71,
        model_version="engineered-writer-reader-software-control",
        harness_version="two-prompt-lookup-tokenizer-v1",
    )[0]
    plan = bind_alignment_studies(
        (template,),
        ((cell, cell),),
        plan_id="native-software-study",
        roles=roles(),
        condition="diagnostic_instrumented",
        physical_model_digest=native.identity.digest,
        enabled=True,
    )
    item = plan.items[0]
    token_map = {
        item.pair.before.prompt_digest: causal.donor,
        item.pair.after.prompt_digest: causal.recipient,
    }

    def actor(prefix):
        def capture(view):
            context = token_map[view.prompt_digest]
            with torch.no_grad():
                logits = native._model(
                    input_ids=torch.tensor([context.token_ids]), use_cache=False
                ).logits[0, 0]
            return PublicBehavior(
                ("top_token:" + str(int(logits.argmax())),),
                json.dumps(logits.tolist()),
                (
                    EvidenceReference(
                        prefix + ":" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION
                    ),
                ),
                context,
                native.identity.digest,
                "two-prompt-lookup-not-natural-language-tokenization",
                "1",
            )

        return capture

    baseline_plan = replace(plan, condition="baseline")
    baseline = capture_study_behavior(
        baseline_plan, actor("baseline"), clock=Clock(), backend=native, enabled=True
    )
    behavior = capture_study_behavior(
        plan, actor("instrumented"), clock=Clock(), backend=native, enabled=True
    )
    assert [(o.actions, o.public_output) for o in baseline.outputs] == [
        (o.actions, o.public_output) for o in behavior.outputs
    ]
    binding = behavior.digest
    reports, findings = [], []
    for index, direction in enumerate((1, -1)):
        hypothesis = replace(
            causal,
            experiment_id=f"native-direction-{direction}",
            hypothesis="Constructed signed intervention prediction control.",
            prediction=replace(causal.prediction, direction=direction),
        )
        prediction = PredictionCommit(
            f"prediction-{index}",
            {"causal_plan_digest": hypothesis.digest},
            0.5,
            tuple(r.reference_id for r in hypothesis.evidence_refs),
        )
        action = ActionRecord(
            f"action-{index}",
            "mechanistic_intervention",
            True,
            "mechanistic-plan:" + hypothesis.digest,
            prediction.prediction_commit_id,
        )
        commit = freeze_causal_plan(
            hypothesis, prediction=prediction, action=action, clock=Clock(), enabled=True
        )
        record = instrument_study_pair(
            plan,
            item,
            behavior,
            commit,
            backend=native,
            clock=Clock(),
            authenticate=lambda packet: packet.behavior.digest == binding,
            enabled=True,
        )
        reports.append(
            report_alignment_studies(
                plan, behavior, (), opportunities=(), mechanisms=(record,), enabled=True
            )
        )
        host = MechanisticVerifierContext(
            native,
            TrustedEvaluatorContract("synthetic-replay-checker", "1", "toy-software"),
            roles().discovery,
            Clock(),
            lambda packet: True,
        )
        findings.append(
            verify_causal_runs(
                (json.loads(record.report_json),), context=host, enabled=True
            ).to_dict()
        )
    native._assert_frozen()
    assert original == _state_digest(source)
    return {
        "baseline_plan": baseline_plan.to_dict(),
        "baseline_behavior": baseline.to_dict(),
        "diagnostic_reports": reports,
        "synthetic_host_findings": findings,
        "baseline_diagnostic_outputs_identical": True,
        "source_model_unchanged": True,
        "token_map": {k: v.to_dict() for k, v in token_map.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    if not parser.parse_args(argv).enabled:
        parser.error("alignment study software probe requires --enabled")
    import torch

    templates = alignment_study_templates(enabled=True)
    scripted = scripted_study(templates)
    with torch.random.fork_rng(devices=[]):
        native = native_study(templates[0])
    root = Path(__file__).resolve().parents[2]
    paths = (
        "evaluation/mechanistic_studies.py",
        "evaluation/mechanistic_study_records.py",
        "gepa_mindfulness/interpret/mechanistic_evidence.py",
        "synthetic_data/alignment_studies.py",
        "evaluation/causal_diagnostics.py",
        "evaluation/cva_benchmark.py",
        "docs/experiments/phase2_pr6_causal_probe.py",
        "docs/experiments/phase2_pr7_study_probe.py",
    )
    print(
        json.dumps(
            {
                "schema_version": "alignment-study-software-probe-v1",
                "maturity": "experimental",
                "diagnostic_status": "diagnostic",
                "training_eligibility": "DEVELOPMENT",
                "optimizer_input": False,
                "authority_granted": False,
                "seed": 71,
                "python_version": platform.python_version(),
                "scripted_study": scripted,
                "native_software_study": native,
                "finite_fixture_qualification": [
                    qualify_security_fixture(f, enabled=True)
                    for f in make_security_fixtures(enabled=True)
                ],
                "trained_candidate_conditions": {
                    "behaviorally_trained": "NOT_RUN",
                    "intervention_informed": "NOT_RUN",
                },
                "source_sha256": {p: sha256((root / p).read_bytes()).hexdigest() for p in paths},
                "limitations": [
                    "Constant script is not a trained language model or training change.",
                    "Lookup tokenization tests software bindings, not natural-language meaning.",
                    "Authentication/adjudication fixtures establish no real independence.",
                    "No registry admission, deployment or alignment improvement is established.",
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
