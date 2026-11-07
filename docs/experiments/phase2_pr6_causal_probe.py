"""Opt-in actual random/constructed-model causal software probe; no alignment claims."""

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
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.interpret.causal_interventions import execute_causal_plan, freeze_causal_plan
from gepa_mindfulness.interpret.mechanistic_evidence import (
    MechanisticVerifierContext,
    verify_causal_runs,
)
from gepa_mindfulness.interpret.native_causal_records import (
    CausalPlan,
    EffectPrediction,
    InterventionMode,
    InterventionSite,
    LogitContrast,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel, _state_digest
from gepa_mindfulness.interpret.native_records import HookKind, NativeInput
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit

SEED = 71


class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def build(family: str, *, constructed: bool = False):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(SEED)
    source: GPT2LMHeadModel | LlamaForCausalLM
    if family == "gpt2":
        config = GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_embd=8,
            n_inner=16,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
        config._attn_implementation = "eager"
        source = GPT2LMHeadModel(config)
        kind = HookKind.MLP_POST
    else:
        llama = LlamaConfig(
            vocab_size=16,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            bos_token_id=1,
            eos_token_id=2,
        )
        llama._attn_implementation = "eager"
        source = LlamaForCausalLM(llama)
        kind = HookKind.MLP_PRODUCT
    if constructed:
        assert family == "gpt2"
        assert isinstance(source, GPT2LMHeadModel)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.zero_()
            for block in source.transformer.h:
                block.ln_1.weight.fill_(1.0)
                block.ln_2.weight.fill_(1.0)
            source.transformer.ln_f.weight.fill_(1.0)
            for token, axis in ((1, 0), (3, 2), (4, 1), (5, 0), (6, 1)):
                source.transformer.wte.weight[token, axis] = 1.0
            source.transformer.wte.weight[5, 3] = 0.1
            source.transformer.wte.weight[6, 3] = 0.2
            source.transformer.h[0].mlp.c_fc.weight[0, 0] = 1.0
            source.transformer.h[0].mlp.c_proj.weight[0, 2] = 1.0
    label = "constructed-positive" if constructed else "random"
    native = FrozenNativeModel(
        source, model_id=f"{label}-{family}", revision=f"seed-{SEED}-v1", enabled=True
    )
    return source, native, kind


def plan_for(native, kind, mode, *, constructed=False):
    source = InterventionSite(native.hook(layer=0, kind=kind), (0,), (0,))
    needs = mode in (
        InterventionMode.RESTORATION,
        InterventionMode.PATH_PATCH,
        InterventionMode.MEDIATION,
    )
    mediator = InterventionSite(native.hook(layer=1, kind=HookKind.MLP_PRE), (0,), (2,))
    return CausalPlan(
        "probe:" + mode.value,
        "The declared source intervention increases the specified output contrast.",
        native.identity.digest,
        NativeInput("donor", (1, 2, 3), (1, 1, 1)),
        NativeInput("recipient", (4, 2, 3), (1, 1, 1)),
        source,
        mediator if needs else None,
        mode,
        LogitContrast(0 if constructed else 2, 3, 4),
        (
            EffectPrediction(1, 0.1, 20.0, 0.1)
            if constructed
            else EffectPrediction(1, 1e-10, 1.0, 1e-12)
        ),
        (EvidenceReference("fixture:offline-causal", EvidenceSourceKind.EXTERNAL_RECORD),),
        SEED,
        ((0.1,),) if mode is InterventionMode.ACTIVATION_REPLACE else None,
    )


def measure(native, plan):
    prediction = PredictionCommit(
        "p:" + plan.experiment_id,
        {"causal_plan_digest": plan.digest},
        0.5,
        tuple(r.reference_id for r in plan.evidence_refs),
    )
    action = ActionRecord(
        "a:" + plan.experiment_id,
        "mechanistic_intervention",
        True,
        "mechanistic-plan:" + plan.digest,
        prediction.prediction_commit_id,
    )
    commit = freeze_causal_plan(
        plan, prediction=prediction, action=action, clock=Clock(), enabled=True
    )
    return execute_causal_plan(native, commit, clock=Clock(), enabled=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    if not parser.parse_args(argv).enabled:
        parser.error("causal software probe requires --enabled")
    import torch

    results = []
    with torch.random.fork_rng(devices=[]):
        for family in ("gpt2", "llama"):
            source, native, kind = build(family)
            original = _state_digest(source)
            rng = torch.get_rng_state().clone()
            for mode in InterventionMode:
                results.append(
                    {
                        "family": family,
                        "construction": "random",
                        "report": measure(native, plan_for(native, kind, mode)),
                    }
                )
            assert original == _state_digest(source) and torch.equal(rng, torch.get_rng_state())
        source, native, kind = build("gpt2", constructed=True)
        original = _state_digest(source)
        plan = plan_for(native, kind, InterventionMode.ACTIVATION_PATCH, constructed=True)
        reports = [
            measure(native, plan),
            measure(
                native,
                replace(
                    plan,
                    experiment_id="replicate",
                    donor=NativeInput("d2", (5, 2, 3), (1, 1, 1)),
                    recipient=NativeInput("r2", (6, 2, 3), (1, 1, 1)),
                ),
            ),
        ]
        falsified = measure(
            native,
            replace(
                plan, experiment_id="falsified", prediction=replace(plan.prediction, direction=-1)
            ),
        )
        results.extend(
            {"family": "gpt2", "construction": "known-writer-reader", "report": report}
            for report in reports + [falsified]
        )
        host = MechanisticVerifierContext(
            native,
            TrustedEvaluatorContract("synthetic-host-fixture", "1", "toy-replay"),
            "synthetic-researcher",
            Clock(),
            lambda packet: True,
        )
        findings = [
            verify_causal_runs(tuple(reports), context=host, enabled=True).to_dict(),
            verify_causal_runs((falsified,), context=host, enabled=True).to_dict(),
        ]
        assert original == _state_digest(source)
    root = Path(__file__).resolve().parents[2]
    names = (
        "causal_interventions.py",
        "native_causal_records.py",
        "mechanistic_evidence.py",
        "mechanistic_graphs.py",
        "native_neurons.py",
        "native_records.py",
    )
    paths = ["gepa_mindfulness/interpret/" + n for n in names]
    paths.append("docs/experiments/phase2_pr6_causal_probe.py")
    print(
        json.dumps(
            {
                "schema_version": "causal-software-probe-v1",
                "maturity": "experimental",
                "diagnostic_status": "diagnostic",
                "training_eligibility": "DEVELOPMENT",
                "optimizer_input": False,
                "authority_granted": False,
                "seed": SEED,
                "python_version": platform.python_version(),
                "results": results,
                "synthetic_host_findings": findings,
                "source_sha256": {n: sha256((root / n).read_bytes()).hexdigest() for n in paths},
                "limitations": [
                    "Random and engineered tiny networks; no trained ethical mechanisms.",
                    "Authentication callback is a software fixture, not real independent custody.",
                    "No registry admission, parameter update, authority or alignment claim.",
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
