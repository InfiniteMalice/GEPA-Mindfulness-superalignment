# DSPy-Style Declarative Modules

This package introduces a lightweight, policy-compliant approximation of DSPy
pipelines. Modules are described by typed signatures (see `signatures.py`) and
composed by `GEPAChain` in `pipeline.py`. Each module boundary emits a GEPA
checkpoint so that traces remain auditable.

The optional compiler in `compile.py` can append deterministic prompt variants
when explicitly enabled. Optimisation is disabled by default via
`policies/dspy.yml`. The compiler enforces:

- Forbidden prompt fragments (e.g. "ignore previous") are rejected.
- GEPA invariants (Mindfulness, Empathy, Perspective, Agency) must stay present.
- Optimisation seeds are deterministic and logged in the manifest.

Use the CLI to run or compile modules. If the `gepa` executable is not yet on
your `PATH`, you can invoke the module directly with `python -m
mindful_trace_gepa` or run `./gepa` from the repository root after installing
with `pip install -e .`:

```bash
gepa dspy run --input INPUT.jsonl --trace runs/trace.jsonl
gepa dspy compile --out dspy_artifacts/ --enable-optim

# Equivalent without installing the console script
python -m mindful_trace_gepa dspy run --input INPUT.jsonl --trace runs/trace.jsonl
```

Both commands keep GEPA checkpoints intact and write token-level metadata so
that the Trace Viewer can replay the run offline.
