# Examples

This directory provides entry points and assets for running the included demos.

* `run_cpu_demo.py` – Thin wrapper that dispatches to the package CPU demo.
* `sample_trace.jsonl`, `self_tracing_sample.jsonl` – Example Circuit Tracer logs.

The full implementation for the demos lives inside the package namespace under
`gepa_mindfulness/examples/`. Run the CPU demo from the project root with:

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
```

or, equivalently, invoke the wrapper in this folder:

```bash
python examples/run_cpu_demo.py
```
