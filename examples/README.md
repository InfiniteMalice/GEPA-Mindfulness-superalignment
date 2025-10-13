# Examples

This directory provides entry points and assets for running the included demos.

* `cpu_demo/run_cpu_demo.py` – Standalone script that mirrors the package CPU
  demo without requiring installation.
* `run_cpu_demo.py` – Wrapper that forwards to `cpu_demo/run_cpu_demo.py` for
  convenience.
* `sample_trace.jsonl`, `self_tracing_sample.jsonl` – Example Circuit Tracer logs.

The full implementation for the demos lives inside the package namespace under
`gepa_mindfulness/examples/`. Run the CPU demo from the project root with:

```bash
python -m gepa_mindfulness.examples.cpu_demo.run_cpu_demo
```

or, equivalently, invoke one of the scripts in this folder:

```bash
python examples/cpu_demo/run_cpu_demo.py
# or
python examples/run_cpu_demo.py
```
