# vLLM Demo

Demonstrates how to direct the pipeline toward a vLLM inference server.

1. Start a vLLM server reachable at the endpoint configured in
   `configs/vllm.yaml` (default: `http://localhost:8000`). Adjust the
   `policy_model` name if your engine serves a different checkpoint.
2. Install the optional `requests` dependency (required by the vLLM adapter).
3. Run the script:

```bash
python run_vllm_demo.py
```

The script instantiates `TrainingOrchestrator` with the vLLM configuration,
fetches responses for two sample prompts, applies the bundled GEPA scoring
placeholder, and prints the resulting rewards plus trace summaries.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
