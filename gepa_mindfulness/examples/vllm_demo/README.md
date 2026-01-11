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

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.
