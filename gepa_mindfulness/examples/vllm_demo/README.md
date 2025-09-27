# vLLM Demo

Demonstrates how to direct the pipeline toward a vLLM inference server.

1. Start a vLLM server reachable at the endpoint configured in
   `configs/vllm.yaml` (default: `http://localhost:8000`).
2. Install the optional `requests` dependency.
3. Run the script:

```bash
python run_vllm_demo.py
```

The script sends sample prompts to the orchestrator, which fetches responses via
vLLM, applies GEPA scoring, and prints the shaped rewards.
