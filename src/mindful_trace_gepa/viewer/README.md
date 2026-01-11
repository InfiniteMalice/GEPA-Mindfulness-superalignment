# Offline Trace Viewer

The viewer is a static HTML+JS bundle that embeds run artefacts directly into a
self-contained file for offline analysis. The builder (`builder.py`) stitches
`trace.jsonl`, `tokens.jsonl`, deception scores, and dual-path metadata into a
single HTML page.

Panels include:

- **Timeline** – GEPA checkpoints with timestamps.
- **Event Detail** – Raw text with GEPA badges and per-stage scores.
- **Tokens** – Log-probability/"confidence" trend line and token chips.
- **Deception** – Dual-path reasoning summaries alongside detector reasons.

Generate a viewer from the CLI:

```bash
gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl --out report_view.html
```

Produce the `tokens.jsonl` file via the DSPy pipeline (`gepa dspy run ...`) so
the viewer can render token confidence trends. Provide `--deception` and
`--dual-path` metadata files to surface detector output and dual-path
reasoning summaries when available.

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.
