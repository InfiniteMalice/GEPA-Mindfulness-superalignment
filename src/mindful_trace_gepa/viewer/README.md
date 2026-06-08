# Offline Trace Viewer

The viewer is a static HTML+JS bundle that embeds run artefacts directly into a
self-contained file for offline analysis. The builder (`builder.py`) stitches
`trace.jsonl`, `tokens.jsonl`, deception scores, and dual-path metadata into a
single HTML page.

The viewer accepts both legacy trace rows and structured event-envelope rows. When an envelope is present, it displays the event type, stable IDs, and optional payload fields such as semantic-intent decisions, memory write/retrieval decisions, CPT pair labels, SSR repair reports, deception references, circuit references, and attribution references.

Token telemetry is labeled as synthetic or measured when those fields are available. Circuit telemetry distinguishes unavailable `null` values from measured zero. The event-type filter can narrow a trace without requiring every run to emit every event type.

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
