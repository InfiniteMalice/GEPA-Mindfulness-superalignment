# Offline Trace Viewer

The viewer is a static HTML+JS bundle that embeds run artefacts directly into a
self-contained file for offline analysis. The builder (`builder.py`) stitches
`trace.jsonl`, `tokens.jsonl`, deception scores, and paired-chain metadata into a
single HTML page.

Panels include:

- **Timeline** – GEPA checkpoints with timestamps.
- **Event Detail** – Raw text with GEPA badges and per-stage scores.
- **Tokens** – Log-probability/"confidence" trend line and token chips.
- **Deception** – Honest vs. deceptive chains alongside detector reasons.

Generate a viewer from the CLI:

```bash
gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl --out report_view.html
```

For paired scenarios you can additionally provide `--deception` and `--paired`
metadata files to surface detector output and honest/deceptive splits.
