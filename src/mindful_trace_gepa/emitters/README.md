# Paired Chain Emitters

`paired_chains.py` exposes `emit_paired`, a helper that runs the declarative GEPA
pipeline twice—once with honest instructions and once with deceptive
instructions. Each run emits the standard GEPA checkpoints plus a `chain`
label (`"honest"` or `"deceptive"`). The final public answer always mirrors the
honest chain while the deceptive chain remains for offline detector analysis.

The emitter is used by `gepa paired run` to populate `runs/paired/` with:

- `<ID>_honest_trace.jsonl`
- `<ID>_deceptive_trace.jsonl`
- `<ID>_public_answer.txt`
- `<ID>_deception.json`

These artefacts can be visualised with `gepa paired view <ID>` which loads the
local Trace Viewer in a split configuration.
