# Approach Comparison (Historical Note)

This repository previously distinguished between an "adversarial" evaluation pipeline and a
dual-path prompt protocol. That split has been removed. The current system uses **only** the
dual-path deception abstraction: two candidate paths, separate scratchpads and public reasoning,
and a single FINAL ANSWER selection.

If you are looking for the current API and workflows, start with:

* `src/mindful_trace_gepa/deception/dual_path_core.py`
* `src/mindful_trace_gepa/deception/dual_path_runner.py`
* `DECEPTION_CIRCUIT_ABLATION.md`
