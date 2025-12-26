# Adversarial Evaluation (Historical Note)

The repository previously shipped a standalone adversarial evaluation pipeline. That system has
been retired in favor of the **dual-path deception abstraction**:

* Two candidate paths per prompt.
* Separate scratchpads and public reasoning for each path.
* A single FINAL ANSWER label: 1, 2, both, or neither.

For the current workflow and API, see:

* `src/mindful_trace_gepa/deception/dual_path_core.py`
* `src/mindful_trace_gepa/deception/dual_path_runner.py`
* `DECEPTION_CIRCUIT_ABLATION.md`
