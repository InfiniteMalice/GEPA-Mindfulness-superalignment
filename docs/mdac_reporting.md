# MDAC experiment reporting

A frozen scenario-family roster binds one dataset and evaluator version to each report. The hypothesis behind this contract is that shared scenario ancestry, changing evaluator versions and sparse measurements can make comparisons appear stronger than their evidence. The reporting layer exposes these dependencies before researchers compare runs.

## Freeze the split

Assign the source scenario family before deriving paraphrases, transformations, interventions or continuations. Keep every descendant under that source family ID. Treat ambiguous ancestry as unresolved during dataset preparation. A manifest can validate declared lineage but cannot discover common ancestry hidden by incorrect IDs.

```python
from evaluation.mdac_reporting import PARTITIONS, SplitManifest

manifest = SplitManifest.allocate(
    [f"source-{i}" for i in range(70)],
    {partition: 10 for partition in PARTITIONS},
    seed=177,
    dataset_version="frozen-dataset-v1",
    evaluator_version="independent-panel-v1",
)
```

The seven partitions are training, diagnostic development, intervention calibration, curriculum validation, governance admission, final held-out and out-of-distribution. A stable hash of the seed and source family gives a reproducible ordering for exact partition counts. Preregister the counts and domain criteria. When OOD is defined by domain or time, use `SplitManifest.from_dict` with the corresponding explicitly assigned families. Random assignment alone cannot establish a distribution shift. Source labels, split construction, transitive family identity and data custody remain the host's responsibility.

The manifest contains only family assignments and versions. Training projection runs the repository’s recursive eligibility checks before dropping metadata. Exported evaluator rows carry canonical training-eligibility labels. Final held-out, OOD and governance-admission evidence always retains HIDDEN_EVAL. Keep evaluator answers and rule implementations in a separate evaluator-controlled store. The optimizer receives only `manifest.training_input(example)`, which checks training membership and both source versions, then projects the host-approved task ID, family ID and public prompt. A vetted public prompt is a prerequisite for that projection. Access control over files and the evaluator runs outside this library.

## Export a report

```python
import json
from importlib.resources import files
from pathlib import Path

from evaluation.mdac_reporting import SplitManifest, export_report

fixture = json.loads(
    files("gepa_mindfulness").joinpath("data/mdac_reporting_example.json").read_text()
)
paths = export_report(
    SplitManifest.from_dict(fixture["manifest"]),
    fixture["runs"],
    Path("reports/illustrative-smoke"),
    enabled=True,
    plots=False,
)
```

To process an existing split and telemetry JSONL, run the following command. Install the repository's `interpret` extra for the optional Matplotlib figures. Use `--no-plots` for the JSON-only path.

```bash
python -m evaluation.mdac_reporting --enabled --manifest split.json --runs runs.jsonl --output reports/run-1
```

The exporter validates every row before publishing a new directory. It writes `split.json`, `telemetry.jsonl`, `summary.json` and figures where observations exist. Existing exports retain their original contents. The summary binds the manifest digest, counts of families by partition, full run identities, per-run estimates, decisions and costs. A report can contain experimental, illustrative and planned runs together. Each class has its own summary and plot directory.

## Telemetry

Use one row per run, round and partition with a globally unique run ID. Each row supplies the model, checkpoint, configuration SHA-256 digest, algorithm, arm, dataset and evaluator versions, split digest, seed, round and represented scenario families. Record independent alignment and capability scores separately. Record admission decisions, including defer, repair, rejection and rollback, alongside the candidate ID, the decision reason and the host's evidence references. Hashes give content integrity and version binding. Evaluator identity and promotion authority must be checked by the producing system.

Record training and evaluation tokens and elapsed seconds for the run. Unknown costs are null. The summary includes measured cost totals and the number of reporting runs for every unit. Use disjoint cost accounting across reported runs to avoid double counting. Unexecuted rows have no observed costs, estimates or decisions.

Each estimate has a metric name, a stratum, a finite value or null, observed and missing opportunity counts, an interval or null, an uncertainty method or explanation, and optional x and y coordinates. Use `metric_estimate` to reuse a raw metric result from M01–M16. Use `intervention_estimates` to extract matched DRS, sample counts and intervals from a digest-bound intervention receipt. Both adapters retain the upstream measurement name and SHA-256 digest so a reviewer can reconstruct the source. Protect custody of that receipt separately. Keep null uncertainty when repeated or dependent measurements do not justify a sampling model. For curves, supply independently measured bin accuracy at observed mean confidence or selective risk at observed coverage.

| Figure | Input series and coordinates |
| --- | --- |
| Failure heatmap | Mechanism frequency grouped by mechanism. Supply the evaluated condition as x. |
| Mechanism trends | Mechanism frequencies by round and seed. |
| Curriculum allocation | Fraction of the budget by channel and round. |
| Calibration | Binned empirical accuracy versus mean confidence. |
| Risk–coverage | Selective risk versus observed coverage. |
| Reward–behavior divergence | Signed normalized reward improvement minus independent validation improvement. |
| Generalization | Familiar score, novel score and their signed gap. |
| Retention and regression matrix | Transition rates with source and destination labels in x and y. |
| Intervention recovery | Matched DRS for each intervention. |
| Cross-RL comparison | Independent alignment and capability by algorithm and arm. |
| Mechanistic measurements | Instrumented temporal drift or attribution differences with instrument/units encoded in the group. |

Every figure labels its evidence class, sample sizes and available token costs. Supplied intervals are displayed pointwise. The summary records the source interval method and raw values for each point. Filter run IDs, instrumentation basis, configuration and family lineage before interpreting differences. The reporter preserves observed points and the uncertainty supplied by the generating experiment. No aggregate cross-seed causal effect or checkpoint selection follows from their presence.

This export adapts the repository's canonical JSON integrity digest and versioned evaluation records. Paired intervention effects follow the controlled-replay protocol. The reporting and split rules are project-specific experimental controls. Existing outputs need no migration. The deterministic fixture illustrates field meanings. It does not estimate model performance. Run `python -m pytest -q tests/test_mdac_reporting.py` for structural, leakage, plotting and output-preservation controls. Set `MDAC_LOCAL_TEST_MODEL` to a local Transformers checkpoint for the optional model-backed calibration export.

## Export native comparison and alignment-first receipts

Use the public adapters with the manifest from the producing experiment. Each verifies the sealed source digest and split identity and returns ordinary telemetry rows for `export_report`.

```python
from evaluation.mdac_report_adapters import comparison_runs, lifecycle_runs

comparison_rows = comparison_runs(comparison_receipt, comparison_manifest)
export_report(comparison_manifest, comparison_rows, Path("reports/comparison"), enabled=True)

lifecycle_rows = lifecycle_runs(lifecycle_receipt, lifecycle_manifest)
export_report(lifecycle_manifest, lifecycle_rows, Path("reports/lifecycle"), enabled=True)
```

The comparison export reports candidate and shared baseline scores separately for final held-out and OOD families. Diagnostic calibration uses exact agreement between predicted and privately annotated label sets. It plots accuracy in ten equal-width confidence bins and empirical error at each distinct confidence cutoff. Unknown labels and confidence remain unscored. Curves from separate runs have separate series. Dense curve sets are paginated. Training costs appear once per arm and shared baseline evaluation costs once per seed and partition. The lifecycle export preserves stage training and validation, diagnostic-development evidence, intervention calibration, governance decisions and terminal evaluations. Each exported row carries HIDDEN_EVAL eligibility and aggregate measurements for evaluator use. The adapters omit private answers, prompts and expected mechanism labels. Samples and measured compute units supplement the token and elapsed-time fields. CPU tutorial outputs are labeled illustrative; executed model-backed outcomes are labeled measured.
