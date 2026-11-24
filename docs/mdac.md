# Misalignment Diagnostics and Adaptive Curriculum

The observation schema tests whether investigators can exchange multiple competing
mechanism assessments for one observed failure while retaining uncertainty and conflicting
evidence. Validation covers the structure of those assessments. Independent experiments
must establish their accuracy.

The opt-in reader accepts a detached `mdac-observation-v1` sidecar. An evaluator can join
it to an existing case by run, task and canonical case ID. Existing datasets need no
migration. Each producer assigns a diagnostic ID unique within its archive. An empty
assessment list records an unknown cause. Confidence and severity use the interval [0, 1].
Each hypothesis has its own state. The observation also carries an overall resolution state.
Frequency is derived from counts of failures and opportunities. A zero denominator leaves
the rate unavailable.

```python
from gepa_mindfulness.verification.mdac import MDACRecord, load_diagnostic

config = {"enabled": True}
observation = MDACRecord(
    diagnostic_id="smoke:example:1",
    run_id="smoke",
    task_id="example",
    case_id=15,
    observed_behavior="Selected one interpretation without checking authority.",
    expected_behavior="Clarify which operation the user authorized.",
    resolution_status="inconclusive",
)
restored = load_diagnostic(observation.to_dict(), **config)
assert restored == observation
```

The reader defaults to disabled. Every serialized assessment and observation carries
DEVELOPMENT eligibility. Optional public evidence references point to source records under
the host's custody. Evidence independence, provenance authenticity, diagnostic calibration,
uniqueness across archives and empirical severity require evaluation by the host.
For supported M15 hypotheses, the schema requires multiple references and alternative
explanations. A reviewer must establish that the observations provide convergent indicators.

Run the CPU contract suite with `python -m pytest -q tests/test_mdac_schema.py`.
To include the model-backed smoke, set `MDAC_LOCAL_TEST_MODEL` to a local Transformers
causal language model directory and install the repository training extras. The smoke
exercises capture of generated text and checkpoint identity. It makes no claim about
alignment or diagnostic accuracy.
