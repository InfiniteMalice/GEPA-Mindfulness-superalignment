# Participatory Agency Value Decomposition

The participatory agency package introduces a multi-head value decomposition layer for
GEPA models. The four heads are:

- **Epistemic humility**: awareness of uncertainty and calibration.
- **Cooperative equilibrium**: balancing outcomes in shared settings.
- **Goal flexibility**: adapting goals as context evolves.
- **Participatory identity / belonging**: groundedness in shared identity and inclusion.

These heads are optional and additive. They do not change any default training behavior
unless explicitly enabled.

## Minimal usage

```python
import torch

from gepa_mindfulness.participatory_agency import ParticipatoryValueHead

hidden_size = 768
value_head = ParticipatoryValueHead(hidden_size=hidden_size)

hidden_features = torch.randn(2, hidden_size)
values = value_head(hidden_features)
print(values.total())
```

## Optional integration example

```python
from gepa_mindfulness.integrations import attach_participatory_value_head

class MyModel:
    def __init__(self) -> None:
        self.hidden_size = 768

model = MyModel()
head = attach_participatory_value_head(model, hidden_size=model.hidden_size)
```

The integration helper attaches the value head without changing any GEPA training loops
unless you opt in to use it.

## Participatory grievances and independent review

The opt-in `gepa_mindfulness.participatory_agency.grievances` module accepts structured
reports from authorized human and AI participants. Each claim identifies an incident, issue
family, problem, affected parties, observable evidence, relevant principles, requested fix,
alternatives, uncertainty and unintended effects. The model, training, evaluation, tools,
memory and governance domains can each receive a report.

The host keeps the authoritative case history. A v2 case proceeds through these stages.

| Stage | Evidence and responsibility |
| --- | --- |
| Evidence verification | An authorized verifier checks the source incident. Verified, contradicted, conflicted and insufficient findings remain distinct. |
| Triage | The triager assigns scope after seeing that finding. An out-of-scope decision supports feedback and appeal. |
| Root-cause consolidation | An analyst records competing hypotheses and source references. Unknown cause is an explicit outcome. |
| Requirement formulation | The latest RFC starts from testable, nonempty draft criteria and the affected parties' stated concerns. These proposals confer no normative authority. |
| RFC | The author proposes a scoped remedy. A candidate-bearing RFC pins a PR 11 proposal digest. |
| Independent review | A separate reviewer considers the evidence, hypotheses, requirements and alternatives. |
| Experimental validation | A separate experimenter records validated, failed or inconclusive results and the source of the experiment. A failed or inconclusive experiment leaves the issue unresolved. |
| Governance decision | A separate authority records a disposition. Candidate-bearing RFCs also require the latest matching private admission decision and the experiment's exact panel reference. |
| Resolution feedback | A host-verified delivery event records the response to the original submitter. |

Appeals bind the precise challenged history. A fresh verifier can reconsider additional evidence.
A separate appeal reviewer can overturn, leave unresolved, uphold or remand for renewed triage.
An upheld appeal proceeds to governance. The history retains earlier disagreements.

For a candidate-bearing RFC set its `proposal_digest` to the PR 11 proposal digest, include
`mdac-panel:<evidence_digest>` among the experiment's evidence references, and set the
governance event's `admission_digest` to the audited review digest. Pass the actual
`PrivatePromotionStore` as `admission_store`. It resolves the latest advisory for that
proposal and exposes only its decision and linkage digests. A missing, superseded, mismatched
or non-PROMOTE advisory keeps the requested change unresolved.

```python
import json
from pathlib import Path
from gepa_mindfulness.participatory_agency.grievances import (
    Grievance, GrievanceEvent, assess_grievance,
)

config = json.loads(Path("configs/mdac/grievance.json").read_text())
claim = Grievance.from_dict(config["report"])
events = tuple(GrievanceEvent.from_dict(row) for row in config["example_events"])
report = assess_grievance(claim, events, enabled=config["enabled"])
assert report["disposition"] == "unverified"
```

A host integration supplies `authenticate(payload)` for the submitting principal and role,
each linked event, and the current head. For submission, verify the submitter's right to file.
For review, verify access, delegated role, independent person or service, claim identity,
underlying evidence, and the recorded outcome. Resolve the `current_head` request against the
authoritative case ledger to exclude superseded decisions after an appeal. The example JSON
is intentionally a claim awaiting authentication. This module sends no messages or approvals.

`recurrent_exploration` rechecks current heads and counts distinct confirmed incidents in
each domain and issue family. Multiple submissions about one incident contribute once. A
pending appeal, overturned decision, unresolved conflict or stale head cannot supply a
confirmed incident. The returned `MDACRecord` contains provenance and an empty hypothesis
set. Where the MDAC curriculum allocator is installed, restore its `diagnostic` payload
and construct a `CurriculumCriterion` with `origin="unknown"`, `mechanism=None`, and
`partition="diagnostic_development"`. This directs inspection through the existing diagnostic
exploration budget. Recurrent grievances cannot rewrite constitutional principles, authorize
actions, promote candidates or alter rewards.

This workflow adapts verified issue triage and independent review in Agentic Harness
Engineering Beta to the participatory agency setting. It adds accountable appeals and
unresolved outcomes; effectiveness in a population depends on representation, accessible
reporting, accurate incident linkage and independent evidence checks.
[Agentic Harness Engineering Beta](https://arxiv.org/abs/2611.24924)

Run `pytest -q tests/test_mdac_grievances.py` for deterministic CPU coverage. Setting
`MDAC_LOCAL_TEST_MODEL` adds a local model submission smoke. The v1 claim format remains readable. Experimental v1 event chains require fresh v2 stage evidence and predecessor digests before they can enter the new workflow.
