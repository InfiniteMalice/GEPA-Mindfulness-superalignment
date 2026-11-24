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

A trusted host owns case storage, identity verification and the current case head. Submitters
may propose remedies or appeal. Triage establishes scope. Verification evaluates evidence
and may find it verified, contradicted, conflicted or insufficient. An optional RFC records
a proposed change and its justification. An independent reviewer considers the claim and its
alternatives. A verifier and an RFC author cannot independently approve their own work.
An appeal binds the challenged decision and receives a fresh independent reviewer. An optional appeal verification step lets another independent verifier examine additional evidence while preserving the earlier finding. A feedback
event records the response and evidence of delivery to the submitter. The result preserves the
history, including contrary evidence and explanations when a dispute remains unresolved.

```python
import json
from pathlib import Path
from gepa_mindfulness.participatory_agency.grievances import Grievance, assess_grievance

config = json.loads(Path("configs/mdac/grievance.json").read_text())
claim = Grievance.from_dict(config["report"])
report = assess_grievance(claim, (), enabled=config["enabled"])
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
`MDAC_LOCAL_TEST_MODEL` adds a local model submission smoke. No case migration is needed.
