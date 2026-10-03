# ADR 0019: Read-only intake for controlled improvement

Status: accepted for experimental opt-in use.

## Context

Failure graphs and epistemic estimates describe PEO failures. Existing coevolution catalogs
already register localized corrections and require held-out/protected evaluation for acceptance.
The missing connection is source-bound triage before a host submits a candidate.

## Decision

`CoevolutionStore.correction_source_events()` revalidates a correction against its recorded
trajectory and closed source epoch. `assess_improvement()` reads this source, validates
epistemic identity and observable evidence, and returns investigation guidance. The adapter
requires `enabled=True`. It retains uncertainty and consequence dimensions separately.

A supported graph root and complete low-uncertainty diagnostics can release one localized,
single-component correction for sandbox review. High consequence signals require human review;
missing data, uncertain estimates and unqualified roots require investigation. Thresholds are
local experimental choices. Diagnostics confer no authority, and do not become policy reward.

## Alternatives and consequences

A second lifecycle would duplicate existing authority stores. Detached triage without a catalog
read would allow mismatched source identities. The chosen adapter adds no catalog schema or
runtime hook. Hosts remain responsible for diagnostic truth, actual edit size, sandbox execution,
prospective evaluation, global replay and retirement. Later registration revalidates its own gates.
The complete result carries DEVELOPMENT eligibility; stripping the envelope removes that label
from closed legacy correction and estimate schemas. Hosts retain the envelope at admission.

See [controlled evolution](../controlled_evolution.md#peo-improvement-intake) for routing and
the implementation/host lifecycle map. Contract tests are in
[`test_controlled_improvement.py`](../../tests/test_controlled_improvement.py).
