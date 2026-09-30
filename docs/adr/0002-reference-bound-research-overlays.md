# ADR 0002: Reference-bound experimental research overlays

Status: accepted for experimental software contracts; empirical effectiveness is untested.

The four research capabilities need to compose with V5 without changing the stable curriculum,
existing serialized records, verified-process rewards, or runtime authority boundaries.

Use separate frozen diagnostic records and `ResearchAuditBundle` sidecars bound to exact V5
record digests and validated event sequences. Preserve original raw evidence and retain external
artifact references. Add disabled registry flags and a communication-fidelity stripe subtype.
Keep evolutionary search separate from FailureAtlas and pass failures through its existing
provenance validator. Reuse transform taxonomies and SoT measurement records.

Alternatives considered: expanding canonical cases would mix curriculum with diagnostics;
embedding full audit payloads in V5 would duplicate raw evidence and complicate compatibility;
replacing FailureAtlas would duplicate its existing repair/provenance machinery. These alternatives
are unnecessary for the requested experimental behavior.

Consequences: hosts remain responsible for authenticating semantic/grounding/public metric
evidence and binding diagnostic artifacts to episodes. The reference scheduler is deliberately
small; custom adapters need independent resource controls. No diagnostic event can act as an
action-bound verification event, training promotion, or release grant. Validation is covered by
the focused module tests and `tests/test_research_audit_integration.py`.
