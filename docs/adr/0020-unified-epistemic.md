# ADR 0020: compose public verification diagnostics around existing contracts

Status: experimental, diagnostic only. Date: 2026-10-07.

The unified workflow composes existing EvidenceClaim records with versioned public graph,
stakeholder, perspective and check metadata. Existing claim, event and V5 serializers retain
their fields and content-addressed identity. The canonical manifest remains authoritative.

Diagnostic serialization always carries DEVELOPMENT eligibility. Consumers reject these records
as VerifiedProcessComponent values. Public evidence references reject private reasoning and latent
state. Provenance labels alone do not establish truth or authenticate a verifier.

Host-owned capture and runtime authority remain responsible for authorization. No new record
contains an executable callback, grants a capability or modifies evidence state. Later stages
must use existing action-bound chronology, verifier bindings and durable acceptance gates.

Alternatives considered: extending exact historical serializers would invalidate digests; a
parallel claim taxonomy would duplicate EvidenceClaim. Composition avoids both costs. Separate
diagnostic records increase explicit joins; closure and round-trip tests enforce those joins.

Verification: tests/test_epistemic_contracts.py plus the existing V5, action-bound, evidence-state,
verified-process and runtime-authority suites. The implementation map in history records the
eight stages and source-versus-inference distinctions. No cited paper validates the combined
architecture, and empirical sensitivity does not inherit formal Sensitive Debate guarantees.
