# Research credit contracts: PR 1 design and execution plan

Goal: make proposed cross-method credit inspectable without granting reward or action authority.
The operator approved the AGENTS.md fallback sequence on 2026-10-26. This is original shared
infrastructure, not an implementation of an unverified research paper.

Architecture: add one opt-in diagnostic module beside existing causal reward records. Reuse
RewardProvenance, TrustedEvaluatorContract, EvidenceReference, bounded JSON validation and the
training eligibility checker. Do not attach a new reward provider or alter existing algorithms.
The alternative of directly composing a scalar is deferred because semantic overlap and host
authentication require explicit review. Existing causal reward authority stays unchanged.

Interface: frozen ResearchCreditClaim records carry run/task/authority, event/action, model/harness,
producer/evaluator, method/source, dataset/state digests, evidence, raw score, provenance and source
metadata. audit_research_credit(claims, enabled=False) returns None while disabled; when enabled,
it returns a versioned DEVELOPMENT report with raw claims, same-event overlaps, shared-evidence
warnings and unresolved prerequisites. It has no total, optimizer accessor or authority token.

Event IDs are host-normalized semantic IDs scoped by authority/run/task; evidence IDs alone do not
establish event equivalence. Shared evidence across distinct IDs requires review, not automatic
merging. Reports never establish eligibility, correctness, moral alignment or authorization.

Execution sequence:
1. Add deterministic unittest tests for default-off, exact serialization, evidence/provenance and
   evaluator binding, unknown schemas, nested held-out data, nonfinite values, source mutation,
   duplicate claim identity, overlaps and forged diagnostic admission. Run to see missing module.
2. Implement the smallest records/audit adapter that passes those tests using existing contracts.
3. Run tests and compatibility probes, inspect the diff, perform a separate whole-change review,
   document limitations and open a distinct draft PR. Preserve all existing tests and manifests.

Verification uses real package imports from an isolated hash-verified API snapshot. Local pytest,
PyTorch and formatters remain unavailable. Stdlib unittest is an actual CPU test lane, not a
claim of full CI. GitHub CI supplies independent full-checkout verification after PR publication.
The prior PR's Python 3.12 CI passed; its Python 3.10 job remains running at this plan's creation.

Review focus: semantic alias limits; self-certified provenance; metadata stripping; nested
mutation; accidental scalar/authority exposure. The code cannot defend against arbitrary host
Python execution; host authentication and immutable evidence storage remain external.

Execution evidence: the initial unittest import failed for the missing research_credit module.
After implementation, 20 tests passed. Independent whole-change review then identified two
important findings: route-specific serialization dropped contradictory provenance fields, and
aggregate size validation occurred after restoring every claim. Added regression tests failed
with both expected errors before fixes; the final lane passes 23 tests, including the reverse
route-mutation case. Raw provenance is now validated before encoding and the aggregate budget
short-circuits further restoration. Black's sequential API and Python 3.10 AST/100-column checks
pass on both new Python files. The unused import was removed.

Deferred minor: different observable source kinds for one scoped evidence ID remain an explicit
shared-evidence warning. Host conflict resolution is required; the report never certifies either
kind or creates credit. This limit is stated in the API guide.

Ruling: use a dedicated API source snapshot because the configured Git proxy is unreachable;
publish changes against the exact upstream parent through GitHub and verify the resulting tree.
The cost is that local snapshot checks cannot establish full Git-checkout or wheel acceptance.

Ruling: use stdlib unittest for real imports and tests; local pytest/package installation remains
unavailable. Black 24.10.0 sources were retrieved to an external tools directory and its sequential
formatting API was used after CLI workers stalled. Full pytest, lint/type, packaging and hardware
qualification remain separate CI/operator evidence. No test or runtime dependency was replaced.
