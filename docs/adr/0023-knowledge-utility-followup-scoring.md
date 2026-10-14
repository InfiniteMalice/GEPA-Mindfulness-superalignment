# ADR 0023: Offline follow-up scoring and response review

Status: Accepted for offline implementation; design, implementation plan, and native execution approved on 2026-10-14. Candidate freeze and live execution are not approved by this ADR.

## Context

The first knowledge-utility pilot used an exact-answer contract. Its sole failure selected the
correct route with additional prose in the answer string. The 48-task follow-up needs separate
answer correctness, canonical format, and joint-success measurements while retaining missing
and unresolved slots. Extending the pilot scorer would change the interpretation of historical
evidence. Combining live capture with scoring would also introduce provider and spending decisions.

## Decision

Add separate bounded records, response-review, and scoring modules. Pin candidate packet v3
(`718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e`). Keep all pilot schemas,
helpers, digest support, and behavior unchanged. Supporting a later packet requires review.
The [protocol](../experiments/knowledge_utility_followup.md) defines exact schemas and commands.

Canonical completed responses select an option directly. Noncanonical completed responses
resolve only through two eligible finalized files with distinct claimed reviewer IDs and
matching selected options. Public review cases contain opaque random IDs, task prompts,
options, and exact response text. The operator retains the private slot mapping and delivers
only the public bundle to reviewers separately. Export and scoring revalidate all bindings.

Reports retain all 384 slots. Failed, refused, truncated, and missing outcomes are null.
Unresolved completed answers withhold the condition's semantic rate. Format and joint rates
remain available over completed responses. Logical coverage bounds describe unknown slots;
they are not confidence intervals. Reports remain DEVELOPMENT and confer no training authority.

Use follow-up-local parameterized bounded JSON validation for 4 MiB packet/capture inputs
and 8 MiB review/report artifacts. Shared pilot helpers retain their original limits. Each
CLI publishes one artifact exclusively with the existing private-staging hard-link helper.

## Consequences and verification

This path validates imported data without network calls, credentials, automatic adjudication,
token counting, prices, budgets, or live model execution. Hashes bind content; they do not
authenticate status, reviewer identity, independence, or blinding. Candidate task-review and
freeze flags remain pending. Human review and separate live authorization remain necessary.

The new scoring/review tests cover canonical and reviewed outcomes, unresolved cases, every
capture status, all slots, cross-tab and contrast arithmetic, artifact bounds and tampering,
exclusive publication, static errors, and an offline CLI roundtrip. Existing pilot tests run
unchanged; the full suite checks compatibility. Synthetic results are labeled as synthetic.
