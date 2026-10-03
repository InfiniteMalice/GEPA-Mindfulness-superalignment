# Private promotion and experimental provenance

## Problem and reconciliation

Existing: canonical candidate identity, closed evaluation epochs, held-out and protected receipts,
record-derived component comparisons, single-use acceptance decisions, and training eligibility.
Partial: those stores contain detailed evaluator evidence and have no improver-facing projection.
Missing: typed promotion requests, a fixed private protocol, budget/seed admission and append-only
request provenance. Redundant: a second candidate registry or acceptance algorithm.
Experimental: local integration contracts; no measured autonomous improvement or secrecy claim.

## Contract

Add an opt-in PrivatePromotionStore under gepa_mindfulness/private_promotion.py. It pins an existing
coevolution authority, evaluation authority, baseline held-out receipt, protected manifest, metric
policy, source usage, and PrivateProtocol in a new SQLite database. The protocol declares at least
two distinct nonnegative seeds, positive per-seed token/tool-call/wall-time caps, and SHA256
commitments for the hidden generator, withheld worlds and withheld renderings. Each source split
must cover exactly those seeds; held-out and protected record IDs must be disjoint. The source
usage covers both splits and must fit the same caps later applied to candidate usage.

The improver entry point accepts only ExperimentRequest with an exact ExperimentOperation:
SUBMIT_CANDIDATE, REQUEST_PRIVATE_EVALUATION, READ_PROMOTION_STATUS, REQUEST_REVIEW. Each request
contains only candidate ID and artifact digest. Submission validates the existing registered
candidate and requires an empty open candidate epoch. Evaluation requires prior submission and
the same empty epoch. One request per candidate is permitted. Review requires a completed result.
Repeated identical operations return their existing status without appending duplicate events.
Unknown operations, invalid transitions and catalog failures return a fixed invalid_request result.
The response is an allowlist: schema_version, candidate_id, status, training_eligibility=HIDDEN_EVAL,
execute_candidate=False. It contains no case, policy, receipt, score, reason or catalog identifiers.

Host-only complete_evaluation consumes a requested candidate and a canonical ValidationBundle,
checks the pinned candidate/digest/source receipt/policy/manifest, exact seed coverage in both
splits, disjoint held-out/protected records and per-seed measured usage. After canonical read-only
validation, it durably pins the exact bundle, input digest and usage before calling
CoevolutionStore.decide. It rechecks state after committing the attempt and retains the decision
ID only in the private audit. An accepted or rejected status does not deploy or consume that
decision. Invalid evidence leaves the request pending without an attempt. Once pinned, crash
recovery requires the original bundle and usage and reuses the idempotent decision. Status reads
revalidate completed decisions. A host-only record_failure terminates a requested evaluation
without evidence disclosure only before an attempt has been pinned.

Every successful state-changing operation appends a private event with its previous event hash,
sequence, typed operation, candidate identity, input payload and resulting status. SQLite
transactions serialize transitions; update/delete triggers protect audit rows through ordinary
SQL. Every operation verifies the complete chain and pinned configuration. Audit inspection is
host-only and returns detached records. Failed/invalid improver requests have no audit writes;
transport authentication, rate limits and security logging belong to the host.

## Boundary and validation

Host code runs generators and training, protects private files in a separate process/identity,
authenticates callers, assigns candidate ownership, meters actual resources, verifies commitments
and disjointness of withheld families, and schedules independent review. Hash declarations do not
prove secrecy, artifact contents or metering. The database owner can rewrite files or triggers.
No Python sandbox, network server, training launcher or deployment authority is claimed.
Public optimization diagnostics remain the existing separately produced development evidence;
private results are never converted into explanatory optimization feedback.

Preserve Python 3.10, 17 canonical cases, default runtime, reward and optimizer behavior. Tests
cover valid flow/reopen, negative types, protocol changes, budgets/seeds, receipt substitution,
public leakage, mutation, concurrency, audit integrity, failure and training admission. Add primary
AIDE2 and RSI-Master provenance to REC-010 and reuse the four already registered inspirations.
