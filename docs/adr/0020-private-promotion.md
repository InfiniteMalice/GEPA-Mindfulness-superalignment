# ADR 0020: Private promotion through bounded experimental requests

Status: Accepted for opt-in experimental integration.

## Context

The existing coevolution stores validate candidate identity and canonical evaluation evidence.
Their detailed records are suitable for a trusted evaluator, but unsuitable as improver feedback.
PR19 needs a bounded interface and durable provenance without a second acceptance algorithm.

## Decision

Add PrivatePromotionStore with one pinned protocol per catalog, typed request operations,
per-seed usage admission and a private append-only event chain. Public responses contain only
candidate ID and status plus schema/eligibility/execution labels. Host-only completion delegates
to the existing coevolution decision and links its identity in private provenance.

Reuse canonical source receipts, protected manifests, complete metric policies and candidate
claims. Add three read-only coevolution accessors to resolve these existing records. Keep hidden
generator/world/rendering artifacts as host-verified commitments; use actual record seeds to
validate coverage. The same fixed resource caps apply to declared baseline and candidate usage.

## Consequences

The promotion request vocabulary cannot edit evaluators, rewards or checkpoint identity. The
adapter remains disabled unless enabled explicitly and adds no default evaluation/training hook.
The host owns process isolation, authentication, candidate ownership, generation, metering,
overall query limits and review. SQLite triggers and hashes protect ordinary catalog operations,
not a privileged file owner. Separate catalog transactions require host-serialized epoch admission;
idempotent existing decisions permit retry after a completion-audit crash.

The module is a promotion integration, not the full research orchestration system described by
RSI-Master. It has no training scheduler, empirical improvement claim or deployment authority.
Contract validation is in tests/test_private_promotion.py; operational contracts and limitations
are in docs/controlled_evolution.md. The new API and registry retain the 17-case V5 baseline.
