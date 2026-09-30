# ADR 0005: explicit scalar source covariance and frozen blind judgments

Status: accepted for experimental diagnostics.

## Context

PR-3 estimates a changing scalar through one configured channel. Multiple monitors can share
errors or copy a peer majority. Their agreement must not silently receive independent-sensor credit.
Existing measurement records already carry source, provenance and a declared correlation group.

## Decision

Add a pure scalar source-fusion API, defaulting to Covariance Intersection. Support a conservative
maximum-marginal bound and an unresolved result with no numbers. With a full source covariance and
provenance, compute a declared convex mean and its exact variance, including singular PSD cases.
Do not optimize weights or add a multidimensional state estimator. Use bounded-size rational
arithmetic without a new dependency; preserve inputs and the actual numerical coefficients.

Reject known zero cross-covariance for shared evidence/groups and post-discussion pairs. Other
known covariance remains usable. This is a conservative policy, not proof of physical correlation.

Add an in-memory cohort collector over existing verified reconciliation events. Release the first
peer packet only after every member has a valid explicitly verified judgment. Seal originals before
release; preserve them for later comparison and force exposure metadata on post-discussion fusion.
The host enforces actual isolation and authenticates identities outside this library.

## Alternatives and consequences

An inverse-covariance optimal mean would need a policy for singular/ill-conditioned matrices and
negative weights. The declared convex mean has transparent behavior and handles perfect copies.
It may be inefficient. Exact PSD checks may reject a rounded empirical covariance; the host must
review its covariance estimator rather than rely on automatic jitter.

Default independent averaging would count duplicated information. CI avoids that reduction only
under valid input error bounds; biased or miscalibrated sources remain a host responsibility.
Full covariance need not be Gaussian, but its numerical assumptions do not establish semantic truth.

No runtime hook, optimizer reward, action authority, raw-observation fabrication or automatic
temporal-prior composition is added. The source fusion result is a diagnostic estimate. Later
evidence/memory contracts and multi-hypothesis handling remain outside PR-4.

## Validation

The [guide](../scalar_fusion.md) gives equations, executable test commands and the independent /
perfectly correlated / unknown comparison. Tests cover strict source covariance, extreme values,
round freezing, verification failure and post-discussion dependence. The
[research register](../recommendations/RESEARCH_TRACEABILITY.md#ref-ci) separates source findings
from repository hypotheses; no published agent or quantum benchmark is reproduced.
