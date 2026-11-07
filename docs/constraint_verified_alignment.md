# Constraint-Verified Alignment (CVA)

Maturity: **Experimental** opt-in finite research prototype. Records and reports are
**Diagnostic**. Real-world verification, SAT/SMT proof formats and empirical alignment improvement
are **Unsupported** by this prototype. Ordinary inference and training do not call these APIs.

## Meaning of a verified result

For a modeled candidate `a`, `S_valid = {a | G(a) AND T(a) AND P(a) AND E(a)}`. Objective achievement
`G` is separate from technical conditions `T`, authorization/scope `P` and independently established
enforceable ethical/safety conditions `E`. These are conjunctive conditions, not tradeable objective
weights. They do not numerically rank the Constitution's three co-equal imperatives or reduce all
ethical values to Boolean predicates. Unresolved normative judgments require review and prevent
verified success. The host owns the formalization and applicable authority ordering.

The finite fragment uses explicit integer domains and conjunctions of typed linear relations
(`eq`, `ne`, `le`, `ge`). Variables, predicates and candidate values contain data only: no expression
evaluation, generated code, endpoints, shell commands or external actions. Arithmetic uses exact
integers. Bounds are 16 variables, 16 values per variable, 128 predicates and at most 4,096 total
assignments; the domain-size cap is checked before Cartesian expansion. Integer inputs are exact
JSON-safe integers; booleans and floats are rejected. The full JSON envelope is bounded to 4 MiB
and depth 32 using the existing validation contract.

An exact conclusion is conditional on the exact constraint-set digest, version, domain, scope and
completeness assumptions. Compact representation does not imply efficient arbitrary feasibility
checking, and the prototype supplies no general decision procedure for moral alignment.

## Typed API reference

All records live in `gepa_mindfulness.verification.cva_records`; each has an explicit versioned
schema, exact `from_dict`/detached `to_dict` and a canonical content `digest`. They reuse the
existing `DiagnosticRecord`, canonical JSON digest, observable `EvidenceReference` and eligibility
contracts. Serialized metadata always declares `training_eligibility="DEVELOPMENT"`,
`maturity="experimental"`, `diagnostic_status="diagnostic"`, `optimizer_input=False` and
`authority_granted=False`. Relabeling the outer envelope TRAIN does not override a nested restriction.

| Record or enum | Contract |
| --- | --- |
| `FiniteVariable(name, values)` | Unique bounded exact integer domain |
| `LinearPredicate(predicate_id, kind, coefficients, relation, bound)` | One closed relation with `PredicateKind.GOAL/TECHNICAL/AUTHORIZATION/ETHICAL` |
| `ConstraintSet` | Task/set/version/objective, variables/predicates, authority hierarchy, authorization scope, original evidence, completeness assumptions and unresolved normative judgments |
| `CandidatePlan` | Exact task/constraint digest, finite assignment, public description and optional original `ActionBinding` |
| `OutcomeCertificate` | Exact task/constraint/candidate binding, certificate type, scope, guarantee, witness or excluded assignments and artifact evidence references |
| `VerificationRecord` | Common subject identity, authority hierarchy, objective, method, actual checker ID/version, host evaluator contract, certificate digest/type, evidence, guarantee, assumptions, uncertainty, advisory next actions, provenance and aware host timestamp |
| `CandidateVerification` | `VerificationRecord`, a `CandidateVerdict` and checked failed-predicate identities |
| `FeasibilityVerification` | `VerificationRecord`, a separate `FeasibilityVerdict` and exact scope |
| `NoncompletionAssessment` | What was attempted/observed, encountered constraints, separate proof reference if any, unknowns, needed information/authorization and legitimate alternatives |
| `SearchReport` | All attempted finite assignments, coverage flag, optional unverified candidate and evidence-limited noncompletion assessment |
| `VerificationInput` | Detached complete task/candidate/certificate subject authenticated by the host |

Candidate verdicts are `VERIFIED_SUCCESS`, `CONSTRAINT_VIOLATION`, `UNRESOLVED`,
`INSUFFICIENT_AUTHORIZATION`, `VERIFIER_UNAVAILABLE` and `INVALID_CERTIFICATE`. Candidate rejection
cannot serialize as `VERIFIED_INFEASIBLE`. Feasibility verdicts separately include
`VERIFIED_FEASIBLE`, `VERIFIED_INFEASIBLE` and the four unresolved/authorization/checker/certificate
conditions. Invalid candidates never establish task-wide infeasibility.

`FeasibilityScope.TECHNICAL` means `G AND T`; `CONSTRAINT_VALID` means `G AND T AND P AND E`.
A technical witness can establish technical feasibility while violating permissions. Such a
certificate cannot establish legitimate candidate success. Changing authorization, evidence,
constraints, versions or assumptions changes the subject digest; earlier certificates cannot be
silently reused. A verified plan is not an observed execution result.

`GuaranteeKind` distinguishes `EXACT`, `CERTIFIED_APPROXIMATION`, `STATISTICAL_ESTIMATE`, `HEURISTIC`
and `UNKNOWN`. The finite checker accepts only exact finite certificates. It does not check an
approximation bound, randomized count, confidence interval or optional solver proof. Unsupported
formats/guarantees remain explicitly rejected; no unverified bound is promoted to exact evidence.

## Construction is separate from independent checking

`gepa_mindfulness.verification.cva_finite` provides:

```python
build_witness_certificate(constraints, candidate, *, evidence_refs,
                          scope=FeasibilityScope.CONSTRAINT_VALID, enabled=False)
build_exhaustion_certificate(constraints, *, scope, evidence_refs, enabled=False)
search_finite(constraints, *, budget, enabled=False)
```

Builders package proposed artifacts and grant no verified status. Witness construction does not
assert that the assignment works. Exhaustion construction records the enumerated rejected rows;
if a satisfying row exists, exclusion coverage is incomplete. Even a fully exhausted search
report remains `NO_AUTHORIZED_SOLUTION_ESTABLISHED` until a separate checker validates a
certificate. A found candidate remains `UNVERIFIED_CANDIDATE_FOUND`.

`gepa_mindfulness.verification.cva` provides:

```python
verify_candidate(constraints, candidate, certificate, *, context=None, enabled=False)
verify_feasibility(constraints, certificate, *, candidate=None, context=None, enabled=False)
bind_candidate_action(candidate, action, prediction)
```

The candidate checker independently recomputes membership and every modeled condition. The
feasibility checker validates a supplied witness with its matching candidate, or checks exclusion
coverage independently of the builder: every row is a distinct domain member, the count equals
the exact product of domain sizes, and every row fails at least one predicate in the declared
scope. It uses separately implemented checking-side arithmetic. Duplicate, missing, out-of-domain,
incorrectly excluded or stale rows cannot establish infeasibility. A checked valid witness proves
feasibility, not optimality unless the objective and separate obligations actually specify it.

An absent certificate stays UNRESOLVED. UNKNOWN is not UNSAT. Strings or model-authored solver
labels cannot substitute for `OutcomeCertificate`; unsupported certificate records return
`INVALID_CERTIFICATE`. No SAT/SMT library or proof parser is installed by this path.

## Trusted host integration

`HostVerificationContext(evaluator, actor_id, clock, authenticate, checker_available=True)` uses
the existing `TrustedEvaluatorContract` and `TrustedClock` interface. The actual deterministic
checker is `gepa-finite-integer-cva`, version `1`; host reviewer identity is recorded separately.
All execution APIs require exact `enabled=True` before reading input/calling a reviewer.

The host must authenticate the complete formalization, original evidence, current authorization
scope, applicable authority hierarchy, certificate source, and genuine reviewer independence.
It supplies the callback from trusted application code, never generated/model/scenario code.
Different role names or models agreeing do not establish independence. Self-review identities are
rejected structurally, but different IDs alone are no authentication guarantee.

The callback receives a detached complete `VerificationInput`. Missing, throwing, mutating,
false or merely truthy authentication remains unresolved; private exception text is not exported.
Missing authorization fails closed. An unavailable checker remains `VERIFIER_UNAVAILABLE`, which
does not establish either existence or absence of a solution. Invalid host clocks fail clearly.
Conflicting evidence identities are rejected before authentication. Output records preserve both
task and certificate evidence (up to 64 distinct references) and all declared uncertainty.

This is a trusted-host Python interface, not a sandbox or identity provider against arbitrary
Python callers. Constructing/deserializing a `VERIFIED_SUCCESS` record does not admit it to any
accepted store. Host publication must independently authenticate the exact finding and still use
the existing [governed evidence commit](VERIFICATION_AND_RUNTIME_AUTHORITY.md), immutable action
events and one-use scoped WRITE authorization. No automatic mapping claims local execution or
observed real-world outcomes from finite plan checking.

`bind_candidate_action` reuses existing `ActionRecord`, `PredictionCommit` and canonical action
digest contracts. The prediction's public outcome must be exactly
`{"cva_candidate_digest": candidate.digest}` before binding. It retains original action/prediction
identities/digests and scope; the host separately validates chronology, evidence and actual grants.
The helper does not execute, consume a grant or certify an action outcome. Advisory
`permitted_next_actions` descriptions convey no permission.

Preserve original task, certificate, candidate, action events and evidence artifacts alongside
their digests in the host audit store. Diagnostics cannot authorize optimizer credit, hidden-thought
penalties, circuit-based punishment or constitutional overrides. Existing ActionOPD/VPO/KnowGPO/
PRM4RL/training and protected promotion owners are unchanged. See the
[verified reward contract](epistemic_process_rewards.md) and [architecture](adr/0026-cva-causal-interpretability.md).

## Verification and limitations

Run `python -m pytest -q tests/test_cva.py tests/test_cva_records.py` and existing verification,
runtime-authority, governed-evidence, reward and V5 regressions. Tests cover valid/invalid witnesses,
technical versus authorized feasibility, independently checked contradictions, incomplete coverage,
failed search, missing authorization, unknown norms, unsupported certificates, record tampering,
round trips, capacity limits, clock/callback failure and nested optimizer rejection.

These are deterministic synthetic software checks, not evidence that a language model learned
graceful failure or distinguishes ethical concepts internally. No external security systems,
model endpoint, training update or mechanistic intervention is executed. White-hat fixtures and
V5 overlays are the next independently reviewed stage. Source/methodological limits remain in
the [primary-source audit](phase2_research_sources.md).
