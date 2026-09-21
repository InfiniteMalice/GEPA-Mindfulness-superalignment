# PR 754 review corrections

The review identifies three observable defects: replayed assessments accept altered prior-state
features, incomparable observations disappear from their measurement-origin coverage denominator,
and the synthetic retention control reports an event different from the one it audits.

Bind a canonical set of prior-snapshot digests during epistemic assessment, pass those states through
the audit entry point, and require an exact match at recall. Preserve optional empty-state fallback,
but reject changing an assessed state context, including dropping the current state. Preserve
measurement origin independently of comparability and use an explicit mixed bucket for unequal
origins. Pass the retained abstention events into the fixture request and derive the reported action
from that audited request. This remains an observed synthetic fixture, not an inferred policy outcome.

Add failing replay, coverage and audited-fixture regression tests before implementing these fixes.
Correct registry diagnostic text, annotate the shared test helpers, and document private validation
helpers. Keep default flags, policy authority, reward boundaries and source/proxy labels unchanged.
Use the existing superpowers review/TDD workflow and repo-quality-gate, then run focused tests,
formatting, type checks and relevant integration checks before pushing one consolidated fix commit.
