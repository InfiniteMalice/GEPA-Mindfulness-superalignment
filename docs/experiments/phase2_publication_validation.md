# Phase 2 publication review validation

Maturity: **Experimental** software corrections and **Diagnostic** evidence. Original research
history, raw measurements and prior validation snapshots are retained unchanged. The maintainer
resumed publishing/merging and confirmed draft handling was unnecessary. PR 831 is merged; PRs
832–840 are open and ready. This report precedes their fresh current-head CI and remaining merges.

The original publication mirrors had green Python 3.12 CI; stages 4–9 failed on Python 3.10 at an
exact float comparison, 2.0000000000000004 versus 2. The fixture now uses absolute tolerance 1e-12,
zero relative tolerance, and explicitly simulates pre-3.12 sequential accumulation. Falsification,
sign errors and original metric computation are unchanged. This is a test portability correction.

Verified review findings were reproduced before fixes: newline-normalizing constitutional hashing;
conflicting certificate source kinds aborting unrelated benchmark rows; private scratchpad spans
contaminating a purported public projection; report geometries fitting the tensor budget but
exceeding four-MiB serialization; root-level files manifests acquiring a forbidden dot prefix.
Three first regressions failed and seven further cases failed before correction. One focused
pass found a mixed colon/bracket private-header case and retained that failure before refinement.

A fresh read-only whole-fix review found one additional Important issue: repeated multiline
private bracket headers were accepted by the legacy parser but missed by the guard. Three new
regressions failed before matching that grammar. The fresh reviewer found no Critical or Minor
issues and no reward, authority or efficacy expansion. No findings are deferred. Original and
failed attempts remain in the adjacent compressed check logs; initial remote CI job evidence is
retained separately with its explicitly identified Git blob digest.

Final focused gate: **104 passed**, 93.21 seconds. Fresh full regression: **7,326 passed, 11 skipped,
18 warnings, 162 subtests**, 576.47 seconds, exit 0. Repository Ruff and Black (866 Python files),
both existing CI mypy commands and explicit types for the four changed modules pass. Full tests
include wheel-resource validation. Original stdout and compressed JUnit retain per-test evidence.
All six comparison studies still run: all **1,104 measured position-level output rows**, original
logits, condition statuses and actual downstream metrics match the original probe exactly.
Original source models and Torch RNG remain unchanged. Wall-time fields are new observations.

CVA invalid proofs remain diagnostic, cannot establish success or infeasibility, and no longer
abort unaffected captures. Private/public overlap is rejected rather than exported. Conservative
report bounds include the frozen source and reject before capture/replay; some shorter concrete
encodings may fit even when the safe bound rejects them. This is an explicit bounded prototype
limitation. Root paths remain traversal-checked. The Constitution itself and the existing V5,
verified-process reward and runtime/promotion authority owners remain unchanged.

The missing repo-quality-gate skill uses the maintainer-approved fallback: scoped plan, test-first
fixes, fresh review, red/green reproduction, formatting/lint/types, full regression and this report.
Beads automation is not invoked. No learned models, training, parameter surgery, punitive circuit
signal or qualified alignment/capability improvement is claimed. Reviewability and original
source pins are preserved while corrections propagate to their stages and descendants through
fast-forward merge commits, without force pushes or history rewriting.
