# Final publication validation

Maturity: **Experimental** software and **Diagnostic**, `DEVELOPMENT` evidence. No optimizer input,
parameter change or promotion authority. Alignment improvement and trained-model capability
retention remain **Unqualified**; trained candidate comparisons remain `NOT_RUN`.

The final review found an explanatory wording mismatch: a correctly `CONTRADICTED` finding still
received a support-worded reason. A new assertion on the existing falsification test reproduced
it. Reasons now explicitly distinguish falsified, nonspecific and supported predictions without
changing their evidence levels, logits or computed effects. All 20 mechanistic-evidence tests pass
in 3.47 seconds. The full offline CPU regression passes **7,338 tests**, with 11 skipped,
18 warnings and 162 subtests, in **586.24 seconds**. Repository Ruff/Black pass and the changed
verifier passes supplemental mypy; the protected CI type gates are unchanged and passed earlier.
Current-head remote CI must pass before merge. Skipped CUDA/Mojo paths remain untested.

The stage-7 CLI was also rerun, preserving actual measurements in `phase2_final_pr7_probe.json.gz`
and explicitly hashing the mechanistic verifier dependency. This adds source provenance, not a
new independent alignment sample. Earlier raw runs, failures, 120-test qualification gate,
7,338-test compatibility gate, ten-stage/136-file historical audit and original-history bundle
remain available. The fresh static review's single Important nested-legacy compatibility finding
was reproduced and fixed; its limitations are recorded in `phase2_prefix_reviewed_validation.md`.

Historical `base_commit` fields identify predecessors. The adjacent snapshot index records tested
full revisions and initial mirrors; historical artifact hashes remain unchanged. The final
publication manifest separately binds current source/artifact bytes. Canonical V5 remains exactly
17 cases, case 0 noncanonical; the Constitution retains 21 numbered sections. Existing reward,
learning-surface and coevolution owners remain unchanged. No diagnostic finding can grant model
training credit or authority. Actual PR timelines determine later check/review/merge status.
