# Phase 2 baseline evidence

Maturity: **Diagnostic** software-validation evidence. Source baseline:
`fcc4142b2a5f0bdd9e9f4f7a8b90d5c72d13d923`. No scientific alignment claim follows.

`manifest.json` binds the preserved artifacts by SHA-256. It records all complete attempts,
including sandbox/environment failures and the final passing full run. Original log paths are
retained for traceability. No credentials or model weights are present.

The final run is `baseline-final-pytest.log`: 7,064 passed, 11 skipped, plus 162 subtests.
JUnit XML is gzip-compressed without a creation timestamp; use `gzip -dc <name>.xml.gz` to read
the complete per-test records. Pytest console and JUnit accounting differ for collection skips
and subtests; preserve the originals rather than rewriting counts.

`dependencies.txt` records the final installed versions, including pip needed by wheel tests.
`baseline-source-hashes.json` freezes the inspected foundational files. Separate logs record
Ruff, Black, CI mypy checks, focused documentation tests and fresh design-review disposition.

Reproduction: install the declared extras in Python 3.12 and run the repository's full pytest
suite with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Local loopback sockets and isolated wheel dependency installation must be permitted. The suite
includes local HTTP fixtures and CPU-DDP; those fixtures are not external target testing.
