# Beads Setup

See [AGENTS.md](../AGENTS.md) for the canonical coding rules that beads will track.
This README focuses on beads-specific setup and ingestion notes.

## Onboarding

Beads has been manually enabled for this repository. Run `bd onboard` from the repository root
to fetch integration instructions and confirm ingestion is configured.

If `bd` is not found, ensure the beads CLI is on your PATH before retrying the onboarding
command. Reach out to the maintainer of the manual install for the expected binary location if
the CLI remains unavailable.

## Workflow expectations

- Keep `AGENTS.md` as the canonical source of repository rules; beads monitors that file.
- Run `bd status` before committing to verify the repository link and ingestion state.
- After updating `AGENTS.md`, rerun `bd onboard` if beads requests refreshed instructions.

## BEADS USAGE

Beads should ingest `AGENTS.md` as the authoritative source of repository rules.
Use this README as a pointer to that file and as a place to record beads-specific guidance or
troubleshooting notes.
