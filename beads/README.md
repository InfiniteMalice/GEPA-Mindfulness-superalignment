# Beads Setup

See [AGENTS.md](../AGENTS.md) for the canonical coding rules that beads will track.
This README focuses on beads-specific setup and ingestion notes.

## Installation

Run the beads installer from the repository root when network access allows:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh \
  | bash
```

If installation fails with a 403 error while fetching the script, retry the command once network
access is restored.

If the beads CLI is available, run `bd onboard` from the repository root to receive integration
instructions. The CLI is currently unavailable here because the installer cannot be fetched due to
the network 403, so onboarding must wait until the installer can be downloaded and executed.

## BEADS USAGE

Beads should ingest `AGENTS.md` as the authoritative source of repository rules.
Use this README as a pointer to that file and as a place to record beads-specific guidance or
troubleshooting notes.
