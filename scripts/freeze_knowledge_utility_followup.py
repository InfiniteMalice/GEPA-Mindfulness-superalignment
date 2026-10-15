"""Prepare acceptance evidence or publish an explicitly declared offline packet freeze."""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party
# Local
from scripts import knowledge_utility_followup_freeze_profile as profile
from scripts.knowledge_utility_capture_records import publish_json
from scripts.knowledge_utility_followup_freeze import (
    MAX_EVIDENCE_BYTES,
    MAX_RECORD_BYTES,
    freeze_followup,
    prepare_acceptance,
)
from scripts.knowledge_utility_followup_records import (
    MAX_INPUT_BYTES,
    bounded_text,
    object_fields,
    read_json_object,
)


def read_evidence(manifest_path: Path) -> dict[str, bytes]:
    """Read each explicitly provided file once; embedded evidence references stay inert."""
    manifest = object_fields(
        read_json_object(manifest_path, max_bytes=MAX_RECORD_BYTES), "schema_version files"
    )
    if manifest["schema_version"] != "knowledge-utility-followup-evidence-inputs-v1":
        raise ValueError("unsupported evidence manifest")
    paths = object_fields(manifest["files"], " ".join(profile.EVIDENCE_PINS))
    evidence: dict[str, bytes] = {}
    total = 0
    for role in sorted(paths):
        bounded_text(paths[role], 4096)
        path = Path(paths[role])
        if not path.is_absolute():
            path = manifest_path.parent / path
        cap = min(MAX_INPUT_BYTES, MAX_EVIDENCE_BYTES - total)
        with path.open("rb") as stream:
            data = stream.read(cap + 1)
        if len(data) > cap:
            raise ValueError("evidence exceeds byte limit")
        total += len(data)
        evidence[role] = data
    return evidence


def main(argv: list[str] | None = None) -> None:
    """Validate offline artifacts and publish one new file without inferring approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "freeze"):
        command = operations.add_parser(operation)
        command.add_argument("--packet", type=Path, required=True)
        command.add_argument("--evidence-manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if operation == "freeze":
            command.add_argument("--candidate", type=Path, required=True)
            command.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        packet = read_json_object(args.packet, max_bytes=MAX_INPUT_BYTES)
        evidence = read_evidence(args.evidence_manifest)
        if args.operation == "prepare":
            result = prepare_acceptance(packet, evidence)
        else:
            candidate = read_json_object(args.candidate, max_bytes=MAX_RECORD_BYTES)
            decision = read_json_object(args.decision, max_bytes=MAX_RECORD_BYTES)
            result = freeze_followup(packet, evidence, candidate, decision)
    except (ValueError, OSError):
        print("cannot process follow-up freeze: invalid input", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        publish_json(args.output, result)
    except OSError:
        print("cannot publish follow-up freeze: file operation failed", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
