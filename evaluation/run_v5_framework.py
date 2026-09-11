"""Emit deterministic V5 evaluation plans without executing a model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from .v5_runner import V5EvaluationCell, plan_v5_cells

_INTEGER_SPELLING = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


def _parse_integer(value: str) -> int:
    """Return one unambiguous decimal integer accepted by the V5 CLI."""

    if _INTEGER_SPELLING.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"invalid integer spelling: {value!r}")
    return int(value)


def _serialize_cell(cell: V5EvaluationCell) -> str:
    """Return one compact JSON object in the documented V5 planned-cell field order."""

    return json.dumps(
        {
            "case_id": cell.case_id,
            "case_version": cell.case_version,
            "stripe_id": cell.stripe_id,
            "subtype": cell.subtype,
            "repeat_id": cell.repeat_id,
            "seed": cell.seed,
            "model_version": cell.model_version,
            "harness_version": cell.harness_version,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _write_jsonl_atomically(output_path: Path, jsonl: str) -> None:
    """Replace one output file only after a same-directory temporary file is closed."""

    parent = output_path.parent
    if not parent.exists():
        raise ValueError(f"output parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise ValueError(f"output parent path is not a directory: {parent}")

    descriptor: int | None = None
    temp_path: Path | None = None
    try:
        descriptor, raw_temp_path = tempfile.mkstemp(
            dir=parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(raw_temp_path)
        temp_file = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = None
        with temp_file:
            temp_file.write(jsonl)
            temp_file.flush()
        os.replace(temp_path, output_path)
        temp_path = None
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not write V5 JSONL output {output_path}: {error}") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated V5 planner command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append", type=_parse_integer, dest="case_ids")
    parser.add_argument("--stripe", action="append", dest="stripe_ids")
    parser.add_argument("--repeats", type=_parse_integer, default=5)
    parser.add_argument("--base-seed", type=_parse_integer, default=0)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--harness-version", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Plan V5 cells and write their deterministic JSONL representation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("model execution is not implemented; pass --dry-run")

    try:
        cells = plan_v5_cells(
            case_ids=args.case_ids,
            stripe_ids=args.stripe_ids,
            repeats=args.repeats,
            base_seed=args.base_seed,
            model_version=args.model_version,
            harness_version=args.harness_version,
        )
        jsonl = "\n".join(_serialize_cell(cell) for cell in cells) + "\n"
        if args.output is not None:
            _write_jsonl_atomically(args.output, jsonl)
    except ValueError as error:
        parser.error(str(error))

    if args.output is None:
        sys.stdout.write(jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
