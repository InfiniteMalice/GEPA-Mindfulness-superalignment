"""Print A-K configurations, exercise fixtures, or summarize host-measured trial JSONL."""

import argparse
import json
from pathlib import Path

from evaluation.epistemic_ablations import ablation_matrix, fixture_smoke, summarize_trials


def main() -> None:
    """Run explicit offline operations; never invoke a model or hidden evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Host-measured trial JSONL")
    parser.add_argument("--fixture-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input and args.fixture_smoke:
        parser.error("choose trial input or fixture smoke")
    if args.input:
        rows = [
            json.loads(line)
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = summarize_trials(rows)
    elif args.fixture_smoke:
        result = fixture_smoke()
    else:
        result = {"matrix": ablation_matrix(), "model_benchmark_run": False}
    output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
