"""Contracts for deterministic reward-integrity preference-pair generation."""

# Standard library
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Third-party
import pytest

# Local
from scripts.build_reward_integrity_rl_dataset import PAIR_RULES, build_dataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "synthetic" / "reward_integrity" / "reward_integrity_curriculum_v1.jsonl"


def build_into(output_dir: Path):
    """Build checked-in source into an isolated directory."""
    output_dir.mkdir()
    return build_dataset(
        SOURCE,
        output_dir / "rl_pairs_v1.jsonl",
        output_dir / "curriculum_manifest.json",
    )


def test_builder_is_byte_deterministic(tmp_path: Path) -> None:
    """A timestamp, nondeterministic ordering, or path leak would break reproducible training."""
    first = build_into(tmp_path / "first")
    second = build_into(tmp_path / "second")

    assert first.pairs.read_bytes() == second.pairs.read_bytes()
    assert first.manifest.read_bytes() == second.manifest.read_bytes()


def test_builder_emits_six_ordered_preference_pairs_per_case_with_provenance(
    tmp_path: Path,
) -> None:
    """Missing rule pairs or source hashes would make preferences incomplete or unauditable."""
    result = build_into(tmp_path / "build")
    pairs = [
        json.loads(line)
        for line in result.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    assert result.case_count == 8
    assert result.pair_count == 48
    assert len(pairs) == 48
    assert [pair["pair_rule"] for pair in pairs[:6]] == [rule[2] for rule in PAIR_RULES]
    assert {pair["source_case_id"] for pair in pairs} == {
        f"reward-integrity-{number:03d}" for number in range(1, 9)
    }
    assert {pair["source_sha256"] for pair in pairs} == {source_sha256}
    assert {pair["source_path"] for pair in pairs} == {
        "data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl"
    }
    assert all(pair["source_line"] in range(1, 9) for pair in pairs)
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)


def test_manifest_hashes_the_source_and_generated_pairs(tmp_path: Path) -> None:
    """An incorrect manifest hash would silently sever dataset provenance."""
    result = build_into(tmp_path / "build")
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))

    assert manifest["case_count"] == 8
    assert manifest["pair_count"] == 48
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert (
        manifest["source_path"]
        == "data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl"
    )
    assert manifest["pairs_sha256"] == hashlib.sha256(result.pairs.read_bytes()).hexdigest()
    assert manifest["pair_rules"] == [rule[2] for rule in PAIR_RULES]


def test_builder_script_runs_from_the_repository_root() -> None:
    """A script-only import path would make the documented rebuild command unusable."""
    completed = subprocess.run(
        [sys.executable, "scripts/build_reward_integrity_rl_dataset.py"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "built 8 cases and 48 preference pairs" in completed.stdout


def test_builder_rejects_duplicate_preference_orderings(tmp_path: Path) -> None:
    """An extra duplicate relation would make the documented six-rule curriculum ambiguous."""
    source = tmp_path / "source.jsonl"
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    records[0]["reward_integrity"]["preference_orderings"].append(
        records[0]["reward_integrity"]["preference_orderings"][0]
    )
    source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid preference orderings"):
        build_dataset(source, tmp_path / "pairs.jsonl", tmp_path / "manifest.json")


def test_addiction_pairs_contain_recovery_mechanisms_in_trainable_text(tmp_path: Path) -> None:
    """Notes omitted from pairs cannot teach the addiction-specific reward-integrity concepts."""
    result = build_into(tmp_path / "build")
    addiction_text = " ".join(
        f"{pair['prompt']} {pair['chosen']} {pair['rejected']}"
        for pair in (
            json.loads(line)
            for line in result.pairs.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if pair["source_case_id"] == "reward-integrity-002"
    ).lower()

    for term in ("motivation", "incentive salience", "habit", "reduced agency", "recovery"):
        assert term in addiction_text
