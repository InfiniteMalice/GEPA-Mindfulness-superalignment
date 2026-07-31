"""Contracts for streaming reward-integrity rollout adapters."""

# Standard library
import hashlib
import json
from pathlib import Path

# Third-party
import pytest

# Local
from gepa_mindfulness.training.adapters.flat_jsonl import FlatJSONLAdapter
from gepa_mindfulness.training.adapters.synthetic_cases import SyntheticCaseAdapter

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = (
    ROOT / "data" / "synthetic" / "reward_integrity" / "reward_integrity_curriculum_v1.jsonl"
)
PAIRS_PATH = ROOT / "data" / "synthetic" / "reward_integrity" / "rl_pairs_v1.jsonl"


def _flat_pair_with(field: str, value: str) -> str:
    """Build a literal valid pair fixture with one hand-selected invalid field."""
    row = {
        "record_id": "reward-integrity-001:grounded_over_proxy",
        "source_case_id": "reward-integrity-001",
        "source_case_version": "1.0",
        "source_path": "data/source.jsonl",
        "source_line": 1,
        "source_sha256": "a" * 64,
        "pair_rule": "grounded_over_proxy",
        "prompt": "Prompt",
        "chosen": "Chosen",
        "rejected": "Rejected",
        "chosen_class": "grounded_success",
        "rejected_class": "proxy_exploitation",
        "chosen_reward_components": {"objective_fidelity": 1.0},
        "rejected_reward_components": {"objective_fidelity": -1.0},
        "diagnostics": {"central": "Question", "supporting": []},
        "schema_version": "reward-integrity-rl-pairs-v1",
    }
    row[field] = json.loads(value)
    return json.dumps(row) + "\n"


def _synthetic_case_with(field: str, value: str) -> str:
    """Build a literal valid rich-case fixture with one hand-selected invalid field."""
    row = {
        "id": "reward-integrity-001",
        "version": "1.0",
        "scenario": {"summary": "Scenario"},
        "reward_integrity": {"central_diagnostic": "Question"},
    }
    row[field] = json.loads(value)
    return json.dumps(row) + "\n"


def _rich_case_with(field: str, value: object) -> str:
    """Build a complete rich-case fixture with one deliberately malformed nested value."""
    row = json.loads(CASES_PATH.read_text(encoding="utf-8").splitlines()[0])
    target = row
    field_parts = field.split(".")
    for part in field_parts[:-1]:
        target = target[part]
    target[field_parts[-1]] = value
    return json.dumps(row) + "\n"


def test_flat_adapter_preserves_pair_provenance() -> None:
    """Dropping pair provenance would make a rollout request unauditable."""
    request = next(FlatJSONLAdapter(PAIRS_PATH).iter_requests())

    assert request.case_id == "reward-integrity-001"
    assert request.metadata["source_case_version"] == "1.0"
    assert request.metadata["source_line"] == 1
    assert request.metadata["source_sha256"] == hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    assert request.metadata["source_path"] == (
        "data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl"
    )


def test_synthetic_case_adapter_adds_file_provenance() -> None:
    """Rich cases need their file hash and line to trace requests back to source."""
    request = next(SyntheticCaseAdapter(CASES_PATH).iter_requests())

    assert request.case_id == "reward-integrity-001"
    assert request.metadata["source_case_version"] == "1.0"
    assert request.metadata["source_line"] == 1
    assert request.metadata["source_sha256"] == hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    assert request.metadata["source_path"] == CASES_PATH.as_posix()


@pytest.mark.parametrize(
    ("adapter", "contents", "expected"),
    [
        (FlatJSONLAdapter, "{not-json}\n", "invalid JSON"),
        (FlatJSONLAdapter, "[]\n", "expected a JSON object"),
        (FlatJSONLAdapter, '{"record_id":"pair-1"}\n', "missing required field 'prompt'"),
        (FlatJSONLAdapter, _flat_pair_with("source_line", '"one"'), "expected an integer"),
        (SyntheticCaseAdapter, "{not-json}\n", "invalid JSON"),
        (SyntheticCaseAdapter, "[]\n", "expected a JSON object"),
        (SyntheticCaseAdapter, '{"id":"case-1"}\n', "missing required field 'version'"),
        (SyntheticCaseAdapter, _synthetic_case_with("version", "1"), "expected a string"),
    ],
)
def test_adapters_reject_malformed_rows_with_path_and_line(
    tmp_path: Path,
    adapter: type[FlatJSONLAdapter] | type[SyntheticCaseAdapter],
    contents: str,
    expected: str,
) -> None:
    """Permissive rows would hide broken data and lose the repair location."""
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=expected) as error:
        next(adapter(path).iter_requests())

    assert f"{path}:1:" in str(error.value)


def test_synthetic_adapter_reports_the_physical_line_after_blank_rows(tmp_path: Path) -> None:
    """Counting only records would send a dataset repair to the wrong source line."""
    path = tmp_path / "blank-lines.jsonl"
    path.write_text('\n\n{"id":"case-1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field 'version'") as error:
        next(SyntheticCaseAdapter(path).iter_requests())

    assert f"{path}:3:" in str(error.value)


@pytest.mark.parametrize(
    ("adapter", "contents"),
    [
        (FlatJSONLAdapter, _flat_pair_with("chosen_reward_components", '{"x": NaN}')),
        (SyntheticCaseAdapter, _rich_case_with("case_metadata", float("inf"))),
    ],
)
def test_adapters_reject_non_standard_json_number_constants(
    tmp_path: Path,
    adapter: type[FlatJSONLAdapter] | type[SyntheticCaseAdapter],
    contents: str,
) -> None:
    """Permissive JSON constants would make invalid numeric metadata appear valid."""
    path = tmp_path / "non-standard-number.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON number constant") as error:
        next(adapter(path).iter_requests())

    assert f"{path}:1:" in str(error.value)


def test_flat_adapter_rejects_non_finite_json_numbers(tmp_path: Path) -> None:
    """A finite-looking JSON exponent must not overflow into an accepted reward component."""
    path = tmp_path / "non-finite-number.jsonl"
    contents = _flat_pair_with("chosen_reward_components", '{"objective_fidelity": 1.0}')
    path.write_text(
        contents.replace('"objective_fidelity": 1.0', '"objective_fidelity": 1e999'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite JSON number") as error:
        next(FlatJSONLAdapter(path).iter_requests())

    assert f"{path}:1:" in str(error.value)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("case_metadata", [], "record.case_metadata must be object"),
        (
            "reward_integrity.response_classes",
            [],
            "record.reward_integrity.response_classes must be object",
        ),
        (
            "reward_integrity.supporting_diagnostics",
            "not-a-list",
            "record.reward_integrity.supporting_diagnostics must be array",
        ),
    ],
)
def test_synthetic_adapter_rejects_malformed_retained_nested_source_data(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    """Unvalidated retained source data would bypass the rich curriculum contract."""
    path = tmp_path / "malformed-rich-case.jsonl"
    path.write_text(_rich_case_with(field, value), encoding="utf-8")

    with pytest.raises(ValueError, match=expected) as error:
        next(SyntheticCaseAdapter(path).iter_requests())

    assert f"{path}:1:" in str(error.value)
