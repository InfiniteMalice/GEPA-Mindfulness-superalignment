"""Contracts for streaming reward-integrity rollout adapters."""

# Standard library
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

# Third-party
import pytest

# Local
from gepa_mindfulness import synthetic_dataset_validation
from gepa_mindfulness.training.adapters.flat_jsonl import FlatJSONLAdapter
from gepa_mindfulness.training.adapters.pair_records import validate_pair_record
from gepa_mindfulness.training.adapters.synthetic_cases import SyntheticCaseAdapter

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = (
    ROOT / "data" / "synthetic" / "reward_integrity" / "reward_integrity_curriculum_v1.jsonl"
)
PAIRS_PATH = ROOT / "data" / "synthetic" / "reward_integrity" / "rl_pairs_v1.jsonl"
COMPONENT_NAMES = (
    "objective_fidelity",
    "feedback_integrity",
    "skill_transfer",
    "reality_contact",
    "exploit_disclosure",
    "long_horizon_agency",
    "benign_creativity",
    "repair_quality",
)


def _flat_pair() -> dict[str, object]:
    """Build one complete literal pair-record fixture independent of production helpers."""
    chosen_components = {name: 0.0 for name in COMPONENT_NAMES}
    chosen_components["objective_fidelity"] = 1.0
    rejected_components = {name: 0.0 for name in COMPONENT_NAMES}
    rejected_components["objective_fidelity"] = -1.0
    return {
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
        "chosen_reward_components": chosen_components,
        "rejected_reward_components": rejected_components,
        "diagnostics": {"central": "Question", "supporting": []},
        "schema_version": "reward-integrity-rl-pairs-v1",
    }


def _flat_pair_with(field: str, value: str) -> str:
    """Build a literal valid pair fixture with one hand-selected invalid field."""
    row = _flat_pair()
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

    assert request.case_id == "reward-integrity-001:grounded_over_proxy"
    assert request.metadata["source_case_id"] == "reward-integrity-001"
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
        ("schema_version", "reward-integrity-rl-pairs-v2", "schema_version"),
        ("source_sha256", "A" * 64, "source_sha256"),
        ("source_sha256", "a" * 63, "source_sha256"),
        ("source_line", 0, "source_line"),
        ("source_case_version", "0", "source_case_version"),
        ("source_case_version", "latest", "source_case_version"),
        ("chosen_class", "reward_tampering", "legal preference relation"),
        ("pair_rule", "agency_over_immediate_reward", "legal preference relation"),
        ("record_id", "unrelated-record", "record_id"),
        ("chosen", 7, "chosen"),
        ("diagnostics", {"central": "Question", "supporting": [], "extra": True}, "unknown"),
    ],
)
def test_flat_pair_validator_rejects_invalid_schema_semantics_and_types(
    field: str,
    value: object,
    expected: str,
) -> None:
    """Pair rows must match the exact versioned relation, provenance, and field contracts."""
    row = _flat_pair()
    row[field] = value

    with pytest.raises(ValueError, match=expected):
        validate_pair_record(row, Path("pairs.jsonl"), 1)


@pytest.mark.parametrize(
    ("field", "component", "value", "expected"),
    [
        ("chosen_reward_components", "objective_fidelity", 1.01, r"\[-1.0, 1.0\]"),
        ("rejected_reward_components", "objective_fidelity", True, "number"),
        ("chosen_reward_components", "unknown_component", 0.0, "exactly"),
    ],
)
def test_flat_pair_validator_requires_exact_bounded_component_maps(
    field: str,
    component: str,
    value: object,
    expected: str,
) -> None:
    """Partial, expanded, boolean, or out-of-range component maps are invalid pair records."""
    row = _flat_pair()
    components = row[field]
    assert isinstance(components, dict)
    components[component] = value

    with pytest.raises(ValueError, match=expected):
        validate_pair_record(row, Path("pairs.jsonl"), 1)


def test_flat_pair_validator_rejects_missing_and_unknown_top_level_fields() -> None:
    """Versioned pair rows are closed records whose required fields cannot disappear."""
    missing = _flat_pair()
    missing.pop("chosen")
    unknown = _flat_pair()
    unknown["private_reasoning"] = "not observable"

    with pytest.raises(ValueError, match="missing required field 'chosen'"):
        validate_pair_record(missing, Path("pairs.jsonl"), 1)
    with pytest.raises(ValueError, match="unknown field 'private_reasoning'"):
        validate_pair_record(unknown, Path("pairs.jsonl"), 1)


def test_flat_pair_validator_rejects_non_string_field_names_cleanly() -> None:
    """Direct validator callers receive a diagnostic, not a mixed-key sorting error."""
    row = _flat_pair()
    row[7] = "invalid"  # type: ignore[index]
    row["extra"] = "invalid"

    with pytest.raises(ValueError, match="field names.*strings"):
        validate_pair_record(row, Path("pairs.jsonl"), 1)


def test_flat_adapter_uses_the_shared_pair_record_validator(tmp_path: Path) -> None:
    """Adapter callers receive the same strict diagnostics as direct validator users."""
    row = _flat_pair()
    row["source_sha256"] = "not-a-sha"
    path = tmp_path / "invalid-pair.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_sha256") as error:
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
        (
            "reward_integrity.response_classes.grounded_success.response",
            7,
            "record.reward_integrity.response_classes.grounded_success.response must be string",
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


def test_synthetic_adapter_rejects_negative_reward_component_without_evidence(
    tmp_path: Path,
) -> None:
    """Negative components without observable evidence would make retained data unauditable."""
    path = tmp_path / "unauditable-rich-case.jsonl"
    path.write_text(
        _rich_case_with(
            "reward_integrity.response_classes.proxy_exploitation.negative_evidence", {}
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="negative component 'objective_fidelity' needs evidence"
    ) as error:
        next(SyntheticCaseAdapter(path).iter_requests())

    assert f"{path}:1:" in str(error.value)


def test_reward_integrity_semantics_follow_loaded_schema_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-valid renamed structures must not be rejected by stale validator constants."""
    schema = deepcopy(synthetic_dataset_validation._load_schema())
    reward_schema = schema["properties"]["reward_integrity"]["properties"]
    response_schema = reward_schema["response_classes"]
    response_schema["required"] = [
        "verified_success" if name == "grounded_success" else name
        for name in response_schema["required"]
    ]
    response_schema["properties"]["verified_success"] = response_schema["properties"].pop(
        "grounded_success"
    )

    component_schema = schema["$defs"]["reward_integrity_component_map"]
    component_schema["required"] = [
        "goal_fidelity" if name == "objective_fidelity" else name
        for name in component_schema["required"]
    ]
    component_schema["properties"]["goal_fidelity"] = component_schema["properties"].pop(
        "objective_fidelity"
    )
    evidence_schema = schema["$defs"]["reward_integrity_response"]["properties"][
        "negative_evidence"
    ]["properties"]
    evidence_schema["goal_fidelity"] = evidence_schema.pop("objective_fidelity")

    record = json.loads(CASES_PATH.read_text(encoding="utf-8").splitlines()[0])
    integrity = record["reward_integrity"]
    integrity["response_classes"]["verified_success"] = integrity["response_classes"].pop(
        "grounded_success"
    )
    integrity["component_targets"]["goal_fidelity"] = integrity["component_targets"].pop(
        "objective_fidelity"
    )
    for response in integrity["response_classes"].values():
        components = response["reward_components"]
        components["goal_fidelity"] = components.pop("objective_fidelity")
        evidence = response["negative_evidence"]
        if "objective_fidelity" in evidence:
            evidence["goal_fidelity"] = evidence.pop("objective_fidelity")

    monkeypatch.setattr(synthetic_dataset_validation, "_SCHEMA_CACHE", schema)

    assert synthetic_dataset_validation.validate_rich_record(record) == []


def test_adapter_import_succeeds_from_an_installed_wheel(tmp_path: Path) -> None:
    """Packaged adapters must not rely on the repository-only scripts directory."""
    wheel_directory = tmp_path / "wheel"
    target_directory = tmp_path / "installed"
    case_path = tmp_path / "rich-case.jsonl"
    wheel_directory.mkdir()
    target_directory.mkdir()
    case_path.write_text(
        CASES_PATH.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8"
    )

    build_root = ROOT
    mapped_drive: str | None = None
    if os.name == "nt":
        for drive_letter in "ZYXWVUT":
            candidate = f"{drive_letter}:"
            mapping = subprocess.run(
                ["subst", candidate, str(ROOT)],
                check=False,
                capture_output=True,
                text=True,
            )
            if mapping.returncode == 0:
                mapped_drive = candidate
                break
        if mapped_drive is None:
            pytest.fail("No temporary drive letter is available for the wheel smoke test.")
        build_root = Path(f"{mapped_drive}/")

    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_directory)],
            cwd=build_root,
            check=True,
            capture_output=True,
            text=True,
        )
        wheel = next(wheel_directory.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = set(archive.namelist())
        assert not any(name.endswith("configs/rl/hybrid_vulkan_grpo.yaml") for name in wheel_names)
        assert not any(name.endswith("mojo/rl_coordinator/main.mojo") for name in wheel_names)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target_directory),
                str(wheel),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        environment = os.environ | {"PYTHONPATH": str(target_directory)}
        wheel_smoke = (
            "from gepa_mindfulness.training.adapters import SyntheticCaseAdapter; "
            "from pathlib import Path; "
            f"request = next(SyntheticCaseAdapter(Path({str(case_path)!r})).iter_requests()); "
            "print(request.case_id)"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                wheel_smoke,
            ],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    finally:
        if mapped_drive is not None:
            subprocess.run(
                ["subst", mapped_drive, "/D"], check=True, capture_output=True, text=True
            )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "reward-integrity-001"
