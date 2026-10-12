"""Untrusted pilot captures cannot change the roster, rubric, or verification status."""

# Standard library
import base64
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts import score_knowledge_utility_pilot as scorer
from scripts.prepare_knowledge_utility_pilot import prepare_pilot
from scripts.score_knowledge_utility_pilot import answer_matches, main, score_pilot


@pytest.fixture(scope="module")
def packet() -> dict[str, Any]:
    """Use the frozen repository packet, never a fabricated approval receipt."""
    return prepare_pilot()


def captures(packet: dict[str, Any], rows=()) -> dict[str, Any]:
    """Bind normalized capture rows to the exact prepared input."""
    return dict(
        schema_version="knowledge-utility-pilot-captures-v1",
        packet_digest=packet["packet_digest"],
        captures=list(rows),
    )


def row(slot, status="completed", text='{"answer":"6"}', reason=None) -> dict[str, Any]:
    """Construct a raw outcome without supplying a correctness flag."""
    return dict(slot_id=slot["slot_id"], status=status, response_text=text, reason=reason)


@pytest.mark.parametrize(
    "text,expected",
    [
        (' {"answer":"6"}\n', True),
        ('{"answer":" 6"}', False),
        ('{"answer":6}', False),
        ('{"answer":true}', False),
        ('{"answer":null}', False),
        ('{"answer":"wrong","answer":"6"}', False),
        ('{"answer":"6","x":1}', False),
        ('{"answer":"6"} trailing', False),
        ('```json\n{"answer":"6"}\n```', False),
        ('{"answer":NaN}', False),
        ("[]", False),
        ("", False),
        ("[" * 10000, False),
    ],
)
def test_exact_answer_parser(text: str, expected: bool) -> None:
    """Only one exact string answer passes; malformed model output is a failed answer."""
    assert answer_matches(text, "6") is expected


def test_all_correct_still_unverified(packet) -> None:
    """Raw success cannot create trusted utility or admission authority."""
    answers = {c["case_id"]: c["expected_answer"] for c in packet["evaluator_cases"]}
    rows = [
        row(s, text=canonical_json({"answer": answers[s["task_id"]]}))
        for s in packet["utility_study"]["expected_slots"]
    ]
    report = score_pilot(packet, captures(packet, rows))
    assert all(c["successes"] == 18 and c["observed"] == 18 for c in report["conditions"].values())
    assert all(c["verified"] is False for c in report["comparisons"].values())
    assert all(c["delta"] == 0 and c["matched_count"] == 18 for c in report["comparisons"].values())
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["optimizer_input"] is False and report["confers_authority"] is False
    digest = report.pop("result_digest")
    assert digest == content_digest(report)


def test_mixed_outcomes_retain_raw_text_and_missing_roster(packet) -> None:
    """Refusal and truncation take precedence even when partial text is correct."""
    slots = packet["utility_study"]["expected_slots"]
    malformed = '{"answer":"wrong","answer":"6"}'
    rows = [
        row(slots[0]),
        row(slots[1], text=malformed),
        row(slots[2], "refused", reason="provider refusal"),
        row(slots[3], "truncated", reason="output limit"),
        row(slots[4], "failed", None, "transport error"),
        row(slots[5], "missing", None, "not sent"),
    ]
    report = score_pilot(packet, captures(packet, rows))
    actual = {r["slot"]["slot_id"]: r for r in report["observations"]}
    assert len(actual) == 72
    assert [actual[s["slot_id"]]["status"] for s in slots[:7]] == [
        "observed",
        "observed",
        "censored",
        "censored",
        "failed",
        "missing",
        "missing",
    ]
    assert [actual[s["slot_id"]]["success"] for s in slots[:7]] == [
        True,
        False,
        None,
        None,
        None,
        None,
        None,
    ]
    sources = report["inputs"]["study"]["observations"]
    source = json.loads(
        next(s["source_json"] for s in sources if s["slot"]["slot_id"] == slots[1]["slot_id"])
    )
    assert base64.b64decode(source["response_utf8_base64"]).decode("utf-8") == malformed


@pytest.mark.parametrize("reseal", [False, True])
def test_packet_tampering_rejected_even_with_recomputed_digest(packet, reseal) -> None:
    """An integrity hash is not permission to substitute a different rubric or task."""
    altered = deepcopy(packet)
    altered["evaluator_cases"][0]["expected_answer"] = "wrong"
    if reseal:
        altered.pop("packet_digest")
        altered["packet_digest"] = content_digest(altered)
    with pytest.raises(ValueError, match="packet"):
        score_pilot(altered, captures(altered))


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "binding", "success", "status"])
def test_bad_capture_structure_rejected(packet, fault) -> None:
    """Invalid captures fail the batch instead of silently selecting or discarding rows."""
    rows = [row(packet["utility_study"]["expected_slots"][0])]
    data = captures(packet, rows)
    if fault == "duplicate":
        data["captures"].append(dict(rows[0]))
    elif fault == "unknown":
        rows[0]["slot_id"] = "not-a-slot"
    elif fault == "binding":
        data["packet_digest"] = "0" * 64
    elif fault == "success":
        rows[0]["success"] = True
    else:
        rows[0]["status"] = "observed"
    with pytest.raises(ValueError):
        score_pilot(packet, data)


def test_cli_round_trip_and_exclusive_output(packet, tmp_path: Path) -> None:
    """The CLI reports missing slots without calls and preserves existing output bytes."""
    packet_path, capture_path, output = (tmp_path / n for n in ("p.json", "c.json", "r.json"))
    packet_path.write_text(canonical_json(packet), encoding="utf-8")
    capture_path.write_text(canonical_json(captures(packet)), encoding="utf-8")
    argv = ["--packet", str(packet_path), "--captures", str(capture_path), "--output", str(output)]
    main(argv)
    before = output.read_bytes()
    report = json.loads(before)
    assert all(c["missing"] == 18 and c["rate"] is None for c in report["conditions"].values())
    with pytest.raises(FileExistsError):
        main(argv)
    assert output.read_bytes() == before
    capture_path.write_text('{"captures":[],"captures":[]}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(argv)


@pytest.mark.parametrize(
    "patch",
    [
        {"response_text": "x" * 8193},
        {"response_text": "\u00e9" * 4097},
        {"response_text": None},
        {"response_text": True},
        {"reason": "unexpected"},
        {"status": "failed", "reason": "error"},
        {"status": "missing", "response_text": None, "reason": " "},
        {"status": "refused", "reason": "x" * 1025},
    ],
)
def test_invalid_row_values_do_not_become_missing_outcomes(packet, patch) -> None:
    """Malformed envelopes fail explicitly; they cannot silently improve denominators."""
    capture = row(packet["utility_study"]["expected_slots"][0])
    capture.update(patch)
    with pytest.raises(ValueError):
        score_pilot(packet, captures(packet, [capture]))


def test_cli_bounds_read_before_decode(tmp_path: Path) -> None:
    """Oversized inputs fail before decoding or creating any output."""
    oversized, output = tmp_path / "large.json", tmp_path / "report.json"
    oversized.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(SystemExit) as error:
        main(["--packet", str(oversized), "--captures", str(oversized), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", ["write", "close"])
def test_cli_cleans_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    """A disk error after exclusive creation does not leave a misleading report file."""
    output = tmp_path / "report.json"
    original_open = Path.open
    monkeypatch.setattr(scorer, "_read_json", lambda path: {})
    monkeypatch.setattr(scorer, "score_pilot", lambda *args: {"result": "test-only"})

    class FailingStream:
        """Inject a write or final-flush failure while retaining a real output handle."""

        def __init__(self, path: Path, *args: Any, **kwargs: Any) -> None:
            """Exclusively create the file using the real filesystem."""
            self.stream = original_open(path, *args, **kwargs)

        def __enter__(self) -> "FailingStream":
            """Expose the fault-injecting stream to the CLI."""
            return self

        def write(self, data: bytes) -> None:
            """Leave a partial write before simulating a storage failure."""
            self.stream.write(data[:4])
            if failure == "write":
                raise OSError("simulated storage failure")

        def __exit__(self, *args: Any) -> None:
            """Close the real handle before simulating the final flush error."""
            self.stream.close()
            if failure == "close":
                raise OSError("simulated flush failure")

    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: FailingStream(path, *a, **kw))
    with pytest.raises(SystemExit) as error:
        main(["--packet", "unused", "--captures", "unused", "--output", str(output)])
    assert error.value.code == 2
    assert "cannot write" in capsys.readouterr().err
    assert not output.exists()


def test_close_failure_preserves_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup cannot delete a completed report published by another writer."""
    output = tmp_path / "report.json"
    replacement = tmp_path / "replacement.json"
    original_open = Path.open
    monkeypatch.setattr(scorer, "_read_json", lambda path: {})
    monkeypatch.setattr(scorer, "score_pilot", lambda *args: {"result": "test-only"})

    class RacingStream:
        """Replace the destination before surfacing a delayed close failure."""

        def __init__(self, path: Path, *args: Any, **kwargs: Any) -> None:
            """Retain the actual open file until the simulated race."""
            self.stream = original_open(path, *args, **kwargs)

        def __enter__(self) -> "RacingStream":
            """Expose the wrapper's file operations."""
            return self

        def write(self, data: bytes) -> None:
            """Write to the original file before replacement."""
            self.stream.write(data)

        def __exit__(self, *args: Any) -> None:
            """Replace the closed file before surfacing its delayed error."""
            self.stream.close()
            with original_open(replacement, "wb") as other:
                other.write(b"another completed report")
            replacement.replace(output)
            raise OSError("simulated final-flush error after replacement")

    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: RacingStream(path, *a, **kw))
    with pytest.raises(SystemExit) as error:
        main(["--packet", "unused", "--captures", "unused", "--output", str(output)])
    assert error.value.code == 2
    with original_open(output, "rb") as stream:
        assert stream.read() == b"another completed report"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("concurrent_output", [False, True])
def test_publication_failure_preserves_destination_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent_output: bool
) -> None:
    """Publishing cannot replace a competing result or leave an unsupported-link partial."""
    output = tmp_path / "report.json"
    original_link = scorer.os.link
    monkeypatch.setattr(scorer, "_read_json", lambda path: {})
    monkeypatch.setattr(scorer, "score_pilot", lambda *args: {"result": "test-only"})

    def racing_link(source: Path, destination: Path) -> None:
        """Inject a competing publication or a filesystem without hard-link support."""
        assert not output.exists()
        assert source.read_bytes() == b'{"result":"test-only"}\n'
        if concurrent_output:
            output.write_bytes(b"another completed report")
            original_link(source, destination)
        else:
            raise OSError("hard links unsupported")

    monkeypatch.setattr(scorer.os, "link", racing_link)
    with pytest.raises(FileExistsError if concurrent_output else SystemExit):
        main(["--packet", "unused", "--captures", "unused", "--output", str(output)])
    if concurrent_output:
        assert output.read_bytes() == b"another completed report"
    else:
        assert not output.exists()
    assert list(tmp_path.iterdir()) == ([output] if concurrent_output else [])
