"""Repository line-ending contracts for byte-hashed provenance artifacts."""

# Standard library
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_jsonl_checkout_preserves_provenance_bytes_with_autocrlf(tmp_path: Path) -> None:
    """A Windows checkout must not change JSONL bytes referenced by provenance hashes."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"],
        cwd=repository,
        check=True,
    )
    shutil.copyfile(ROOT / ".gitattributes", repository / ".gitattributes")

    artifact = repository / "data" / "provenance.jsonl"
    artifact.parent.mkdir()
    canonical_bytes = b'{"id":"one"}\n{"id":"two"}\n'
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    artifact.write_bytes(canonical_bytes)
    subprocess.run(
        ["git", "add", ".gitattributes", "data/provenance.jsonl"],
        cwd=repository,
        check=True,
    )

    artifact.unlink()
    subprocess.run(
        ["git", "checkout-index", "--force", "--", "data/provenance.jsonl"],
        cwd=repository,
        check=True,
    )
    checked_out_bytes = artifact.read_bytes()

    assert checked_out_bytes == canonical_bytes
    assert hashlib.sha256(checked_out_bytes).hexdigest() == canonical_sha256
