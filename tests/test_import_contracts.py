"""Import contract tests for installed third-party dependencies."""

from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


def test_matplotlib_import_uses_installed_dependency() -> None:
    """The repository must not shadow the real matplotlib package."""

    import matplotlib

    repo_root = Path(__file__).resolve().parents[1]
    module_path = Path(matplotlib.__file__).resolve()

    assert not (module_path.is_relative_to(repo_root) and ".venv" not in module_path.parts)


def test_pyproject_uses_discovery_and_declares_build_tooling() -> None:
    """Packaging config should avoid stale package lists and include wheel tooling."""

    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    setuptools_config = pyproject["tool"]["setuptools"]
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]

    packages_config = setuptools_config["packages"]

    assert isinstance(packages_config, dict)
    assert "find" in packages_config
    assert any(dependency.startswith("build>=") for dependency in dev_deps)
