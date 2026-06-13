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


def test_source_tree_import_shims_are_removed() -> None:
    """Clean installs must not depend on root-level import mutation shims."""

    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "sitecustomize.py").exists()
    assert not (repo_root / "mindful_trace_gepa" / "__init__.py").exists()


def test_base_dependencies_stay_light_and_heavy_stacks_are_extras() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]
    base_names = {dependency.split(">=")[0].split("==")[0] for dependency in dependencies}

    assert "torch" not in base_names
    assert "transformers" not in base_names
    assert "matplotlib" not in base_names
    for extra in ("base", "train", "interpret", "dspy", "pdf", "vllm", "all"):
        assert extra in optional
    assert any(dependency.startswith("torch") for dependency in optional["train"])
    assert any(dependency.startswith("matplotlib") for dependency in optional["interpret"])
