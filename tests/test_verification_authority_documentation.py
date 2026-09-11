"""Keep verification and runtime-authority documentation tied to tested boundaries."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
CONTRACT_PATH = ROOT / "docs" / "VERIFICATION_AND_RUNTIME_AUTHORITY.md"


def test_readme_links_the_verification_and_runtime_authority_contract() -> None:
    """The repository orientation links the implemented contract by its canonical path."""

    readme = README_PATH.read_text(encoding="utf-8")

    contract_link = (
        "[Verification and runtime authority](docs/VERIFICATION_AND_RUNTIME_AUTHORITY.md)"
    )
    assert contract_link in readme


def test_contract_names_the_distinct_boundaries_and_trust_limits() -> None:
    """The public contract does not collapse state, verification, causality, or trust roles."""

    contract = " ".join(CONTRACT_PATH.read_text(encoding="utf-8").split())

    required_text = (
        "`WorldStateChange` is an observed artifact transition",
        "`EvidenceClaim` does not create or prove a `WorldStateChange`",
        "`LocalVerificationResult` does not imply `RelationalVerificationResult`",
        "Event order alone does not create a causal edge",
        "`AuthorityGrantRegistry.enroll()` does not authenticate grant issuers",
        "`TrustedClock` is an injected trust boundary",
        "`RecoveryStateStore.enroll_classification()` does not authenticate verifier output",
        "`select_recovery()` reserves a proposal but does not consume its budget transition",
        "`consume_recovery()` performs the authoritative budget transition",
    )
    for text in required_text:
        assert text in contract


def test_contract_lists_normative_verification_paths() -> None:
    """Every normative boundary points readers to the exact automated test path."""

    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    expected_paths = (
        "tests/test_world_evidence_state.py",
        "tests/test_verifier_interfaces.py",
        "tests/test_action_bound_event_sequence.py",
        "tests/test_failure_graph.py",
        "tests/test_runtime_authority.py",
        "tests/test_bounded_recovery.py",
    )

    for path in expected_paths:
        assert f"`{path}`" in contract
