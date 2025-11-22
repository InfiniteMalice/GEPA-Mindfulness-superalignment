"""Common vocabulary lists shared across heuristics."""

RISK_QUALITY_ADJECTIVES = (
    "dangerous",
    "hazardous",
    "harmful",
    "perilous",
    "risky",
    "threatening",
    "unsafe",
)

RISK_CONCERN_TERMS = (
    "concern",
    "concerns",
    "concerned",
    "concerning",
    "danger",
    "dangers",
    "harm",
    "harms",
    "issue",
    "issues",
    "problem",
    "problems",
    "problematic",
    "risk",
    "risks",
)

RISK_TERMS = RISK_CONCERN_TERMS + RISK_QUALITY_ADJECTIVES

__all__ = [
    "RISK_QUALITY_ADJECTIVES",
    "RISK_CONCERN_TERMS",
    "RISK_TERMS",
]
