"""Common vocabulary lists shared across heuristics."""

RISK_QUALITY_ADJECTIVES = (
    "dangerous",
    "harmful",
    "hazardous",
    "perilous",
    "risky",
    "threatening",
    "unsafe",
)

RISK_CONCERN_TERMS = (
    "concern",
    "concerned",
    "concerning",
    "concerns",
    "danger",
    "dangers",
    "harm",
    "harms",
    "issue",
    "issues",
    "problem",
    "problematic",
    "problems",
    "risk",
    "risks",
)

RISK_TERMS = RISK_CONCERN_TERMS + RISK_QUALITY_ADJECTIVES

__all__ = [
    "RISK_QUALITY_ADJECTIVES",
    "RISK_CONCERN_TERMS",
    "RISK_TERMS",
]
