from __future__ import annotations

import json
from pathlib import Path

from scripts.labels_export import export_low_confidence


def test_export_low_confidence_uses_dataclass(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "id": "example-run",
                "final": {"mindfulness": 3, "compassion": 2},
                "confidence": {"mindfulness": 0.6, "compassion": 0.8},
                "per_tier": [
                    {
                        "tier": "judge",
                        "scores": {
                            "mindfulness": 3,
                            "compassion": 2,
                            "integrity": 4,
                            "prudence": 1,
                        },
                        "confidence": {
                            "mindfulness": 0.6,
                            "compassion": 0.8,
                            "integrity": 0.9,
                            "prudence": 0.5,
                        },
                        "meta": {},
                    }
                ],
                "reasons": ["low mindfulness"],
            }
        ),
        encoding="utf-8",
    )

    rows = export_low_confidence(scores_path, threshold=0.7)

    assert rows == [
        {
            "id": "example-run",
            "dimension": "mindfulness",
            "confidence": 0.6,
            "score": 3,
            "reasons": ["low mindfulness"],
        }
    ]
