"""Test that all infrastructure files exist and are valid."""

import json
from pathlib import Path

import yaml


def test_prompts_exist():
    assert Path("prompts/judge/gepa_wisdom_judge_prompt.txt").exists()
    assert Path("prompts/judge/gepa_wisdom_judge_schema.json").exists()

    # Validate schema
    with open("prompts/judge/gepa_wisdom_judge_schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    assert "properties" in schema
    assert "mindfulness" in schema["properties"]


def test_configs_exist():
    assert Path("configs/scoring.yml").exists()
    assert Path("configs/classifier/default.yml").exists()

    # Validate YAML
    with open("configs/scoring.yml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "weights" in config
    assert "abstention_thresholds" in config


def test_label_infrastructure():
    assert Path("datasets/labels/label_schema.json").exists()
    assert Path("datasets/labels/annotation_guide.md").exists()
    assert Path("datasets/labels/examples/seed_labels.jsonl").exists()

    # Validate seed labels
    with open("datasets/labels/examples/seed_labels.jsonl", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f]
    assert len(labels) >= 3
    for label in labels:
        assert "mindfulness" in label
        assert "score" in label["mindfulness"]
        assert 0 <= label["mindfulness"]["score"] <= 4


def test_scripts_exist():
    assert Path("scripts/labels_export.py").exists()
    assert Path("scripts/labels_import.py").exists()
    assert Path("scripts/train_classifier.py").exists()

    # Check executability
    for script in ["labels_export.py", "labels_import.py", "train_classifier.py"]:
        path = Path("scripts") / script
        assert path.stat().st_mode & 0o111  # executable bit


def test_docs_exist():
    assert Path("docs/scoring_pipeline.md").exists()
    content = Path("docs/scoring_pipeline.md").read_text(encoding="utf-8")
    assert "Three Tiers" in content or "three-tier" in content.lower()
