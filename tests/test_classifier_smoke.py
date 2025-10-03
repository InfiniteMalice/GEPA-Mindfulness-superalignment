from pathlib import Path

from mindful_trace_gepa.scoring.classifier import Tier2Classifier, load_classifier_from_config
from mindful_trace_gepa.scoring.schema import DIMENSIONS


def test_classifier_trains_and_predicts(tmp_path):
    labels_path = Path("datasets/labels/examples/seed_labels.jsonl")
    config_path = Path("configs/classifier/default.yml")
    out_dir = tmp_path / "artifacts"

    classifier = load_classifier_from_config(config_path)
    rows = [
        __import__("json").loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    classifier.fit(rows)
    classifier.save(out_dir)

    loaded = Tier2Classifier()
    loaded.load(out_dir)
    tier = loaded.predict(rows[0]["meta"]["trace_text"])
    assert tier.tier == "classifier"
    assert set(tier.scores.keys()) == set(DIMENSIONS)
