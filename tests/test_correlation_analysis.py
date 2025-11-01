"""Tests for correlation analysis utilities."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

from gepa_mindfulness.experiments.correlation_analysis import CorrelationAnalyzer


def test_correlation_analysis_workflow(tmp_path) -> None:
    results_dir = tmp_path
    records = []
    for idx in range(12):
        records.append(
            {
                "example_id": f"ex_{idx}",
                "gepa_components": {
                    "mindfulness": float(idx),
                    "empathy": float(idx) / 2.0,
                },
                "gepa_aggregate": float(idx) / 3.0,
                "ag_metrics": {
                    "path_coherence": float(idx) / 4.0,
                    "entropy": float(idx) / 5.0,
                    "centrality_concentration": float(idx) / 6.0,
                    "average_path_length": float(idx) / 7.0,
                },
            }
        )
    results_file = results_dir / "dataset_results.jsonl"
    with results_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    analyzer = CorrelationAnalyzer(results_dir=results_dir)
    correlations = analyzer.analyze_gepa_ag_correlations()
    assert correlations
    heatmap_path = results_dir / "heatmap.png"
    report_path = results_dir / "report.md"
    analyzer.visualize_correlations(output_path=heatmap_path)
    analyzer.generate_report(output_path=report_path)
    assert heatmap_path.exists()
    assert report_path.exists()
