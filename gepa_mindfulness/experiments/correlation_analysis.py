"""Correlation analysis between GEPA scores and attribution graph metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


@dataclass
class CorrelationResult:
    """Statistical summary for a single correlation pair."""

    variable1: str
    variable2: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    interpretation: str


class CorrelationAnalyzer:
    """Analyse stored evaluation results for cross-metric trends."""

    def __init__(self, *, results_dir: Path) -> None:
        """Load results from ``results_dir`` into memory."""

        self.results_dir = results_dir
        self.data: List[Dict[str, object]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load JSONL result files produced by the baseline evaluator."""

        for filepath in self.results_dir.glob("*_results.jsonl"):
            with filepath.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self.data.append(json.loads(line))

    def analyze_gepa_ag_correlations(self) -> List[CorrelationResult]:
        """Compute pairwise correlations between GEPA and AG metrics."""

        if not self.data:
            return []

        gepa_fields = [
            "mindfulness",
            "empathy",
            "perspective",
            "agency",
            "reduce_suffering",
            "increase_prosperity",
            "increase_knowledge",
            "gepa_aggregate",
        ]
        ag_fields = [
            "path_coherence",
            "entropy",
            "centrality_concentration",
            "average_path_length",
        ]

        results: List[CorrelationResult] = []
        for gepa in gepa_fields:
            for ag in ag_fields:
                correlation = self._compute_correlation(var1=gepa, var2=ag)
                if correlation is not None:
                    results.append(correlation)

        results.sort(key=lambda item: abs(item.pearson_r), reverse=True)
        return results

    def _compute_correlation(
        self,
        *,
        var1: str,
        var2: str,
    ) -> Optional[CorrelationResult]:
        """Compute correlation statistics for ``var1`` and ``var2``."""

        values1: List[float] = []
        values2: List[float] = []
        for item in self.data:
            gepa_components = item.get("gepa_components", {})
            ag_metrics = item.get("ag_metrics", {})

            if var1 == "gepa_aggregate":
                value1 = item.get("gepa_aggregate")
            else:
                value1 = gepa_components.get(var1)
            value2 = ag_metrics.get(var2)

            if value1 is None or value2 is None:
                continue
            values1.append(float(value1))
            values2.append(float(value2))

        if len(values1) < 10:
            return None

        pearson_val, pearson_p = pearsonr(values1, values2)
        spearman_val, spearman_p = spearmanr(values1, values2)
        interpretation = self._interpret_correlation(r=pearson_val, p=pearson_p)
        return CorrelationResult(
            variable1=var1,
            variable2=var2,
            pearson_r=pearson_val,
            pearson_p=pearson_p,
            spearman_r=spearman_val,
            spearman_p=spearman_p,
            interpretation=interpretation,
        )

    @staticmethod
    def _interpret_correlation(*, r: float, p: float) -> str:
        """Return a short textual interpretation for correlation results."""

        if p >= 0.05:
            return "Not significant"
        magnitude = abs(r)
        if magnitude < 0.3:
            strength = "Weak"
        elif magnitude < 0.6:
            strength = "Moderate"
        else:
            strength = "Strong"
        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} correlation (p={p:.4f})"

    def visualize_correlations(self, *, output_path: Path) -> None:
        """Render and save a correlation heatmap as ``output_path``."""

        gepa_vars = [
            "mindfulness",
            "empathy",
            "perspective",
            "agency",
            "reduce_suffering",
            "increase_prosperity",
            "increase_knowledge",
            "gepa_aggregate",
        ]
        ag_vars = [
            "path_coherence",
            "entropy",
            "centrality_concentration",
            "average_path_length",
        ]
        matrix = np.zeros((len(gepa_vars), len(ag_vars)))
        for row, gepa in enumerate(gepa_vars):
            for col, ag in enumerate(ag_vars):
                result = self._compute_correlation(var1=gepa, var2=ag)
                matrix[row, col] = result.pearson_r if result is not None else 0.0

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            xticklabels=ag_vars,
            yticklabels=gepa_vars,
            cmap="RdBu_r",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
        )
        plt.title("GEPA vs Attribution Graph Correlations")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def generate_report(self, *, output_path: Path) -> None:
        """Generate a Markdown report summarising correlation findings."""

        correlations = self.analyze_gepa_ag_correlations()
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("# Phase 0: Correlation Analysis Report\n\n")
            handle.write(f"- Total examples analysed: {len(self.data)}\n")
            handle.write(f"- Correlations computed: {len(correlations)}\n\n")

            strong = [
                corr for corr in correlations if abs(corr.pearson_r) > 0.6 and corr.pearson_p < 0.05
            ]
            if strong:
                handle.write("## Strong Correlations\n\n")
                for corr in strong:
                    handle.write(
                        "- **{}** vs **{}**: r={:.3f}, p={:.4f} — {}\n".format(
                            corr.variable1,
                            corr.variable2,
                            corr.pearson_r,
                            corr.pearson_p,
                            corr.interpretation,
                        )
                    )
                handle.write("\n")
            else:
                handle.write(
                    "No strong correlations detected. Attribution graphs "
                    "offer additional signal.\n\n"
                )

            handle.write("## All Correlations\n\n")
            handle.write("| GEPA Component | AG Metric | r | p-value |\n")
            handle.write("| --- | --- | --- | --- |\n")
            for corr in correlations:
                handle.write(
                    "| {} | {} | {:.3f} | {:.4f} |\n".format(
                        corr.variable1,
                        corr.variable2,
                        corr.pearson_r,
                        corr.pearson_p,
                    )
                )

            handle.write("\n## Recommendation\n\n")
            if not strong:
                handle.write(
                    "✅ Proceed to Phase 1 — AG metrics are not redundant with GEPA scores.\n"
                )
            elif all(abs(corr.pearson_r) > 0.9 for corr in strong):
                handle.write(
                    "❌ Re-evaluate AG integration — metrics align almost perfectly with GEPA.\n"
                )
            else:
                handle.write(
                    "⚠️ Investigate further — partial correlations warrant targeted follow-up.\n"
                )
