"""Correlation analysis between GEPA scores and attribution graph metrics."""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


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

        pearson_val, pearson_p = _pearsonr(values1, values2)
        spearman_val, spearman_p = _spearmanr(values1, values2)
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

        _render_heatmap(matrix=matrix, output_path=output_path)

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


def _render_heatmap(*, matrix: np.ndarray, output_path: Path) -> None:
    """Render a simple PNG heatmap from ``matrix`` values."""

    clipped = np.clip(matrix, -1.0, 1.0)
    cell_size = 32
    expanded = clipped.repeat(cell_size, axis=0).repeat(cell_size, axis=1)
    positive = np.clip(expanded, 0.0, 1.0)
    negative = np.clip(-expanded, 0.0, 1.0)
    neutral = 1.0 - np.abs(expanded)
    red = (positive * 255).astype(np.uint8)
    green = (neutral * 200).astype(np.uint8)
    blue = (negative * 255).astype(np.uint8)
    rgb = np.dstack((red, green, blue))

    height, width, _ = rgb.shape
    raw_rows = []
    for row in rgb:
        raw_rows.append(b"\x00" + row.tobytes())
    payload = b"".join(raw_rows)
    compressed = zlib.compress(payload)

    with output_path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")

        def write_chunk(name: bytes, data: bytes) -> None:
            handle.write(len(data).to_bytes(4, "big"))
            handle.write(name)
            handle.write(data)
            checksum = zlib.crc32(name)
            checksum = zlib.crc32(data, checksum)
            handle.write(checksum.to_bytes(4, "big"))

        header = (
            width.to_bytes(4, "big")
            + height.to_bytes(4, "big")
            + b"\x08"
            + b"\x02"
            + b"\x00\x00\x00"
        )
        write_chunk(b"IHDR", header)
        write_chunk(b"IDAT", compressed)
        write_chunk(b"IEND", b"")


def _pearsonr(values1: List[float], values2: List[float]) -> tuple[float, float]:
    """Return Pearson's r and a two-tailed p-value."""

    x = np.asarray(values1, dtype=float)
    y = np.asarray(values2, dtype=float)
    x -= np.mean(x)
    y -= np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0, 1.0
    r = float(np.dot(x, y) / denom)
    r = max(min(r, 1.0), -1.0)
    return r, _p_value_from_t(r=r, sample_size=len(x))


def _spearmanr(values1: List[float], values2: List[float]) -> tuple[float, float]:
    """Return Spearman's rho and a two-tailed p-value."""

    ranks1 = _rankdata(np.asarray(values1, dtype=float))
    ranks2 = _rankdata(np.asarray(values2, dtype=float))
    r, p = _pearsonr(ranks1.tolist(), ranks2.tolist())
    return r, p


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to ``values`` similarly to ``scipy.stats.rankdata``."""

    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[sorter] = np.arange(len(values), dtype=float)

    sorted_vals = values[sorter]
    _, indices = np.unique(sorted_vals, return_index=True)
    counts = np.diff(np.append(indices, len(sorted_vals)))
    for start, count in zip(indices, counts):
        if count > 1:
            avg = (2 * start + count - 1) / 2.0
            ranks[sorter[start : start + count]] = avg

    return ranks + 1.0


def _p_value_from_t(*, r: float, sample_size: int) -> float:
    """Compute a two-tailed p-value using Student's t-distribution."""

    df = sample_size - 2
    if df <= 0:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    scale = math.sqrt(max(1.0 - r * r, 1e-12))
    t_value = abs(r) * math.sqrt(df) / scale
    tail = 1.0 - _student_t_cdf(t_value, df)
    return min(1.0, max(0.0, 2.0 * tail))


def _student_t_cdf(value: float, df: int) -> float:
    """Return the CDF of Student's t-distribution with ``df`` degrees of freedom."""

    if df <= 0:
        return 0.5
    if math.isnan(value):
        return math.nan
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    x = df / (df + value * value)
    beta = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    if value >= 0:
        return 1.0 - 0.5 * beta
    return 0.5 * beta


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Evaluate the regularized incomplete beta function ``I_x(a, b)``."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""

    max_iter = 200
    epsilon = 3e-7
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x
        denominator = (qam + m2) * (a + m2)
        aa = numerator / denominator
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        numerator = -(a + m) * (qab + m) * x
        denominator = (a + m2) * (qap + m2)
        aa = numerator / denominator
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h
