"""Optional offline figures for separately labeled MDAC evidence cohorts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .mdac_reporting import PLOTS

_TITLES = {
    "failure_heatmap": "Mechanism frequency by evaluated condition",
    "mechanism_trends": "Mechanism frequency across rounds",
    "curriculum_allocation": "Curriculum budget allocation",
    "calibration": "Calibration of answered cases",
    "risk_coverage": "Selective risk and coverage",
    "reward_behavior_divergence": "Training reward minus independent improvement",
    "generalization": "Familiar and novel performance and gap",
    "retention_regression": "Retention and regression transitions",
    "intervention_recovery": "Matched intervention recovery",
    "cross_rl": "Independent alignment and capability by RL method",
    "mechanistic": "Instrumented trajectory and attribution measurements",
}


def _cell_text(row: dict[str, Any]) -> str:
    interval = row["interval"]
    bounds = "" if interval is None else f"\n[{interval[0]:.2g}, {interval[1]:.2g}]"
    return f"{row['value']:.3g}\nn={row['n']}{bounds}"


def _heatmap(ax: Any, rows: list[dict[str, Any]], name: str) -> None:
    # Keep per-run cells instead of pooling unlike series or averaging repeat rates.
    cells = []
    for r in rows:
        if name == "failure_heatmap":
            column = r["group"]
            condition = str(r["x"]) if r["x"] is not None else r["arm"]
            label = f"{r['run_id']} · {condition}"
        else:
            column = f"{r['metric']} · {r['x']} → {r['y']}"
            label = f"{r['run_id']} · {r['group']}"
        cells.append((label, column, r))
    columns = sorted({c for _, c, _ in cells})
    labels = sorted({row_label for row_label, _, _ in cells})
    matrix = [[float("nan")] * len(columns) for _ in labels]
    for label, column, row in cells:
        iy, ix = labels.index(label), columns.index(column)
        matrix[iy][ix] = row["value"]
        ax.text(
            ix,
            iy,
            _cell_text(row),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if row["value"] < 0.45 else "black",
        )
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.figure.colorbar(im, ax=ax, shrink=0.75, label="Rate")
    ax.set_xticks(range(len(columns)), columns, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)


def _series(ax: Any, rows: list[dict[str, Any]], name: str) -> None:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    within_run = name in ("calibration", "risk_coverage")
    for r in rows:
        key = tuple(
            r[k]
            for k in (
                "metric",
                "group",
                "arm",
                "rl_method",
                "partition",
                "seed",
                "dataset_version",
                "evaluator_version",
            )
        )
        if within_run:
            key += (r["run_id"], r["round"])
        groups[key].append(r)
    categories = sorted({f"{r['rl_method']} · {r['arm']}" for r in rows})
    for key, points in groups.items():
        if within_run:
            column = "x"
        else:
            column = "round"
        points.sort(key=lambda r: (r[column], r["run_id"]))
        xs = (
            [categories.index(f"{r['rl_method']} · {r['arm']}") for r in points]
            if name == "cross_rl"
            else [r[column] for r in points]
        )
        ys = [r["value"] for r in points]
        series = f"{key[0]} · {key[1]} · {key[2]} · {key[3]} · {key[4]} · seed {key[5]}"
        if within_run:
            series += f" · {points[0]['run_id']} · round {points[0]['round']}"
        # No interpolation across tied coordinates or independent RL arms.
        joined = len(xs) == len(set(xs)) and name != "cross_rl"
        line = ax.plot(xs, ys, linestyle="-" if joined else "", marker="o", label=series)[0]
        for x, y, r in zip(xs, ys, points):
            if r["interval"] is not None:
                lo, hi = r["interval"]
                ax.vlines(x, lo, hi, color=line.get_color(), linewidth=1.2)
            ax.annotate(
                f"n={r['n']}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7
            )
    if name == "calibration":
        ax.plot([0, 1], [0, 1], linestyle=":", color="0.5")
        ax.set(
            xlabel="Mean confidence in bin", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1)
        )
    elif name == "risk_coverage":
        ax.set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1), ylim=(0, 1))
    elif name == "cross_rl":
        ax.set_xticks(range(len(categories)), categories, rotation=20, ha="right")
        ax.set_ylabel("Independent score")
    else:
        ax.set_xlabel("Evaluation round")
        ax.set_ylabel("Estimate")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=7)


def render_plots(cohorts: dict[str, Any], output: Path) -> list[str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    files = []
    for kind, cohort in cohorts.items():
        for name, metrics in PLOTS.items():
            rows = [
                r for m in metrics for r in cohort["metrics"].get(m, []) if r["value"] is not None
            ]
            if not rows:
                continue
            heat = name in ("failure_heatmap", "retention_regression")
            width = min(20, max(8, len({r["group"] for r in rows}) * 0.85)) if heat else 9
            height = min(18, max(5, len({r["run_id"] for r in rows}) * 0.65)) if heat else 5.5
            fig = Figure(figsize=(width, height), dpi=120)
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            if heat:
                _heatmap(ax, rows, name)
            else:
                _series(ax, rows, name)
            ids = {r["run_id"] for r in rows}
            costs = [r["cost"] for r in cohort["runs"] if r["run_id"] in ids and r["cost"]]

            def cost(name: str) -> str:
                values = [c[name] for c in costs if c[name] is not None]
                return (
                    f"{sum(values):g} ({len(values)}/{len(ids)} runs)" if values else "unavailable"
                )

            ax.set_title(_TITLES[name] + " · " + kind)
            fig.text(
                0.02,
                0.02,
                f"Train tokens {cost('training_tokens')}   |   "
                f"Eval tokens {cost('evaluation_tokens')}"
                f"   |   Missing opportunities by point: {min(r['missing'] for r in rows)}–"
                f"{max(r['missing'] for r in rows)}",
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0.065, 1, 1))
            path = output / kind / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            files.append(f"plots/{kind}/{name}.png")
    return files
