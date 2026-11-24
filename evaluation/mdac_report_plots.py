"""Optional offline figures for separately labeled MDAC evidence cohorts."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from textwrap import fill
from typing import Any

from .causal_records import content_digest
from .mdac_reporting import PLOTS

_TITLES = {
    "failure_heatmap": "Mechanism frequency by evaluated condition",
    "mechanism_trends": "Mechanism frequency across rounds",
    "curriculum_allocation": "Curriculum budget allocation",
    "calibration": "Confidence calibration",
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


def _display_label(label: str) -> str:
    short = re.sub(r"[0-9a-fA-F]{32,}", lambda match: match[0][:9] + "…", label)
    return fill(short, 45) if len(short) > 55 else short


def _run_label(row: dict[str, Any]) -> str:
    if len(row["run_id"]) <= 42:
        return row["run_id"]
    return (
        f"{row['arm']} · {row['rl_method']} · {row['partition']} · "
        f"r{row['round']} s{row['seed']} [{content_digest(row['run_id'])[:6]}]"
    )


def _heatmap(ax: Any, rows: list[dict[str, Any]], name: str) -> None:
    # Keep per-run cells instead of pooling unlike series or averaging repeat rates.
    cells = []
    row_labels: dict[tuple[Any, ...], str] = {}
    column_labels: dict[tuple[Any, ...], str] = {}
    for r in rows:
        column: tuple[Any, ...]
        key: tuple[Any, ...]
        if name == "failure_heatmap":
            column = (r["group"],)
            column_labels[column] = r["group"]
            key = (r["run_id"], r["x"], r["y"])
            condition = str(r["x"]) if r["x"] is not None else r["arm"]
            label = f"{_run_label(r)} · {condition}"
            if r["y"] is not None:
                label += f" · {r['y']}"
        elif name == "retention_regression":
            column = (r["metric"], r["x"], r["y"])
            column_labels[column] = f"{r['metric']} · {r['x']} → {r['y']}"
            key = (r["run_id"], r["group"])
            label = f"{_run_label(r)} · {r['group']}"
        else:
            key = (r["run_id"], r["x"], r["y"])
            label = _run_label(r)
            coordinates = [str(r[k]) for k in ("x", "y") if r[k] is not None]
            if coordinates:
                label += " · " + " / ".join(coordinates)
            column = (r["metric"], r["group"])
            column_labels[column] = f"{r['metric']} · {r['group']}"
        row_labels[key] = label
        cells.append((key, column, r))
    columns = sorted(column_labels, key=lambda k: (column_labels[k], repr(k)))
    keys = sorted(row_labels, key=lambda k: (row_labels[k], repr(k)))
    matrix = [[float("nan")] * len(columns) for _ in keys]
    for key, column, row in cells:
        iy, ix = keys.index(key), columns.index(column)
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
    signed = any(r["value"] < 0 or r["value"] > 1 for r in rows)
    limit = max(abs(r["value"]) for r in rows) if signed else 1
    im = ax.imshow(
        matrix,
        cmap="RdBu" if signed else "viridis",
        vmin=-limit if signed else 0,
        vmax=limit,
        aspect="auto",
    )
    ax.figure.colorbar(im, ax=ax, shrink=0.75, label="Estimate")
    ax.set_xticks(
        range(len(columns)),
        [_display_label(column_labels[c]) for c in columns],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(keys)), [_display_label(row_labels[k]) for k in keys])


def _series_key(r: dict[str, Any], name: str) -> tuple[Any, ...]:
    identity = tuple(
        r[k]
        for k in (
            "metric",
            "group",
            "arm",
            "rl_method",
            "model",
            "partition",
            "seed",
            "dataset_version",
            "evaluator_version",
            "config_hash",
        )
    )
    coordinates: tuple[Any, ...]
    if name in ("calibration", "risk_coverage"):
        coordinates = (r["run_id"], r["round"])
    elif name == "cross_rl":
        coordinates = (r["run_id"], r["checkpoint"], r["round"], r["x"])
    else:
        coordinates = (
            r.get("trajectory_id"),
            r["run_id"] if "trajectory_id" not in r else None,
            r["x"],
        )
    return identity + (tuple(sorted(r["family_ids"])), r["y"]) + coordinates


def _series(ax: Any, rows: list[dict[str, Any]], name: str) -> None:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    within_run = name in ("calibration", "risk_coverage")
    for r in rows:
        groups[_series_key(r, name)].append(r)

    def checkpoint_key(r: dict[str, Any]) -> tuple:
        return (r["rl_method"], r["arm"], r["round"], r["checkpoint"], r["partition"], r["run_id"])

    categories = sorted({checkpoint_key(r) for r in rows}) if name == "cross_rl" else []
    positions = {key: i for i, key in enumerate(categories)}
    for points in groups.values():
        if within_run:
            column = "x"
        else:
            column = "round"
        points.sort(key=lambda r: (r[column], r["run_id"]))
        xs = (
            [positions[checkpoint_key(r)] for r in points]
            if name == "cross_rl"
            else [r[column] for r in points]
        )
        ys = [r["value"] for r in points]
        first = points[0]
        series = (
            f"{first['metric']} · {first['group']} · {first['model']} · "
            f"{first['arm']} · {first['rl_method']} · {first['partition']} · seed {first['seed']}"
        )
        coordinates = [first["y"]] if within_run else [first["x"], first["y"]]
        for coordinate in coordinates:
            if coordinate is not None:
                series += f" · {coordinate}"
        if within_run:
            run = first["run_id"]
            shown = run if len(run) <= 42 else content_digest(run)[:9]
            series += f" · {shown} · round {first['round']}"
        elif name == "cross_rl":
            series += f" · round {first['round']} · {first['checkpoint']}"
        else:
            series += " · " + first.get("trajectory_id", first["run_id"])
        joined = (
            len(xs) > 1
            and len(xs) == len(set(xs))
            and name != "cross_rl"
            and (within_run or "trajectory_id" in first)
        )
        line = ax.plot(
            xs, ys, linestyle="-" if joined else "", marker="o", label=_display_label(series)
        )[0]
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
        labels = [
            _display_label(f"{method} · {arm} · round {step} · {checkpoint} · {part}")
            for method, arm, step, checkpoint, part, _ in categories
        ]
        ax.set_xticks(range(len(categories)), labels, rotation=20, ha="right")
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
            all_ids = sorted({r["run_id"] for r in rows})
            paged = name in ("calibration", "risk_coverage") and len(all_ids) > 4
            batches = (
                [all_ids[i : i + 4] for i in range(0, len(all_ids), 4)] if paged else [all_ids]
            )
            all_rows = rows
            for page, batch in enumerate(batches, 1):
                rows = [r for r in all_rows if r["run_id"] in batch]
                filename = f"{name}-p{page:02d}" if paged else name
                series_count = len({_series_key(r, name) for r in rows})
                heat = name in ("failure_heatmap", "retention_regression") or (
                    series_count > 10 and name not in ("calibration", "risk_coverage")
                )
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
                        f"{sum(values):g} ({len(values)}/{len(ids)} runs)"
                        if values
                        else "unavailable"
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
                path = output / kind / f"{filename}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path)
                files.append(f"plots/{kind}/{filename}.png")
    return files
