"""Step 1d diagnostics - plot the RF-A cross-validation report.

Reads stage3_1_report_{SITE}_{YEAR}.json and renders one figure per framework:
training balance, per-class scores, per-fold stability, the confusion matrix,
and feature importance. A comparison figure is added when several frameworks
are present.

These are DIAGNOSTIC plots of a cross-validation over training polygons. They
are not a Step 6 accuracy assessment, which needs the Olofsson area-weighted
protocol on an independent probability sample (instructions5.md section 6). The
figures say so on their face so a number lifted from one cannot be mistaken for
map accuracy.

Colour follows two rules from the visualization guidance:

  identity   class marks use the locked section 3 CLASS_COLORS, ALWAYS beside a
             text label. The palette fails a normal-vision separation check on
             grass vs bare (delta-E 12.6, below the floor of 15), so colour is
             never the only channel carrying class identity here.
  magnitude  the confusion matrix and importance bars use a single-hue
             light-to-dark ramp, deliberately outside the class palette so a
             magnitude cell is never mistaken for a class mark.

Runs are numbered and separated on disk: figures are read from and written to
stage3_classification/run{N}/, matching run_stage3_1_random_forest_ground_truth_classification.py, so a baseline
run stays intact and comparable after the inputs change.

Usage:  python run_stage3_2_generate_ground_truth_classification_plots.py config/srer_2022.json --run 2
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from constants import CLASS_COLORS, CLASS_LABELS, CLASS_ORDER
from helpers import resolve_config_path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e4e4e2"
SURFACE = "#fcfcfb"
# single hue, light to dark, chosen outside the class palette so magnitude is
# never read as identity
MAGNITUDE = LinearSegmentedColormap.from_list("magnitude", ["#f4f6f7", "#cfdde2", "#8fb3bf", "#4d8496", "#1f5566"])
BAR = "#4d8496"


def style_axes(ax):
    """Recessive grid and axes so the marks carry the chart.

    Inputs:  ax - a matplotlib Axes
    Outputs: None, the axes are styled in place
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, axis="both", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def plot_training_balance(ax, counts):
    """Training pixels per class - the imbalance that drives everything else.

    Shown first because polygon counts mislead here: shrub has the most polygons
    and the fewest pixels, so a per-class score cannot be read without it.

    Inputs:  ax - Axes; counts - dict of {class label: n pixels}
    Outputs: None
    """
    values = [counts.get(name, 0) for name in CLASS_ORDER]
    total = sum(values) or 1
    bars = ax.bar(
        CLASS_ORDER,
        values,
        color=[CLASS_COLORS[c] for c in CLASS_LABELS],
        width=0.62,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{value / total:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.set_title("training pixels per class", fontsize=10, color=INK, loc="left")
    ax.set_ylim(0, max(values) * 1.28 if values else 1)
    ax.set_yticks([])
    ax.grid(False)
    style_axes(ax)
    ax.grid(False)


def plot_per_class(ax, per_class):
    """Recall, precision and F1 per class, grouped.

    Per class rather than pooled: overall accuracy here is dominated by bare and
    grass and barely moves on shrub, the class carrying the difficulty
    (instructions5.md section 6.4).

    Inputs:  ax - Axes; per_class - dict of {label: {recall, precision, f1}}
    Outputs: None
    """
    metrics = ["recall", "precision", "f1"]
    width = 0.26
    positions = np.arange(len(CLASS_ORDER))
    alphas = [0.45, 0.72, 1.0]
    for i, metric in enumerate(metrics):
        values = [per_class.get(name, {}).get(metric, 0.0) or 0.0 for name in CLASS_ORDER]
        colors = [CLASS_COLORS[c] for c in CLASS_LABELS]
        ax.bar(
            positions + (i - 1) * width,
            values,
            width=width * 0.92,
            color=colors,
            alpha=alphas[i],
            edgecolor=SURFACE,
            linewidth=1.2,
            label=metric,
        )
    for i, name in enumerate(CLASS_ORDER):
        f1 = per_class.get(name, {}).get("f1", 0.0) or 0.0
        ax.text(i + width, f1 + 0.03, f"{f1:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0, 1.12)
    ax.set_title("per-class scores", fontsize=10, color=INK, loc="left")
    # metric identity rides on shade, so it needs a legend of its own - the bar
    # colour is already spent on class identity and cannot carry a second job
    proxies = [Patch(facecolor=MUTED, alpha=alpha, edgecolor=SURFACE) for alpha in alphas]
    ax.legend(
        proxies,
        metrics,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        handlelength=1.1,
        columnspacing=1.1,
        labelcolor=MUTED,
    )
    style_axes(ax)


def plot_folds(ax, folds):
    """Per-fold macro-F1 and overall accuracy, to show stability across tiles.

    A score that swings between held-out tiles is a warning that the training
    set is too small or the tiles differ compositionally - both true here.

    Inputs:  ax - Axes; folds - list of fold dicts
    Outputs: None
    """
    names = [f["held_out"].replace("_", "\n") for f in folds]
    macro = [f["macro_f1"] for f in folds]
    overall = [f["overall"] for f in folds]
    positions = np.arange(len(folds))
    ax.bar(positions - 0.17, macro, width=0.32, color=BAR, edgecolor=SURFACE, linewidth=1.2)
    ax.bar(
        positions + 0.17,
        overall,
        width=0.32,
        color=BAR,
        alpha=0.4,
        edgecolor=SURFACE,
        linewidth=1.2,
    )
    for x, value in zip(positions - 0.17, macro):
        ax.text(x, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=INK)
    for x, value in zip(positions + 0.17, overall):
        ax.text(x, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "per fold: macro-F1 (solid) vs overall (pale)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    style_axes(ax)


def plot_confusion(ax, matrix):
    """Row-normalized confusion matrix with raw counts annotated.

    Rows are truth and sum to 1, so the diagonal reads as recall. Counts are
    kept in the cell because a rate over a small support is easy to over-read.

    Inputs:  ax - Axes; matrix - 4x4 nested list of counts
    Outputs: None
    """
    counts = np.array(matrix, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    ax.imshow(rates, cmap=MAGNITUDE, vmin=0, vmax=1, aspect="auto")
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            value = rates[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}\n{int(counts[i, j]):,}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if value > 0.55 else INK,
            )
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("predicted", fontsize=9, color=MUTED)
    ax.set_ylabel("true", fontsize=9, color=MUTED)
    ax.set_title(
        "confusion, row-normalized (diagonal = recall)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(False)


def plot_importance(ax, importance, top_n):
    """Feature importance, highest first.

    Magnitude, so a single-hue ramp rather than class colours - these are
    features, not classes.

    Inputs:  ax - Axes; importance - dict of {feature: value}; top_n - int
    Outputs: None
    """
    items = sorted(importance.items(), key=lambda kv: -kv[1])[:top_n][::-1]
    names = [name for name, _ in items]
    values = [value for _, value in items]
    shades = [MAGNITUDE(0.35 + 0.6 * (v / max(values))) for v in values]
    ax.barh(names, values, color=shades, height=0.68, edgecolor=SURFACE, linewidth=1.2)
    for i, value in enumerate(values):
        ax.text(
            value + max(values) * 0.02,
            i,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_title(f"feature importance, top {len(items)}", fontsize=10, color=INK, loc="left")
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")


def figure_for_framework(framework, block, site, year, out_dir, run):
    """Render the five-panel diagnostic figure for one framework.

    Inputs:  framework - key string; block - the report section; site, year;
             out_dir - Path to write into
    Outputs: Path written
    """
    fig = plt.figure(figsize=(15.5, 9.0), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26, left=0.06, right=0.97, top=0.86, bottom=0.08)

    plot_training_balance(fig.add_subplot(grid[0, 0]), block.get("train_pixels_per_class", {}))
    pooled = block.get("pooled", {})
    plot_per_class(fig.add_subplot(grid[0, 1]), pooled.get("per_class", {}))
    plot_folds(fig.add_subplot(grid[0, 2]), block.get("folds", []))
    plot_confusion(fig.add_subplot(grid[1, 0]), block.get("confusion", [[0] * 4] * 4))
    if block.get("feature_importance"):
        plot_importance(fig.add_subplot(grid[1, 1:]), block["feature_importance"], 14)

    features = block.get("features", [])
    macro = pooled.get("macro_f1", float("nan"))
    overall = pooled.get("overall", float("nan"))
    fig.suptitle(
        f"Step 1d  RF-A_{framework}  ·  run {run}  ·  {site} {year}",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=15,
        color=INK,
    )
    fig.text(
        0.06,
        0.925,
        f"macro-F1 {macro:.3f}   overall {overall:.3f}   ·   {len(features)} features: {', '.join(features)}",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.902,
        "DIAGNOSTIC - leave-one-tile-out over training polygons. Not a Step 6 accuracy assessment (that needs the Olofsson protocol on an independent sample).",
        ha="left",
        fontsize=8.5,
        color="#a04a2f",
    )

    path = out_dir / f"stage3_1_diagnostics_{framework}_{site}_{year}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_comparison(frameworks, site, year, out_dir):
    """Per-class F1 across frameworks - the A-E comparison, when more than one ran.

    Frameworks vary by input group only, so a difference here is attributable to
    the inputs (instructions5.md section 4.1).

    Inputs:  frameworks - dict of {key: report block}; site, year; out_dir
    Outputs: Path written, or None when fewer than two frameworks are present
    """
    keys = [k for k in sorted(frameworks) if frameworks[k].get("pooled")]
    if len(keys) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    width = 0.8 / len(CLASS_ORDER)
    positions = np.arange(len(keys))
    for i, name in enumerate(CLASS_ORDER):
        values = [frameworks[k]["pooled"]["per_class"].get(name, {}).get("f1", 0.0) or 0.0 for k in keys]
        offset = (i - (len(CLASS_ORDER) - 1) / 2) * width
        ax.bar(
            positions + offset,
            values,
            width=width * 0.9,
            color=CLASS_COLORS[CLASS_ORDER.index(name)],
            edgecolor=SURFACE,
            linewidth=1.4,
            label=name,
        )
        for x, value in zip(positions + offset, values):
            ax.text(x, value + 0.015, f"{value:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"RF-A_{k}" for k in keys])
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "per-class F1 by RF-A variant · variants differ by input features only",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=len(CLASS_ORDER),
        loc="upper left",
        bbox_to_anchor=(0, 1.0),
    )
    style_axes(ax)
    path = out_dir / f"stage3_1_framework_comparison_{site}_{year}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--run",
        default="1",
        help="run label - a number, optionally suffixed (e.g. 3 or 3_smoke). Reads and writes under run{LABEL}/",
    )
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    out_dir = resolve_config_path(config["results_root"], "stage3_classification", f"run{args.run}")
    report_path = out_dir / f"stage3_1_report_{site}_{year}.json"
    if not report_path.exists():
        print(f"no report at {report_path} - run run_stage3_1_random_forest_ground_truth_classification.py --run {args.run} first")
        return
    report = json.loads(report_path.read_text())
    frameworks = report.get("frameworks", {})
    if not frameworks:
        print("no frameworks in the report - run run_stage3_1_random_forest_ground_truth_classification.py first")
        return

    cap = report.get("max_pixels_per_polygon")
    print(f"Step 1d diagnostics - {site} {year} - run {report.get('run', args.run)}")
    print(f"polygon subsampling: {('max ' + str(cap) + ' px per polygon') if cap else 'OFF'}")
    print("=" * 62)
    for framework in sorted(frameworks):
        block = frameworks[framework]
        path = figure_for_framework(framework, block, site, year, out_dir, report.get("run", args.run))
        pooled = block.get("pooled", {})
        shrub = pooled.get("per_class", {}).get("shrub", {})
        print(f"[{framework}] macro-F1 {pooled.get('macro_f1', float('nan')):.3f}   shrub F1 {shrub.get('f1', float('nan')):.3f}   -> {path.name}")

    path = figure_comparison(frameworks, site, year, out_dir)
    if path:
        print(f"[all] framework comparison -> {path.name}")
    print(f"\nwrote to {out_dir}")


if __name__ == "__main__":
    main()
