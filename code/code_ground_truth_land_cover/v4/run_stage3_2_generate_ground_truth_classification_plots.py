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

  identity class marks use the locked section 3 CLASS_COLORS, ALWAYS beside a
             text label. The palette fails a normal-vision separation check on
             grass vs bare (delta-E 12.6, below the floor of 15), so colour is
             never the only channel carrying class identity here.
  magnitude the confusion matrix and feature-importance bars use a single-hue
             light-to-dark ramp, deliberately outside the class palette so a
             magnitude cell is never mistaken for a class mark.

Runs are numbered and separated on disk: figures are read from and written to
stage3_classification/run{N}/, matching run_stage3_1_random_forest_ground_truth_classification.py, so a baseline
run stays intact and comparable after the inputs change.

Usage: python run_stage3_2_generate_ground_truth_classification_plots.py config/srer_2022.json --run 2
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from constants import SEVENTY

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
MAGNITUDE_COLORMAP = LinearSegmentedColormap.from_list("magnitude", ["#f4f6f7", "#cfdde2", "#8fb3bf", "#4d8496", "#1f5566"])
BAR_COLOR = "#4d8496"


def apply_recessive_axis_style(axes):
    """Recessive grid and axes so the marks carry the chart.

    Inputs: axes - a matplotlib Axes
    Outputs: None, the axes are styled in place
    """
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8, length=3)
    axes.grid(True, axis="both", color=GRID, linewidth=0.6, alpha=0.9)
    axes.set_axisbelow(True)


def plot_training_balance(axes, train_pixels_per_class):
    """Training pixels per class - the imbalance that drives everything else.

    Shown first because polygon counts mislead here: shrub has the most polygons
    and the fewest pixels, so a per-class score cannot be read without it.

    Inputs: axes - Axes; train_pixels_per_class - dict of {class label: n pixels}
    Outputs: None
    """
    values = [train_pixels_per_class.get(name, 0) for name in CLASS_ORDER]
    total = sum(values) or 1
    bars = axes.bar(
        CLASS_ORDER,
        values,
        color=[CLASS_COLORS[class_code] for class_code in CLASS_LABELS],
        width=0.62,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{value / total:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    axes.set_title("training pixels per class", fontsize=10, color=INK, loc="left")
    axes.set_ylim(0, max(values) * 1.28 if values else 1)
    axes.set_yticks([])
    axes.grid(False)
    apply_recessive_axis_style(axes)
    axes.grid(False)


def plot_per_class_scores(axes, per_class_scores):
    """Recall, precision and F1 per class, grouped.

    Per class rather than pooled: overall accuracy here is dominated by bare and
    grass and barely moves on shrub, the class carrying the difficulty
    (instructions5.md section 6.4).

    Inputs: axes - Axes; per_class_scores - dict of {label: {recall, precision, f1}}
    Outputs: None
    """
    metric_names = ["recall", "precision", "f1"]
    bar_width = 0.26
    positions = np.arange(len(CLASS_ORDER))
    metric_alphas = [0.45, 0.72, 1.0]
    class_colours = [CLASS_COLORS[class_code] for class_code in CLASS_LABELS]
    for metric_index, metric in enumerate(metric_names):
        values = [per_class_scores.get(name, {}).get(metric, 0.0) or 0.0 for name in CLASS_ORDER]
        axes.bar(
            positions + (metric_index - 1) * bar_width,
            values,
            width=bar_width * 0.92,
            color=class_colours,
            alpha=metric_alphas[metric_index],
            edgecolor=SURFACE,
            linewidth=1.2,
            label=metric,
        )
    for class_index, name in enumerate(CLASS_ORDER):
        f1_score = per_class_scores.get(name, {}).get("f1", 0.0) or 0.0
        axes.text(class_index + bar_width, f1_score + 0.03, f"{f1_score:.2f}", ha="center", fontsize=8, color=INK)
    axes.set_xticks(positions)
    axes.set_xticklabels(CLASS_ORDER)
    axes.set_ylim(0, 1.12)
    axes.set_title("per-class scores", fontsize=10, color=INK, loc="left")
    # metric identity rides on shade, so it needs a legend of its own - the bar
    # colour is already spent on class identity and cannot carry a second job
    metric_shade_proxies = [Patch(facecolor=MUTED, alpha=alpha, edgecolor=SURFACE) for alpha in metric_alphas]
    axes.legend(
        metric_shade_proxies,
        metric_names,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        handlelength=1.1,
        columnspacing=1.1,
        labelcolor=MUTED,
    )
    apply_recessive_axis_style(axes)


def plot_fold_scores(axes, folds):
    """Per-fold macro-F1 and overall accuracy, to show stability across tiles.

    A score that swings between held-out tiles is a warning that the training
    set is too small or the tiles differ compositionally - both true here.

    Inputs: axes - Axes; folds - list of fold dicts
    Outputs: None
    """
    fold_names = [fold["held_out"].replace("_", "\n") for fold in folds]
    macro_f1_per_fold = [fold["macro_f1"] for fold in folds]
    overall_per_fold = [fold["overall"] for fold in folds]
    positions = np.arange(len(folds))
    axes.bar(positions - 0.17, macro_f1_per_fold, width=0.32, color=BAR_COLOR, edgecolor=SURFACE, linewidth=1.2)
    axes.bar(
        positions + 0.17,
        overall_per_fold,
        width=0.32,
        color=BAR_COLOR,
        alpha=0.4,
        edgecolor=SURFACE,
        linewidth=1.2,
    )
    for position, value in zip(positions - 0.17, macro_f1_per_fold):
        axes.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=INK)
    for position, value in zip(positions + 0.17, overall_per_fold):
        axes.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=MUTED)
    axes.set_xticks(positions)
    axes.set_xticklabels(fold_names, fontsize=7)
    axes.set_ylim(0, 1.15)
    axes.set_title(
        "per fold: macro-F1 (solid) vs overall (pale)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    apply_recessive_axis_style(axes)


def plot_confusion_matrix(axes, confusion_counts):
    """Row-normalized confusion matrix with the raw counts annotated.

    Rows are truth and sum to 1, so the diagonal reads as recall. Counts are
    kept in the cell because a rate over a small support is easy to over-read.

    Inputs: axes - Axes; confusion_counts - 4x4 nested list of pixel counts
    Outputs: None
    """
    counts = np.array(confusion_counts, dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    axes.imshow(rates, cmap=MAGNITUDE_COLORMAP, vmin=0, vmax=1, aspect="auto")
    for true_row in range(len(CLASS_ORDER)):
        for predicted_column in range(len(CLASS_ORDER)):
            value = rates[true_row, predicted_column]
            axes.text(
                predicted_column,
                true_row,
                f"{value:.2f}\n{int(counts[true_row, predicted_column]):,}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if value > 0.55 else INK,
            )
    axes.set_xticks(range(len(CLASS_ORDER)))
    axes.set_yticks(range(len(CLASS_ORDER)))
    axes.set_xticklabels(CLASS_ORDER)
    axes.set_yticklabels(CLASS_ORDER)
    axes.set_xlabel("predicted", fontsize=9, color=MUTED)
    axes.set_ylabel("true", fontsize=9, color=MUTED)
    axes.set_title(
        "confusion, row-normalized (diagonal = recall)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    axes.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)
    axes.grid(False)


def plot_feature_importance(axes, feature_importance, n_top_features):
    """Feature importance, highest first.

    Magnitude, so a single-hue ramp rather than class colors - these are
    features, not classes.

    Inputs: axes - Axes; feature_importance - dict of {feature: weight};
            n_top_features - int
    Outputs: None
    """
    ranked = sorted(feature_importance.items(), key=lambda pair: -pair[1])[:n_top_features][::-1]
    feature_names = [name for name, _ in ranked]
    weights = [weight for _, weight in ranked]
    shades = [MAGNITUDE_COLORMAP(0.35 + 0.6 * (weight / max(weights))) for weight in weights]
    axes.barh(feature_names, weights, color=shades, height=0.68, edgecolor=SURFACE, linewidth=1.2)
    for row, value in enumerate(weights):
        axes.text(
            value + max(weights) * 0.02,
            row,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color=INK,
        )
    axes.set_xlim(0, max(weights) * 1.2)
    axes.set_title(f"feature importance, top {len(ranked)}", fontsize=10, color=INK, loc="left")
    apply_recessive_axis_style(axes)
    axes.grid(True, axis="x", color=GRID, linewidth=0.6)
    axes.grid(False, axis="y")


def figure_for_framework(framework, framework_report, site, year, out_dir, run):
    """Render the five-panel diagnostic figure for one framework.

    Inputs: framework - key string; framework_report - the report section; site, year;
             out_dir - Path to write into
    Outputs: Path written
    """
    fig = plt.figure(figsize=(15.5, 9.0), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26, left=0.06, right=0.97, top=0.86, bottom=0.08)

    plot_training_balance(fig.add_subplot(grid[0, 0]), framework_report.get("train_pixels_per_class", {}))
    pooled = framework_report.get("pooled", {})
    plot_per_class_scores(fig.add_subplot(grid[0, 1]), pooled.get("per_class", {}))
    plot_fold_scores(fig.add_subplot(grid[0, 2]), framework_report.get("folds", []))
    plot_confusion_matrix(fig.add_subplot(grid[1, 0]), framework_report.get("confusion", [[0] * 4] * 4))
    if framework_report.get("feature_importance"):
        plot_feature_importance(fig.add_subplot(grid[1, 1:]), framework_report["feature_importance"], 14)

    feature_names = framework_report.get("features", [])
    macro_f1 = pooled.get("macro_f1", float("nan"))
    overall_accuracy = pooled.get("overall", float("nan"))
    fig.suptitle(
        f"Step 1d RF-A_{framework} · run {run} · {site} {year}",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=15,
        color=INK,
    )
    fig.text(
        0.06,
        0.925,
        f"macro-F1 {macro_f1:.3f} overall {overall_accuracy:.3f} · {len(feature_names)} features: {', '.join(feature_names)}",
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


def figure_framework_comparison(frameworks, site, year, out_dir):
    """Per-class F1 across frameworks - the A-E comparison, when more than one ran.

    Frameworks vary by input group only, so a difference here is attributable to
    the inputs (instructions5.md section 4.1).

    Inputs: frameworks - dict of {framework key: its report section}; site, year; out_dir
    Outputs: Path written, or None when fewer than two frameworks are present
    """
    framework_keys = [key for key in sorted(frameworks) if frameworks[key].get("pooled")]
    if len(framework_keys) < 2:
        return None
    fig, axes = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    width = 0.8 / len(CLASS_ORDER)
    positions = np.arange(len(framework_keys))
    for class_index, name in enumerate(CLASS_ORDER):
        values = [frameworks[key]["pooled"]["per_class"].get(name, {}).get("f1", 0.0) or 0.0 for key in framework_keys]
        offset = (class_index - (len(CLASS_ORDER) - 1) / 2) * width
        axes.bar(
            positions + offset,
            values,
            width=width * 0.9,
            color=CLASS_COLORS[CLASS_ORDER.index(name)],
            edgecolor=SURFACE,
            linewidth=1.4,
            label=name,
        )
        for position, value in zip(positions + offset, values):
            axes.text(position, value + 0.015, f"{value:.2f}", ha="center", fontsize=7, color=INK)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"RF-A_{key}" for key in framework_keys])
    axes.set_ylim(0, 1.12)
    axes.set_title(
        "per-class F1 by RF-A variant · variants differ by input features only",
        fontsize=11,
        color=INK,
        loc="left",
    )
    axes.legend(
        frameon=False,
        fontsize=9,
        ncol=len(CLASS_ORDER),
        loc="upper left",
        bbox_to_anchor=(0, 1.0),
    )
    apply_recessive_axis_style(axes)
    path = out_dir / f"stage3_1_framework_comparison_{site}_{year}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--run",
        default="1",
        help="run label - a number, optionally suffixed (e.g. 3 or 3_smoke). Reads and writes under run{LABEL}/",
    )
    args = parser.parse_args()

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

    max_pixels_per_polygon = report.get("max_pixels_per_polygon")
    print(f"Step 1d diagnostics - {site} {year} - run {report.get('run', args.run)}")
    print(f"polygon subsampling: {('max ' + str(max_pixels_per_polygon) + ' px per polygon') if max_pixels_per_polygon else 'OFF'}")
    print("=" * SEVENTY)
    for framework in sorted(frameworks):
        framework_report = frameworks[framework]
        figure_path = figure_for_framework(framework, framework_report, site, year, out_dir, report.get("run", args.run))
        pooled = framework_report.get("pooled", {})
        shrub_scores = pooled.get("per_class", {}).get("shrub", {})
        print(f"[{framework}] macro-F1 {pooled.get('macro_f1', float('nan')):.3f} shrub F1 {shrub_scores.get('f1', float('nan')):.3f} -> {figure_path.name}")

    comparison_path = figure_framework_comparison(frameworks, site, year, out_dir)
    if comparison_path:
        print(f"[all] framework comparison -> {comparison_path.name}")
    print(f"\nwrote to {out_dir}")


if __name__ == "__main__":
    main()
