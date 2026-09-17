"""Stage 5_2 - plot the RF-B phenology regression report.

Reads stage5_1_report_{SITE}_{YEAR}.json and renders one figure per model plus a
model comparison. Every panel answers a question that the JSON can state but not
show.

THE HEADLINE PANEL IS ERROR BY MIXEDNESS, not pooled error. Pure and near-pure
blocks are the easy case and dominate the count, so a pooled MAE can look
respectable while the model fails completely on mixtures - which is the regime
this project exists to resolve. The constant-mean baseline is drawn on that
panel as a reference line, because an error figure without its null model is
uninterpretable.

THE RAW SUM PANEL IS THE SECOND MOST IMPORTANT. Neither model enforces the
simplex, so whether four independently regressed fractions reconstruct full
coverage is a property of the model worth seeing rather than normalising away.

Colour follows the same two rules as stage 3_2:

  identity - class marks use the locked section 3 CLASS_COLORS, always beside a
             text label, because the palette fails a normal-vision separation
             check on grass against bare and colour must never be the only
             channel carrying class identity
  magnitude - error and count use a single-hue ramp deliberately outside the
             class palette, so a magnitude cell is never read as a class

AN OFF YEAR IS LABELLED ON THE FIGURE ITSELF. Stage 4 produced a fractional
cover map for the training year only, so that is the year any model is scored
against. When a run's phenology year is not the training year, the scores still
describe the training year, and the figure says so rather than leaving it in the
report alone, because a figure is what gets copied into a slide.

ARGUMENTS

    config
        positional, required. Site config JSON, e.g.
        config/srer_2022.json.
    --run
        required. Run LABEL, which is the directory name under
        `stage5_phenology_model_prediction/` with the leading `run`
        stripped. Stage 5_1 now fits one feature set, so a label is normally
        just the stage 4 run number. The historical `5` (retired timing-only)
        and `5_timing_evi` directories are still readable, since this script
        reads whatever report the label points at.

The script needs nothing else: every score, the feature set, the phenology year
and the training-year fields are read out of that run's stage5_1_report JSON,
so it is never told separately what it is plotting.

EXAMPLE COMMANDS

    conda activate LCSC

    # the current run
    python run_stage5_2_generate_phenology_regression_plots.py config/srer_2022.json --run 6

    # a historical run, retired feature sets included
    python run_stage5_2_generate_phenology_regression_plots.py config/srer_2022.json --run 5_timing_evi
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from constants import CLASS_COLORS, CLASS_LABELS, CLASS_NAMES
from helpers import resolve_config_path

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e4e4e2"
SURFACE = "#fcfcfb"
BASELINE_COLOR = "#a04a2f"
BAR_COLOR = "#4d8496"
TEST_COLOR = "#1f5566"
MIXEDNESS_ORDER = ["pure", "slightly mixed", "mixed", "highly mixed"]


def apply_recessive_axis_style(axes):
    """Recessive grid and axes so the marks carry the chart."""
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8, length=3)
    axes.grid(True, axis="both", color=GRID, linewidth=0.6, alpha=0.9)
    axes.set_axisbelow(True)


def plot_per_class_error(axes, splits, metric, title):
    """One metric per class, with every available split side by side.

    Bars grouped per class rather than one panel per split: the comparison that
    matters is train against test for the SAME class, and separating them into
    panels makes the eye do work the layout should do.

    SPLITS ARE PASSED IN, NOT ASSUMED, because out-of-bag scores only exist in
    reports written after they were added. An older report simply plots fewer
    bars instead of crashing.

    Inputs: axes; splits - list of (label, scores dict, alpha); metric - "mae",
            "bias" or "r2"; title - str
    Outputs: None
    """
    positions = np.arange(len(CLASS_NAMES))
    class_colours = [CLASS_COLORS[code] for code in CLASS_LABELS]
    width = 0.8 / len(splits)
    all_values = []
    for split_index, (label, scores, alpha) in enumerate(splits):
        values = [scores["per_class"][name][metric] or 0.0 for name in CLASS_NAMES]
        all_values.extend(values)
        offset = (split_index - (len(splits) - 1) / 2) * width
        axes.bar(positions + offset, values, width=width * 0.9, color=class_colours, alpha=alpha, edgecolor=SURFACE, linewidth=1.2, label=label)
        for position, value in zip(positions + offset, values):
            axes.text(position, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=6.5, color=INK if alpha == 1.0 else MUTED)
    if metric == "bias":
        axes.axhline(0.0, color=INK, linewidth=0.8)
    spread = max(all_values + [0.0]) - min(all_values + [0.0])
    axes.margins(y=0.22 if spread else 0.1)
    axes.set_xticks(positions)
    axes.set_xticklabels(CLASS_NAMES)
    # title padded so the legend anchored just above the axes clears it
    axes.set_title(title, fontsize=10, color=INK, loc="left", pad=20)
    # legend ABOVE the axes, never "best": with eight bars there is no free
    # corner, and matplotlib placed it over the tallest bar on every render
    axes.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.005), labelcolor=MUTED, handlelength=1.1, columnspacing=1.2)
    apply_recessive_axis_style(axes)


def plot_error_by_mixedness(axes, by_mixedness, baseline_mae, title):
    """MAE by how mixed the block is, with the null model as a reference line.

    THIS IS THE HEADLINE PANEL. Pure blocks are the easy case, so a model that
    only works on them is not a fractional-cover model. The baseline line is
    what makes the numbers mean anything: a bar above it is worse than
    predicting the site mean.

    Inputs: axes; by_mixedness - dict of {stratum: scores}; baseline_mae - float;
            title - str
    Outputs: None
    """
    labels = [label for label in MIXEDNESS_ORDER if label in by_mixedness]
    values = [by_mixedness[label]["mae"] for label in labels]
    counts = [by_mixedness[label]["n"] for label in labels]
    positions = np.arange(len(labels))
    axes.bar(positions, values, width=0.62, color=BAR_COLOR, edgecolor=SURFACE, linewidth=1.4)
    for position, value, count in zip(positions, values, counts):
        # value above the bar, count INSIDE it: stacking both above collided with
        # the baseline line and its label on every render
        axes.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        axes.text(position, value * 0.06, f"n={count:,}", ha="center", va="bottom", fontsize=7, color=SURFACE)
    axes.axhline(baseline_mae, color=BASELINE_COLOR, linewidth=1.4, linestyle="--")
    axes.text(-0.45, baseline_mae * 0.97, f"baseline {baseline_mae:.3f}", ha="left", va="top", fontsize=8, color=BASELINE_COLOR)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylim(0, max(max(values), baseline_mae) * 1.25)
    axes.set_title(title, fontsize=10, color=INK, loc="left")
    apply_recessive_axis_style(axes)


def plot_per_tile_error(axes, folds, test_scores, test_tile_ids, title):
    """MAE per held-out tile, train folds and test tiles distinguished by shade.

    Tiles are the holdout unit, so tile-to-tile spread is the honest read on
    stability - the same argument as the stage 3 fold spread.

    Inputs: axes; folds - list of fold dicts; test_scores; test_tile_ids; title
    Outputs: None
    """
    fold_names = [fold["held_out"].replace("_", "\n") for fold in folds]
    fold_values = [fold["mae"] for fold in folds]
    labels = fold_names + ["test tiles\npooled"]
    values = fold_values + [test_scores["mae"]]
    colours = [BAR_COLOR] * len(fold_values) + [TEST_COLOR]
    positions = np.arange(len(labels))
    axes.bar(positions, values, width=0.66, color=colours, edgecolor=SURFACE, linewidth=1.2)
    for position, value in zip(positions, values):
        axes.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=7)
    axes.set_title(title, fontsize=10, color=INK, loc="left")
    apply_recessive_axis_style(axes)


def plot_raw_sum(axes, cross_validated, test_scores, site_wide, title):
    """How far the four predicted fractions are from summing to one.

    Neither model enforces the simplex. A model needing large corrections is
    saying something about its own coherence, and renormalising first would hide
    it, so the raw sum is reported at every scope that has one.

    Inputs: axes; cross_validated, test_scores - score dicts; site_wide - dict
            or None; title - str
    Outputs: None
    """
    scopes, means, deviations = [], [], []
    for label, scores in (("cross-validated", cross_validated), ("test tiles", test_scores)):
        scopes.append(label)
        means.append(scores["raw_sum_before_renormalisation"]["mean"])
        deviations.append(scores["raw_sum_before_renormalisation"]["mean_absolute_deviation_from_one"])
    if site_wide:
        scopes.append("site-wide")
        means.append(site_wide["raw_sum_mean"])
        deviations.append(site_wide["raw_sum_mean_absolute_deviation_from_one"])
    positions = np.arange(len(scopes))
    axes.bar(positions, means, width=0.5, color=BAR_COLOR, edgecolor=SURFACE, linewidth=1.4)
    axes.axhline(1.0, color=BASELINE_COLOR, linewidth=1.4, linestyle="--")
    axes.text(len(scopes) - 0.5, 1.0, " exactly 1", ha="right", va="bottom", fontsize=8, color=BASELINE_COLOR)
    for position, mean_value, deviation in zip(positions, means, deviations):
        axes.text(position, mean_value, f"{mean_value:.4f}\nmean abs dev {deviation:.4f}", ha="center", va="bottom", fontsize=7, color=INK)
    axes.set_xticks(positions)
    axes.set_xticklabels(scopes, fontsize=8)
    axes.set_ylim(0, max(max(means), 1.0) * 1.25)
    axes.set_title(title, fontsize=10, color=INK, loc="left")
    apply_recessive_axis_style(axes)


def available_splits(model_report):
    """The score splits this report actually carries, in train-to-test order.

    Out-of-bag arrives first because it is the most optimistic: it is a RANDOM
    holdout, so autocorrelated neighbours sit on both sides of it. Reading the
    three left to right shows the price of each step toward an honest estimate.

    Inputs: model_report - the per-model report section
    Outputs: list of (label, scores dict, alpha)
    """
    splits = []
    if "out_of_bag" in model_report:
        splits.append(("out of bag, random holdout", model_report["out_of_bag"], 0.35))
    splits.append(("cross-validated, leave one tile out", model_report["cross_validated"], 0.62))
    splits.append(("held-out test tiles", model_report["test_tiles"], 1.0))
    return splits


def plot_importance_heatmap(axes, feature_names, per_class_increase, title):
    """Permutation importance as features by classes, the magnitude ramp.

    A heatmap rather than 52 bars. The cell is the MAE increase when that
    feature is shuffled, so a bright row is a feature the model leans on and a
    bright cell says which class it leans on it for.

    Inputs: axes; feature_names; per_class_increase - {class: [per feature]};
            title
    Outputs: None
    """
    matrix = np.array([per_class_increase[name] for name in CLASS_NAMES])
    image = axes.imshow(matrix, aspect="auto", cmap="BuPu", vmin=0.0)
    axes.set_xticks(np.arange(len(feature_names)))
    axes.set_xticklabels(feature_names, rotation=90, fontsize=7)
    axes.set_yticks(np.arange(len(CLASS_NAMES)))
    axes.set_yticklabels(CLASS_NAMES, fontsize=8)
    largest = matrix.max() if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axes.text(column, row, f"{value:.3f}".lstrip("0"), ha="center", va="center", fontsize=6, color=SURFACE if value > largest * 0.55 else INK)
    axes.set_title(title, fontsize=10, color=INK, loc="left")
    axes.tick_params(colors=MUTED, length=0)
    return image


def figure_feature_importance(model_name, model_report, report, output_directory):
    """Which features the model uses, per class, in training and out of tile.

    THE TRAIN AND TEST PANELS ARE THE POINT. A feature that scores high on the
    train rows and low on the test tiles was being used to memorise rather than
    to generalise, and only the pair shows that.

    Impurity importance is drawn as a cross-check with a caveat: the joint model
    reports ONE vector for all four outputs, because its trees split on the four
    together, so per-class impurity exists only for the independent model.

    Inputs: model_name; model_report; report; output_directory
    Outputs: Path written, or None when the report predates feature importance
    """
    importance = model_report.get("feature_importance")
    if not importance:
        return None
    feature_names = importance["features"]
    figure = plt.figure(figsize=(14.5, 7.6), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 2, hspace=0.65, wspace=0.16, left=0.07, right=0.97, top=0.82, bottom=0.14)

    plot_importance_heatmap(figure.add_subplot(grid[0, 0]), feature_names, importance["permutation_train"]["per_class_mae_increase"], "permutation importance, TRAIN rows - MAE increase when shuffled")
    plot_importance_heatmap(figure.add_subplot(grid[0, 1]), feature_names, importance["permutation_test"]["per_class_mae_increase"], "permutation importance, HELD-OUT TEST TILES")

    axes = figure.add_subplot(grid[1, 0])
    train_pooled = np.mean([importance["permutation_train"]["per_class_mae_increase"][name] for name in CLASS_NAMES], axis=0)
    test_pooled = np.mean([importance["permutation_test"]["per_class_mae_increase"][name] for name in CLASS_NAMES], axis=0)
    positions = np.arange(len(feature_names))
    axes.bar(positions - 0.2, train_pooled, width=0.38, color=BAR_COLOR, alpha=0.55, edgecolor=SURFACE, linewidth=1.1, label="train rows")
    axes.bar(positions + 0.2, test_pooled, width=0.38, color=TEST_COLOR, edgecolor=SURFACE, linewidth=1.1, label="test tiles")
    axes.set_xticks(positions)
    axes.set_xticklabels(feature_names, rotation=90, fontsize=7)
    axes.set_title("pooled over the four classes, train against test", fontsize=10, color=INK, loc="left", pad=18)
    axes.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005), labelcolor=MUTED)
    apply_recessive_axis_style(axes)

    axes = figure.add_subplot(grid[1, 1])
    impurity = importance["impurity"]
    impurity_labels = list(impurity)
    width = 0.8 / len(impurity_labels)
    for label_index, label in enumerate(impurity_labels):
        offset = (label_index - (len(impurity_labels) - 1) / 2) * width
        axes.bar(positions + offset, impurity[label], width=width * 0.9, edgecolor=SURFACE, linewidth=1.0, label=label, color=BAR_COLOR if len(impurity_labels) == 1 else CLASS_COLORS[CLASS_LABELS[label_index % len(CLASS_LABELS)]])
    axes.set_xticks(positions)
    axes.set_xticklabels(feature_names, rotation=90, fontsize=7)
    axes.set_title("impurity importance, a biased cross-check", fontsize=10, color=INK, loc="left", pad=18)
    axes.legend(frameon=False, fontsize=7, ncol=min(4, len(impurity_labels)), loc="lower left", bbox_to_anchor=(0, 1.005), labelcolor=MUTED)
    apply_recessive_axis_style(axes)

    figure.suptitle(f"RF-B {model_name} - feature importance - {report['site']} {report['ground_truth_year']}", x=0.07, y=0.965, ha="left", fontsize=15, color=INK)
    figure.text(0.07, 0.915, f"permutation importance is the MAE increase when one feature is shuffled, {importance['permutation_test']['n_repeats']} repeats, {importance['permutation_test']['n_rows_used']:,} test rows and {importance['permutation_train']['n_rows_used']:,} train rows. Impurity importance is computed on training data and favours features with more distinct values, so it is a cross-check only.", ha="left", fontsize=8.5, color=MUTED)

    path = output_directory / f"stage5_2_feature_importance_{model_name}_{report['site']}_{report['ground_truth_year']}.png"
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def write_metric_tables(report, output_directory):
    """The same numbers as CSV, because a figure cannot be sorted or pasted.

    Two long-format tables: one row per model, split, class and metric, and one
    row per model, split, class and feature for importance. Long format so they
    drop straight into a pivot without reshaping.

    Inputs: report; output_directory
    Outputs: list of Paths written
    """
    written = []
    metrics_path = output_directory / f"stage5_2_per_class_metrics_{report['site']}_{report['ground_truth_year']}.csv"
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "split", "class", "mae", "rmse", "bias", "r2", "n"])
        for model_name, model_report in report["models"].items():
            for label, scores, _ in available_splits(model_report):
                for class_name in CLASS_NAMES:
                    per_class = scores["per_class"][class_name]
                    writer.writerow([model_name, label, class_name, f"{per_class['mae']:.6f}", f"{per_class['rmse']:.6f}", f"{per_class['bias']:.6f}", f"{per_class['r2']:.6f}", scores["n"]])
    written.append(metrics_path)

    importance_rows = []
    for model_name, model_report in report["models"].items():
        importance = model_report.get("feature_importance")
        if not importance:
            continue
        for split_key in ("permutation_train", "permutation_test"):
            for class_name in CLASS_NAMES:
                for feature_name, value in zip(importance["features"], importance[split_key]["per_class_mae_increase"][class_name]):
                    importance_rows.append([model_name, split_key.replace("permutation_", ""), class_name, feature_name, f"{value:.6f}"])
        for label, values in importance["impurity"].items():
            for feature_name, value in zip(importance["features"], values):
                importance_rows.append([model_name, "impurity", label, feature_name, f"{value:.6f}"])
    if importance_rows:
        importance_path = output_directory / f"stage5_2_feature_importance_{report['site']}_{report['ground_truth_year']}.csv"
        with open(importance_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "split", "class_or_output", "feature", "mae_increase_or_impurity"])
            writer.writerows(importance_rows)
        written.append(importance_path)
    return written


def figure_for_model(model_name, model_report, report, output_directory):
    """The four-panel diagnostic figure for one model.

    Inputs: model_name; model_report - the report section; report - the whole
            report, for the baseline and run metadata; output_directory
    Outputs: Path written
    """
    figure = plt.figure(figsize=(15.5, 8.4), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 3, hspace=0.42, wspace=0.26, left=0.06, right=0.97, top=0.84, bottom=0.09)
    cross_validated = model_report["cross_validated"]
    test_scores = model_report["test_tiles"]
    baseline_mae = report["constant_mean_baseline"]["mae"]

    plot_error_by_mixedness(figure.add_subplot(grid[0, 0]), model_report["cross_validated_by_mixedness"], baseline_mae, "MAE by how mixed the block is, cross-validated")
    splits = available_splits(model_report)
    plot_per_class_error(figure.add_subplot(grid[0, 1]), splits, "mae", "per-class MAE")
    plot_per_class_error(figure.add_subplot(grid[0, 2]), splits, "bias", "per-class bias, signed")
    plot_per_tile_error(figure.add_subplot(grid[1, 0]), model_report["folds"], test_scores, report["test_tiles"], "MAE per held-out tile")
    plot_per_class_error(figure.add_subplot(grid[1, 1]), splits, "r2", "per-class R squared")
    plot_raw_sum(figure.add_subplot(grid[1, 2]), cross_validated, test_scores, model_report.get("site_wide"), "raw sum of the four fractions, before renormalisation")

    figure.suptitle(f"RF-B {model_name} - phenology {report['phenology_year']} to fractional cover - {report['site']} {report['ground_truth_year']}", x=0.06, y=0.965, ha="left", fontsize=15, color=INK)
    figure.text(
        0.06,
        0.925,
        f"pooled MAE {cross_validated['mae']:.4f} cross-validated, {test_scores['mae']:.4f} on test tiles, against a constant-mean baseline of {baseline_mae:.4f} - target {report['target']}, framework RF-A_{report['framework']}, {len(report['features'])} phenology characteristics",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    if not report.get("is_training_year", True):
        figure.text(0.06, 0.898, f"OFF YEAR - phenology year {report['phenology_year']}, fitted and scored against {report['ground_truth_year']} ground truth, the only year with a stage 4 map. These scores describe the training year.", ha="left", fontsize=8.5, color=BASELINE_COLOR)

    path = output_directory / f"stage5_2_diagnostics_{model_name}_{report['site']}_{report['ground_truth_year']}.png"
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def figure_model_comparison(report, output_directory):
    """Joint against independent, on the measures that would distinguish them.

    If the two are indistinguishable, the joint target is buying nothing and the
    four classes are being predicted independently anyway - which is the whole
    reason both were fitted.

    Inputs: report; output_directory
    Outputs: Path written
    """
    model_names = [name for name in ("joint", "independent") if name in report["models"]]
    if len(model_names) < 2:
        return None
    figure, axes_pair = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)

    axes = axes_pair[0]
    positions = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(model_names)
    for model_index, model_name in enumerate(model_names):
        per_class = report["models"][model_name]["cross_validated"]["per_class"]
        values = [per_class[name]["mae"] for name in CLASS_NAMES]
        offset = (model_index - (len(model_names) - 1) / 2) * width
        axes.bar(positions + offset, values, width=width * 0.9, color=BAR_COLOR, alpha=0.55 if model_name == "joint" else 1.0, edgecolor=SURFACE, linewidth=1.2, label=model_name)
        for position, value in zip(positions + offset, values):
            axes.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    axes.set_xticks(positions)
    axes.set_xticklabels(CLASS_NAMES)
    axes.set_title("per-class MAE by model, cross-validated", fontsize=11, color=INK, loc="left")
    axes.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    apply_recessive_axis_style(axes)

    axes = axes_pair[1]
    labels = [label for label in MIXEDNESS_ORDER if label in report["models"][model_names[0]]["cross_validated_by_mixedness"]]
    positions = np.arange(len(labels))
    for model_index, model_name in enumerate(model_names):
        by_mixedness = report["models"][model_name]["cross_validated_by_mixedness"]
        values = [by_mixedness[label]["mae"] for label in labels]
        offset = (model_index - (len(model_names) - 1) / 2) * width
        axes.bar(positions + offset, values, width=width * 0.9, color=BAR_COLOR, alpha=0.55 if model_name == "joint" else 1.0, edgecolor=SURFACE, linewidth=1.2, label=model_name)
    axes.axhline(report["constant_mean_baseline"]["mae"], color=BASELINE_COLOR, linewidth=1.4, linestyle="--")
    axes.text(len(labels) - 0.5, report["constant_mean_baseline"]["mae"], " baseline", ha="right", va="bottom", fontsize=8, color=BASELINE_COLOR)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_title("MAE by mixedness, by model", fontsize=11, color=INK, loc="left")
    axes.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    apply_recessive_axis_style(axes)

    difference = report["models"]["joint"]["cross_validated"]["mae"] - report["models"]["independent"]["cross_validated"]["mae"]
    figure.suptitle(f"RF-B model comparison - joint minus independent MAE {difference:+.4f}", x=0.06, y=0.98, ha="left", fontsize=13, color=INK)
    figure.text(0.06, 0.93, "A difference near zero means the joint four-column target buys nothing and the classes are predicted independently anyway.", ha="left", fontsize=9, color=MUTED)
    figure.tight_layout(rect=[0, 0, 1, 0.90])

    path = output_directory / f"stage5_2_model_comparison_{report['site']}_{report['ground_truth_year']}.png"
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot the RF-B phenology regression report.")
    parser.add_argument("config", help="site config JSON")
    parser.add_argument("--run", required=True, help="run label, e.g. 5")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    site, year = config["site"], config["year"]
    output_directory = resolve_config_path(config["results_root"], "stage5_phenology_model_prediction", f"run{args.run}")
    report_path = output_directory / f"stage5_1_report_{site}_{year}.json"
    if not report_path.exists():
        raise SystemExit(f"FAIL - no report at {report_path}. Run run_stage5_1_fit_phenology_fractional_cover.py --run {args.run} first.")
    report = json.loads(report_path.read_text())
    # A PREDICT-MODE RUN CARRIES MAPS, NOT SCORES. Stage 4 has ground truth for
    # the training year only, so an off-year run fits nothing and scores nothing
    # and there is no panel here that could be drawn from it.
    if report.get("mode") == "predict" or "constant_mean_baseline" not in report:
        raise SystemExit(f"FAIL - {report_path} is a predict-mode report for phenology year {report.get('phenology_year')}. It carries maps, not scores, so there is nothing to plot. Point --run at a training-year run.")

    print(f"RF-B diagnostics - {site} {year} - run {args.run}")
    print(f"phenology year {report['phenology_year']}, target {report['target']}, framework RF-A_{report['framework']}")
    if not report.get("is_training_year", True):
        print(f"OFF YEAR - scores describe the training year {report.get('training_year', report['ground_truth_year'])}, not {report['phenology_year']}")
    print(f"constant-mean baseline MAE {report['constant_mean_baseline']['mae']:.4f}")
    print("=" * 74)

    for model_name in sorted(report["models"]):
        model_report = report["models"][model_name]
        path = figure_for_model(model_name, model_report, report, output_directory)
        cross_validated = model_report["cross_validated"]
        mixed_strata = model_report["cross_validated_by_mixedness"]
        worst_label = max(mixed_strata, key=lambda label: mixed_strata[label]["mae"])
        print(f"[{model_name}] MAE {cross_validated['mae']:.4f} cross-validated, {model_report['test_tiles']['mae']:.4f} test, worst stratum {worst_label} at {mixed_strata[worst_label]['mae']:.4f} -> {path.name}")

    comparison_path = figure_model_comparison(report, output_directory)
    if comparison_path:
        print(f"[both] model comparison -> {comparison_path.name}")
    print(f"\nwrote to {output_directory}")


if __name__ == "__main__":
    main()
