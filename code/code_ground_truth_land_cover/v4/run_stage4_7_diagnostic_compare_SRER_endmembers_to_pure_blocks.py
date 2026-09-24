"""Stage 4_7 diagnostic - hand-drawn SRER end members against RF-A's pure blocks.

THE CHECK THE WHOLE TRANSFER RESTS ON. Stage 5 normalises every site's features
against its own end members, so if SRER's end members are wrong, every model
trained on them and every site predicted from them is wrong in the same way and
nothing downstream would reveal it.

TWO INDEPENDENT ROUTES TO THE SAME FOUR CLASS CENTRES

    hand end members - polygons drawn on 10 cm imagery, read through stage 4_6's
        buffer and QA rules, summarised by stage 4_7
    pure blocks - RF-A's own 1 m classification, aggregated by stage 4_1 to 3 m
        blocks where at least min_pure of the valid pixels share one class

They share no step: one is a person reading imagery, the other is a classifier
reading reflectance and height. Agreement means two routes reached the same
place, which is what makes the end member statistics safe to normalise with.

NEITHER SIDE IS GROUND TRUTH, so a disagreement says look, not which to believe.

THE MEDIANS ARE COMPARABLE, THE BOX WIDTHS ARE NOT. The hand box is the spread
across 15 to 20 polygon medians; the pure box is the spread across thousands of
individual blocks. The pure box will be far wider and that is arithmetic, not a
finding. The verdict below is therefore built on the medians alone.

WHAT IT DOES

    1. read stage 4_7's SRER statistics, the hand side, as recorded
    2. read pure_endmember_{framework}_SRER_{year}.tif from stage 4_1
    3. check that raster sits cell for cell on the PLSP production grid
    4. apply THE SAME PLSP FILTERS as the hand side: NumCycles == 1, QA 1 or 2,
       and no fill in any of the 13 features
    5. take a 10% random sample per class, with a fixed seed, so the figure is
       reproducible and the run is quick
    6. draw both sides, hand on top, pure below, SHARING Y LIMITS per feature
    7. write a table: hand median, pure median, difference, and that difference
       as a fraction of the pure IQR

READING THE TABLE. difference_in_pure_iqr is the number that matters. Below 0.5
the two routes agree well within the pure blocks' own spread; above 1.0 the hand
end members sit outside the middle half of what RF-A calls that class, which
needs explaining before stage 5 normalises anything with them.

INPUTS

    config/endmembers/SRER_2022_endmembers.json
    {endmembers}/stage4_7/{SITE}_{year}_endmember_stats.json, from stage 4_7
    stage4_aggregation/run{N}/stage4_2_planet_blocks/pure_endmember_{framework}_SRER_{year}.tif
    the PLSP netCDF for the site's imagery year

OUTPUTS, overwritten every run, in {endmembers}/stage4_7/ beside stage 4_7's own

    {SITE}_{year}_endmembers_against_pure_blocks.png
    {SITE}_{year}_endmembers_against_pure_blocks.csv

ARGUMENTS

    --run
        required. The stage 4 run label, e.g. 5.
    --framework
        RF-A framework to compare against, default C, the one RF-B trains on.

The share of pure blocks sampled per class is DEFAULT_SAMPLE_SHARE, a property
of the analysis rather than of a run, so it is set here and not on the command
line.

EXAMPLE COMMANDS

    conda activate LCSC
    python run_stage4_7_diagnostic_compare_SRER_endmembers_to_pure_blocks.py --run 5
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import run_stage4_4_download_site_endmember_tiles as download_stage
import run_stage4_7_compute_endmember_statistics as stats_stage
from constants import CLASS_COLORS, CLASS_LABELS, SEVENTY
from helpers import endmember_imagery_year, endmember_stats_directory, planet_blocks_directory

CONFIG_NAME = "SRER_2022_endmembers.json"
DEFAULT_FRAMEWORK = "C"
DEFAULT_SAMPLE_SHARE = 0.1
SAMPLE_SEED = 20220817  # fixed so two runs draw the same blocks
PURE_NODATA = 255
AGREEMENT_GOOD = 0.5  # difference as a fraction of the pure IQR
AGREEMENT_POOR = 1.0


def read_pure_block_classes(planet_blocks, framework, site, year, grid):
    """RF-A's pure block class per cell, checked onto the PLSP production grid.

    The hand cells are indexed by (row, column) on the production grid, and the
    pure blocks must be read on that same grid or the two sides would describe
    different ground while looking perfectly plausible.

    Inputs: planet_blocks - directory; framework; site; year; grid
    Outputs: uint8 array [rows, columns], PURE_NODATA where not pure
    """
    path = planet_blocks / f"pure_endmember_{framework}_{site}_{year}.tif"
    if not path.exists():
        raise SystemExit(f"FAIL - no pure end member raster at {path}; run stage 4_1 for framework {framework}")
    with rasterio.open(path) as dataset:
        problems = []
        if (dataset.width, dataset.height) != (len(grid["x_centres"]), len(grid["y_centres"])):
            problems.append(f"size {dataset.width} x {dataset.height} against production {len(grid['x_centres'])} x {len(grid['y_centres'])}")
        if abs(dataset.transform.c - grid["left"]) > 1e-6 or abs(dataset.transform.f - grid["top"]) > 1e-6:
            problems.append(f"origin {dataset.transform.c}, {dataset.transform.f} against production {grid['left']}, {grid['top']}")
        if abs(dataset.transform.a - grid["pixel_size"]) > 1e-9:
            problems.append(f"pixel size {dataset.transform.a} against production {grid['pixel_size']}")
        if problems:
            raise SystemExit(f"FAIL - {path.name} is not on the production grid: " + "; ".join(problems))
        return dataset.read(1), path


def sample_pure_blocks(pure_classes, usable, feature_stack, sample_share):
    """A fixed random sample of usable pure blocks per class, with their features.

    THE SAME PLSP FILTERS AS THE HAND SIDE are applied first, through the usable
    mask: NumCycles == 1, QA 1 or 2 and no fill anywhere in the 13 features.
    Sampling before filtering would compare filtered hand cells against
    unfiltered blocks, which is the comparison this check exists to avoid.

    Inputs: pure_classes; usable - bool array; feature_stack [13, rows, cols];
            sample_share - float
    Outputs: dict of class_code to {values [n, 13], n_pure, n_usable, n_sampled}
    """
    generator = np.random.default_rng(SAMPLE_SEED)
    sampled = {}
    for class_code in sorted(CLASS_LABELS):
        is_class = pure_classes == class_code
        rows, columns = np.where(is_class & usable)
        take = max(1, int(round(len(rows) * sample_share)))
        chosen = generator.choice(len(rows), size=min(take, len(rows)), replace=False) if len(rows) else np.array([], dtype=int)
        values = feature_stack[:, rows[chosen], columns[chosen]].T.astype("float64") if len(chosen) else np.empty((0, feature_stack.shape[0]))
        sampled[class_code] = {"values": values, "n_pure": int(is_class.sum()), "n_usable": int(len(rows)), "n_sampled": int(len(chosen))}
    return sampled


def pure_percentiles(values):
    """Median, p25, p75, p5 and p95 per feature for one class's pure blocks.

    Inputs: values - array [n_blocks, n_features]
    Outputs: dict of lists, matching stage 4_7's summary keys
    """
    percentiles = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
    return {"p5": percentiles[0].tolist(), "p25": percentiles[1].tolist(), "median": percentiles[2].tolist(), "p75": percentiles[3].tolist(), "p95": percentiles[4].tolist()}


def agreement_rows(hand_statistics, pure_statistics, feature_names):
    """One row per class and feature: both medians, their difference, and the verdict.

    THE DIFFERENCE IS SCALED BY THE PURE IQR, not by an absolute threshold,
    because the 13 features are in days and in EVI2 units and no single number
    could serve both. A day of difference means little where a class spans 40
    days and a lot where it spans 4.

    Inputs: hand_statistics - 4_7's classes dict; pure_statistics; feature_names
    Outputs: list of row dicts
    """
    rows = []
    for class_code in sorted(CLASS_LABELS):
        class_name = CLASS_LABELS[class_code]
        hand = hand_statistics[class_name]
        pure = pure_statistics[class_code]
        for feature_index, feature_name in enumerate(feature_names):
            hand_median = hand["median"][feature_index]
            pure_median = pure["median"][feature_index]
            pure_iqr = pure["p75"][feature_index] - pure["p25"][feature_index]
            difference = hand_median - pure_median
            scaled = abs(difference) / pure_iqr if pure_iqr else float("nan")
            verdict = "agrees" if scaled <= AGREEMENT_GOOD else ("differs" if scaled > AGREEMENT_POOR else "borderline")
            rows.append({"class": class_name, "feature": feature_name, "hand_median": round(hand_median, 4), "pure_median": round(pure_median, 4), "difference": round(difference, 4), "pure_iqr": round(pure_iqr, 4), "difference_in_pure_iqr": round(scaled, 3), "verdict": verdict})
    return rows


def plot_boxes(axes, summaries, feature_index, hand_median=None):
    """Four class boxes for one feature, from recorded percentiles.

    Inputs: axes; summaries - {class_code: summary dict}; feature_index;
            hand_median - draw the hand value as a line, for the pure panel
    Outputs: (low, high) of the whiskers, so both rows can share limits
    """
    boxes = []
    for class_code in sorted(CLASS_LABELS):
        summary = summaries[class_code]
        boxes.append({"label": CLASS_LABELS[class_code][:2], "med": summary["median"][feature_index], "q1": summary["p25"][feature_index], "q3": summary["p75"][feature_index], "whislo": summary["p5"][feature_index], "whishi": summary["p95"][feature_index], "fliers": []})
    drawn = axes.bxp(boxes, showfliers=False, patch_artist=True, widths=0.6, medianprops={"color": stats_stage.INK, "linewidth": 1.4}, whiskerprops={"color": stats_stage.MUTED, "linewidth": 1.0}, capprops={"color": stats_stage.MUTED, "linewidth": 1.0})
    for patch, class_code in zip(drawn["boxes"], sorted(CLASS_LABELS)):
        patch.set_facecolor(CLASS_COLORS[class_code])
        patch.set_edgecolor(stats_stage.INK)
        patch.set_linewidth(0.9)
        patch.set_alpha(0.75)
    if hand_median is not None:
        for position, class_code in enumerate(sorted(CLASS_LABELS), start=1):
            axes.plot([position - 0.42, position + 0.42], [hand_median[class_code]] * 2, color=stats_stage.INK, linewidth=1.2, linestyle=":")
    stats_stage.apply_recessive_axis_style(axes)
    return min(box["whislo"] for box in boxes), max(box["whishi"] for box in boxes)


def figure_comparison(hand_summaries, pure_summaries, feature_names, site, year, sampled, output_path):
    """Hand end members on top, pure blocks below, sharing y limits per feature.

    SHARED LIMITS ARE THE POINT. Two panels auto-scaled independently make any
    pair of distributions look alike, which is exactly the illusion this figure
    has to avoid.

    Inputs: hand_summaries; pure_summaries; feature_names; site; year; sampled;
            output_path
    Outputs: Path written
    """
    figure, theme = stats_stage.start_themed_figure(max(16.0, 1.5 * len(feature_names)), 9.2)
    grid_spec = figure.add_gridspec(2, len(feature_names), hspace=0.30, wspace=0.62, left=0.042, right=0.99, top=0.83, bottom=0.08)
    for feature_index, feature_name in enumerate(feature_names):
        top_axes = figure.add_subplot(grid_spec[0, feature_index])
        bottom_axes = figure.add_subplot(grid_spec[1, feature_index])
        hand_low, hand_high = plot_boxes(top_axes, hand_summaries, feature_index)
        hand_medians = {class_code: hand_summaries[class_code]["median"][feature_index] for class_code in sorted(CLASS_LABELS)}
        pure_low, pure_high = plot_boxes(bottom_axes, pure_summaries, feature_index, hand_median=hand_medians)
        low, high = min(hand_low, pure_low), max(hand_high, pure_high)
        margin = (high - low) * 0.1 or 1.0
        top_axes.set_ylim(low - margin, high + margin)
        bottom_axes.set_ylim(low - margin, high + margin)
        top_axes.set_title(feature_name, color=stats_stage.INK, fontsize=9)
    # total_sampled = sum(entry["n_sampled"] for entry in sampled.values())
    figure.suptitle(f"{site} {year} - hand end members against RF-A pure blocks", color=stats_stage.INK, fontsize=13, y=0.955)
    figure.text(0.042, 0.895, "top: hand polygons, the spread across polygon medians. bottom: pure blocks, the spread across individual blocks, with the hand median dotted on each box.", color=stats_stage.MUTED, fontsize=8.5, ha="left")
    return stats_stage.finish_themed_figure(figure, theme, output_path)


def main():
    parser = argparse.ArgumentParser(description="Compare SRER's hand-drawn end members with RF-A's pure blocks.")
    parser.add_argument("--run", required=True, help="stage 4 run label, e.g. 5")
    parser.add_argument("--framework", default=DEFAULT_FRAMEWORK, help="RF-A framework, default C")
    args = parser.parse_args()

    config = json.loads((stats_stage.CONFIG_DIRECTORY / CONFIG_NAME).read_text())
    site = config["site"]
    year = endmember_imagery_year(config)
    stats_directory = endmember_stats_directory(config["results_root"], args.run)
    statistics_path = stats_directory / f"{site}_{year}_endmember_stats.json"
    if not statistics_path.exists():
        raise SystemExit(f"FAIL - no stage 4_7 statistics at {statistics_path}; run stage 4_7 first")
    hand_statistics = json.loads(statistics_path.read_text())
    feature_names = hand_statistics["feature_names"]

    print(f"Stage 4_7 diagnostic - {site} {year} hand end members against RF-A pure blocks, framework {args.framework}")
    print("=" * SEVENTY)

    grid = download_stage.read_production_grid(config)
    download_stage.check_crs_agrees(config, grid)
    netcdf_path = stats_stage.lsp_netcdf_path(config, year)
    feature_stack, netcdf_feature_names, usable, diagnostics = stats_stage.read_feature_cube(netcdf_path, config, grid)
    if netcdf_feature_names != feature_names:
        raise SystemExit(f"FAIL - the statistics were written for {feature_names} but the netCDF gives {netcdf_feature_names}")
    print(f"PLSP {year} {Path(netcdf_path).name}, usable phenology {diagnostics['usable_phenology']:.2%}")

    planet_blocks = planet_blocks_directory(config["results_root"], args.run)
    pure_classes, pure_path = read_pure_block_classes(planet_blocks, args.framework, site, year, grid)
    print(f"pure blocks {pure_path.name}, on the production grid")

    sampled = sample_pure_blocks(pure_classes, usable, feature_stack, DEFAULT_SAMPLE_SHARE)
    pure_summaries = {}
    for class_code in sorted(CLASS_LABELS):
        entry = sampled[class_code]
        if entry["n_sampled"] == 0:
            raise SystemExit(f"FAIL - no usable pure block for {CLASS_LABELS[class_code]}")
        pure_summaries[class_code] = pure_percentiles(entry["values"])
        print(f"{CLASS_LABELS[class_code]}: {entry['n_pure']:,} pure blocks, {entry['n_usable']:,} usable, {entry['n_sampled']:,} sampled ({DEFAULT_SAMPLE_SHARE:.0%})")

    hand_summaries = {class_code: hand_statistics["classes"][CLASS_LABELS[class_code]] for class_code in sorted(CLASS_LABELS)}
    rows = agreement_rows(hand_statistics["classes"], pure_summaries, feature_names)
    stem = f"{site}_{year}_endmembers_against_pure_blocks"
    table_path = stats_directory / f"{stem}.csv"
    with open(table_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_path = figure_comparison(hand_summaries, pure_summaries, feature_names, site, year, sampled, stats_directory / f"{stem}.png")

    print("\nagreement, difference between medians as a fraction of the pure IQR")
    for class_code in sorted(CLASS_LABELS):
        class_name = CLASS_LABELS[class_code]
        class_rows = [row for row in rows if row["class"] == class_name]
        worst = max(class_rows, key=lambda row: row["difference_in_pure_iqr"])
        counts = {verdict: sum(1 for row in class_rows if row["verdict"] == verdict) for verdict in ("agrees", "borderline", "differs")}
        print(f"{class_name}: {counts['agrees']} agree, {counts['borderline']} borderline, {counts['differs']} differ of {len(class_rows)}; worst {worst['feature']} hand {worst['hand_median']} against pure {worst['pure_median']}, {worst['difference_in_pure_iqr']} IQR")
    print("\n" + "=" * SEVENTY)
    print(f"wrote {table_path.name} and {figure_path.name} in {stats_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
