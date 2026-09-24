"""Stage 4_7 - end member statistics per site, from the hand-drawn polygons.

Reads each site's end member polygons, finds the PLSP cells they cover, and
records what the four classes look like in phenology space at that site. Stage 5
normalises every site's features against these statistics, which is what lets
ONE RF-B, trained at SRER 2022, be applied everywhere without retraining.

STRICTLY READ-ONLY FOR POLYGONS, as in stage 4_6. Nothing here writes to a
polygon file, and the held-out choice is recorded in the report only.

ONLY SITES THAT PASS STAGE 4_6 ARE PROCESSED. The gate is recomputed here by
calling stage 4_6 directly rather than reading its report, so the two stages
cannot disagree about which polygons count. Sites with any other status are
listed with that status and produce no statistics.

WHICH CELLS ARE READ, per counting polygon

    the WHOLE 3 m cells the polygon covers after the 0.5 m inward buffer, the
        same rule stage 4_6 gates on
    kept only where the phenology is usable: NumCycles == 1, QA 1 or 2, and no
        fill in any of the 13 features
    drawn against kept is recorded per class, as the QA pass rate

THE POLYGON IS THE SAMPLE, NOT THE PIXEL. Each polygon contributes its own
median per feature, and the class statistics are taken across those polygon
medians. A polygon covering 30 cells therefore does not outvote one covering 1.

FEATURES, the same 13 stage 5_1 fits on, read from PLSP_Layers.csv by layer
number through stage 5_1's own reader so the two cannot drift apart

    timing, layers 2 to 8 - OGI, 50PCGI, OGMx, Peak, OGD, 50PCGD, OGMn
    durations, derived - DurGU, DurGD, LOS
    greening, layers 9 to 11 - EVImax, EVIamp, EVIarea

THE YEAR IS THE SITE'S IMAGERY YEAR, and the tier follows the year rather than
being configured: production where that year was released, stage otherwise. At
SRER 2022 is a stage file while 2017 to 2021 are production.

HELD-OUT POLYGONS. 3 per class are kept out of the statistics for the transfer
check in stage 5_2. They are chosen HERE, by a stable hash of the site and the
polygon id, so the labeller cannot pick them and two runs over the same polygons
hold out the same ones. Adding polygons can still displace one, which is
harmless because statistics are only computed once a site PASSES stage 4_6.

THE CLASS-ORDER FLAGS ARE RECORDED, NEVER APPLIED. The G3 greening transform,
(x - bare) / (tree - bare), assumes bare is least green and tree most green. An
evergreen juniper site can break that, and dividing by a flipped or tiny
denominator would send features to the model backwards with no error. Each
greening feature is therefore flagged per site, for stage 5 to act on.

NO POLYGON IS EVER CHOSEN OR EDITED BY LOOKING AT ITS PLSP VALUES. Those values
are what these statistics measure.

INPUTS, per site

    config/endmembers/{SITE}_{year}_endmembers.json, from stage 4_3
    endmember_polygons_{site_name}_{year}.gpkg, hand-drawn through stage 4_5
    the PLSP netCDF for the site's imagery year
    everything stage 4_6 needs, since its gate is recomputed

OUTPUTS, overwritten every run

    EVERY OUTPUT GOES TO {endmembers}/stage4_7/, NAMED BY SITE CODE, so the eight
    sites' statistics sit together and can be read against one another.

    {endmembers}/stage4_7/{SITE}_{year}_endmember_stats.json
    {endmembers}/stage4_7/{SITE}_{year}_endmember_cells.csv
    {endmembers}/stage4_7/endmember_stats_summary.csv, one row per site, class and
        feature
    {endmembers}/stage4_7/{SITE}_{year}_endmember_class_spread.png, one panel per
        feature, the four class boxes in each
    {endmembers}/stage4_7/{SITE}_{year}_endmember_polygon_counts.png, polygons
        drawn against polygons passing the buffer and QA rules, per class per
        tile

ARGUMENTS

    --run
        required. The stage 4 run label the endmembers folder sits under,
        e.g. 5.

EXAMPLE COMMANDS

    conda activate LCSC
    python run_stage4_7_compute_endmember_statistics.py --run 5
"""

import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_stage4_4_download_site_endmember_tiles as download_stage
import run_stage4_6_check_endmember_labeling_progress as progress_stage
import run_stage5_1_fit_phenology_fractional_cover as model_stage
import xarray as xr
from aquarel import load_theme
from constants import CLASS_COLORS, CLASS_LABELS, SEVENTY
from helpers import endmember_imagery_year, endmember_polygon_path, endmember_stats_directory, resolve_config_path

CONFIG_DIRECTORY = Path(__file__).resolve().parent / "config" / "endmembers"
HOLD_OUT_PER_CLASS = 3
GREENING_FEATURE_NAMES = ("EVImax", "EVIamp", "EVIarea")
CLASS_ORDER_FOR_G3 = (0, 1, 2, 3)  # bare, grass, shrub, tree - least green to most green
SUMMARY_NAME = "endmember_stats_summary.csv"
REPORT_NAME = "stage4_7_endmember_statistics.json"

# FIGURE STYLE, the aquarel arctic_dark theme used by the site pixel quality
# figures in selected_sites_info, so the project's figures read as one set. The
# theme is applied per figure rather than at import, because applying a theme
# edits global rcParams and this module must not change how anything else draws.
# The colours below are that theme's own Nord values, named here because the
# marks are placed by hand rather than by a seaborn palette.
FIGURE_THEME = "arctic_dark"
FIGURE_FONT = "DejaVu Sans"
FIGURE_COLOR = "#3B4252"  # theme figure.facecolor
SURFACE = "#4C566A"  # theme axes.facecolor
INK = "#ECEFF4"  # theme text.color
MUTED = "#D8DEE9"  # theme axes.edgecolor
GRID = "#ECEFF4"
DRAWN_COLOR = "#D8DEE9"


def lsp_netcdf_path(config, plsp_year):
    """The PLSP netCDF for one site-year, production tier or stage tier.

    WHICH TIER HOLDS A YEAR IS NOT CONFIGURED, IT IS DISCOVERED. Production
    files carry the full site identity in the name, stage files are PLSP_{Y}.nc
    only, and a year moves from stage to production as the product is released.
    Both spellings are tried in both folders, production first, so a released
    year is preferred over its stage copy. This mirrors stage 5_1's finder,
    which reads the same files from the fuller site config.

    Inputs: config - an end member config; plsp_year - int
    Outputs: Path
    """
    site_directory_name = config["site_name"]
    site_without_suffix = site_directory_name[: -len("_NEON")] if site_directory_name.endswith("_NEON") else site_directory_name
    # THE PRODUCTION NAME DIFFERS BETWEEN SITE TYPES. A NEON site's file carries
    # the _NEON segment, US-xSR_NEON_Santa_Rita_..., while an AmeriFlux site's
    # does not, US-Wkg_Walnut_Gulch_.... Both are tried rather than assumed.
    candidates = []
    for subdirectory in ("PLSP_production_nc", "PLSP_stage_nc"):
        for file_name in (f"{config['site_id']}_NEON_{site_without_suffix}_PLSP_{plsp_year}.nc", f"{config['site_id']}_{site_directory_name}_PLSP_{plsp_year}.nc", f"PLSP_{plsp_year}.nc"):
            candidate = resolve_config_path(config["planet_data_root"], subdirectory, site_directory_name, file_name)
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"FAIL - no PLSP netCDF for {config['site']} {plsp_year}. Tried:\n  " + "\n  ".join(str(candidate) for candidate in candidates))


def check_netcdf_on_production_grid(dataset, grid, netcdf_path):
    """Stop unless the netCDF sits cell for cell on the production grid.

    Stage 4_6 indexes QA by (row, column) on the production grid, and the cells
    found there are read straight out of this array. A one-row or half-pixel
    disagreement would pair every polygon with the wrong phenology and produce
    plausible-looking statistics of the wrong ground, with no error raised.

    Stage files are also not guaranteed to store y in the same direction as
    production files, so the two coordinate arrays are compared as stored, and
    a reversed y is reported rather than quietly accepted.

    Inputs: dataset - open xarray Dataset; grid - from read_production_grid;
            netcdf_path - for the message
    Outputs: None, raises SystemExit on any mismatch
    """
    x_values = np.asarray(dataset["x"].values, dtype="float64")
    y_values = np.asarray(dataset["y"].values, dtype="float64")
    problems = []
    if x_values.size != grid["x_centres"].size or y_values.size != grid["y_centres"].size:
        problems.append(f"size {x_values.size} x {y_values.size} against production {grid['x_centres'].size} x {grid['y_centres'].size}")
    else:
        if not np.allclose(x_values, grid["x_centres"], atol=1e-6):
            problems.append(f"x centres differ, first {x_values[0]} against {grid['x_centres'][0]}")
        if not np.allclose(y_values, grid["y_centres"], atol=1e-6):
            reversed_note = " (y runs the other way)" if np.allclose(y_values[::-1], grid["y_centres"], atol=1e-6) else ""
            problems.append(f"y centres differ, first {y_values[0]} against {grid['y_centres'][0]}{reversed_note}")
    if problems:
        raise SystemExit(f"FAIL - {Path(netcdf_path).name} is not on the production grid: " + "; ".join(problems))


def read_feature_cube(netcdf_path, config, grid):
    """The 13 features and the usable mask for one site-year, on the production grid.

    Stage 5_1's readers are called rather than copied. They own the order of
    operations that matters here: mask fill to NaN, cast, scale, and only then
    subtract dates to form the durations. Subtracting raw fill values instead
    yields 32767 - 32767 = 0, a plausible zero-day duration that no later check
    would catch.

    Inputs: netcdf_path; config; grid
    Outputs: (feature_stack [13, rows, columns], feature_names, usable bool
             [rows, columns], quality diagnostics dict)
    """
    specification = model_stage.read_layer_specification(config, None)
    with xr.open_dataset(netcdf_path, mask_and_scale=False) as dataset:
        check_netcdf_on_production_grid(dataset, grid, netcdf_path)
        quality_mask, diagnostics = model_stage.read_quality_mask(dataset, specification)
        feature_stack, feature_names = model_stage.read_phenology_feature_stack(dataset, specification)
    all_features_finite = np.all(np.isfinite(feature_stack), axis=0)
    usable = quality_mask & all_features_finite
    diagnostics["all_features_finite"] = float(all_features_finite.mean())
    diagnostics["usable_phenology"] = float(usable.mean())
    return feature_stack, feature_names, usable, diagnostics


def whole_cells_in(geometry, grid):
    """The (row, column) of every PLSP cell the polygon covers after the inward buffer.

    THE RULE IS STAGE 4_6'S, IMPORTED, NOT RESTATED. 4_6 gates a polygon on
    having at least one such cell, and 4_7 then reads exactly those cells, so
    the gate and the sample can never mean different things.

    Inputs: geometry - shapely polygon in the grid CRS; grid
    Outputs: list of (row, column)
    """
    buffered = geometry.buffer(-progress_stage.INWARD_BUFFER_M)
    if buffered.is_empty:
        return []
    return [(row, column) for row, column, cell in progress_stage.cells_near(buffered, grid) if buffered.covers(cell)]


def is_held_out(feature_id, ranked_ids):
    """Whether one polygon is held out of the statistics for the stage 5_2 check.

    THE CHOICE IS A HASH, NOT A DRAW. A random sample would differ on every run
    of this script, so the transfer check would silently change its meaning
    between runs. Hashing the site and the polygon id fixes each polygon's rank
    for good, so two runs over the same polygons always hold out the same ones.

    ADDING POLYGONS CAN STILL DISPLACE ONE, because the top three of a longer
    list may include a newcomer. That is harmless in practice: statistics are
    only computed once a site PASSES stage 4_6, by which point labelling there
    has stopped, so the set is settled before anything downstream uses it.

    Inputs: feature_id; ranked_ids - the class's counting ids in hash order
    Outputs: bool
    """
    return feature_id in ranked_ids[:HOLD_OUT_PER_CLASS]


def hash_rank(site_name, feature_id):
    """The stable sort key that decides the held-out polygons.

    Inputs: site_name; feature_id
    Outputs: str, the hex digest
    """
    return hashlib.sha256(f"{site_name}:{feature_id}".encode()).hexdigest()


# NOTE stats per polygon
def summarise_polygon_medians(polygon_medians):
    """Median, IQR, p5 and p95 ACROSS POLYGONS, one feature at a time.

    Inputs: polygon_medians - array [n_polygons, n_features]
    Outputs: dict of lists, one value per feature
    """
    percentiles = np.percentile(polygon_medians, [5, 25, 50, 75, 95], axis=0)
    return {
        "median": percentiles[2].tolist(),
        "iqr": (percentiles[3] - percentiles[1]).tolist(),
        "p25": percentiles[1].tolist(),
        "p75": percentiles[3].tolist(),
        "p5": percentiles[0].tolist(),
        "p95": percentiles[4].tolist(),
    }


def check_class_order(class_statistics, feature_names):
    """Flag greening features whose class order breaks the G3 assumption.

    G3 rescales a greening feature onto a bare-to-tree axis. That is only
    meaningful where bare really is the least green class and tree the most: an
    evergreen juniper can hold a LOWER EVI amplitude than the grass beneath it,
    which flips the denominator's sign and mirrors every normalised value.
    A denominator small against the two end classes' spread is just as bad, as
    it amplifies noise instead of scaling.

    NOTHING IS APPLIED HERE. The flags are recorded for stage 5 to act on, which
    is why they name the feature rather than change it.

    Inputs: class_statistics - {class_name: summary dict}; feature_names
    Outputs: dict of {feature_name: flag dict} for the greening features only
    """
    flags = {}
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name not in GREENING_FEATURE_NAMES:
            continue
        medians = [class_statistics[CLASS_LABELS[code]]["median"][feature_index] for code in CLASS_ORDER_FOR_G3]
        spreads = [class_statistics[CLASS_LABELS[code]]["iqr"][feature_index] for code in CLASS_ORDER_FOR_G3]
        bare_value, tree_value = medians[0], medians[-1]
        denominator = tree_value - bare_value
        pooled_spread = (spreads[0] + spreads[-1]) / 2.0
        flags[feature_name] = {
            "class_medians": {CLASS_LABELS[code]: medians[position] for position, code in enumerate(CLASS_ORDER_FOR_G3)},
            "order_expected": [CLASS_LABELS[code] for code in CLASS_ORDER_FOR_G3],
            "order_observed": [CLASS_LABELS[code] for _, code in sorted(zip(medians, CLASS_ORDER_FOR_G3))],
            "denominator_tree_minus_bare": denominator,
            "pooled_end_class_iqr": pooled_spread,
            "order_broken": medians != sorted(medians),
            "denominator_too_small": abs(denominator) <= pooled_spread,
        }
    return flags


def statistics_for_site(config, stage4_run, gate_report):
    """Every kept cell, every polygon median and the class statistics for one site.

    Inputs: config; stage4_run; gate_report - this site's stage 4_6 result
    Outputs: (site statistics dict, list of cell rows for the CSV)
    """
    site_name = config["site_name"]
    plsp_year = endmember_imagery_year(config)
    polygon_path = endmember_polygon_path(config, stage4_run)
    grid = download_stage.read_production_grid(config)
    download_stage.check_crs_agrees(config, grid)
    netcdf_path = lsp_netcdf_path(config, plsp_year)
    feature_stack, feature_names, usable, diagnostics = read_feature_cube(netcdf_path, config, grid)

    polygons = gpd.read_file(polygon_path, layer=progress_stage.POLYGON_LAYER_NAME, fid_as_index=True)
    if polygons.crs is not None and polygons.crs.to_string() != config["expected_crs"]:
        polygons = polygons.to_crs(config["expected_crs"])
    counting = {int(entry["fid"]): entry for entry in gate_report["polygons"] if entry["counts"]}

    # DRAWN AGAINST PASSED, PER TILE, FOR THE FIGURE'S SECOND ROW. Drawn counts
    # every polygon stage 4_6 placed in a tile, whatever became of it; passed
    # counts those that cleared the buffer and QA rules. A class that is drawn
    # plenty but passes little is a labelling problem, and only the pair shows
    # it.
    area_ids = sorted({entry["area_id"] for entry in gate_report["polygons"] if entry["area_id"]})
    per_area_counts = {area_id: {class_name: {"drawn": 0, "passed": 0} for class_name in CLASS_LABELS.values()} for area_id in area_ids}
    for entry in gate_report["polygons"]:
        if entry["area_id"] is None or entry["class_code"] is None:
            continue
        class_name = CLASS_LABELS[entry["class_code"]]
        per_area_counts[entry["area_id"]][class_name]["drawn"] += 1
        if entry["counts"]:
            per_area_counts[entry["area_id"]][class_name]["passed"] += 1

    ids_by_class = {code: [] for code in CLASS_LABELS}
    for feature_id, entry in counting.items():
        ids_by_class[entry["class_code"]].append(feature_id)
    ranked_ids = {code: sorted(ids, key=lambda one_id: hash_rank(site_name, one_id)) for code, ids in ids_by_class.items()}

    cell_rows = []
    medians_by_class = {code: [] for code in CLASS_LABELS}
    held_out_by_class = {code: [] for code in CLASS_LABELS}
    drawn_cells = {code: 0 for code in CLASS_LABELS}
    kept_cells = {code: 0 for code in CLASS_LABELS}
    dropped_polygons = {code: 0 for code in CLASS_LABELS}
    for feature_id, entry in sorted(counting.items()):
        class_code = entry["class_code"]
        geometry = polygons.geometry.loc[feature_id]
        cells = whole_cells_in(geometry, grid)
        drawn_cells[class_code] += len(cells)
        kept = [(row, column) for row, column in cells if usable[row, column]]
        kept_cells[class_code] += len(kept)
        if not kept:
            dropped_polygons[class_code] += 1
            continue
        values = np.array([[feature_stack[feature_index, row, column] for feature_index in range(len(feature_names))] for row, column in kept], dtype="float64")
        polygon_median = np.median(values, axis=0)
        held_out = is_held_out(feature_id, ranked_ids[class_code])
        if held_out:
            held_out_by_class[class_code].append(feature_id)
        else:
            medians_by_class[class_code].append(polygon_median)
        for (row, column), cell_values in zip(kept, values):
            cell_row = {"site": config["site"], "site_name": site_name, "plsp_year": plsp_year, "class": CLASS_LABELS[class_code], "fid": feature_id, "area_id": entry["area_id"], "held_out": int(held_out), "row": row, "column": column, "x": float(grid["x_centres"][column]), "y": float(grid["y_centres"][row])}
            cell_row.update({name: float(value) for name, value in zip(feature_names, cell_values)})
            cell_rows.append(cell_row)

    class_statistics = {}
    for class_code, class_name in CLASS_LABELS.items():
        polygon_medians = np.array(medians_by_class[class_code], dtype="float64")
        if polygon_medians.size == 0:
            raise SystemExit(f"FAIL - {config['site']} {class_name} has no polygon left for statistics; stage 4_6 passed but every polygon lost all its cells to QA")
        summary = summarise_polygon_medians(polygon_medians)
        summary.update(
            {
                "n_polygons_counting": len(ids_by_class[class_code]),
                "n_polygons_in_statistics": int(polygon_medians.shape[0]),
                "n_polygons_held_out": len(held_out_by_class[class_code]),
                "n_polygons_dropped_by_qa": dropped_polygons[class_code],
                "held_out_fids": sorted(held_out_by_class[class_code]),
                "n_cells_drawn": drawn_cells[class_code],
                "n_cells_kept": kept_cells[class_code],
                "qa_pass_rate": (kept_cells[class_code] / drawn_cells[class_code]) if drawn_cells[class_code] else 0.0,
            }
        )
        class_statistics[class_name] = summary

    class_median_stack = np.array([class_statistics[CLASS_LABELS[code]]["median"] for code in sorted(CLASS_LABELS)], dtype="float64")
    site_statistics = {
        "site": config["site"],
        "site_name": site_name,
        "plsp_year": plsp_year,
        "plsp_file": str(netcdf_path),
        "plsp_tier": "production" if "PLSP_production_nc" in str(netcdf_path) else "stage",
        "polygon_file": str(polygon_path),
        "feature_names": feature_names,
        "hold_out_per_class": HOLD_OUT_PER_CLASS,
        "inward_buffer_m": progress_stage.INWARD_BUFFER_M,
        "accepted_qa_values": list(progress_stage.ACCEPTED_QA_VALUES),
        "quality_diagnostics": diagnostics,
        "polygons_per_area": per_area_counts,
        "classes": class_statistics,
        "m_median_of_class_medians": np.median(class_median_stack, axis=0).tolist(),
        "class_order_flags": check_class_order(class_statistics, feature_names),
    }
    return site_statistics, cell_rows


def write_site_files(directory, site_statistics, cell_rows):
    """The per-site statistics JSON and the per-cell CSV, both overwritten.

    NAMED BY SITE CODE, SRER_2022_..., and written to the shared stage4_7 folder
    rather than to the site's own. Stage 5 reads one site's statistics after
    another, and the diagnostics compare sites, so a single folder of short
    names beats eight folders of long ones.

    Inputs: directory - the shared stats folder; site_statistics; cell_rows
    Outputs: (stats path, cells path)
    """
    site_code, plsp_year = site_statistics["site"], site_statistics["plsp_year"]
    stats_path = directory / f"{site_code}_{plsp_year}_endmember_stats.json"
    cells_path = directory / f"{site_code}_{plsp_year}_endmember_cells.csv"
    stats_path.write_text(json.dumps(site_statistics, indent=2) + "\n")
    with open(cells_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    return stats_path, cells_path


def write_summary(summary_path, all_statistics):
    """One row per site, class and feature, so sites can be compared in one file.

    Inputs: summary_path; all_statistics - list of site statistics dicts
    Outputs: None
    """
    field_names = ["site", "site_name", "plsp_year", "class", "feature", "n_polygons", "n_cells", "median", "iqr", "p5", "p95", "qa_pass_rate"]
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for site_statistics in all_statistics:
            for class_name, summary in site_statistics["classes"].items():
                for feature_index, feature_name in enumerate(site_statistics["feature_names"]):
                    writer.writerow(
                        {
                            "site": site_statistics["site"],
                            "site_name": site_statistics["site_name"],
                            "plsp_year": site_statistics["plsp_year"],
                            "class": class_name,
                            "feature": feature_name,
                            "n_polygons": summary["n_polygons_in_statistics"],
                            "n_cells": summary["n_cells_kept"],
                            "median": round(summary["median"][feature_index], 4),
                            "iqr": round(summary["iqr"][feature_index], 4),
                            "p5": round(summary["p5"][feature_index], 4),
                            "p95": round(summary["p95"][feature_index], 4),
                            "qa_pass_rate": round(summary["qa_pass_rate"], 4),
                        }
                    )


def apply_recessive_axis_style(axes):
    """Recessive grid and axes so the marks carry the chart, as in stage 5_2."""
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=MUTED, labelsize=8, length=3)
    axes.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.25)
    axes.set_axisbelow(True)


def plot_feature_boxes(axes, site_statistics, feature_index, feature_name):
    """One feature's four class boxes, drawn from the recorded percentiles.

    THE BOXES ARE THE STATISTICS, NOT A SECOND CALCULATION. matplotlib's bxp
    takes the five numbers summarise_polygon_medians already recorded, so the
    figure cannot disagree with the JSON: box p25 to p75, line at the median,
    whiskers at p5 and p95. Each box is a spread ACROSS POLYGON MEDIANS, not
    across cells, which is the unit the statistics are built on.

    Inputs: axes; site_statistics; feature_index - column in the feature
            vectors; feature_name - for the title
    Outputs: None
    """
    boxes = []
    for class_code in sorted(CLASS_LABELS):
        summary = site_statistics["classes"][CLASS_LABELS[class_code]]
        boxes.append({"label": CLASS_LABELS[class_code][:2], "med": summary["median"][feature_index], "q1": summary["p25"][feature_index], "q3": summary["p75"][feature_index], "whislo": summary["p5"][feature_index], "whishi": summary["p95"][feature_index], "fliers": []})
    drawn = axes.bxp(boxes, showfliers=False, patch_artist=True, widths=0.6, medianprops={"color": INK, "linewidth": 1.4}, whiskerprops={"color": MUTED, "linewidth": 1.0}, capprops={"color": MUTED, "linewidth": 1.0})
    for patch, class_code in zip(drawn["boxes"], sorted(CLASS_LABELS)):
        patch.set_facecolor(CLASS_COLORS[class_code])
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.9)
        patch.set_alpha(0.75)
    apply_recessive_axis_style(axes)
    # EACH PANEL IS STRETCHED TO ITS OWN FEATURE'S RANGE, p5 to p95 across the
    # four classes plus a tenth of that range as margin. A shared or automatic
    # limit leaves timing features, whose classes differ by a few days, as four
    # flat lines.
    low = min(box["whislo"] for box in boxes)
    high = max(box["whishi"] for box in boxes)
    margin = (high - low) * 0.1 or 1.0
    axes.set_ylim(low - margin, high + margin)
    axes.set_title(feature_name, color=INK, fontsize=9)


def plot_polygon_counts(axes, site_statistics):
    """Polygons drawn against polygons passing the buffer and QA rules, per tile.

    Grouped by class, one pair of bars per tile, with the passed bar drawn over
    the drawn bar rather than beside it: the question is what share of the
    labelling effort survived, and a pale total behind a solid kept bar answers
    it without doubling the bar count.

    Inputs: axes; site_statistics
    Outputs: None
    """
    per_area = site_statistics["polygons_per_area"]
    area_ids = list(per_area)
    class_names = [CLASS_LABELS[class_code] for class_code in sorted(CLASS_LABELS)]
    positions, drawn_heights, passed_heights, colors, tick_positions = [], [], [], [], []
    position = 0.0
    for class_code in sorted(CLASS_LABELS):
        class_name = CLASS_LABELS[class_code]
        first_position = position
        for area_id in area_ids:
            positions.append(position)
            drawn_heights.append(per_area[area_id][class_name]["drawn"])
            passed_heights.append(per_area[area_id][class_name]["passed"])
            colors.append(CLASS_COLORS[class_code])
            position += 1.0
        tick_positions.append((first_position + position - 1.0) / 2.0)
        position += 1.0
    axes.bar(positions, drawn_heights, width=0.82, color=DRAWN_COLOR, edgecolor=INK, linewidth=0.5, alpha=0.55, label="drawn")
    axes.bar(positions, passed_heights, width=0.82, color=colors, edgecolor=INK, linewidth=0.7, label="passed buffer and QA")
    axes.axhline(progress_stage.MIN_COUNTING_PER_CLASS_PER_TILE, color=MUTED, linewidth=1.0, linestyle="--")
    apply_recessive_axis_style(axes)
    axes.set_xticks(positions)
    axes.set_xticklabels(area_ids * len(class_names), color=MUTED, fontsize=8)
    # THE CLASS NAME SITS UNDER ITS OWN GROUP, in axes coordinates rather than
    # as a second set of ticks: two tick label sets on one axis collide, and
    # the tile ids are the ones that must stay legible.
    for tick_position, class_name in zip(tick_positions, class_names):
        axes.text(tick_position, -0.24, class_name, transform=axes.get_xaxis_transform(), color=INK, fontsize=11, va="top", ha="center")
    axes.tick_params(axis="x", length=0)
    axes.set_ylim(0, max(drawn_heights + [progress_stage.MIN_COUNTING_PER_CLASS_PER_TILE]) * 1.35)
    axes.set_ylabel("# of polygons labeled", color=MUTED, fontsize=9)
    axes.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper right", ncols=2)


def start_themed_figure(width, height):
    """Open a figure under the arctic_dark theme, returning it and the theme.

    The theme is applied per figure rather than at import, because applying one
    edits global rcParams and this module must not change how anything else
    draws. finish_themed_figure puts those defaults back.

    Inputs: width; height - inches
    Outputs: (figure, theme)
    """
    theme = load_theme(FIGURE_THEME)
    theme.apply()
    matplotlib.rcParams["font.family"] = FIGURE_FONT
    return plt.figure(figsize=(width, height), facecolor=FIGURE_COLOR), theme


def finish_themed_figure(figure, theme, output_path, rotate_xticklabels=None):
    """Save one themed figure and restore matplotlib's defaults.

    ROTATION IS APPLIED AFTER THE THEME'S TRANSFORMS, not before. apply_transforms
    rewrites the tick labels and drops any rotation set earlier, which is how
    the tile ids on the counts figure ended up overlapping each other.

    Inputs: figure; theme; output_path; rotate_xticklabels - degrees, or None
    Outputs: Path written
    """
    theme.apply_transforms()
    if rotate_xticklabels is not None:
        for axes in figure.axes:
            plt.setp(axes.get_xticklabels(), rotation=rotate_xticklabels, ha="right", rotation_mode="anchor")
    figure.savefig(output_path, dpi=150, facecolor=FIGURE_COLOR)
    plt.close(figure)
    matplotlib.rcdefaults()
    matplotlib.use("Agg")
    return output_path


def figure_class_spread(site_statistics, output_path):
    """Figure one: each feature's four class boxes, one panel per feature.

    THE PANELS FILL THE FIGURE HEIGHT. This was the top row of a two-row
    figure, where a box covering a few days of the year was squeezed into a
    band too shallow to read. On its own the row gets the whole canvas, and
    each panel's y axis is set to its own feature's range, so the spread that
    matters is the one being shown.

    Inputs: site_statistics; output_path
    Outputs: Path written
    """
    feature_names = site_statistics["feature_names"]
    figure, theme = start_themed_figure(max(16.0, 1.5 * len(feature_names)), 7.6)
    grid_spec = figure.add_gridspec(1, len(feature_names), wspace=0.62, left=0.042, right=0.99, top=0.80, bottom=0.10)
    for feature_index, feature_name in enumerate(feature_names):
        plot_feature_boxes(figure.add_subplot(grid_spec[0, feature_index]), site_statistics, feature_index, feature_name)
    figure.suptitle(f"{site_statistics['site']} end members from PLSP {site_statistics['plsp_year']}", color=INK, fontsize=13, y=0.95)
    figure.text(0.042, 0.875, f"boxplots for median pixel per polygon, box p25 to p75 and whiskers p5 to p95, {HOLD_OUT_PER_CLASS} polygons per class held out", color=MUTED, fontsize=8.5, ha="left")
    return finish_themed_figure(figure, theme, output_path)


def figure_polygon_counts(site_statistics, output_path):
    """Figure two: polygons drawn against polygons passing, per class per tile.

    Inputs: site_statistics; output_path
    Outputs: Path written
    """
    figure, theme = start_themed_figure(max(12.0, 1.4 * len(CLASS_LABELS) * len(site_statistics["polygons_per_area"])), 6.4)
    grid_spec = figure.add_gridspec(1, 1, left=0.06, right=0.99, top=0.84, bottom=0.30)
    plot_polygon_counts(figure.add_subplot(grid_spec[0, 0]), site_statistics)
    figure.suptitle(f"{site_statistics['site']} end members from  PLSP {site_statistics['plsp_year']}", color=INK, fontsize=13, y=0.96)
    return finish_themed_figure(figure, theme, output_path, rotate_xticklabels=35)


def print_site(site_statistics):
    """One site's statistics for the console, without column padding.

    Inputs: site_statistics
    Outputs: None
    """
    print(f"\n{site_statistics['site']} {site_statistics['site_name']}, PLSP {site_statistics['plsp_year']} ({site_statistics['plsp_tier']} tier, {Path(site_statistics['plsp_file']).name})")
    feature_names = site_statistics["feature_names"]
    for class_name, summary in site_statistics["classes"].items():
        held_out = ", ".join(str(one_id) for one_id in summary["held_out_fids"]) or "none"
        print(f"{class_name}: {summary['n_polygons_in_statistics']} polygons in stats, {summary['n_polygons_held_out']} held out ({held_out}), cells {summary['n_cells_kept']} of {summary['n_cells_drawn']} kept, QA pass {summary['qa_pass_rate']:.1%}")
        greening = ", ".join(f"{name} {summary['median'][feature_names.index(name)]:.3f}" for name in GREENING_FEATURE_NAMES if name in feature_names)
        print(f"{class_name} medians: {greening}")
    for feature_name, flag in site_statistics["class_order_flags"].items():
        if flag["order_broken"] or flag["denominator_too_small"]:
            problems = ", ".join(name for name, is_set in (("order broken", flag["order_broken"]), ("denominator too small", flag["denominator_too_small"])) if is_set)
            print(f"FLAG {feature_name}: {problems}; observed order {' < '.join(flag['order_observed'])}, tree - bare {flag['denominator_tree_minus_bare']:.3f}, pooled IQR {flag['pooled_end_class_iqr']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compute end member statistics for every site that passes stage 4_6.")
    parser.add_argument("--run", required=True, help="stage 4 run label the endmembers folder sits under, e.g. 5")
    args = parser.parse_args()

    config_paths = sorted(CONFIG_DIRECTORY.glob("*_endmembers.json"))
    if not config_paths:
        raise SystemExit(f"FAIL - no end member configs in {CONFIG_DIRECTORY}; run stage 4_3 first")

    print("Stage 4_7 - end member statistics")
    print("=" * SEVENTY)
    print(f"the polygon is the sample: each polygon's median first, then the median across polygons, {HOLD_OUT_PER_CLASS} per class held out")

    all_statistics = []
    statuses = {}
    results_root = None
    for config_path in config_paths:
        config = json.loads(config_path.read_text())
        results_root = config["results_root"]
        gate_report = progress_stage.check_site(config, args.run)
        if gate_report["status"] != "PASS":
            statuses[config["site"]] = f"skipped, stage 4_6 says {gate_report['status']}"
            print(f"\n{config['site']} {config['site_name']}: skipped, stage 4_6 says {gate_report['status']}")
            continue
        site_statistics, cell_rows = statistics_for_site(config, args.run, gate_report)
        site_statistics["config"] = config_path.name
        all_statistics.append(site_statistics)
        stats_directory = endmember_stats_directory(config["results_root"], args.run)
        stats_directory.mkdir(parents=True, exist_ok=True)
        stats_path, cells_path = write_site_files(stats_directory, site_statistics, cell_rows)
        print_site(site_statistics)
        print(f"wrote {stats_path.name} and {cells_path.name}")
        statuses[config["site"]] = f"statistics from {sum(summary['n_polygons_in_statistics'] for summary in site_statistics['classes'].values())} polygons"

    endmembers_root = endmember_stats_directory(results_root, args.run)
    endmembers_root.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage4_run": args.run,
        "hold_out_per_class": HOLD_OUT_PER_CLASS,
        "summary": statuses,
        "sites": all_statistics,
    }
    (endmembers_root / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    if all_statistics:
        write_summary(endmembers_root / SUMMARY_NAME, all_statistics)
    for site_statistics in all_statistics:
        stem = f"{site_statistics['site']}_{site_statistics['plsp_year']}_endmember"
        spread_path = figure_class_spread(site_statistics, endmembers_root / f"{stem}_class_spread.png")
        counts_path = figure_polygon_counts(site_statistics, endmembers_root / f"{stem}_polygon_counts.png")
        print(f"wrote {spread_path.name} and {counts_path.name}")
    print("\n" + "=" * SEVENTY)
    print("summary: " + ", ".join(f"{site}: {status}" for site, status in statuses.items()))
    print(f"wrote {endmembers_root / REPORT_NAME}" + (f" and {SUMMARY_NAME}" if all_statistics else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
