"""Stage 4_6 - check end member labelling progress at every site.

Reads each site's hand-drawn end member polygons and says whether the site has
enough of them for stage 4_7 to compute end member statistics. Run it as often
as you like while labelling.

STRICTLY READ-ONLY FOR POLYGONS. Nothing here writes to, fills, fixes or drops
from a polygon file. Polygons that do not count are listed with the reason and
left exactly as drawn. The only file written is the progress report, which is
overwritten on every run.

EVERY SITE IS CHECKED ON EVERY RUN: every config in config/endmembers/.

WHAT HAPPENS TO EACH POLYGON, in this order

    skipped, and not counted anywhere:
        below 1 m2 - slivers and stray clicks
        no class_code
        outside downloaded imagery - not wholly inside one downloaded NEON
            tile, or the NAIP window
        outside the PLSP footprint

    counting, only if both of these hold:
        at least one WHOLE 3 m PLSP cell lies inside the polygon after a 0.5 m
            inward buffer, the rule stage 4_7 reads cells by; the buffer allows
            for misregistration between the imagery and PlanetScope, so the
            polygon must reach at least 0.5 m past every edge of the cell
        EVERY PLSP CELL THE POLYGON TOUCHES IS QA-USABLE, QA 1 or 2, so no part
            of a polygon sits on phenology stage 4_7 would discard

    Otherwise the polygon is drawn but not counting, and the report says which
    rule failed.

THE GATE, per class, 0 bare, 1 grass, 2 shrub, 3 tree:

    at least 15 counting polygons in total
    at least 5 counting polygons in EVERY downloaded tile, or in the NAIP window

A NEON site has three tiles, so each class needs 5 in each of them. A tile whose
imagery has not been downloaded cannot hold polygons, so it fails the gate.

SITE STATUS

    PASS - the gate is met
    FAIL - the gate is not met; every shortfall is listed
    PENDING QA - the site has no PLSP QA layer yet, because its PLSP year has
        not been generated, so the QA rule cannot be applied. Counts use the
        geometry rule alone, and the gate result is shown but not final.
    NOT STARTED - no polygon file, or an empty one
    NOT READY - stage 4_4 or 4_5 has not produced what this check needs

INPUTS, per site, from its end member folder

    endmember_polygons_{site_name}_{year}.gpkg, layer endmember_polygons
    tile_selection_{site_name}.json, from stage 4_4
    plsp_qa_{site_name}_{year}.tif, from stage 4_5, where PLSP exists
    the PLSP grid, from the site's production file

OUTPUT, overwritten every run

    {results_root}/stage4_aggregation/run{N}/stage4_6_labeling_progress/stage4_6_labeling_progress.json

ARGUMENTS

    --run
        required. The stage 4 run label the labelling folder sits under,
        e.g. 5.

EXAMPLE COMMANDS

    conda activate LCSC
    python run_stage4_6_check_endmember_labeling_progress.py --run 5
"""

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import run_stage4_4_download_site_endmember_tiles as download_stage
from constants import CLASS_LABELS, SEVENTY
from helpers import endmember_directory, endmember_imagery_year, endmember_polygon_path, labeling_progress_directory, resolve_config_path

CONFIG_DIRECTORY = Path(__file__).resolve().parent / "config" / "endmembers"
POLYGON_LAYER_NAME = "endmember_polygons"
PLSP_GRID_SIZE = 3.0  # 3m pixel
MIN_POLYGON_AREA_M2 = PLSP_GRID_SIZE * PLSP_GRID_SIZE  # NOTE polygon must cover at least 1 entire PLSP grid cell
INWARD_BUFFER_M = 0.5  # buffer around plsp grid that must also be same class
MIN_COUNTING_PER_CLASS = 15
MIN_COUNTING_PER_CLASS_PER_TILE = 5
ACCEPTED_QA_VALUES = (1, 2)
REPORT_NAME = "stage4_6_labeling_progress.json"
NEON_TILE_SIZE_M = 1000


def labelled_areas(config, directory):
    """The areas polygons may be drawn in: the downloaded NEON tiles or NAIP window.

    Inputs: config; directory - the site's end member folder
    Outputs: (dict of area id to {geometry, role, downloaded}, problem string or None)
    """
    from shapely.geometry import box

    selection_path = directory / f"tile_selection_{config['site_name']}.json"
    if not selection_path.exists():
        return {}, "stage 4_4 has not run, so there are no downloaded tiles to check against"
    selection = json.loads(selection_path.read_text())
    areas = {}
    if selection.get("is_neon"):
        # data_root IS READ ONLY ON THE NEON BRANCH. An AmeriFlux end member
        # config has no data_root at all: stage 4_3 gives it naip_root instead,
        # because NAIP is found by a STAC search rather than under a NEON Data
        # Portal folder. Resolving it before the branch raised KeyError at
        # every AmeriFlux site.
        site_directory = resolve_config_path(config["data_root"], config["site_name"])
        for selected in selection["selected"]:
            tile_id = selected["tile_id"]
            easting, northing = (int(value) for value in tile_id.split("_"))
            rgb_path = download_stage.neon_download.expected_paths(config, site_directory, tile_id, "rgb")[0]
            areas[tile_id] = {"geometry": box(easting, northing, easting + NEON_TILE_SIZE_M, northing + NEON_TILE_SIZE_M), "role": selected["role"], "downloaded": rgb_path.exists()}
    else:
        window = selection["window"]
        areas["naip_window"] = {"geometry": box(window["left"], window["bottom"], window["right"], window["top"]), "role": "window", "downloaded": Path(selection["imagery_path"]).exists()}
    return areas, None


def read_qa_on_grid(qa_path, grid):
    """The PLSP QA layer, checked to sit on the production grid.

    Inputs: qa_path; grid - from read_production_grid
    Outputs: int array [rows, columns], or None when the layer does not exist
    """
    if not qa_path.exists():
        return None
    with rasterio.open(qa_path) as dataset:
        transform = dataset.transform
        if not (math.isclose(transform.c, grid["left"]) and math.isclose(transform.f, grid["top"]) and math.isclose(transform.a, grid["pixel_size"])):
            raise SystemExit(f"FAIL - {qa_path.name} is not on the production grid")
        if (dataset.width, dataset.height) != (len(grid["x_centres"]), len(grid["y_centres"])):
            raise SystemExit(f"FAIL - {qa_path.name} size differs from the production grid")
        return dataset.read(1)


def cells_near(geometry, grid):
    """Every PLSP cell whose square overlaps a geometry's bounding box.

    Inputs: geometry - shapely geometry in the grid CRS; grid
    Outputs: list of (row, column, cell square)
    """
    from shapely.geometry import box

    pixel = grid["pixel_size"]
    columns, rows = len(grid["x_centres"]), len(grid["y_centres"])
    minimum_x, minimum_y, maximum_x, maximum_y = geometry.bounds
    first_column = max(0, math.floor((minimum_x - grid["left"]) / pixel))
    last_column = min(columns - 1, math.ceil((maximum_x - grid["left"]) / pixel) - 1)
    first_row = max(0, math.floor((grid["top"] - maximum_y) / pixel))
    last_row = min(rows - 1, math.ceil((grid["top"] - minimum_y) / pixel) - 1)
    cells = []
    for row in range(first_row, last_row + 1):
        for column in range(first_column, last_column + 1):
            cell_left = grid["left"] + column * pixel
            cell_top = grid["top"] - row * pixel
            cells.append((row, column, box(cell_left, cell_top - pixel, cell_left + pixel, cell_top)))
    return cells


def assess_polygon(geometry, grid, qa):
    """Whole cells after the inward buffer, touched cells, and the QA test.

    A cell counts as touched only when it shares area with the polygon, so a
    polygon edge lying exactly on a cell boundary does not pull in its
    neighbour.

    Inputs: geometry; grid; qa - array or None
    Outputs: dict of whole_cells, touched_cells, touched_cells_not_qa_usable
             (None when there is no QA layer)
    """
    buffered = geometry.buffer(-INWARD_BUFFER_M)
    whole_cells = 0 if buffered.is_empty else sum(1 for _, _, cell in cells_near(buffered, grid) if buffered.covers(cell))
    touched = [(row, column) for row, column, cell in cells_near(geometry, grid) if geometry.intersection(cell).area > 0]
    not_usable = None
    if qa is not None:
        not_usable = sum(1 for row, column in touched if int(qa[row, column]) not in ACCEPTED_QA_VALUES)
    return {"whole_cells": whole_cells, "touched_cells": len(touched), "touched_cells_not_qa_usable": not_usable}


def check_site(config, stage4_run):
    """Every polygon's fate and the gate result for one site.

    Inputs: config - a site end member config; stage4_run
    Outputs: report dict for the site
    """
    from shapely.geometry import box

    site_name = config["site_name"]
    year = endmember_imagery_year(config)
    directory = endmember_directory(config, stage4_run)
    polygon_path = endmember_polygon_path(config, stage4_run)
    qa_path = directory / f"plsp_qa_{site_name}_{year}.tif"
    report = {"site": config["site"], "site_name": site_name, "imagery_year": year, "polygon_file": str(polygon_path), "status": None, "qa_available": qa_path.exists(), "areas": {}, "classes": {}, "shortfalls": [], "skipped": {}, "polygons": []}

    areas, problem = labelled_areas(config, directory)
    if problem:
        report["status"] = "NOT READY"
        report["shortfalls"].append(problem)
        return report
    report["areas"] = {area_id: {"role": area["role"], "downloaded": area["downloaded"]} for area_id, area in areas.items()}

    if not polygon_path.exists():
        report["status"] = "NOT READY"
        report["shortfalls"].append("no polygon file; run stage 4_5 for this site")
        return report
    polygons = gpd.read_file(polygon_path, layer=POLYGON_LAYER_NAME, fid_as_index=True)
    if polygons.empty:
        report["status"] = "NOT STARTED"
        return report
    if polygons.crs is not None and polygons.crs.to_string() != config["expected_crs"]:
        polygons = polygons.to_crs(config["expected_crs"])

    grid = download_stage.read_production_grid(config)
    download_stage.check_crs_agrees(config, grid)
    footprint = box(grid["left"], grid["bottom"], grid["right"], grid["top"])
    qa = read_qa_on_grid(qa_path, grid)

    counting = {code: {area_id: 0 for area_id in areas} for code in CLASS_LABELS}
    drawn = {code: 0 for code in CLASS_LABELS}
    not_counting = {code: {} for code in CLASS_LABELS}
    for feature_id, feature in polygons.iterrows():
        geometry = feature.geometry
        entry = {"fid": int(feature_id), "class_code": None, "area_m2": 0.0 if geometry is None else round(float(geometry.area), 2), "area_id": None, "skip_reason": None, "counts": False, "not_counting_reason": None}
        class_value = feature.get("class_code")
        if geometry is None or geometry.is_empty or geometry.area < MIN_POLYGON_AREA_M2:
            entry["skip_reason"] = "below 1 m2"
        elif class_value is None or (isinstance(class_value, float) and np.isnan(class_value)):
            entry["skip_reason"] = "no class_code"
        else:
            entry["class_code"] = int(class_value)
            inside = [area_id for area_id, area in areas.items() if area["geometry"].covers(geometry)]
            if not inside or not areas[inside[0]]["downloaded"]:
                entry["skip_reason"] = "outside downloaded imagery"
            elif not footprint.covers(geometry):
                entry["skip_reason"] = "outside the PLSP footprint"
            else:
                entry["area_id"] = inside[0]
        if entry["skip_reason"]:
            report["skipped"][entry["skip_reason"]] = report["skipped"].get(entry["skip_reason"], 0) + 1
            report["polygons"].append(entry)
            continue

        code = entry["class_code"]
        drawn[code] += 1
        entry.update(assess_polygon(geometry, grid, qa))
        if entry["whole_cells"] == 0:
            reason = f"no whole {PLSP_GRID_SIZE} m cell after the {INWARD_BUFFER_M} m inward buffer"
            entry["not_counting_reason"] = reason
        elif entry["touched_cells_not_qa_usable"]:
            reason = "touches a cell that is not QA-usable"
            entry["not_counting_reason"] = f"touches {entry['touched_cells_not_qa_usable']} cell(s) that are not QA-usable"
        else:
            reason = None
            entry["counts"] = True
            counting[code][entry["area_id"]] += 1
        if reason:
            not_counting[code][reason] = not_counting[code].get(reason, 0) + 1
        report["polygons"].append(entry)

    for code, class_name in CLASS_LABELS.items():
        total = sum(counting[code].values())
        report["classes"][class_name] = {"drawn": drawn[code], "counting": total, "counting_per_area": counting[code], "not_counting": not_counting[code]}
        if total < MIN_COUNTING_PER_CLASS:
            report["shortfalls"].append(f"{class_name}: {total} of {MIN_COUNTING_PER_CLASS} in total")
        for area_id, area_count in counting[code].items():
            if area_count < MIN_COUNTING_PER_CLASS_PER_TILE:
                note = "" if areas[area_id]["downloaded"] else ", imagery not downloaded"
                report["shortfalls"].append(f"{class_name}: {area_count} of {MIN_COUNTING_PER_CLASS_PER_TILE} in {area_id} ({areas[area_id]['role']}{note})")

    gate_met = not report["shortfalls"]
    if qa is None:
        report["status"] = "PENDING QA"
        report["gate_met_on_geometry_alone"] = gate_met
    else:
        report["status"] = "PASS" if gate_met else "FAIL"
    return report


def print_site(site_report):
    """One site's result for the console, without column padding.

    Inputs: site_report
    Outputs: None
    """
    print()
    print("-" * SEVENTY)
    print(f"{site_report['site']} {site_report['site_name']}, imagery {site_report['imagery_year']}, QA {'available' if site_report['qa_available'] else 'not available'}: {site_report['status']}")
    print()
    if site_report["status"] in ("NOT READY", "NOT STARTED"):
        for shortfall in site_report["shortfalls"]:
            print(f"- {shortfall}")
        return
    area_ids = list(site_report["areas"])
    print("areas: " + ", ".join(f"{area_id} {site_report['areas'][area_id]['role']}{'' if site_report['areas'][area_id]['downloaded'] else ' NOT DOWNLOADED'}" for area_id in area_ids))
    skipped_total = sum(site_report["skipped"].values())
    print(f"skipped {skipped_total}" + (": " + ", ".join(f"{reason} {count}" for reason, count in site_report["skipped"].items()) if skipped_total else ""))
    for class_name, values in site_report["classes"].items():
        per_area = ", ".join(str(values["counting_per_area"][area_id]) for area_id in area_ids)
        not_counting = "; ".join(f"{reason} {count}" for reason, count in values["not_counting"].items())
        print(f"{class_name}: counting {values['counting']} of {MIN_COUNTING_PER_CLASS}, per area {per_area} (min {MIN_COUNTING_PER_CLASS_PER_TILE} each), drawn {values['drawn']}" + (f", not counting: {not_counting}" if not_counting else ""))
    if site_report["status"] == "PENDING QA":
        print(f"gate on geometry alone: {'met' if site_report['gate_met_on_geometry_alone'] else 'not met'}, final once PLSP QA exists")
    for shortfall in site_report["shortfalls"]:
        print(f"- {shortfall}")


def main():
    parser = argparse.ArgumentParser(description="Check end member labelling progress at every site.")
    parser.add_argument("--run", required=True, help="stage 4 run label the endmembers folder sits under, e.g. 5")
    args = parser.parse_args()

    config_paths = sorted(CONFIG_DIRECTORY.glob("*_endmembers.json"))
    if not config_paths:
        raise SystemExit(f"FAIL - no end member configs in {CONFIG_DIRECTORY}; run stage 4_3 first")

    print("Stage 4_6 - end member labelling progress")
    print("=" * SEVENTY)
    print(f"gate: {MIN_COUNTING_PER_CLASS} counting polygons per class, and {MIN_COUNTING_PER_CLASS_PER_TILE} per class in every downloaded tile")
    print(f"a polygon counts if it is at least {MIN_POLYGON_AREA_M2:g} m2, holds a whole {PLSP_GRID_SIZE:g} m cell after a {INWARD_BUFFER_M:g} m inward buffer, and touches only QA-usable cells")

    site_reports = []
    results_root = None
    for config_path in config_paths:
        config = json.loads(config_path.read_text())
        results_root = config["results_root"]
        site_report = check_site(config, args.run)
        site_report["config"] = config_path.name
        site_reports.append(site_report)
        print_site(site_report)

    report_path = labeling_progress_directory(results_root, args.run) / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "stage4_run": args.run,
        "rules": {"min_polygon_area_m2": MIN_POLYGON_AREA_M2, "inward_buffer_m": INWARD_BUFFER_M, "min_counting_per_class": MIN_COUNTING_PER_CLASS, "min_counting_per_class_per_tile": MIN_COUNTING_PER_CLASS_PER_TILE, "accepted_qa_values": list(ACCEPTED_QA_VALUES)},
        "summary": {site_report["site"]: site_report["status"] for site_report in site_reports},
        "sites": site_reports,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + "=" * SEVENTY)
    print("summary: " + ", ".join(f"{site} {status}" for site, status in report["summary"].items()))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
