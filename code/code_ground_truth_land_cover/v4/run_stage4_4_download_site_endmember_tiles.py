"""Stage 4_4 - choose and download the imagery on which end members are drawn.

ONE SCRIPT FOR EVERY SITE. The site's end member config, written by stage 4_3, names the site; the selected-sites
table says whether it is a NEON site or an AmeriFlux site, and that decides the
branch. Both branches write the same report shape, so stage 4_5 reads one format.

WHY THIS STAGE EXISTS. RF-B is trained once, at SRER 2022, and applied to other
sites by expressing each site's PLSP features relative to that site's own end
members. End members are hand-drawn polygons of bare, grass, shrub and tree.
This script only fetches the imagery they are drawn on. It builds no ground
truth map, and RF-A never runs outside SRER. See steps/stage4_steps.md.

NEON SITES - three 1 km AOP tiles, chosen by canopy height

    1. list every CHM tile flown for the site in its imagery year
    2. keep tiles lying FULLY inside the PLSP footprint, taken from the site's
       production PLSP grid, so every end member falls where PLSP exists
    3. download CHM for all of those tiles, which is cheap
    4. measure each tile from its CHM:

           N_total is every pixel position in the tile
           N_valid is the pixels with a CHM value, not nodata
           N_woody is the valid pixels with CHM >= H_GRASS_MAX (0.7 m)
           N_tree is the valid pixels with CHM >= H_TREE_MIN (2.0 m)

           flown_share = N_valid / N_total
           woody_share = N_woody / N_valid
           tree_share = N_tree / N_valid

       N_woody COUNTS shrub and tree pixels together; nothing is subtracted.
       By the section 3 height rule, CHM below 0.7 m is bare or grass, 0.7 m up
       to 2.0 m is shrub, and 2.0 m and above is tree.

    5. among tiles with flown_share >= 0.95, choose three by woody_share:

           low - the least woody tile, supplying BARE and GRASS polygons
           high - the most woody tile with tree_share > 0, supplying TREE
           medium - the tile closest to the median woody_share, supplying SHRUB

       No two chosen tiles may touch, including diagonally, so the three are
       spread across the site. If that cannot be met the constraint is relaxed
       for the medium tile and the report says so.

    6. write the per-tile table and the selection
    7. download 10 cm RGB for the three chosen tiles only

AMERIFLUX SITES - one NAIP window centred on the phenocam

    1. read phenocam 1 from the selected-sites table
    2. build a window of about 2 x 2 km centred on it, in the site's PLSP CRS,
       SNAPPED TO THE PLSP 3 m GRID and an odd number of cells wide so the
       phenocam's own cell sits at the centre, and check it lies inside the
       PLSP footprint
    3. find the NAIP quarter-quads that window touches, for the site's imagery
       year, on the Planetary Computer STAC, keeping the latest acquisition per
       quad and checking the quads cover the whole window
    4. read ONLY the window from each, reproject to the PLSP CRS at NAIP's
       native resolution, mosaic, and write one 4-band GeoTIFF, red, green,
       blue and near infrared, on the window's exact bounds

    An existing window is left alone unless --force is passed.

THE PLSP GRID COMES FROM THE PRODUCTION TIER. Its x/y arrays were checked equal
to the stage tier at every site that has both, and GDAL reads its CRS correctly
where it cannot read the stage tier's (results/stage4_results.md section 9). The
production file is found by globbing, because its name does not always follow
site_name: Onaqui's is US-xNQ_NEON_Onaqui-Ault. The UTM zone is read from its
crs variable and must match the config's expected_crs, because zones differ
between sites.

OUTPUTS

    NEON imagery and CHM -> {data_root}/{site_name}/, sub-folders from the
        site config's products block, the layout SRER already uses
    NAIP window -> {naip_root}/{site_name}/
    reports -> {results_root}/stage4_aggregation/run{N}/stage4_6_labeling_progress/{site_name}/
        tile_stats_{site_name}.csv - one row per candidate tile, NEON only
        tile_selection_{site_name}.json - what was chosen and why

THE SELECTION IS WRITTEN TO THE REPORT, NOT INTO THE SITE CONFIG. Configs are
hand-maintained and carry long notes, and rewriting one with json.dumps would
reformat the whole file. Stage 4_5 reads the selection from the report.

SITE CONFIG KEYS READ

    site, site_name, data_root, results_root, expected_crs, products (rgb, chm),
    parameters.H_GRASS_MAX, parameters.H_TREE_MIN, phenocam_csv, year
    optional stage4_4_endmember_tiles block:
        imagery_year - defaults to year
        min_flown_share - defaults to 0.95
        naip_window_m - defaults to 2000
    optional planet_data_root and naip_root, with defaults below

ARGUMENTS

    config
        positional, required. The site config JSON.
    --run
        required. The stage 4 run label the endmembers folder sits under,
        e.g. 5.
    --select-only
        flag. Choose and report, but download no RGB. Use it to inspect the
        choice before fetching imagery. CHM is still downloaded, since the
        choice is made from it.
    --force
        flag. Fetch imagery again even where it already exists. Without it an
        existing NAIP window is left alone, because polygons may already be
        drawn on it.

EXAMPLE COMMANDS

    conda activate LCSC

    # choose the three tiles and stop, to review the report first
    python run_stage4_4_download_site_endmember_tiles.py config/endmembers/SRER_2022_endmembers.json --run 5 --select-only

    # choose and download
    python run_stage4_4_download_site_endmember_tiles.py config/endmembers/SRER_2022_endmembers.json --run 5

The NEON API token is read from NEON_DATA_API_TOKEN in the environment.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import netCDF4
import numpy as np
import rasterio
import run_stage1_1_download_neon_tiles as neon_download
from constants import SEVENTY
from helpers import endmember_directory, endmember_imagery_year, read_phenocam_positions, read_selected_site_row, resolve_config_path, resolve_script_relative

DEFAULT_PLANET_DATA_ROOT = "~/Dropbox/planet/data/planet"
DEFAULT_NAIP_ROOT = "~/Dropbox/planet/data/NAIP"
DEFAULT_MIN_FLOWN_SHARE = 0.95
DEFAULT_NAIP_WINDOW_M = 2000
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_BAND_COUNT = 4
NAIP_BAND_NAMES = ("red", "green", "blue", "nir")
NAIP_FALLBACK_RESOLUTION_M = 0.6  # only used if an item does not report its gsd
TILE_SIZE_M = neon_download.TILE_SIZE_M
ROLE_DESCRIPTIONS = {"low": "expected bare and grass", "medium": "expected shrub", "high": "expected tree"}


def stage_settings(config):
    """The optional stage 4_4 settings, with their defaults filled in.

    Inputs: config - site config dict
    Outputs: dict with imagery_year, min_flown_share, naip_window_m
    """
    block = config.get("stage4_4_endmember_tiles", {})
    return {"imagery_year": endmember_imagery_year(config), "min_flown_share": float(block.get("min_flown_share", DEFAULT_MIN_FLOWN_SHARE)), "naip_window_m": float(block.get("naip_window_m", DEFAULT_NAIP_WINDOW_M))}


def read_production_grid(config):
    """The site's PLSP grid and CRS, from any production-tier file.

    Every production year was checked to share one grid, so the first file
    found is as good as any. The filename is globbed rather than built, because
    it does not always follow site_name.

    Inputs: config - site config dict
    Outputs: dict with the file, x and y cell-centre arrays, pixel size, the
             footprint's outer edges, and the EPSG code derived from the file
    """
    planet_root = config.get("planet_data_root") or config.get("stage1_3_planet_grid", {}).get("planet_data_root") or DEFAULT_PLANET_DATA_ROOT
    production_directory = resolve_config_path(planet_root, "PLSP_production_nc", config["site_name"])
    candidates = sorted(glob.glob(str(production_directory / "*PLSP_*.nc")))
    if not candidates:
        raise SystemExit(f"FAIL - no production PLSP file in {production_directory}")
    with netCDF4.Dataset(candidates[0]) as dataset:
        x_centres = np.asarray(dataset.variables["x"][:], dtype="float64")
        y_centres = np.asarray(dataset.variables["y"][:], dtype="float64")
        central_meridian = float(dataset.variables["crs"].getncattr("longitude_of_central_meridian"))
    pixel_size = abs(x_centres[1] - x_centres[0])
    half_pixel = pixel_size / 2
    utm_zone = int((central_meridian + 183) / 6)
    return {
        "file": candidates[0],
        "x_centres": x_centres,
        "y_centres": y_centres,
        "pixel_size": pixel_size,
        "left": float(x_centres.min() - half_pixel),
        "right": float(x_centres.max() + half_pixel),
        "bottom": float(y_centres.min() - half_pixel),
        "top": float(y_centres.max() + half_pixel),
        "epsg": f"EPSG:326{utm_zone:02d}",
    }


def check_crs_agrees(config, grid):
    """Stop if the config's CRS is not the zone the PLSP file is in.

    Inputs: config; grid - from read_production_grid
    Outputs: None, raises on mismatch
    """
    if config["expected_crs"] != grid["epsg"]:
        raise SystemExit(f"FAIL - config expected_crs is {config['expected_crs']} but the production PLSP grid is in {grid['epsg']} (central meridian from {Path(grid['file']).name})")


def tiles_inside_footprint(tile_ids, grid):
    """NEON tile ids whose whole 1 km square lies inside the PLSP footprint.

    The footprint is the rectangle covered by the production grid, so a
    rectangle test is exact and needs no boundary file.

    Inputs: tile_ids - list of "easting_northing" strings; grid
    Outputs: sorted list of tile ids
    """
    inside = []
    for tile_id in tile_ids:
        easting, northing = (int(value) for value in tile_id.split("_"))
        if easting >= grid["left"] and easting + TILE_SIZE_M <= grid["right"] and northing >= grid["bottom"] and northing + TILE_SIZE_M <= grid["top"]:
            inside.append(tile_id)
    return sorted(inside)


def download_product_for_tiles(config, site_directory, tile_ids, product_key, imagery_year, token):
    """Fetch one NEON product for the given tiles, skipping files already present.

    Reuses stage 1_1's listing, signed-URL fetch and config-driven layout, so
    the two stages can never disagree about where a file belongs.

    Inputs: config; site_directory - Path; tile_ids; product_key - "chm" or
            "rgb"; imagery_year - int; token - NEON API token
    Outputs: (placed, failed, not_listed) counts
    """
    if product_key not in config.get("products", {}):
        raise SystemExit(f"FAIL - the site config has no products.{product_key} block, so the file layout is unknown")
    missing = [tile_id for tile_id in tile_ids if not all(path.exists() for path in neon_download.expected_paths(config, site_directory, tile_id, product_key))]
    print(f"{product_key}: {len(tile_ids) - len(missing)} of {len(tile_ids)} tiles already present, {len(missing)} to fetch")
    if not missing:
        return 0, 0, 0
    listing, month = neon_download.product_files(neon_download.PRODUCTS[product_key]["dpid"], config["site"], imagery_year, token)
    by_name = {entry["name"]: entry for entry in listing}
    placed = failed = not_listed = 0
    for tile_id in missing:
        for target_path in neon_download.expected_paths(config, site_directory, tile_id, product_key):
            if target_path.exists():
                continue
            entry = by_name.get(target_path.name)
            if entry is None:
                print(f"NOT IN LISTING {target_path.name}")
                not_listed += 1
                continue
            try:
                if neon_download.fetch(entry["url"], target_path, entry.get("size")):
                    placed += 1
                else:
                    print(f"SIZE MISMATCH {target_path.name}")
                    failed += 1
            except Exception as error:
                print(f"FAILED {target_path.name}: {error}")
                failed += 1
    print(f"{product_key} ({month}): placed {placed}, failed {failed}, not in listing {not_listed}")
    return placed, failed, not_listed


def measure_tile_canopy(chm_path, woody_height_minimum, tree_height_minimum):
    """flown_share, woody_share and tree_share for one CHM tile.

    Inputs: chm_path - Path; woody_height_minimum - H_GRASS_MAX in m;
            tree_height_minimum - H_TREE_MIN in m
    Outputs: dict of the counts and the three shares
    """
    with rasterio.open(chm_path) as dataset:
        heights = dataset.read(1)
        nodata = dataset.nodata
    is_valid = np.isfinite(heights)
    if nodata is not None:
        is_valid &= heights != nodata
    pixels_total = int(heights.size)
    pixels_valid = int(is_valid.sum())
    pixels_woody = int((is_valid & (heights >= woody_height_minimum)).sum())
    pixels_tree = int((is_valid & (heights >= tree_height_minimum)).sum())
    return {
        "pixels_total": pixels_total,
        "pixels_valid": pixels_valid,
        "pixels_woody": pixels_woody,
        "pixels_tree": pixels_tree,
        "flown_share": pixels_valid / pixels_total if pixels_total else 0.0,
        "woody_share": pixels_woody / pixels_valid if pixels_valid else 0.0,
        "tree_share": pixels_tree / pixels_valid if pixels_valid else 0.0,
    }


def tiles_touch(first_tile_id, second_tile_id):
    """Whether two 1 km tiles share an edge or a corner.

    Inputs: two tile id strings
    Outputs: bool
    """
    first_easting, first_northing = (int(value) for value in first_tile_id.split("_"))
    second_easting, second_northing = (int(value) for value in second_tile_id.split("_"))
    return abs(first_easting - second_easting) <= TILE_SIZE_M and abs(first_northing - second_northing) <= TILE_SIZE_M


def select_low_medium_high(tile_measurements, min_flown_share):
    """Choose the low, medium and high tiles by woody_share.

    THE HIGH TILE IS CHOSEN FIRST, because it carries the hardest requirement:
    it must contain tree, and every class needs end members. Low is then the
    least woody tile not touching high, and medium the tile closest to the
    median woody_share touching neither.

    Inputs: tile_measurements - {tile_id: measurement dict}; min_flown_share
    Outputs: (selection dict of role -> tile_id, notes list, median woody_share)
    """
    notes = []
    eligible = {tile_id: values for tile_id, values in tile_measurements.items() if values["flown_share"] >= min_flown_share}
    if len(eligible) < 3:
        raise SystemExit(f"FAIL - only {len(eligible)} tiles are at least {min_flown_share:.0%} flown; three are needed")

    with_tree = sorted((tile_id for tile_id, values in eligible.items() if values["tree_share"] > 0), key=lambda tile_id: -eligible[tile_id]["woody_share"])
    if not with_tree:
        raise SystemExit("FAIL - no eligible tile contains any CHM at or above H_TREE_MIN, so no tree end members are possible at this site. Missing classes are handled per site; see steps/stage4_steps.md section 6.")
    high = with_tree[0]

    by_woody_ascending = sorted(eligible, key=lambda tile_id: eligible[tile_id]["woody_share"])
    low_candidates = [tile_id for tile_id in by_woody_ascending if tile_id != high and not tiles_touch(tile_id, high)]
    if not low_candidates:
        raise SystemExit("FAIL - every eligible tile touches the high tile, so a separate low tile cannot be chosen")
    low = low_candidates[0]

    median_woody = statistics.median(values["woody_share"] for values in eligible.values())
    remaining = [tile_id for tile_id in eligible if tile_id not in (low, high)]
    by_distance_to_median = sorted(remaining, key=lambda tile_id: abs(eligible[tile_id]["woody_share"] - median_woody))
    separated = [tile_id for tile_id in by_distance_to_median if not tiles_touch(tile_id, low) and not tiles_touch(tile_id, high)]
    if separated:
        medium = separated[0]
    else:
        medium = by_distance_to_median[0]
        notes.append("no remaining tile avoids both low and high, so the medium tile touches one of them")

    if not eligible[low]["woody_share"] <= eligible[medium]["woody_share"] <= eligible[high]["woody_share"]:
        notes.append("the medium tile's woody_share does not lie between low and high")
    return {"low": low, "medium": medium, "high": high}, notes, median_woody


def write_tile_table(path, tile_measurements, selection, min_flown_share):
    """One row per candidate tile, with its role if chosen.

    Inputs: path; tile_measurements; selection; min_flown_share
    Outputs: None
    """
    role_of = {tile_id: role for role, tile_id in selection.items()}
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile_id", "flown_share", "woody_share", "tree_share", "pixels_total", "pixels_valid", "pixels_woody", "pixels_tree", "eligible", "role"])
        for tile_id in sorted(tile_measurements):
            values = tile_measurements[tile_id]
            writer.writerow([tile_id, f"{values['flown_share']:.6f}", f"{values['woody_share']:.6f}", f"{values['tree_share']:.6f}", values["pixels_total"], values["pixels_valid"], values["pixels_woody"], values["pixels_tree"], values["flown_share"] >= min_flown_share, role_of.get(tile_id, "")])


def run_neon_site(config, site_row, settings, grid, report_directory, select_only, force):
    """The NEON branch: list, filter, CHM, measure, choose, then RGB.

    Inputs: config; site_row; settings; grid; report_directory - Path;
            select_only - bool
    Outputs: report dict
    """
    token = os.environ.get("NEON_DATA_API_TOKEN")
    if not token:
        raise SystemExit("FAIL - NEON_DATA_API_TOKEN is not set in the environment")
    imagery_year = settings["imagery_year"]
    site_directory = resolve_config_path(config["data_root"], config["site_name"])
    woody_height_minimum = config["parameters"]["H_GRASS_MAX"]
    tree_height_minimum = config["parameters"]["H_TREE_MIN"]

    print(f"\n1 - listing CHM tiles flown at {config['site']} in {imagery_year}")
    flown, month = neon_download.list_available_tiles(config["site"], imagery_year, token)
    if not flown:
        raise SystemExit(f"FAIL - no {config['site']} CHM tiles listed for {imagery_year}")
    print(f"{len(flown)} tiles flown ({month})")

    candidates = tiles_inside_footprint(flown, grid)
    print(f"\n2 - {len(candidates)} of them lie fully inside the PLSP footprint")
    if len(candidates) < 3:
        raise SystemExit("FAIL - fewer than three flown tiles lie fully inside the PLSP footprint")

    print("\n3 - CHM for every candidate tile")
    download_product_for_tiles(config, site_directory, candidates, "chm", imagery_year, token)

    print(f"\n4 - measuring canopy, woody at CHM >= {woody_height_minimum} m, tree at CHM >= {tree_height_minimum} m")
    measurements = {}
    for tile_id in candidates:
        chm_path = neon_download.expected_paths(config, site_directory, tile_id, "chm")[0]
        if not chm_path.exists():
            print(f"CHM still missing for {tile_id}, excluded")
            continue
        measurements[tile_id] = measure_tile_canopy(chm_path, woody_height_minimum, tree_height_minimum)

    print(f"\n5 - choosing low, medium and high among tiles at least {settings['min_flown_share']:.0%} flown")
    selection, notes, median_woody = select_low_medium_high(measurements, settings["min_flown_share"])
    eligible_count = sum(values["flown_share"] >= settings["min_flown_share"] for values in measurements.values())
    print(f"{eligible_count} eligible, median woody_share {median_woody:.4f}")
    for role in ("low", "medium", "high"):
        values = measurements[selection[role]]
        print(f"{role} {selection[role]} - woody {values['woody_share']:.4f}, tree {values['tree_share']:.4f}, flown {values['flown_share']:.4f} - {ROLE_DESCRIPTIONS[role]}")
    for note in notes:
        print(f"NOTE {note}")

    table_path = report_directory / f"tile_stats_{config['site_name']}.csv"
    write_tile_table(table_path, measurements, selection, settings["min_flown_share"])
    print(f"\n6 - wrote {table_path}")

    if select_only:
        print("\n7 - skipped, --select-only parameter enabled")
    else:
        print("\n7 - RGB for the three chosen tiles")
        download_product_for_tiles(config, site_directory, list(selection.values()), "rgb", imagery_year, token)

    return {
        "source": "NEON",
        "imagery_year": imagery_year,
        "neon_month": month,
        "thresholds": {"woody_height_minimum_m": woody_height_minimum, "tree_height_minimum_m": tree_height_minimum, "min_flown_share": settings["min_flown_share"]},
        "tiles_flown": len(flown),
        "tiles_inside_footprint": len(candidates),
        "tiles_measured": len(measurements),
        "tiles_eligible": eligible_count,
        "median_woody_share": median_woody,
        "selected": [{"role": role, "end_member_classes": ROLE_DESCRIPTIONS[role], "tile_id": selection[role], **measurements[selection[role]]} for role in ("low", "medium", "high")],
        "notes": notes,
        "tile_table": str(table_path),
        "rgb_downloaded": not select_only,
        "imagery_directory": str(site_directory),
    }


def naip_window_bounds(phenocam, grid, window_m):
    """A window of about window_m a side, centred on the phenocam, on the PLSP grid.

    SNAPPED TO THE PLSP 3 m GRID, and an odd number of cells wide, so the cell
    holding the phenocam is exactly central and every PLSP cell in the window
    is whole. At 2000 m that is 667 cells, 2001 m.

    Inputs: phenocam - {latitude, longitude}; grid; window_m - float
    Outputs: dict of left, right, bottom, top, cells, and the phenocam's
             projected easting and northing
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", grid["epsg"], always_xy=True)
    easting, northing = transformer.transform(phenocam["longitude"], phenocam["latitude"])
    pixel = grid["pixel_size"]
    cells = int(round(window_m / pixel))
    if cells % 2 == 0:
        cells += 1
    half = cells // 2
    column = int(np.floor((easting - grid["left"]) / pixel))
    row = int(np.floor((grid["top"] - northing) / pixel))
    bounds = {"left": grid["left"] + pixel * (column - half), "right": grid["left"] + pixel * (column + half + 1), "top": grid["top"] - pixel * (row - half), "bottom": grid["top"] - pixel * (row + half + 1), "cells": cells, "phenocam_easting": easting, "phenocam_northing": northing}
    if bounds["left"] < grid["left"] or bounds["right"] > grid["right"] or bounds["bottom"] < grid["bottom"] or bounds["top"] > grid["top"]:
        raise SystemExit(f"FAIL - the {cells * pixel:.0f} m window around {phenocam['name']} runs outside the PLSP footprint")
    return bounds


def window_polygon_in_lonlat(bounds, crs):
    """The window as a lon/lat polygon, for the STAC search.

    The four corners are not enough on their own: a UTM rectangle's edges bow
    slightly in lon/lat, so each edge is sampled as well and the search polygon
    contains the true footprint rather than cutting its middle.

    Inputs: bounds - from naip_window_bounds; crs - the PLSP EPSG string
    Outputs: GeoJSON-style dict
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    steps = 8
    eastings, northings = [], []
    for step in range(steps + 1):
        fraction = step / steps
        eastings.append(bounds["left"] + fraction * (bounds["right"] - bounds["left"]))
        northings.append(bounds["bottom"] + fraction * (bounds["top"] - bounds["bottom"]))
    edge = [(easting, bounds["bottom"]) for easting in eastings]
    edge += [(bounds["right"], northing) for northing in northings]
    edge += [(easting, bounds["top"]) for easting in reversed(eastings)]
    edge += [(bounds["left"], northing) for northing in reversed(northings)]
    ring = [list(transformer.transform(easting, northing)) for easting, northing in edge]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


def quarter_quad_key(item_id):
    """The quarter-quad an item belongs to, without its acquisition date.

    A NAIP id is state_m_3110917_nw_12_060_20211128, and the trailing date is
    the only part that changes between acquisitions of the same quad. Dropping
    it groups the acquisitions so the latest can be kept.

    Inputs: item_id - str
    Outputs: str
    """
    parts = item_id.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8 else item_id


def search_naip_quarter_quads(bounds, crs, imagery_year):
    """The NAIP quarter-quads a window touches, for one year, latest per quad.

    THE PLANETARY COMPUTER IS THE SOURCE and its assets are signed, because the
    hrefs expire. A 2 km window falls on 1 to 4 quads depending on where it
    lands, so the count is not assumed; what IS checked is that the returned
    footprints cover the whole window, since a missing quad would leave a blank
    corner that only shows up when someone tries to label in it.

    Inputs: bounds - from naip_window_bounds; crs - the PLSP EPSG string;
            imagery_year - int
    Outputs: list of STAC items, one per quarter-quad
    """
    import planetary_computer
    import pystac_client
    from shapely.geometry import shape

    search_polygon = window_polygon_in_lonlat(bounds, crs)
    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_STAC, modifier=planetary_computer.sign_inplace)
    found = list(catalog.search(collections=["naip"], intersects=search_polygon, datetime=f"{imagery_year}-01-01/{imagery_year}-12-31").items())
    if not found:
        raise SystemExit(f"FAIL - NAIP has no imagery over this window in {imagery_year}")
    latest_by_quad = {}
    for item in found:
        key = quarter_quad_key(item.id)
        if key not in latest_by_quad or item.datetime > latest_by_quad[key].datetime:
            latest_by_quad[key] = item
    items = [latest_by_quad[key] for key in sorted(latest_by_quad)]
    covered = shape(items[0].geometry)
    for item in items[1:]:
        covered = covered.union(shape(item.geometry))
    if not covered.covers(shape(search_polygon)):
        raise SystemExit(f"FAIL - the {len(items)} NAIP quarter-quad(s) found for {imagery_year} do not cover the whole window")
    return items


def read_naip_window_mosaic(items, bounds, crs, output_path):
    """Read only the window from each quarter-quad and write one GeoTIFF.

    ONLY THE WINDOW IS READ. Each quarter-quad is a cloud-optimised GeoTIFF of
    roughly 3.75 x 3.75 minutes, and a WarpedVRT onto the window's own grid
    makes rasterio fetch just the overlapping blocks, so a 2 km window costs a
    few tens of megabytes instead of the whole quad.

    ALL FOUR BANDS ARE KEPT, red, green, blue and near infrared. The near
    infrared separates green grass from bare soil and from shrub far better
    than the visible bands, and dropping it to save space would be paid for
    during labelling.

    NATIVE RESOLUTION IS KEPT, 0.6 m for these years, so the imagery is never
    resampled to a coarser grid than it was flown at. Reprojection from NAIP's
    NAD83 UTM to the site's PLSP CRS is unavoidable, since the polygons are
    drawn in the PLSP CRS, and is done once here with bilinear resampling.

    Quads are written in order and only where nothing has been written yet, so
    a seam takes whole pixels from one quad rather than blending two.

    WHICH PIXELS A QUAD ACTUALLY HOLDS IS TAKEN FROM ITS FOOTPRINT, not from
    the warped mask. NAIP quads declare no nodata, so a WarpedVRT reports every
    pixel as valid and returns zeros beyond the quad's edge. Trusting that mask
    let the first quad claim the whole window and write black over ground it
    does not cover: 24.4 million of 44.5 million pixels came out empty at WKG
    while a second quad that covered them was skipped. The footprint is
    rasterised onto the window's own grid instead, and all-zero pixels inside
    it are treated as missing too, so a later quad can still fill them.

    Inputs: items; bounds; crs; output_path - Path
    Outputs: (Path written, list of per-item dicts for the report)
    """
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from shapely.geometry import shape
    from shapely.ops import transform as transform_geometry

    resolution = min(float(item.properties.get("gsd", NAIP_FALLBACK_RESOLUTION_M)) for item in items)
    width = int(round((bounds["right"] - bounds["left"]) / resolution))
    height = int(round((bounds["top"] - bounds["bottom"]) / resolution))
    destination_transform = from_origin(bounds["left"], bounds["top"], resolution, resolution)
    mosaic = np.zeros((NAIP_BAND_COUNT, height, width), dtype="uint8")
    filled = np.zeros((height, width), dtype=bool)
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    sources = []
    for item in items:
        href = item.assets["image"].href
        footprint = transform_geometry(to_grid, shape(item.geometry))
        inside_footprint = geometry_mask([footprint], out_shape=(height, width), transform=destination_transform, invert=True)
        if not inside_footprint.any():
            sources.append({"id": item.id, "acquired": item.datetime.date().isoformat(), "state": item.properties.get("naip:state"), "gsd_m": item.properties.get("gsd"), "source_crs": None, "pixels_contributed": 0})
            print(f"{item.id} {item.datetime.date()} does not reach the window, skipped")
            continue
        with rasterio.open(href) as dataset:
            if dataset.count < NAIP_BAND_COUNT:
                raise SystemExit(f"FAIL - {item.id} has {dataset.count} bands, expected {NAIP_BAND_COUNT}")
            with WarpedVRT(dataset, crs=crs, transform=destination_transform, width=width, height=height, resampling=rasterio.enums.Resampling.bilinear) as warped:
                values = warped.read(indexes=list(range(1, NAIP_BAND_COUNT + 1)))
            source_crs = dataset.crs.to_string()
        valid = inside_footprint & values.any(axis=0)
        new_ground = valid & ~filled
        mosaic[:, new_ground] = values[:, new_ground]
        filled |= new_ground
        sources.append({"id": item.id, "acquired": item.datetime.date().isoformat(), "state": item.properties.get("naip:state"), "gsd_m": item.properties.get("gsd"), "source_crs": source_crs, "pixels_contributed": int(new_ground.sum())})
        print(f"{item.id} {item.datetime.date()} {source_crs} contributed {new_ground.sum() / (height * width):.1%} of the window")
    if not filled.all():
        raise SystemExit(f"FAIL - {(~filled).sum()} of {height * width} window pixels have no NAIP data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "width": width, "height": height, "count": NAIP_BAND_COUNT, "dtype": "uint8", "crs": crs, "transform": destination_transform, "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512, "nodata": 0}
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(mosaic)
        destination.descriptions = NAIP_BAND_NAMES
    return output_path, sources


def run_naip_site(config, site_row, settings, grid, report_directory, select_only, force):
    """The AmeriFlux branch: phenocam, window, then the NAIP fetch.

    Inputs: config; site_row; settings; grid; report_directory; select_only;
            force - refetch a window that already exists
    Outputs: report dict
    """
    cameras = read_phenocam_positions(site_row)
    if not cameras:
        raise SystemExit(f"FAIL - no phenocam position recorded for {config['site_name']}, so there is nothing to centre the NAIP window on")
    phenocam = cameras[0]
    print(f"\n1 centring on phenocam {phenocam['number']}, {phenocam['name']}, at {phenocam['latitude']:.5f} {phenocam['longitude']:.5f}")
    bounds = naip_window_bounds(phenocam, grid, settings["naip_window_m"])
    print(f"2 window {bounds['cells']} x {bounds['cells']} cells, {bounds['right'] - bounds['left']:.0f} m, left {bounds['left']:.1f} bottom {bounds['bottom']:.1f} right {bounds['right']:.1f} top {bounds['top']:.1f}, {grid['epsg']}")
    naip_directory = resolve_config_path(config.get("naip_root", DEFAULT_NAIP_ROOT), config["site_name"])
    output_path = naip_directory / f"naip_{settings['imagery_year']}_{config['site_name']}_window.tif"
    report = {"source": "NAIP", "imagery_year": settings["imagery_year"], "phenocam": phenocam, "window": bounds, "imagery_path": str(output_path), "imagery_directory": str(naip_directory)}
    if select_only:
        print("3 skipped, --select-only")
        return report
    # AN EXISTING WINDOW IS NEVER REPLACED WITHOUT --force. Polygons may already
    # be drawn on it, and swapping the imagery under them would silently change
    # what every one of those polygons means.
    if output_path.exists() and not force:
        print(f"3 exists, left untouched: {output_path.name}. Pass --force to fetch it again.")
        report["naip_downloaded"] = True
        report["naip_reused"] = True
        return report
    print(f"3 NAIP {settings['imagery_year']} quarter-quads touching the window")
    items = search_naip_quarter_quads(bounds, grid["epsg"], settings["imagery_year"])
    print(f"{len(items)} quarter-quad(s): {', '.join(item.id for item in items)}")
    written_path, sources = read_naip_window_mosaic(items, bounds, grid["epsg"], output_path)
    print(f"4 wrote {written_path} at {written_path.stat().st_size / 1e6:.1f} MB")
    report["naip_downloaded"] = True
    report["naip_sources"] = sources
    return report


def main():
    parser = argparse.ArgumentParser(description="Choose and download the imagery end members are drawn on, for a NEON or AmeriFlux site.")
    parser.add_argument("config", help="site config JSON")
    parser.add_argument("--run", required=True, help="stage 4 run label the endmembers folder sits under, e.g. 5")
    parser.add_argument("--select-only", action="store_true", help="choose and report, download no RGB or NAIP")
    parser.add_argument("--force", action="store_true", help="fetch imagery again even where it already exists")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    settings = stage_settings(config)
    site_row = read_selected_site_row(resolve_script_relative(config["phenocam_csv"]), config["site_name"])
    is_neon = site_row.get("is_neon", "").strip() in ("1", "1.0", "True", "true")
    grid = read_production_grid(config)
    check_crs_agrees(config, grid)

    report_directory = endmember_directory(config, args.run)
    report_directory.mkdir(parents=True, exist_ok=True)

    print(f"Stage 4_4 - end member imagery - {config['site_name']} ({site_row.get('site_id', '')}, {site_row.get('neon_domain', '')})")
    print("=" * SEVENTY)
    print(f"site type {'NEON' if is_neon else 'AmeriFlux'}, imagery year {settings['imagery_year']}")
    print(f"PLSP grid {len(grid['x_centres'])} x {len(grid['y_centres'])} at {grid['pixel_size']:.0f} m, {grid['epsg']}, from {Path(grid['file']).name}")
    print(f"footprint left {grid['left']:.1f} right {grid['right']:.1f} bottom {grid['bottom']:.1f} top {grid['top']:.1f}")

    runner = run_neon_site if is_neon else run_naip_site
    branch_report = runner(config, site_row, settings, grid, report_directory, args.select_only, args.force)

    report = {
        "site": config["site"],
        "site_name": config["site_name"],
        "site_id": site_row.get("site_id"),
        "neon_domain": site_row.get("neon_domain"),
        "is_neon": is_neon,
        "stage4_run": args.run,
        "plsp_grid": {"file": grid["file"], "epsg": grid["epsg"], "pixel_size_m": grid["pixel_size"], "left": grid["left"], "right": grid["right"], "bottom": grid["bottom"], "top": grid["top"]},
        "phenocams": read_phenocam_positions(site_row),
        **branch_report,
    }
    report_path = report_directory / f"tile_selection_{config['site_name']}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * SEVENTY)
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
