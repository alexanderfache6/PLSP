import csv
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


def resolve_config_path(root, *parts):
    """Expand a config path and join sub-paths onto it."""
    return Path(str(root)).expanduser().joinpath(*parts)


def expand_path(root, *parts):
    return os.path.join(os.path.expanduser(str(root)), *parts)


def resolve_script_relative(relative_path):
    """Resolve a config path that is written relative to the v4 script directory.

    Config entries such as plsp_layers_csv and phenocam_csv are recorded
    relative to where the scripts live, not to the config file or the shell's
    working directory. Resolving them here, once, keeps every stage agreeing.

    Inputs: relative_path - str from the config
    Outputs: absolute Path
    """
    return (Path(__file__).resolve().parent / relative_path).resolve()


def endmember_imagery_year(config):
    """The year a site's end members are drawn in: imagery AND PLSP.

    NEON AOP or NAIP, whichever the site has, closest to 2022. Set per site in
    the optional stage4_4_endmember_tiles block, defaulting to the config year.

    THE PLSP YEAR FOLLOWS THE IMAGERY YEAR. A site whose closest flight is 2023,
    such as SJER, is labelled against PLSP 2023 and its end member statistics
    come from PLSP 2023. Only RF-B's training is fixed to SRER 2022.

    Inputs: config - site config dict
    Outputs: int
    """
    return int(config.get("stage4_4_endmember_tiles", {}).get("imagery_year", config["year"]))


def planet_blocks_directory(results_root, stage4_run):
    """Where stage 4_1 writes, and stages 4_2, 5_1 and 5_3 read, the Planet-block products.

    stage4_aggregation/run{N}/stage4_2_planet_blocks/, beside the run's
    stage4_6_labeling_progress/ and stage4_7_stats/ folders, so a run's three
    kinds of product never share one. One place for the path, so writer and readers cannot disagree about
    it: 4_1 writes here, and 4_2, 5_1, 5_3 and the 4_7 diagnostic read from it.
    It was plain planet_blocks/ until 2026-09-18, when runs 4 and 5 were moved
    and their QGIS projects repointed.

    Inputs: results_root - str or Path, may start with ~; stage4_run - e.g. "5"
    Outputs: Path, not created here
    """
    return resolve_config_path(results_root, "stage4_aggregation", f"run{stage4_run}", "stage4_2_planet_blocks")


def labeling_progress_directory(results_root, stage4_run):
    """Where stage 4_4, 4_5 and 4_6 write: the labelling side of stage 4.

    stage4_aggregation/run{N}/stage4_6_labeling_progress/, holding one folder
    per site plus 4_6's own report. Named for the stage that gates it, beside
    run{N}/stage4_2_planet_blocks/ and run{N}/stage4_7_stats/, so a run's three
    kinds of product are told apart by their folder rather than by memory.

    Inputs: results_root - str or Path, may start with ~; stage4_run
    Outputs: Path, not created here
    """
    return resolve_config_path(results_root, "stage4_aggregation", f"run{stage4_run}", "stage4_6_labeling_progress")


def endmember_directory(config, stage4_run):
    """Where every stage 4_4, 4_5 and 4_6 file for ONE SITE lives.

    One place for the path, so the three stages cannot disagree about it. Stage
    4_7's outputs are NOT here: they go to the shared stage4_7_stats folder,
    because they are read site against site.

    Inputs: config; stage4_run - run label, e.g. "5"
    Outputs: Path, not created here
    """
    return labeling_progress_directory(config["results_root"], stage4_run) / config["site_name"]


def endmember_polygon_path(config, stage4_run):
    """The hand-drawn end member polygon GeoPackage for one site.

    Named by the IMAGERY year the polygons are drawn on, which differs by site.
    THIS FILE HOLDS HAND LABELS AND IS NEVER DELETED OR REWRITTEN by any stage.

    Inputs: config; stage4_run
    Outputs: Path
    """
    return endmember_directory(config, stage4_run) / f"endmember_polygons_{config['site_name']}_{endmember_imagery_year(config)}.gpkg"


def endmember_stats_directory(results_root, stage4_run):
    """Where every stage 4_7 output lands, for all sites together.

    stage4_aggregation/run{N}/stage4_7_stats/, beside the labelling folder that
    holds polygons and imagery reports. The statistics, figures and tables are
    read site against site, so they live together rather than scattered one per
    site folder. Files are named by SITE CODE, SRER_2022_..., not by the long
    site_name.

    Inputs: results_root - str or Path, may start with ~; stage4_run - e.g. "5"
    Outputs: Path, not created here
    """
    return resolve_config_path(results_root, "stage4_aggregation", f"run{stage4_run}", "stage4_7_stats")


def read_selected_site_row(csv_path, site_name):
    """The row for one site from the selected-sites table.

    Matched on site_name, which is also the directory name under the PLSP
    product tiers and the NEON and NAIP data roots, so one key ties them all.

    Inputs: csv_path - Path to 01_selected_sites_short_2.csv; site_name
    Outputs: dict of that row's columns, as strings
    """
    with open(csv_path, newline="") as handle:
        for site_row in csv.DictReader(handle):
            if site_row["site_name"] == site_name:
                return site_row
    raise SystemExit(f"FAIL - site_name {site_name} not found in {csv_path}")


def read_phenocam_positions(site_row):
    """Phenocam 1 and 2 for a site, whichever are recorded.

    Stdlib only, because stage 4_5 calls this from the QGIS environment. A
    camera is kept only when its name and both coordinates are present and
    finite: the table writes a missing camera as an empty cell or as NaN, and
    SJER has neither camera.

    Inputs: site_row - dict from read_selected_site_row
    Outputs: list of {number, name, latitude, longitude}, possibly empty
    """
    positions = []
    for camera_number in (1, 2):
        camera_name = (site_row.get(f"phenocam{camera_number}") or "").strip()
        latitude_text = (site_row.get(f"(p{camera_number})latitude") or "").strip()
        longitude_text = (site_row.get(f"(p{camera_number})longitude") or "").strip()
        if not camera_name or camera_name.lower() == "nan" or not latitude_text or not longitude_text:
            continue
        latitude, longitude = float(latitude_text), float(longitude_text)
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            continue
        positions.append({"number": camera_number, "name": camera_name, "latitude": latitude, "longitude": longitude})
    return positions


def read_rgb_at_scale(path, scale_m):
    """Read NEON RGB decimated to scale_m, with unflown ground masked to NaN.

    SHARED BY STAGE 1_5 AND STAGE 1_6 ON PURPOSE. Both need RGB at
    TEXTURE_SCALE, and both were reading it with their own copy of this
    function - so when the unflown-ground defect was found in one, the other
    still had it. One implementation, one place to fix.

    NEON RGB DECLARES NO NODATA, AND THAT IS A DEFECT, NOT A CURIOSITY. Ground
    the camera did not fly is written as all-zero, and with no declared nodata
    GDAL returns those zeros as ordinary data. Nothing downstream can tell them
    from genuinely black ground: zero is finite, so the usable-pixel test in
    stage 3 accepts it and the classifier assigns a land-cover class to ground
    that was never photographed.

    Measured at SRER, 517000_3531000: 1.137% of the tile is all-zero RGB while
    CHM nodata is only 0.083% - the lidar flew it, the camera did not. Those
    pixels became k-means cluster 4 in its entirety and bled into clusters 5, 6,
    9 and 12, which made the stage 2 coverage gate demand hand labels over
    unphotographed ground. In stage 1_6 the same zeros entered the POOLED luma
    histogram, dragging the site-wide 20th percentile down and re-cutting shadow
    on every tile. No other tile at SRER is affected.

    A PRODUCT'S FLIGHT COVERAGE MUST BE TESTED ON THAT PRODUCT. Checking CHM
    says nothing about the camera - the instruments have different footprints,
    and the tile-eligibility check in instructions5.md section 2A missed this
    case by testing CHM alone.

    The zero test runs at NATIVE resolution, before decimation: averaging first
    blends black and lit pixels into a dark-but-nonzero fringe that no threshold
    recovers cleanly. The majority rule then matches the shadow aggregation in
    stage 1_6, so the boundary lands in one place rather than eroding or
    dilating the hole.

    Inputs: path - RGB GeoTIFF; scale_m - target pixel size in metres
    Outputs: (arr float32 [3, h, w] with NaN where unflown, transform, crs, bounds)
    """
    with rasterio.open(path) as ds:
        width = int(round((ds.bounds.right - ds.bounds.left) / scale_m))
        height = int(round((ds.bounds.top - ds.bounds.bottom) / scale_m))
        unflown_native = (ds.read() == 0).all(axis=0).astype("float32")
        arr = ds.read(
            out_shape=(ds.count, height, width),
            resampling=Resampling.average,
            out_dtype="float32",
        )
        transform = from_bounds(*ds.bounds, width, height)
        unflown = np.empty((height, width), dtype="float32")
        reproject(
            source=unflown_native,
            destination=unflown,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=transform,
            dst_crs=ds.crs,
            resampling=Resampling.average,
        )
        arr[:, unflown > 0.5] = np.nan
        return arr, transform, ds.crs, ds.bounds
