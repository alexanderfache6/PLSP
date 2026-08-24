#!/usr/bin/env python3
"""Step 0 - PlanetScope grid definition and 1 m analysis grid alignment (instructions5.md section 5 Step 3, section 11 checks 10-12).

THE PLANETSCOPE LSP NETCDF IS THE ONLY AUTHORITY FOR THE GRID. Nothing else in
the project defines it. CRS, pixel size and origin are read from that file's
coordinate arrays - not from a config value, not from a NEON raster, not from an
assumption - because instructions5.md section 5 Step 3 makes this grid the spine
of the pipeline: every 1 m product, every N x N block and every fraction estimate
is positioned relative to it, so an error here propagates silently into all of
them and cannot be corrected downstream.

The grid is year-invariant for a site, so it is derived from whichever LSP year
is on disk (`stage1_3_planet_grid.grid_source_year` in config). At SRER the 2022 product
is still pending generation and Step 5 is gated on it, but Steps 3 and 4 are not:
2021 defines the same grid.

WHAT THIS SCRIPT DECIDES, once, for every later step to read:

    planet_pixel_m     PlanetScope cell size, measured from the coordinate array
    N                  planet_pixel_m / analysis_grid_m, required to be an integer
    origin             the grid's cell-EDGE origin in native UTM
    epsg               derived from the netCDF crs variable, cross-checked
                       against config `expected_crs`

WHAT IT CHECKS, and why each one can ruin the pipeline silently:

    uniform spacing    a non-uniform coordinate array is not a grid at all, and
                       every block would be a different size on the ground
    square pixels      N is one number; non-square pixels need two
    integer N          a fractional N means 1 m cells straddle Planet cell
                       boundaries everywhere, and no aggregation is well defined
    integer edges      cell edges must land on whole metres or the 1 m analysis
                       grid cannot be nested inside them without a remainder
    tile congruence    per tile, the offset of the NEON tile origin modulo
                       planet_pixel_m. A non-zero offset does NOT break nesting -
                       the 1 m grid still nests globally - but it means the tile
                       boundary cuts THROUGH Planet cells, so N x N blocks at the
                       tile edge draw pixels from two tiles. Aggregation that
                       runs per tile in isolation produces partial blocks there
                       and must either mosaic first or carry a halo.
    footprint overlap  a NEON tile can lie partly outside the LSP footprint, in
                       which case part of its ground truth has no Planet pixel to
                       aggregate into and cannot enter Steps 3-6 at all

TILES ARE CROPPED TO THE PLANETSCOPE FOOTPRINT, which is the smaller SRER focus
area rather than the full AOP flight box (`stage1_3_planet_grid.crop_to_planet_footprint`).
Ground truth outside the footprint has no Planet pixel to aggregate into, so
carrying it forward would inflate ground-truth area against a Planet denominator
that does not exist. The cropped extents are written as their own layer so the
loss is visible rather than implied.

OUTPUTS -> `stage1_data_and_features/qa/`:

    planet_grid_{SITE}_{YEAR}.json   machine-readable: parameters and checks
    planet_grid_{SITE}_{YEAR}.gpkg   the QGIS verification layers, below
    planet_{VAR}_{SITE}_{YEAR}.tif   one real LSP layer on the measured grid,
                                     default EVIamp - see below

VISUAL VERIFICATION IS PART OF THE STEP, NOT AN OPTIONAL EXTRA. instructions5.md
requires both grids exported as GeoPackage in native UTM so alignment can be
confirmed by eye against the NEON tiles before anything is computed. A numeric
check confirms the arithmetic is self-consistent; it cannot confirm the grid sits
where the imagery sits. Layers written:

    planet_footprint            LSP extent, one polygon
    tile_footprints             one polygon per configured tile, carrying the
                                offsets, the congruence verdict and the overlap
                                fraction as attributes
    tile_footprints_cropped     the same tiles clipped to the LSP footprint -
                                what actually enters Steps 3-6
    verification_windows        the small windows the cell grids are drawn in
    planet_cells_verification   planet_pixel_m cells inside those windows
    analysis_cells_verification analysis_grid_m cells inside those windows

A REAL LSP LAYER IS EXPORTED ALONGSIDE THE GRIDS, and it is the check that
matters most. Empty cell outlines can only be compared against each other; they
cannot show that the grid corresponds to the Planet data it is supposed to
address. `verification_variable` (default `EVIamp`) is written as a GeoTIFF whose
transform is built from the origin and pixel size THIS SCRIPT MEASURED - not from
anything GDAL infers - so a measurement error puts the layer visibly off the AOP
imagery. Loading the netCDF directly in QGIS would re-read the same
georeferencing the measurement came from and could not catch a mistake in it.

Fill is masked to NaN and the scale factor applied before anything else, per
instructions5.md section 5.3 trap 1.

THE CELL GRIDS ARE DRAWN IN WINDOWS, NOT OVER THE WHOLE SITE, and that is
deliberate. A full-site 3 m grid is 11.1 million polygons and a full-site 1 m
grid is 100 million; QGIS will open neither usefully. Two windows per tile answer
the question instead: one STRADDLING the tile's south-west corner, where any
congruence offset is visible as the tile boundary slicing a Planet cell, and one
at the tile CENTRE, where nesting is checked away from any edge effect.

Usage: python run_stage1_3_define_planet_grid.py config/srer_2022.json
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from helpers import resolve_config_path
from rasterio.transform import from_origin
from shapely.geometry import box

HERE = Path(__file__).resolve().parent

PASS, FAIL, REPORT = "PASS", "FAIL", "REPORT"

COORD_TOLERANCE_M = 1e-6


def lsp_path(config):
    """Full path to the LSP netCDF the grid is derived from.

    instructions5.md section 5.1 flags the naming trap: the directory carries the
    `_NEON` suffix and the filename does not. Both are derived here from the one
    `site_name` config key so the two strings cannot drift apart.
    """
    grid = config["stage1_3_planet_grid"]
    site_dir = config["site_name"]
    site_name = site_dir[: -len("_NEON")] if site_dir.endswith("_NEON") else site_dir
    filename = f"{config['ameriflux_id']}_NEON_{site_name}_PLSP_{grid['grid_source_year']}.nc"
    return resolve_config_path(grid["planet_data_root"], grid["lsp_subdir"], site_dir, filename)


def epsg_from_crs_variable(crs_attrs):
    """UTM EPSG code implied by the netCDF CF grid-mapping attributes.

    The LSP product stores a CF `crs` variable with transverse Mercator
    parameters rather than an EPSG code, so the zone is recovered from the
    central meridian and the hemisphere from the false northing.
    """
    if crs_attrs.get("grid_mapping_name") != "transverse_mercator":
        return None
    central_meridian = float(crs_attrs["longitude_of_central_meridian"])
    zone = int(round((central_meridian + 183.0) / 6.0))
    northern = float(crs_attrs.get("false_northing", 0.0)) == 0.0
    return f"EPSG:{(32600 if northern else 32700) + zone}"


def uniform_step(values):
    """The single spacing of a coordinate array, or None if it is not uniform."""
    steps = np.diff(np.asarray(values, dtype=np.float64))
    if steps.size == 0:
        return None
    if np.ptp(steps) > COORD_TOLERANCE_M:
        return None
    return float(np.mean(steps))


def read_grid(nc_path):
    """Grid geometry measured from the LSP netCDF coordinate arrays.

    Coordinates in the product are cell CENTRES; everything downstream works in
    cell EDGES, so the half-pixel shift is applied here once and nowhere else.
    """
    with xr.open_dataset(nc_path) as ds:
        x = ds["x"].values.astype(np.float64)
        y = ds["y"].values.astype(np.float64)
        crs_attrs = dict(ds["crs"].attrs) if "crs" in ds.variables else {}
    step_x = uniform_step(x)
    step_y = uniform_step(y)
    grid = {
        "nc_path": str(nc_path),
        "nx": int(x.size),
        "ny": int(y.size),
        "step_x": step_x,
        "step_y": step_y,
        "epsg": epsg_from_crs_variable(crs_attrs),
        "crs_attributes": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in crs_attrs.items()},
    }
    if step_x is None or step_y is None:
        return grid
    grid["planet_pixel_m"] = abs(step_x)
    grid["x_min"] = float(x.min() - abs(step_x) / 2.0)
    grid["x_max"] = float(x.max() + abs(step_x) / 2.0)
    grid["y_min"] = float(y.min() - abs(step_y) / 2.0)
    grid["y_max"] = float(y.max() + abs(step_y) / 2.0)
    grid["origin_x"] = grid["x_min"]
    grid["origin_y"] = grid["y_max"]
    return grid


def is_whole(value, tolerance=COORD_TOLERANCE_M):
    """True when a float sits on a whole number within tolerance."""
    return abs(value - round(value)) <= tolerance


def tile_alignment(tile, grid, tile_size_m):
    """Congruence and footprint overlap for one NEON tile.

    `offset_x`/`offset_y` are the tile origin's position modulo the Planet pixel.
    Zero means the tile boundary coincides with a Planet cell edge; anything else
    means the boundary cuts through a cell, which is the condition that makes
    per-tile aggregation produce partial blocks at the edge.
    """
    easting, northing = (int(v) for v in tile.split("_"))
    pixel = grid["planet_pixel_m"]
    offset_x = (easting - grid["x_min"]) % pixel
    offset_y = (northing - grid["y_max"]) % pixel
    tile_geom = box(easting, northing, easting + tile_size_m, northing + tile_size_m)
    footprint = box(grid["x_min"], grid["y_min"], grid["x_max"], grid["y_max"])
    overlap = tile_geom.intersection(footprint)
    return {
        "tile": tile,
        "easting": easting,
        "northing": northing,
        "offset_x_m": round(float(offset_x), 6),
        "offset_y_m": round(float(offset_y), 6),
        "congruent": bool(offset_x <= COORD_TOLERANCE_M and offset_y <= COORD_TOLERANCE_M),
        "overlap_fraction": round(float(overlap.area / tile_geom.area), 6),
        "cropped_area_m2": round(float(overlap.area), 3),
        "geometry": tile_geom,
        "cropped_geometry": overlap,
    }


def cell_grid(x_min, y_min, x_max, y_max, size, origin_x, origin_y):
    """Cells of the given size covering a window, snapped to the grid origin.

    Snapping to the origin rather than to the window is the whole point: the
    cells drawn must be the actual Planet cells, so that a tile boundary passing
    through one is visible as such.
    """
    start_x = origin_x + np.floor((x_min - origin_x) / size) * size
    start_y = origin_y + np.floor((y_min - origin_y) / size) * size
    cells = []
    xs = np.arange(start_x, x_max, size)
    ys = np.arange(start_y, y_max, size)
    for cell_x in xs:
        for cell_y in ys:
            cells.append(box(cell_x, cell_y, cell_x + size, cell_y + size))
    return cells


def verification_windows(tiles, grid, tile_size_m, window_m):
    """Two windows per tile: one straddling the south-west corner, one central.

    The corner window is where a congruence offset shows up, because the tile
    boundary runs through it. The centre window checks nesting in the interior,
    where no edge effect can explain away a misalignment.
    """
    windows = []
    half = window_m / 2.0
    for tile in tiles:
        easting, northing = (int(v) for v in tile.split("_"))
        centre = easting + tile_size_m / 2.0, northing + tile_size_m / 2.0
        windows.append(
            {
                "tile": tile,
                "position": "corner",
                "geometry": box(easting - half, northing - half, easting + half, northing + half),
            }
        )
        windows.append(
            {
                "tile": tile,
                "position": "centre",
                "geometry": box(
                    centre[0] - half,
                    centre[1] - half,
                    centre[0] + half,
                    centre[1] + half,
                ),
            }
        )
    return windows


def build_cell_layers(windows, grid, analysis_grid_m):
    """Planet cells and 1 m analysis cells inside every verification window."""
    planet_records, analysis_records = [], []
    for window in windows:
        bounds = window["geometry"].bounds
        for geom in cell_grid(*bounds, grid["planet_pixel_m"], grid["x_min"], grid["y_max"]):
            planet_records.append(
                {
                    "tile": window["tile"],
                    "position": window["position"],
                    "geometry": geom,
                }
            )
        for geom in cell_grid(*bounds, analysis_grid_m, grid["x_min"], grid["y_max"]):
            analysis_records.append(
                {
                    "tile": window["tile"],
                    "position": window["position"],
                    "geometry": geom,
                }
            )
    return planet_records, analysis_records


def export_verification_raster(nc_path, variable, grid, out_path):
    """One LSP layer written as a GeoTIFF on the MEASURED grid, for visual verification.

    This is the strongest alignment check available. The GeoTIFF's transform is
    built from the origin and pixel size that this script measured - not from
    anything GDAL infers - so if the measurement were wrong, the exported layer
    would land off the imagery in QGIS and the error would be visible. Loading
    the netCDF directly instead would re-read the same georeferencing the
    measurement came from and could not catch a mistake in it.

    Fill is masked to NaN and the scale factor applied BEFORE anything else, per
    instructions5.md section 5.3 trap 1: the fill value 32767 is Int16 max, and
    arithmetic on the raw integers turns fill into plausible-looking values.
    """
    with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
        if variable not in ds.data_vars:
            print(f"SKIP verification raster - {variable} not in {nc_path.name}")
            return None
        band = ds[variable]
        raw = band.values
        fill = band.attrs.get("_FillValue")
        scale = float(band.attrs.get("scale", 1.0))
        offset = float(band.attrs.get("offset", 0.0))
        long_name = str(band.attrs.get("long_name", variable)).strip()
    data = raw.astype(np.float32)
    if fill is not None:
        data[raw == fill] = np.nan
    data = data * scale + offset
    transform = from_origin(grid["x_min"], grid["y_max"], grid["planet_pixel_m"], grid["planet_pixel_m"])
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": grid["epsg"],
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, long_name)
    finite = np.isfinite(data)
    return {
        "variable": variable,
        "long_name": long_name,
        "path": str(out_path),
        "scale": scale,
        "fill_pixels": int((~finite).sum()),
        "valid_min": float(np.nanmin(data)),
        "valid_median": float(np.nanmedian(data)),
        "valid_max": float(np.nanmax(data)),
    }


def run_checks(grid, analysis_grid_m, expected_crs, tile_rows):
    """The section 11 checks 10-12 block, each with its observed value."""
    checks = []

    def add(number, name, status, observed, blocker=False):
        checks.append(
            {
                "check": str(number),
                "name": name,
                "status": status,
                "blocker": bool(blocker),
                "observed": observed,
            }
        )

    uniform = grid["step_x"] is not None and grid["step_y"] is not None
    add(
        10,
        "LSP coordinate arrays are uniformly spaced",
        PASS if uniform else FAIL,
        {"step_x": grid["step_x"], "step_y": grid["step_y"]},
        blocker=True,
    )
    if not uniform:
        return checks

    square = abs(abs(grid["step_x"]) - abs(grid["step_y"])) <= COORD_TOLERANCE_M
    add(
        10.1,
        "LSP pixels are square",
        PASS if square else FAIL,
        {"planet_pixel_m": grid["planet_pixel_m"]},
        blocker=True,
    )

    ratio = grid["planet_pixel_m"] / analysis_grid_m
    integer_n = is_whole(ratio)
    add(
        11,
        "N = planet_pixel_m / analysis_grid_m is an integer",
        PASS if integer_n else FAIL,
        {"ratio": ratio, "N": int(round(ratio)) if integer_n else None},
        blocker=True,
    )

    whole_edges = is_whole(grid["x_min"]) and is_whole(grid["y_max"])
    add(
        11.1,
        "Planet cell edges land on whole metres",
        PASS if whole_edges else FAIL,
        {"origin_x": grid["x_min"], "origin_y": grid["y_max"]},
        blocker=True,
    )

    crs_match = expected_crs is None or grid["epsg"] == expected_crs
    add(
        12,
        "LSP CRS matches config expected_crs",
        PASS if crs_match else FAIL,
        {"lsp_epsg": grid["epsg"], "expected_crs": expected_crs},
        blocker=True,
    )

    congruent = [r["tile"] for r in tile_rows if r["congruent"]]
    add(
        12.1,
        "NEON tile origins are congruent with the Planet grid",
        REPORT,
        {
            "congruent_tiles": congruent,
            "congruent_count": len(congruent),
            "tile_count": len(tile_rows),
            "consequence": "Non-congruent tiles have their 1 km boundary cutting through Planet cells. Nesting of the 1 m grid is unaffected, but N x N blocks at a tile edge draw 1 m pixels from two tiles, so Step 3 must mosaic or carry a halo rather than aggregate each tile in isolation.",
        },
    )

    partial = {r["tile"]: r["overlap_fraction"] for r in tile_rows if r["overlap_fraction"] < 1.0}
    empty = [r["tile"] for r in tile_rows if r["overlap_fraction"] <= 0.0]
    add(
        12.2,
        "Configured tiles against the LSP footprint (tiles are cropped to it)",
        PASS if not partial else REPORT,
        {
            "partial_tiles": partial,
            "wholly_outside": empty,
            "consequence": "Tiles are cropped to the PlanetScope footprint, which is the smaller SRER focus area. Ground truth outside it has no Planet pixel to aggregate into and cannot enter Steps 3-6, so the cropped-away area must be excluded from every ground-truth denominator, not merely ignored.",
        },
        blocker=bool(empty),
    )
    return checks


def main():
    parser = argparse.ArgumentParser(description="Define the PlanetScope grid and the nested 1 m analysis grid, and export both for QGIS verification.")
    parser.add_argument("config", help="site config JSON, e.g. config/srer_2022.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    site, year = config["site"], config["year"]
    grid_cfg = config["stage1_3_planet_grid"]
    analysis_grid_m = float(config["stage1_5_generate_features"]["analysis_grid_m"])
    tile_size_m = float(grid_cfg["neon_tile_size_m"])
    window_m = float(grid_cfg["verification_window_m"])

    nc_path = lsp_path(config)
    print(f"LSP grid source: {nc_path}")
    if not nc_path.exists():
        raise SystemExit(f"FAIL - LSP netCDF not found: {nc_path}")

    grid = read_grid(nc_path)
    tile_rows = [tile_alignment(tile, grid, tile_size_m) for tile in config["tiles"]]
    checks = run_checks(grid, analysis_grid_m, config.get("expected_crs"), tile_rows)

    planet_pixel_m = grid.get("planet_pixel_m")
    n_factor = int(round(planet_pixel_m / analysis_grid_m)) if planet_pixel_m else None

    print("")
    print(f"CRS               {grid['epsg']} (config expects {config.get('expected_crs')})")
    print(f"planet_pixel_m    {planet_pixel_m}")
    print(f"analysis_grid_m   {analysis_grid_m}")
    print(f"N                 {n_factor}")
    print(f"grid size         {grid['nx']} x {grid['ny']} Planet cells")
    print(f"origin (edge)     {grid['x_min']}, {grid['y_max']}")
    print(f"extent            x [{grid['x_min']}, {grid['x_max']}] y [{grid['y_min']}, {grid['y_max']}]")
    print("")
    print("tile role offset_x offset_y congruent overlap")
    for row in tile_rows:
        role = config["tiles"][row["tile"]]
        print(f"{row['tile']:17s} {role:5s} {row['offset_x_m']:8.1f} {row['offset_y_m']:8.1f} {str(row['congruent']):9s} {row['overlap_fraction'] * 100:5.1f}%")
    print("")
    for check in checks:
        print(f"[{check['status']:6s}] {check['check']:5s} {check['name']}")

    out_dir = resolve_config_path(config["results_root"], "stage1_data_and_features", "qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = out_dir / f"planet_grid_{site}_{year}.gpkg"
    json_path = out_dir / f"planet_grid_{site}_{year}.json"

    crs = grid["epsg"] or config.get("expected_crs")
    footprint = gpd.GeoDataFrame(
        [
            {
                "site": site,
                "source": nc_path.name,
                "planet_pixel_m": planet_pixel_m,
                "nx": grid["nx"],
                "ny": grid["ny"],
                "geometry": box(grid["x_min"], grid["y_min"], grid["x_max"], grid["y_max"]),
            }
        ],
        crs=crs,
    )
    tiles_gdf = gpd.GeoDataFrame(
        [
            {
                **{k: v for k, v in row.items() if k != "cropped_geometry"},
                "role": config["tiles"][row["tile"]],
            }
            for row in tile_rows
        ],
        crs=crs,
    )
    cropped_rows = [
        {
            **{k: v for k, v in row.items() if k not in ("geometry", "cropped_geometry")},
            "role": config["tiles"][row["tile"]],
            "geometry": row["cropped_geometry"],
        }
        for row in tile_rows
        if not row["cropped_geometry"].is_empty
    ]
    cropped_gdf = gpd.GeoDataFrame(cropped_rows, crs=crs)
    windows = verification_windows(list(config["tiles"]), grid, tile_size_m, window_m)
    windows_gdf = gpd.GeoDataFrame(windows, crs=crs)
    planet_records, analysis_records = build_cell_layers(windows, grid, analysis_grid_m)
    planet_gdf = gpd.GeoDataFrame(planet_records, crs=crs)
    analysis_gdf = gpd.GeoDataFrame(analysis_records, crs=crs)

    footprint.to_file(gpkg_path, layer="planet_footprint", driver="GPKG")
    tiles_gdf.to_file(gpkg_path, layer="tile_footprints", driver="GPKG")
    cropped_gdf.to_file(gpkg_path, layer="tile_footprints_cropped", driver="GPKG")
    windows_gdf.to_file(gpkg_path, layer="verification_windows", driver="GPKG")
    planet_gdf.to_file(gpkg_path, layer="planet_cells_verification", driver="GPKG")
    analysis_gdf.to_file(gpkg_path, layer="analysis_cells_verification", driver="GPKG")

    variable = grid_cfg.get("verification_variable", "EVIamp")
    raster_path = out_dir / f"planet_{variable}_{site}_{grid_cfg['grid_source_year']}.tif"
    verification_raster = export_verification_raster(nc_path, variable, grid, raster_path)

    report = {
        "site": site,
        "year": year,
        "grid_source": {
            "path": str(nc_path),
            "year": grid_cfg["grid_source_year"],
            "note": grid_cfg.get("grid_source_year_note"),
        },
        "grid": {
            "epsg": grid["epsg"],
            "planet_pixel_m": planet_pixel_m,
            "analysis_grid_m": analysis_grid_m,
            "N": n_factor,
            "nx": grid["nx"],
            "ny": grid["ny"],
            "origin_x": grid.get("x_min"),
            "origin_y": grid.get("y_max"),
            "x_min": grid.get("x_min"),
            "x_max": grid.get("x_max"),
            "y_min": grid.get("y_min"),
            "y_max": grid.get("y_max"),
            "crs_attributes": grid["crs_attributes"],
        },
        "tiles": [{k: v for k, v in row.items() if k not in ("geometry", "cropped_geometry")} for row in tile_rows],
        "checks": checks,
        "outputs": {
            "geopackage": str(gpkg_path),
            "planet_cells_drawn": len(planet_records),
            "analysis_cells_drawn": len(analysis_records),
            "verification_raster": verification_raster,
        },
    }
    json_path.write_text(json.dumps(report, indent=2))

    print("")
    print(f"wrote {gpkg_path}")
    print(f"wrote {json_path}")
    if verification_raster:
        print(f"wrote {raster_path} ({verification_raster['long_name']}, {verification_raster['valid_min']:.4f} - {verification_raster['valid_max']:.4f}, {verification_raster['fill_pixels']} fill pixels)")

    blocking = [c for c in checks if c["status"] == FAIL and c["blocker"]]
    if blocking:
        print("")
        print(f"BLOCKING FAILURES: {len(blocking)} - Step 3 must not run")
        raise SystemExit(1)
    print("")
    print("Grid defined. Next: run run_stage1_4_create_qgis_grid_verification_project.py in the LCSC_QGIS environment and confirm alignment by eye before any Step 3 aggregation.")


if __name__ == "__main__":
    main()
