"""Stage 4_1 - aggregate the 1 m ground truth to PlanetScope 3x3 blocks (instructions5.md section 5 Step 3).

Turns the stage 3 per-pixel classification into per-Planet-pixel fractional
cover: for every block of N x N one-meter cells, what share of it is bare,
grass, shrub and tree. These fractions are the training target for RF-B and the
quantity the Step 6 area estimates are built on.

NO PLANETSCOPE QA MASKING HAPPENS HERE, deliberately. A block is reduced below
N^2 valid pixels only by the GROUND TRUTH side - classifier nodata, which already
carries the shadow mask and the tile edge margin folded into it. The PLSP QA
layers (NumCycles, QA) mask the PHENOLOGY side and are applied later, when RF-B
training pairs these fractions against Planet metrics. Mixing the two here would
discard ground truth that is perfectly good, on account of a defect in a
different dataset.

THE GRID IS READ, NEVER RE-DERIVED. Stage 1_3 measured it from the LSP netCDF
and it was verified by eye in QGIS; this script reads that record and asserts the
netCDF still agrees. Two independent derivations of the same grid is exactly how
a half-pixel disagreement gets into a pipeline unnoticed.

NOTHING IS RESAMPLED, SHIFTED OR REPROJECTED. The 1 m cells already nest exactly
inside the Planet cells - whole-meter edges on both sides, 3 divides evenly - so
assigning a pixel to a block is integer arithmetic on coordinates and nothing
more. See instructions5.md section 5 Step 3 for why shifting the tiles to remove
their offsets would be actively wrong.

AGGREGATION RUNS OVER A SITE-WIDE MOSAIC, not tile by tile. No SRER tile is
congruent with the Planet grid in both axes, so tile boundaries cut through
Planet cells and a block on a shared edge draws its pixels from two tiles.
Blocking each tile alone would emit partial blocks around every perimeter, drop
them, and silently delete the tile edges. Only two adjacency groups exist at SRER
(511000_3527000/3528000/3529000 and 515000_3530000/3531000); the other tiles are
isolated and their perimeter blocks are unrecoverable by any method, which the
report quantifies rather than hides.

THREE FRACTION ESTIMATES, all produced, because they answer different questions
and their disagreement is itself diagnostic (instructions5.md section 5 Step 3):

    hard count            count of plurality labels / n_valid. The reference
                          AREA fraction - what "percent cover" conventionally
                          means, and what Step 6 validates against
    soft mean             mean of the per-pixel probability vectors. The better
                          target for RF-B, because sub-pixel mixture is exactly
                          what the phenology model is asked to predict and
                          hardening throws that signal away
    confidence weighted   hard labels weighted by per-pixel prediction_quality.
                          NOT an area fraction - see the caveat below

CONFIDENCE WEIGHTING COMES FROM THE 1 M CLASSIFIER'S OWN PROBABILITIES.
w_i = prediction_quality = 1 - H(p_i)/log(4), written by stage 3. A block that is
95% grass assembled from coin-flip pixels is a weaker measurement than one that
is 92% grass called confidently, and the weighted estimate is what separates
them. `block_prediction_quality` (mean w_i) is carried forward as its own band
for end-member ranking at stage 5, sample weighting at stage 6, and stratified
accuracy at stage 7.

    CAVEAT, stated so it is not misread: the confidence-weighted estimate is no
    longer strictly an area fraction and will not agree with a manual area
    estimate. Use it as a diagnostic and a sensitivity check, never as the
    headline product. Where it diverges sharply from the hard count, the block is
    ambiguity-dominated and worth inspecting.

RETENTION IS A COUNT, NOT A PERCENTAGE: a block is kept at 8 or 9 valid pixels of
9, at most one masked pixel (`stage4_1_aggregation.min_valid_pixels_per_block`).
At N = 3 only three cuts exist - 7, 8 or 9 - and the old "75% valid" wording
quantised to 7, admitting blocks with 22% of their area unobserved while
appearing to enforce 75%. The report gives the full 0-9 histogram and the
retained totals at BOTH 8 and 7, so the cost of the stricter cut is a measured
number rather than an assumption.

THE SOURCE CLASSIFICATION'S AREA BIAS IS CARRIED INTO THE REPORT. Computed from
the stage 3 confusion matrix, it is the one error that does NOT cancel when
pixels are aggregated: random per-pixel error averages out over 9 samples, a
systematic bias does not. At run 3 / RF-A_C shrub is under-predicted by 14.7%,
so every block's shrub fraction is low by roughly that proportion. Outputs are
provisional until that is addressed.

Outputs -> `stage4_aggregation/`, one set per framework, site-wide on the full
Planet grid so the arrays align 1:1 with the LSP netCDF:

    fraction_hard_count_{fw}_{SITE}_{YEAR}.tif          float32, 4 bands
    fraction_soft_mean_{fw}_{SITE}_{YEAR}.tif           float32, 4 bands
    fraction_confidence_weighted_{fw}_{SITE}_{YEAR}.tif float32, 4 bands
    block_prediction_quality_{fw}_{SITE}_{YEAR}.tif     float32
    valid_pixel_count_{fw}_{SITE}_{YEAR}.tif            uint8, 0..9, RAW count
                                                        before the retention rule
    stage4_1_report_{SITE}_{YEAR}.json

`valid_pixel_count` deliberately records the count BEFORE retention, so the
histogram stays inspectable after the fact; the fraction rasters are NaN wherever
the block failed the rule.

Usage: python run_stage4_1_aggregate_to_planet_blocks.py config/srer_2022.json --run 3 --frameworks C
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from constants import CLASS_CODES, CLASS_NAMES, NODATA
from helpers import resolve_config_path
from rasterio.transform import from_origin

COORDINATE_TOLERANCE_M = 1e-6


def load_grid(qa_dir, site, year):
    """The Planet grid as measured and verified by stage 1_3.

    Read, never re-derived. Deriving it a second time here would create two
    independent sources of truth for the same geometry, which is how a
    half-pixel disagreement enters a pipeline without anyone noticing.
    """
    path = qa_dir / f"planet_grid_{site}_{year}.json"
    if not path.exists():
        raise SystemExit(f"FAIL - {path} not found. Run run_stage1_3_define_planet_grid.py first.")
    report = json.loads(path.read_text())
    grid = report["grid"]
    if grid["N"] is None or grid["planet_pixel_m"] is None:
        raise SystemExit("FAIL - the stage 1_3 report holds no usable grid; its blocking checks did not pass.")
    return grid, report


def assert_netcdf_agrees(config, grid):
    """Re-open the LSP netCDF only to confirm it still matches the recorded grid.

    Cheap insurance against the product being regenerated on a different grid
    while the stage 1_3 report on disk goes stale.
    """
    settings = config["stage1_3_planet_grid"]
    site_dir = config["site_name"]
    site_name = site_dir[: -len("_NEON")] if site_dir.endswith("_NEON") else site_dir
    filename = f"{config['ameriflux_id']}_NEON_{site_name}_PLSP_{settings['grid_source_year']}.nc"
    path = resolve_config_path(settings["planet_data_root"], settings["lsp_subdir"], site_dir, filename)
    if not path.exists():
        print(f"WARNING - LSP netCDF not found at {path}, grid agreement not re-checked")
        return
    with xr.open_dataset(path) as ds:
        x = ds["x"].values.astype(np.float64)
        y = ds["y"].values.astype(np.float64)
    pixel = abs(float(np.mean(np.diff(x))))
    x_min = float(x.min() - pixel / 2.0)
    y_max = float(y.max() + pixel / 2.0)
    checks = [
        ("planet_pixel_m", pixel, grid["planet_pixel_m"]),
        ("origin_x", x_min, grid["origin_x"]),
        ("origin_y", y_max, grid["origin_y"]),
        ("nx", float(x.size), float(grid["nx"])),
        ("ny", float(y.size), float(grid["ny"])),
    ]
    for name, observed, recorded in checks:
        if abs(observed - recorded) > COORDINATE_TOLERANCE_M:
            raise SystemExit(f"FAIL - LSP netCDF disagrees with the stage 1_3 record on {name}: {observed} vs {recorded}. Re-run run_stage1_3_define_planet_grid.py.")
    print("LSP netCDF agrees with the recorded grid")


def block_index(transform, shape, grid):
    """Flat Planet-block index for every 1 m pixel of a tile, plus an in-grid mask.

    Integer arithmetic on the pixel's own coordinates: each 1 m cell lies wholly
    inside exactly one Planet cell, so the cell's upper-left corner identifies
    its block unambiguously. Pixels falling outside the Planet grid are masked
    here, which applies the footprint crop as a side effect.

    Inputs:  transform - the tile's affine; shape - (rows, cols); grid - the dict
             from load_grid
    Outputs: (flat_index int64 array, inside bool array), both tile-shaped
    """
    rows, cols = shape
    pixel = grid["planet_pixel_m"]
    left = transform.c + np.arange(cols, dtype=np.float64)
    top = transform.f - np.arange(rows, dtype=np.float64)
    block_col = np.floor((left - grid["origin_x"]) / pixel).astype(np.int64)
    block_row = np.floor((grid["origin_y"] - top) / pixel).astype(np.int64)
    col_ok = (block_col >= 0) & (block_col < grid["nx"])
    row_ok = (block_row >= 0) & (block_row < grid["ny"])
    inside = row_ok[:, None] & col_ok[None, :]
    flat = block_row[:, None] * grid["nx"] + block_col[None, :]
    return flat, inside


def coverage_masks(paths, shape):
    """Split a tile's invalid pixels into flight-coverage, shadow, and the rest.

    WHY THIS MATTERS AND IS NOT COSMETIC. A pixel outside the AOP flight zone and
    a pixel lost to tree shadow are both nodata in the classification, but they
    mean opposite things. Unflown ground was never observed and its tile is
    simply smaller than 1 km2; shadowed ground was observed and then discarded,
    which is a real loss concentrated against woody canopy. Reporting them as one
    number makes a 96%-unflown tile look like a 96%-shadowed one.

    CHM is the binding product for flight coverage at SRER - measured on
    511000_3532000, CHM is nodata over 95.91% of the tile against 74.96% for RGB
    and 66.79% for SAVI - so the CHM nodata mask is the flight-coverage test.

    Inputs:  paths - dict that may hold chm and shadow file paths; shape - the
             tile's (rows, cols) at 1 m
    Outputs: dict of boolean masks, any of which may be None if the file is absent
    """
    masks = {"unflown": None, "shadow": None}
    if paths.get("chm") and paths["chm"].exists():
        with rasterio.open(paths["chm"]) as ds:
            chm = ds.read(1)
            nodata = ds.nodata
        if nodata is not None and chm.shape == shape:
            masks["unflown"] = chm == nodata  # NOTE where there is no chm data, that means the pixels were missed by flights
    if paths.get("shadow") and paths["shadow"].exists():
        with rasterio.open(paths["shadow"]) as ds:
            shadow = ds.read(1)
        if shadow.shape == shape:
            masks["shadow"] = shadow == 1  # NOTE there is shadow data but pixels marked as nodata are bad quality
    return masks


def accumulate_site_tiles(paths, grid, totals):
    """Add one tile's valid pixels into the site-wide block accumulators.

    Inputs:  paths - dict with classification, probability, quality file paths;
             grid; totals - the accumulator dict, modified in place
    Outputs: dict of per-tile counts, and the unique block ids the tile touched
    """
    with rasterio.open(paths["classification"]) as ds:
        classification = ds.read(1)
        transform, shape = ds.transform, ds.shape
    with rasterio.open(paths["probability"]) as ds:
        probability = ds.read().astype(np.float32)
    with rasterio.open(paths["quality"]) as ds:
        quality = ds.read(1).astype(np.float32)

    flat, inside = block_index(transform, shape, grid)
    labelled = classification != NODATA
    finite = np.isfinite(quality)
    valid = labelled & finite & inside

    flat_valid = flat[valid]
    weights = quality[valid]
    n_cells = grid["nx"] * grid["ny"]

    totals["valid_count"] += np.bincount(flat_valid, minlength=n_cells).astype(np.int32)
    totals["weight_sum"] += np.bincount(flat_valid, weights=weights, minlength=n_cells).astype(np.float32)
    for index, code in enumerate(CLASS_CODES):
        is_class = valid & (classification == code)
        flat_class = flat[is_class]
        totals["hard"][index] += np.bincount(flat_class, minlength=n_cells).astype(np.int32)
        totals["weighted"][index] += np.bincount(flat_class, weights=quality[is_class], minlength=n_cells).astype(np.float32)
        totals["soft"][index] += np.bincount(flat_valid, weights=probability[index][valid], minlength=n_cells).astype(np.float32)

    masks = coverage_masks(paths, shape)
    unflown = masks["unflown"]
    shadow = masks["shadow"]
    flown_total = int((~unflown).sum()) if unflown is not None else int(classification.size)
    lost_shadow = int((shadow & ~unflown).sum()) if shadow is not None and unflown is not None else (int(shadow.sum()) if shadow is not None else None)
    accounted = np.zeros(shape, dtype=bool)
    if unflown is not None:
        accounted |= unflown
    if shadow is not None:
        accounted |= shadow
    stats = {
        "pixels_total": int(classification.size),
        "pixels_unflown": int(unflown.sum()) if unflown is not None else None,
        "pixels_flown": flown_total,
        "flight_coverage": float(flown_total / classification.size),
        "pixels_lost_to_shadow": lost_shadow,
        "pixels_lost_other": int(((~labelled) & ~accounted).sum()),
        "pixels_masked_total": int((~labelled).sum()),
        "pixels_outside_footprint": int((labelled & ~inside).sum()),
        "pixels_used": int(valid.sum()),
        "used_share_of_flown": float(valid.sum() / flown_total) if flown_total else None,
    }
    return stats, np.unique(flat_valid)


def area_bias_from_confusion(confusion):
    """Per-class area bias of the source classification, from the stage 3 confusion.

    Rows are true, columns predicted, so the column total is the area the
    classifier assigns to a class and the row total is the area that truly
    belongs to it. This is the one error that survives aggregation: random
    per-pixel error averages out over the 9 pixels of a block, a systematic
    over- or under-prediction does not.
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    true_total = matrix.sum(axis=1)
    predicted_total = matrix.sum(axis=0)
    bias = {}
    for index, name in enumerate(CLASS_NAMES):
        reference = true_total[index]
        bias[name] = float((predicted_total[index] - reference) / reference) if reference else None
    return bias


def write_raster(path, data, grid, dtype, nodata, descriptions=None):
    """Write one product on the full Planet grid, so it aligns 1:1 with the LSP arrays."""
    data = np.atleast_3d(data.T).T if data.ndim == 2 else data
    count = data.shape[0]
    transform = from_origin(
        grid["origin_x"],
        grid["origin_y"],
        grid["planet_pixel_m"],
        grid["planet_pixel_m"],
    )
    profile = {
        "driver": "GTiff",
        "height": grid["ny"],
        "width": grid["nx"],
        "count": count,
        "dtype": dtype,
        "crs": grid["epsg"],
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))
        for band in range(count):
            dst.set_band_description(
                band + 1,
                (descriptions or CLASS_NAMES)[band] if count > 1 else (descriptions or ["value"])[0],
            )


def main():
    parser = argparse.ArgumentParser(description="Aggregate the 1 m ground-truth classification to PlanetScope N x N blocks.")
    parser.add_argument("config", help="site config JSON, e.g. config/srer_2022.json")
    parser.add_argument("--run", required=True, help="stage 3 run label to aggregate, e.g. 3")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["C"],
        help="RF-A framework letters (default: C)",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    site, year = config["site"], config["year"]
    settings = config["stage4_1_aggregation"]
    min_valid = int(settings["min_valid_pixels_per_block"])

    results = Path(str(config["results_root"])).expanduser()
    qa_dir = results / "stage1_data_and_features" / "qa"
    shadow_dir = results / "stage1_data_and_features" / "shadow"
    run_dir = results / "stage3_classification" / f"run{args.run}"
    out_dir = results / "stage4_aggregation"
    data_dir = Path(str(config["data_root"])).expanduser() / config["site_name"]
    chm_product = config["products"]["chm"]
    out_dir.mkdir(parents=True, exist_ok=True)

    grid, _ = load_grid(qa_dir, site, year)
    grid_px_size_factor = int(grid["N"])
    per_block = grid_px_size_factor * grid_px_size_factor
    assert_netcdf_agrees(config, grid)

    print("")
    print(f"planet pixel {grid['planet_pixel_m']} m, N = {grid_px_size_factor}, block = {per_block} one-meter pixels")
    print(f"grid {grid['nx']} x {grid['ny']} blocks, origin {grid['origin_x']}, {grid['origin_y']}")
    print(f"retention: keep blocks with >= {min_valid} of {per_block} valid pixels")
    print(f"source: run {args.run}, frameworks {' '.join(args.frameworks)}")
    print("")

    stage3_report = json.loads((run_dir / f"stage3_1_report_{site}_{year}.json").read_text())
    n_cells = grid["nx"] * grid["ny"]
    report = {
        "site": site,
        "year": year,
        "source_run": args.run,
        "grid": {k: grid[k] for k in ("epsg", "planet_pixel_m", "N", "nx", "ny", "origin_x", "origin_y")},
        "min_valid_pixels_per_block": min_valid,
        "planet_qa_applied": False,
        "planet_qa_note": "No PlanetScope QA masking at this stage. Blocks lose pixels only to ground-truth nodata, which already carries the shadow mask and tile edge margin. PLSP QA (NumCycles, QA) masks the phenology side and is applied at RF-B training.",
        "frameworks": {},
    }

    for framework in args.frameworks:
        framework_dir = run_dir / framework
        if not framework_dir.is_dir():
            print(f"SKIP framework {framework} - {framework_dir} not found")
            continue
        print(f"--- framework {framework}")

        totals = {
            "valid_count": np.zeros(n_cells, dtype=np.int32),
            "weight_sum": np.zeros(n_cells, dtype=np.float32),
            "hard": [np.zeros(n_cells, dtype=np.int32) for _ in CLASS_CODES],
            "soft": [np.zeros(n_cells, dtype=np.float32) for _ in CLASS_CODES],
            "weighted": [np.zeros(n_cells, dtype=np.float32) for _ in CLASS_CODES],
        }
        tile_stats, tile_blocks = {}, {}

        for tile in config["tiles"]:
            paths = {
                "classification": framework_dir / f"classification_{framework}_{site}_{tile}_{year}.tif",
                "probability": framework_dir / f"class_probability_{framework}_{site}_{tile}_{year}.tif",
                "quality": framework_dir / f"prediction_quality_{framework}_{site}_{tile}_{year}.tif",
            }
            if not all(p.exists() for p in paths.values()):
                print(f"[{tile}] SKIP - missing stage 3 outputs")
                continue
            paths["chm"] = data_dir / chm_product["folder"] / chm_product["pattern"].format(tile=tile)
            paths["shadow"] = shadow_dir / f"shadow_mask_ref_{site}_{tile}_{year}.tif"
            stats, blocks = accumulate_site_tiles(paths, grid, totals)
            tile_stats[tile] = stats
            tile_blocks[tile] = blocks
            shadow_pct = stats["pixels_lost_to_shadow"] / stats["pixels_flown"] if stats["pixels_lost_to_shadow"] is not None and stats["pixels_flown"] else 0.0
            print(f"[{tile}] flown {stats['flight_coverage']:7.2%} | used {stats['pixels_used']:>7} ({stats['used_share_of_flown']:.2%} of flown) | shadow {shadow_pct:5.2%} of flown | other {stats['pixels_lost_other']:>6} | outside footprint {stats['pixels_outside_footprint']:>6}")

        valid_count = totals["valid_count"]
        touched = valid_count > 0
        keep = valid_count >= min_valid
        denominator = np.where(touched, valid_count, 1).astype(np.float32)
        weight_total = np.where(totals["weight_sum"] > 0, totals["weight_sum"], np.nan)

        hard = np.stack([totals["hard"][i] / denominator for i in range(4)]).astype(np.float32)
        soft = np.stack([totals["soft"][i] / denominator for i in range(4)]).astype(np.float32)
        weighted = np.stack([totals["weighted"][i] / weight_total for i in range(4)]).astype(np.float32)
        block_quality = (totals["weight_sum"] / denominator).astype(np.float32)

        drop = ~keep
        for stack in (hard, soft, weighted):
            stack[:, drop] = np.nan
        block_quality[drop] = np.nan

        shape = (4, grid["ny"], grid["nx"])
        single = (grid["ny"], grid["nx"])
        stem = f"{framework}_{site}_{year}"
        write_raster(
            out_dir / f"fraction_hard_count_{stem}.tif",
            hard.reshape(shape),
            grid,
            "float32",
            np.nan,
        )
        write_raster(
            out_dir / f"fraction_soft_mean_{stem}.tif",
            soft.reshape(shape),
            grid,
            "float32",
            np.nan,
        )
        write_raster(
            out_dir / f"fraction_confidence_weighted_{stem}.tif",
            weighted.reshape(shape),
            grid,
            "float32",
            np.nan,
        )
        write_raster(
            out_dir / f"block_prediction_quality_{stem}.tif",
            block_quality.reshape(1, *single),
            grid,
            "float32",
            np.nan,
            ["block_prediction_quality"],
        )
        write_raster(
            out_dir / f"valid_pixel_count_{stem}.tif",
            valid_count.reshape(1, *single),
            grid,
            "uint8",
            0,
            ["valid_pixel_count"],
        )

        histogram = {str(k): int(((valid_count == k) & touched).sum()) for k in range(per_block + 1)}
        touched_total = int(touched.sum())
        retained = {str(cut): int((valid_count >= cut).sum()) for cut in range(1, per_block + 1)}
        bias = area_bias_from_confusion(stage3_report["frameworks"][framework]["confusion"])

        per_tile = {}
        for tile, blocks in tile_blocks.items():
            counts = valid_count[blocks]
            per_tile[tile] = {
                "role": config["tiles"][tile],
                "blocks_touched": int(blocks.size),
                "blocks_retained": int((counts >= min_valid).sum()),
                "blocks_full": int((counts == per_block).sum()),
                "retention": float((counts >= min_valid).mean()) if blocks.size else None,
            }

        mean_fraction = {name: float(np.nanmean(hard[i][keep])) for i, name in enumerate(CLASS_NAMES)} if keep.any() else {}
        report["frameworks"][framework] = {
            "tiles": tile_stats,
            "per_tile_blocks": per_tile,
            "blocks_touched": touched_total,
            "blocks_retained": int(keep.sum()),
            "retention": float(keep.sum() / touched_total) if touched_total else None,
            "valid_pixel_histogram": histogram,
            "valid_pixel_histogram_pct": {k: (v / touched_total if touched_total else None) for k, v in histogram.items()},
            "retained_at_cut": retained,
            "mean_hard_fraction": mean_fraction,
            "source_area_bias": bias,
            "source_area_bias_note": "Per-class area bias of the run 3 classification, from its confusion matrix (rows true, columns predicted). Systematic, so it does NOT cancel across the 9 pixels of a block - every block's fraction inherits it. These outputs are provisional until it is addressed.",
        }

        print("")
        print(f"blocks touched {touched_total}, retained at >= {min_valid} valid: {int(keep.sum())} ({keep.sum() / touched_total:.2%})")
        print("valid pixels per block:")
        for k in range(per_block + 1):
            n = histogram[str(k)]
            print(f"{k:>2} of {per_block}   {n:>9}   {n / touched_total:6.2%}" if touched_total else f"{k:>2} of {per_block}   {n:>9}")
        print("")
        print(f"retained at cut 9: {retained['9']}, at 8: {retained['8']}, at 7: {retained['7']}")
        print("mean hard fraction over retained blocks:")
        for name in CLASS_NAMES:
            print(f"{name:<6} {mean_fraction.get(name, float('nan')):.4f}")
        print("source area bias carried in:")
        for name in CLASS_NAMES:
            print(f"{name:<6} {bias[name]:+.2%}")
        print("")

    report_path = out_dir / f"stage4_1_report_{site}_{year}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
