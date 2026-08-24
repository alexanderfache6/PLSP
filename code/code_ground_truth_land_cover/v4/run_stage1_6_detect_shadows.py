"""Step 1c - shadow detection (instructions5.md Step 1c, from instructions2.md Phase 3).

Two passes, because the luma threshold is pooled across tiles:

  pass 1  accumulate the Rec. 709 luma histogram at 0.6 m over every tile,
          then take the pooled 20th percentile
  pass 2  per tile, flag shadow where luma < threshold AND B > R, aggregate to
          the 1 m grid by > 70% areal majority, then resolve it

A percentile rather than an absolute cut is required by R3: NEON RGB and NAIP
differ in radiometry and bit depth, so a fixed DN threshold would not mean the
same thing at the transfer site, while "darkest 20% of this site" does.

Resolution rule: shadow within SHADOW_TREE_RADIUS of CHM >= H_TREE_MIN becomes
tree; all remaining shadow is masked to nodata and excluded from training, from
Step 3 aggregation denominators, and from accuracy assessment.

This is the REFERENCE (CHM-bearing) resolution, so it is the D/E path. Frameworks
A-C have no CHM and must run the same proximity test against their own predicted
tree mask, which does not exist until Step 1d.

Output per tile: stage1_data_and_features/shadow/shadow_mask_ref_{SITE}_{tile}_{YEAR}.tif
    0 = not shadow
    1 = shadow resolved to tree
    2 = shadow masked to nodata
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from helpers import resolve_config_path
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
LUMA_MAX = 255.0


def tile_paths(config, site_dir, tile):
    p = config["products"]
    return {
        "rgb": site_dir / p["rgb"]["folder"] / p["rgb"]["pattern"].format(tile=tile),
        "chm": site_dir / p["chm"]["folder"] / p["chm"]["pattern"].format(tile=tile),
    }


# NOTE used to scale rgb 0.1 features to 0.6m features to match NAIP
def scale_rgb_raster_to_common_layer(path, scale_m):
    with rasterio.open(path) as ds:
        width = int(round((ds.bounds.right - ds.bounds.left) / scale_m))
        height = int(round((ds.bounds.top - ds.bounds.bottom) / scale_m))
        arr = ds.read(
            out_shape=(ds.count, height, width),
            resampling=Resampling.average,
            out_dtype="float32",
        )
        return arr, from_bounds(*ds.bounds, width, height), ds.crs


def generate_rec709_luma(rgb):
    """Rec. 709 luma - the same definition Step 1a writes into the feature stack."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def calculate_blue_fraction(rgb):
    """Chromatic blue b = B/(R+G+B) - a relative blue shift, brightness removed."""
    return rgb[2] / (rgb[0] + rgb[1] + rgb[2] + 1e-6)


def _percentile_from_histogram(counts, edges, percentile):
    cum = np.cumsum(counts)
    target = cum[-1] * percentile / 100.0
    i = min(int(np.searchsorted(cum, target)), len(counts) - 1)
    below = cum[i - 1] if i else 0
    frac = (target - below) / max(counts[i], 1)
    return float(edges[i] + frac * (edges[i + 1] - edges[i]))


def pooled_thresholds(config, site_dir, tiles, bins):
    """Pooled luma and blue-fraction percentiles across tiles, via histograms."""
    s1c = config["stage1_6_detect_shadows"]
    luma_edges = np.linspace(0.0, LUMA_MAX, bins + 1)
    blue_edges = np.linspace(0.0, 1.0, bins + 1)
    luma_hist = np.zeros(bins, dtype=np.int64)
    blue_hist = np.zeros(bins, dtype=np.int64)
    for tile in tiles:
        rgb, _, _ = scale_rgb_raster_to_common_layer(
            tile_paths(config, site_dir, tile)["rgb"],
            config["stage1_5_generate_features"]["texture_scale_m"],
        )
        luma_hist += np.histogram(generate_rec709_luma(rgb).ravel(), bins=luma_edges)[0]
        blue_hist += np.histogram(calculate_blue_fraction(rgb).ravel(), bins=blue_edges)[0]
    return (
        _percentile_from_histogram(luma_hist, luma_edges, s1c["luma_percentile"]),
        _percentile_from_histogram(blue_hist, blue_edges, s1c["blue_percentile"]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    s1c = config["stage1_6_detect_shadows"]
    site, year = config["site"], config["year"]
    site_dir = resolve_config_path(config["data_root"], config["site_name"])
    out_dir = resolve_config_path(config["results_root"], "stage1_data_and_features", "shadow")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = list(config["tiles"])
    pool_tiles = [t for t, role in config["tiles"].items() if role == "train"] if s1c["pool_over"] == "train" else all_tiles

    print(f"pass 1: pooling luma over {len(pool_tiles)} tile(s) [{s1c['pool_over']}]")
    threshold, blue_threshold = pooled_thresholds(config, site_dir, pool_tiles, s1c["hist_bins"])
    mode = s1c["blue_shift_mode"]
    print(f"pooled p{s1c['luma_percentile']} luma threshold = {threshold:.3f}")
    if mode == "percentile":
        print(f"pooled p{s1c['blue_percentile']} blue-fraction threshold = {blue_threshold:.4f}")
    else:
        print("blue-shift rule: absolute B > R (spec wording)")
    print()

    scale = config["stage1_5_generate_features"]["texture_scale_m"]
    radius = config["parameters"]["SHADOW_TREE_RADIUS"]
    h_tree = config["parameters"]["H_TREE_MIN"]
    summary = []

    for tile in all_tiles:
        paths = tile_paths(config, site_dir, tile)
        rgb, tr06, crs = scale_rgb_raster_to_common_layer(paths["rgb"], scale)
        luma = generate_rec709_luma(rgb)

        # dark AND blue-shifted: shadow is lit by sky rather than sun, so it is
        # relatively blue; the second term rejects dark-but-neutral soil
        # R3: the luma cut is a percentile so it transfers, but an absolute
        # B > R is an absolute threshold and is inoperative on reddish desert
        # soil (measured at SRER: true for 0.22% of pixels). The percentile
        # form keeps the same physics - shadow is sky-lit and relatively blue -
        # without assuming blue ever exceeds red.
        if mode == "percentile":
            blue = calculate_blue_fraction(rgb) > blue_threshold
        else:
            blue = rgb[2] > rgb[0]
        shadow06 = (luma < threshold) & blue

        with rasterio.open(paths["chm"]) as ds:
            grid = dict(transform=ds.transform, crs=ds.crs, width=ds.width, height=ds.height)
            chm = ds.read(1, masked=True)
            profile = ds.profile

        # areal fraction of 0.6 m shadow inside each 1 m cell, then > 70% majority
        frac = np.empty((grid["height"], grid["width"]), dtype="float32")
        reproject(
            source=np.ascontiguousarray(shadow06, dtype="float32"),
            destination=frac,
            src_transform=tr06,
            src_crs=crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            resampling=Resampling.average,
        )
        shadow1m = frac > s1c["majority_fraction"]

        # resolve: shadow near a tall object is that object's own shadow
        chm_valid = chm.filled(0.0)
        chm_valid = np.where(chm_valid > config["thresholds"]["chm_max_valid_m"], 0.0, chm_valid)
        tall = chm_valid >= h_tree
        dist = ndi.distance_transform_edt(~tall) if tall.any() else np.full(chm.shape, np.inf)
        to_tree = shadow1m & (dist <= radius)
        to_nodata = shadow1m & ~to_tree

        mask = np.zeros(shadow1m.shape, dtype="uint8")
        mask[to_tree] = 1
        mask[to_nodata] = 2

        out_profile = dict(profile, dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(out_dir / f"shadow_mask_ref_{site}_{tile}_{year}.tif", "w", **out_profile) as ds:
            ds.write(mask, 1)

        n = shadow1m.size
        row = {
            "tile": tile,
            "role": config["tiles"][tile],
            "shadow_pct_06m": round(100.0 * float(shadow06.mean()), 3),
            "shadow_pct_1m": round(100.0 * float(shadow1m.mean()), 3),
            "resolved_to_tree_pct": round(100.0 * float(to_tree.sum()) / n, 3),
            "masked_to_nodata_pct": round(100.0 * float(to_nodata.sum()) / n, 3),
            "tall_px_pct": round(100.0 * float(tall.mean()), 3),
        }
        summary.append(row)
        print(f"[{tile}] shadow 0.6m={row['shadow_pct_06m']:>6.2f}%  1m={row['shadow_pct_1m']:>6.2f}%  ->tree={row['resolved_to_tree_pct']:>5.2f}%  ->nodata={row['masked_to_nodata_pct']:>5.2f}%")

    spec = {
        "site": site,
        "year": year,
        "luma_percentile": s1c["luma_percentile"],
        "pooled_threshold": round(threshold, 4),
        "pool_over": s1c["pool_over"],
        "pool_tiles": pool_tiles,
        "computed_at_m": scale,
        "majority_fraction": s1c["majority_fraction"],
        "blue_shift_mode": mode,
        "blue_shift_rule": (f"chromatic b = B/(R+G+B) > p{s1c['blue_percentile']} (pooled) = {blue_threshold:.4f}" if mode == "percentile" else "B > R (absolute)"),
        "blue_threshold": round(blue_threshold, 5),
        "shadow_tree_radius_m": radius,
        "h_tree_min_m": h_tree,
        "mask_codes": {
            "0": "not shadow",
            "1": "resolved to tree",
            "2": "masked to nodata",
        },
        "path": "reference (CHM-based). Frameworks A-C repeat this test against their own predicted tree mask at Step 1d.",
        "tiles": summary,
    }
    (out_dir / f"shadow_{site}_{year}_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nwrote {out_dir / f'shadow_{site}_{year}_spec.json'}")


if __name__ == "__main__":
    main()
