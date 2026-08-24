"""Step 0 - data audit and grid definition (instructions5.md section 5 Step 0, section 11).

Runs the section 11 checklist against a site/year and writes both a machine-readable
`data_audit_{SITE}_{YEAR}.json` and a human-readable summary into `stage1_data_and_features/qa/`.

Checks are grouped as in the spec:
  11.1 inventory                 1-5
  11.2 georeferencing/alignment  6-9    (10-12 need the PlanetScope LSP product)
  11.3 radiometry/value sanity   13-19
  11.4 class/sampling sanity     19a-23 (need Step 1 outputs or user input)
  11.5 cross-site               24-29   (transfer site only)
  11.6 PlanetScope LSP          30-35   (need the LSP product)

Checks that cannot run yet are recorded as DEFERRED with the reason, so the audit
states what is outstanding rather than silently omitting it.

All site-specific values come from the config file (R8) - nothing is hard-coded here.

Usage:  python run_stage1_2_data_audit.py config/srer_2022.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent

PASS, FAIL, REPORT, DEFERRED = "PASS", "FAIL", "REPORT", "DEFERRED"


class Audit:
    """Collects check results in spec order."""

    def __init__(self):
        self.checks = []

    def add(self, num, name, status, observed, blocker=False):
        self.checks.append(
            {
                "check": str(num),
                "name": name,
                "status": status,
                "blocker": bool(blocker),
                "observed": observed,
            }
        )
        return status

    def defer(self, num, name, reason):
        return self.add(num, name, DEFERRED, {"reason": reason})

    @property
    def blocking_failures(self):
        return [c for c in self.checks if c["status"] == FAIL and c["blocker"]]


def resolve(root, *parts):
    return Path(str(root)).expanduser().joinpath(*parts)


def tile_paths(config, site_dir, tile):
    """Every expected file for one tile, keyed by product/index."""
    p = config["products"]
    out = {
        "rgb": site_dir / p["rgb"]["folder"] / p["rgb"]["pattern"].format(tile=tile),
        "chm": site_dir / p["chm"]["folder"] / p["chm"]["pattern"].format(tile=tile),
    }
    for index in p["vi"]["indices"]:
        out[f"vi_{index}"] = site_dir / p["vi"]["folder"] / p["vi"]["pattern"].format(tile=tile, index=index)
    return out


def edge_magnitude(arr):
    """Gradient magnitude, so two different modalities can be compared on structure.

    RGB brightness and CHM height are not correlated in value (vegetation is dark
    and tall), but their edges coincide, which is what coregistration means here.
    """
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float64))
    gy, gx = np.gradient(arr)
    mag = np.hypot(gx, gy)
    return (mag - mag.mean()) / (mag.std() or 1.0)


def subpixel_shift(a, b, upsample=20):
    """Sub-pixel shift of b relative to a, on edge magnitude.

    Uses skimage's matrix-multiply DFT upsampling rather than a hand-rolled
    parabola: the separable 1-D parabolic fit this replaces is biased whenever
    the correlation peak is broad, which is the normal case in low-contrast
    rangeland. Returns (dy, dx) in pixels.
    """
    from skimage.registration import phase_cross_correlation

    ea, eb = edge_magnitude(a), edge_magnitude(b)
    shift, _, _ = phase_cross_correlation(ea, eb, upsample_factor=upsample)
    return float(shift[0]), float(shift[1])


def shift_field(a, b, window_px, upsample=20, min_std=1e-6):
    """Per-subwindow shifts, so one number per tile becomes a distribution.

    A single global estimate cannot say whether an offset is significant, nor
    distinguish a rigid translation from rotation/scale. Estimating a shift in
    each window gives a median, a spread, and the spatial structure of the
    offset field.
    """
    n = min(a.shape[0], b.shape[0], a.shape[1], b.shape[1])
    rows = []
    for y in range(0, n - window_px + 1, window_px):
        for x in range(0, n - window_px + 1, window_px):
            wa = a[y : y + window_px, x : x + window_px]
            wb = b[y : y + window_px, x : x + window_px]
            # a flat window carries no edges to register
            if wa.std() < min_std or wb.std() < min_std:
                continue
            dy, dx = subpixel_shift(wa, wb, upsample)
            rows.append(
                {
                    "y": y + window_px // 2,
                    "x": x + window_px // 2,
                    "dy": round(dy, 4),
                    "dx": round(dx, 4),
                    "offset": round(float(np.hypot(dx, dy)), 4),
                }
            )
    return rows


def summarize_field(rows, limit_m):
    """Median offset, robust spread, and whether it clears the threshold."""
    if not rows:
        return {"n_windows": 0}
    dys = np.array([r["dy"] for r in rows])
    dxs = np.array([r["dx"] for r in rows])
    offs = np.array([r["offset"] for r in rows])
    med_dy, med_dx = float(np.median(dys)), float(np.median(dxs))
    # MAD scaled to a normal-equivalent sigma, then the SE of the median
    mad = float(np.median(np.abs(offs - np.median(offs)))) * 1.4826
    se = mad / max(np.sqrt(len(offs)), 1.0)
    median_offset = float(np.hypot(med_dx, med_dy))
    return {
        "n_windows": len(rows),
        "median_dy_m": round(med_dy, 3),
        "median_dx_m": round(med_dx, 3),
        "median_vector_offset_m": round(median_offset, 3),
        "median_of_offsets_m": round(float(np.median(offs)), 3),
        "mad_sigma_m": round(mad, 3),
        "se_of_median_m": round(se, 4),
        "ci95_m": [
            round(median_offset - 1.96 * se, 3),
            round(median_offset + 1.96 * se, 3),
        ],
        "exceeds_limit": bool(median_offset - 1.96 * se > limit_m),
        "frac_windows_over_limit": round(float((offs > limit_m).mean()), 3),
    }


def best_shift(a, b, max_shift):
    """Integer shift of b vs a maximizing edge correlation, searched within max_shift.

    A bounded search is deliberate: an unbounded FFT correlation across modalities
    picks far-field spurious peaks (observed: a 369 m 'offset' on a grid that
    checks 7 and 8 prove is aligned to the millimetre).
    """
    ea, eb = edge_magnitude(a), edge_magnitude(b)
    m = max_shift
    core = ea[m:-m, m:-m]
    h, w = core.shape
    n = 2 * m + 1
    scores = np.full((n, n), -np.inf)
    for iy, dy in enumerate(range(-m, m + 1)):
        for ix, dx in enumerate(range(-m, m + 1)):
            window = eb[m + dy : m + dy + h, m + dx : m + dx + w]
            if window.shape == core.shape:
                scores[iy, ix] = float((core * window).mean())
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    best_score = float(scores[iy, ix])

    # parabolic sub-pixel refinement: the grid search quantizes at 1 m, which is
    # also the blocker threshold, so an integer answer cannot resolve the check
    def refine(prev, mid, nxt):
        denom = prev - 2.0 * mid + nxt
        return 0.0 if denom == 0 else 0.5 * (prev - nxt) / denom

    sub_y = refine(scores[iy - 1, ix], scores[iy, ix], scores[iy + 1, ix]) if 0 < iy < n - 1 else 0.0
    sub_x = refine(scores[iy, ix - 1], scores[iy, ix], scores[iy, ix + 1]) if 0 < ix < n - 1 else 0.0
    dy = (iy - m) + float(np.clip(sub_y, -0.5, 0.5))
    dx = (ix - m) + float(np.clip(sub_x, -0.5, 0.5))
    return round(dy, 3), round(dx, 3), best_score

    # ---------------------------------------------------------------- 11.1 inventory


def check_inventory(audit, config, site_dir):
    tiles = list(config["tiles"])
    expected, missing = [], []
    for tile in tiles:
        for key, path in tile_paths(config, site_dir, tile).items():
            expected.append(str(path))
            if not path.exists():
                missing.append(f"{tile}/{key}: {path}")

    n_products = 3
    audit.add(
        1,
        f"All {len(tiles)} tiles present for all {n_products} products",
        PASS if not missing else FAIL,
        {
            "expected_files": len(expected),
            "found": len(expected) - len(missing),
            "missing": missing,
        },
        blocker=True,
    )

    # 2 - SAVI/EVI filenames follow the NDVI pattern (never verified before)
    pattern_ok, pattern_detail = True, {}
    for tile in tiles:
        for index in config["products"]["vi"]["indices"]:
            path = tile_paths(config, site_dir, tile)[f"vi_{index}"]
            pattern_detail[f"{tile}/{index}"] = path.exists()
            pattern_ok &= path.exists()
    audit.add(
        2,
        "SAVI/EVI filenames match the NDVI pattern",
        PASS if pattern_ok else FAIL,
        pattern_detail,
        blocker=True,
    )

    # 3, 4 - dimensions per product
    for num, key, label in ((3, "rgb", "RGB"), (4, "chm", "CHM")):
        sizes, bad = {}, []
        spec = config["products"][key]
        for tile in tiles:
            path = tile_paths(config, site_dir, tile)[key]
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                sizes[tile] = [ds.width, ds.height]
                if [ds.width, ds.height] != spec["expected_size"]:
                    bad.append(tile)
        audit.add(
            num,
            f"{label} tile dimensions = {spec['expected_size']} at {spec['expected_res']} m",
            PASS if not bad else REPORT,
            {"sizes": sizes, "unexpected": bad},
        )
        # 4 also covers the VI grid
    vi_sizes, vi_bad = {}, []
    spec = config["products"]["vi"]
    for tile in tiles:
        path = tile_paths(config, site_dir, tile)["vi_SAVI"]
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            vi_sizes[tile] = [ds.width, ds.height]
            if [ds.width, ds.height] != spec["expected_size"]:
                vi_bad.append(tile)
    audit.add(
        "4b",
        f"VI tile dimensions = {spec['expected_size']} at {spec['expected_res']} m",
        PASS if not vi_bad else REPORT,
        {"sizes": vi_sizes, "unexpected": vi_bad},
    )

    # 5 - no duplicate or overlapping footprints
    bounds = {}
    for tile in tiles:
        path = tile_paths(config, site_dir, tile)["chm"]
        if path.exists():
            with rasterio.open(path) as ds:
                bounds[tile] = [round(v, 3) for v in ds.bounds]
    overlaps = []
    keys = list(bounds)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ax, ay = bounds[a][0], bounds[a][1]
            axm, aym = bounds[a][2], bounds[a][3]
            bx, by, bxm, bym = bounds[b]
            if ax < bxm and bx < axm and ay < bym and by < aym:
                overlaps.append([a, b])
    audit.add(
        5,
        "No duplicate or overlapping tile footprints",
        PASS if not overlaps else REPORT,
        {
            "bounds": bounds,
            "overlaps": overlaps,
            "duplicates": len(bounds) != len(set(map(tuple, bounds.values()))),
        },
    )
    return bounds

    # ------------------------------------------------- 11.2 georeferencing and alignment


def check_georeferencing(audit, config, site_dir):
    tiles = list(config["tiles"])
    crs_seen, origins, _ = {}, {}, []
    for tile in tiles:
        for key, path in tile_paths(config, site_dir, tile).items():
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                crs_seen.setdefault(str(ds.crs), []).append(f"{tile}/{key}")
                origins[f"{tile}/{key}"] = [ds.transform.c, ds.transform.f]

    expected_crs = config["expected_crs"]
    utm_zone = None
    if len(crs_seen) == 1:
        only = next(iter(crs_seen))
        if only.startswith("EPSG:326"):
            utm_zone = f"{int(only.split(':')[1]) - 32600}N"
        elif only.startswith("EPSG:327"):
            utm_zone = f"{int(only.split(':')[1]) - 32700}S"
    audit.add(
        6,
        "CRS identical across all products and tiles; UTM zone recorded",
        PASS if list(crs_seen) == [expected_crs] else FAIL,
        {
            "crs_found": {k: len(v) for k, v in crs_seen.items()},
            "expected": expected_crs,
            "utm_zone": utm_zone,
        },
        blocker=True,
    )

    # 7 - VI and CHM 1 m grids share an identical origin
    offsets = {}
    for tile in tiles:
        paths = tile_paths(config, site_dir, tile)
        if not (paths["chm"].exists() and paths["vi_SAVI"].exists()):
            continue
        with rasterio.open(paths["chm"]) as c, rasterio.open(paths["vi_SAVI"]) as v:
            offsets[tile] = [
                round(v.transform.c - c.transform.c, 6),
                round(v.transform.f - c.transform.f, 6),
            ]
    bad7 = [t for t, o in offsets.items() if o != [0.0, 0.0]]
    audit.add(
        7,
        "VI and CHM 1 m grids share an identical origin (no half-pixel offset)",
        PASS if not bad7 else FAIL,
        {"vi_minus_chm_origin_m": offsets, "mismatched": bad7},
        blocker=True,
    )

    # 8 - RGB 10 cm grid nests exactly within the 1 m grid
    nesting = {}
    for tile in tiles:
        paths = tile_paths(config, site_dir, tile)
        if not (paths["chm"].exists() and paths["rgb"].exists()):
            continue
        with rasterio.open(paths["chm"]) as c, rasterio.open(paths["rgb"]) as r:
            dx, dy = r.transform.c - c.transform.c, r.transform.f - c.transform.f
            ratio = c.res[0] / r.res[0]
            nesting[tile] = {
                "origin_delta_m": [round(dx, 6), round(dy, 6)],
                "res_ratio": round(ratio, 6),
                "nests": abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(ratio - round(ratio)) < 1e-6,
            }
    bad8 = [t for t, v in nesting.items() if not v["nests"]]
    audit.add(
        8,
        "RGB 10 cm grid nests exactly within the 1 m grid",
        PASS if not bad8 else FAIL,
        {"per_tile": nesting, "not_nesting": bad8},
        blocker=True,
    )

    # 9 - coregistration: RGB (degraded to 1 m) against CHM and SAVI
    limit = config["thresholds"]["coreg_blocker_m"]
    coreg = {}
    worst = 0.0
    for tile in tiles:
        paths = tile_paths(config, site_dir, tile)
        if not all(paths[k].exists() for k in ("rgb", "chm", "vi_SAVI")):
            continue
        with rasterio.open(paths["rgb"]) as r:
            # green band, averaged 10x10 to the 1 m grid
            rgb = r.read(2, out_dtype="float32")
            f = int(round(1.0 / r.res[0]))
            h, w = rgb.shape[0] // f * f, rgb.shape[1] // f * f
            rgb1m = rgb[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
        with rasterio.open(paths["chm"]) as c:
            chm = c.read(1, masked=True).filled(0.0)
        with rasterio.open(paths["vi_SAVI"]) as v:
            savi = v.read(1, masked=True).filled(0.0)
        n = min(rgb1m.shape[0], chm.shape[0], savi.shape[0])
        entry = {}
        window = int(config["thresholds"].get("coreg_window_px", 100))
        upsample = int(config["thresholds"].get("coreg_upsample", 20))
        # all three pairings: CHM vs SAVI involves no camera, so it separates a
        # camera registration problem from a whole-block geolocation difference
        pairs = (
            ("rgb_chm", rgb1m[:n, :n], chm[:n, :n]),
            ("rgb_savi", rgb1m[:n, :n], savi[:n, :n]),
            ("chm_savi", chm[:n, :n], savi[:n, :n]),
        )
        for label, a, b in pairs:
            gdy, gdx = subpixel_shift(a, b, upsample)
            rows = shift_field(a, b, window, upsample)
            stats = summarize_field(rows, limit)
            stats["global_dy_m"] = round(gdy, 3)
            stats["global_dx_m"] = round(gdx, 3)
            stats["global_offset_m"] = round(float(np.hypot(gdx, gdy)), 3)
            entry[label] = stats
            if label != "chm_savi":
                worst = max(worst, stats.get("median_vector_offset_m", 0.0))
        coreg[tile] = entry
        # significance, not a point estimate: fail only if the lower bound of the
        # median's 95% CI is above the limit on a camera-bearing pair
    significant = [f"{tile}/{label}" for tile, entry in coreg.items() for label, s in entry.items() if label != "chm_savi" and s.get("exceeds_limit")]
    audit.add(
        9,
        f"Coregistration, per-window median offset (> {limit} m, CI-significant, is a blocker)",
        PASS if not significant else FAIL,
        {
            "per_tile": coreg,
            "worst_median_offset_m": round(worst, 3),
            "limit_m": limit,
            "significantly_over_limit": significant,
        },
        blocker=True,
    )

    for num, name in (
        (10, "Planet LSP CRS matches and footprint covers all tiles"),
        (11, "Planet pixel size read from file; derive N"),
        (12, "Planet grid origin recorded; 1 m blocks tile it exactly"),
    ):
        audit.defer(num, name, "PlanetScope LSP product for SRER 2022 not yet produced")

        # ------------------------------------------------ 11.3 radiometry and value sanity


def check_radiometry(audit, config, site_dir):
    tiles = list(config["tiles"])

    # 13 - VI scale factor
    vi_stats = {}
    out_of_range = []
    for tile in tiles:
        for index in config["products"]["vi"]["indices"]:
            path = tile_paths(config, site_dir, tile)[f"vi_{index}"]
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                arr = ds.read(1, masked=True)
                key = f"{tile}/{index}"
                vi_stats[key] = {
                    "dtype": ds.dtypes[0],
                    "scales": list(ds.scales),
                    "offsets": list(ds.offsets),
                    "nodata": ds.nodata,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
                # the [-1, 1] gate is for SAVI/NDVI only: EVI is unbounded by
                # construction and routinely exceeds it over bright soil, so it
                # is reported rather than treated as an unapplied scale factor
                if index in ("SAVI", "NDVI") and not (-1.0 <= float(arr.min()) and float(arr.max()) <= 1.0):
                    out_of_range.append(key)
    evi_out = [k for k, v in vi_stats.items() if k.endswith("/EVI") and not (-1.0 <= v["min"] and v["max"] <= 1.0)]
    audit.add(
        13,
        "VI scale factor applied; SAVI/NDVI land in [-1, 1]",
        PASS if not out_of_range else FAIL,
        {
            "per_index": vi_stats,
            "out_of_range_savi_ndvi": out_of_range,
            "evi_outside_unit_range": evi_out,
            "note": "EVI is unbounded by design; listed for information, not a failure",
        },
        blocker=True,
    )

    # 14 - RGB band count and dtype
    rgb_info, rgb_bad = {}, []
    for tile in tiles:
        path = tile_paths(config, site_dir, tile)["rgb"]
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            info = {
                "bands": ds.count,
                "dtype": ds.dtypes[0],
                "colorinterp": [str(c) for c in ds.colorinterp],
            }
            rgb_info[tile] = info
            if ds.count != 3 or ds.dtypes[0] != "uint8":
                rgb_bad.append(tile)
    audit.add(
        14,
        "RGB band count 3 and dtype uint8; no alpha band",
        PASS if not rgb_bad else FAIL,
        {"per_tile": rgb_info, "unexpected": rgb_bad},
        blocker=True,
    )

    # 15 - nodata declared and applied per product
    nodata = {}
    undeclared = []
    for tile in tiles:
        for key, path in tile_paths(config, site_dir, tile).items():
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                nodata[f"{tile}/{key}"] = ds.nodata
                if ds.nodata is None:
                    undeclared.append(f"{tile}/{key}")
                    # RGB carries no nodata value; recorded as an accepted decision, not a defect
    accepted = config.get("decisions", {}).get("rgb_nodata_undeclared")
    audit.add(
        15,
        "Nodata/fill declared per product and applied, not left as a raw sentinel",
        PASS if (not undeclared or accepted) else REPORT,
        {
            "per_file": nodata,
            "undeclared": sorted(set(undeclared)),
            "decision": accepted,
        },
    )

    # 16 - CHM range, and the man-made-structure mask it justifies
    chm_stats = {}
    mask_limit = config["thresholds"]["chm_max_valid_m"]
    total_masked = 0
    for tile in tiles:
        path = tile_paths(config, site_dir, tile)["chm"]
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            arr = ds.read(1, masked=True)
            over = int((arr > mask_limit).sum())
            valid = int(arr.count())
            total_masked += over
            kept = arr[arr <= mask_limit]
            chm_stats[tile] = {
                "min": round(float(arr.min()), 3),
                "max_raw": round(float(arr.max()), 3),
                "max_after_mask": round(float(kept.max()), 3) if kept.count() else None,
                "negatives": int((arr < 0).sum()),
                "masked_px": over,
                "masked_pct": round(100.0 * over / valid, 6) if valid else None,
            }
    audit.add(
        16,
        f"CHM range; values > {mask_limit:g} m masked as man-made structures",
        PASS,
        {
            "per_tile": chm_stats,
            "chm_max_valid_m": mask_limit,
            "total_masked_px": total_masked,
            "decision": config.get("decisions", {}).get("chm_max_valid_m"),
        },
    )

    # 17 - CHM noise floor, proxied by the low tail over low-SAVI (bare) pixels
    savi_bare_max = config["parameters"]["SAVI_BARE_MAX"]
    noise = {}
    for tile in tiles:
        paths = tile_paths(config, site_dir, tile)
        if not (paths["chm"].exists() and paths["vi_SAVI"].exists()):
            continue
        with rasterio.open(paths["chm"]) as c, rasterio.open(paths["vi_SAVI"]) as v:
            chm = c.read(1, masked=True)
            savi = v.read(1, masked=True)
        bare = np.asarray(chm[(savi < savi_bare_max) & ~savi.mask & ~chm.mask])
        if bare.size:
            noise[tile] = {
                "n_low_savi_px": int(bare.size),
                "frac_exactly_zero": round(float((bare == 0).mean()), 4),
                "p50": round(float(np.percentile(bare, 50)), 3),
                "p95": round(float(np.percentile(bare, 95)), 3),
                "p99": round(float(np.percentile(bare, 99)), 3),
                "max": round(float(bare.max()), 3),
            }
    audit.add(
        17,
        f"CHM distribution over low-SAVI (< {savi_bare_max}) pixels - input to H_GRASS_MAX",
        REPORT,
        {
            "per_tile": noise,
            "configured_H_GRASS_MAX_m": config["parameters"]["H_GRASS_MAX"],
            "suggested_H_GRASS_MAX_m": None,
            "note": (
                "NO value is suggested from this proxy. Low SAVI is not the same as bare: "
                "the p95 of 1.3-1.7 m is woody vegetation, not sensor noise, so using it "
                "would set H_GRASS_MAX near H_TREE_MIN and collapse the shrub class. "
                "The median is 0.0 m on every tile, which is the only defensible read here. "
                "Section 3 safeguard 2 requires visually-confirmed bare polygons - "
                "deferred to that step."
            ),
        },
    )

    # 18 - VI nodata percentage (bidirectional mosaics carry flightline gaps)
    gaps, flagged = {}, []
    flag_pct = config["thresholds"]["vi_nodata_flag_pct"]
    for tile in tiles:
        path = tile_paths(config, site_dir, tile)["vi_SAVI"]
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            arr = ds.read(1, masked=True)
            pct = 100.0 * float(np.ma.getmaskarray(arr).sum()) / arr.size
            gaps[tile] = round(pct, 4)
            if pct > flag_pct:
                flagged.append(tile)
    audit.add(
        18,
        f"Per-tile nodata % in the VI mosaic (> {flag_pct}% flags the tile)",
        PASS if not flagged else REPORT,
        {"nodata_pct": gaps, "flagged": flagged},
    )

    audit.defer(19, "Visual scan for cloud, cloud shadow and mosaic seams in RGB", "manual step")

    # ------------------------------------------------------ deferred check groups


def check_phenocam(audit, config, site_dir):
    """23 - locate the phenocam(s) on the tile grid and record which tile holds them."""
    csv_rel = config.get("phenocam_csv")
    if not csv_rel:
        return audit.defer(23, "Phenocam location vs tile grid", "phenocam_csv not set in config")
    csv_path = (HERE / csv_rel).resolve()
    if not csv_path.exists():
        return audit.defer(23, "Phenocam location vs tile grid", f"not found: {csv_path}")

    import csv as _csv

    from pyproj import Transformer

    row = None
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            if (r.get("site_id") or "").strip() == config["ameriflux_id"]:
                row = r
                break
    if row is None:
        return audit.defer(
            23,
            "Phenocam location vs tile grid",
            f"{config['ameriflux_id']} not in {csv_path.name}",
        )

        # tile bounds from the CHM, which defines the 1 m analysis grid
    bounds = {}
    for tile in config["tiles"]:
        path = tile_paths(config, site_dir, tile)["chm"]
        if path.exists():
            with rasterio.open(path) as ds:
                bounds[tile] = ds.bounds

    transformer = Transformer.from_crs("EPSG:4326", config["expected_crs"], always_xy=True)
    cams, tiles_hit = {}, []
    for n in (1, 2):
        name = (row.get(f"phenocam{n}") or "").strip()
        lat, lon = row.get(f"(p{n})latitude"), row.get(f"(p{n})longitude")
        if not name or not lat or not lon:
            continue
        easting, northing = transformer.transform(float(lon), float(lat))
        inside = [t for t, b in bounds.items() if b.left <= easting <= b.right and b.bottom <= northing <= b.top]
        tiles_hit += inside
        cams[name] = {
            "lat": float(lat),
            "lon": float(lon),
            "easting": round(easting, 2),
            "northing": round(northing, 2),
            "in_tiles": inside,
            "tile_roles": [config["tiles"][t] for t in inside],
        }

    roles = {config["tiles"][t] for t in tiles_hit}
    if not tiles_hit:
        status, note = REPORT, "phenocam falls outside every configured tile"
    elif roles == {"train"}:
        status, note = PASS, "phenocam sits in a train tile"
    else:
        status, note = (
            REPORT,
            "phenocam sits in a TEST tile - instructions5.md section 2A states it is in the "
            "511000 train block, which the coordinates contradict. Using it as an independent "
            "phenology reference (Step 5) is still valid, but it is held-out data, so it cannot "
            "also inform training or labeling without leaking the test block.",
        )
    return audit.add(
        23,
        "Phenocam location vs the tile grid",
        status,
        {
            "source": str(csv_path),
            "crs": config["expected_crs"],
            "phenocams": cams,
            "tiles_hit": sorted(set(tiles_hit)),
            "roles_hit": sorted(roles),
            "note": note,
        },
    )


def check_deferred(audit):
    for num, name, reason in (
        ("19a", "Class rasters use the locked section 3 codes", "needs Step 1 outputs"),
        (
            20,
            "Per-class pixel counts under the reference rules",
            "needs Step 1 outputs",
        ),
        (21, "Train and test blocks contain all four classes", "needs Step 1 outputs"),
        (22, "Expected pure end-member counts per class", "needs Step 3/4"),
        (
            "22a",
            "Distribution of the 1 m and block confidence layers",
            "needs Step 1/3",
        ),
        ("22b", "Hard vs soft fraction agreement per block", "needs Step 3"),
    ):
        audit.defer(num, name, reason)
    for num in range(24, 30):
        audit.defer(
            num,
            "Cross-site check (section 11.5)",
            "transfer site (US-Wkg) not yet processed",
        )
    for num in (30, "30a", "30b", "30c", 31, 32, 33, 34, 35):
        audit.defer(
            num,
            "PlanetScope LSP product check (section 11.6)",
            "PLSP SRER 2022 not yet produced",
        )

        # ------------------------------------------------------------------------ report


def write_report(audit, config, out_dir, site, year):
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for c in audit.checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    payload = {
        "site": site,
        "year": year,
        "config": config,
        "summary": counts,
        "blocking_failures": [c["check"] for c in audit.blocking_failures],
        "checks": audit.checks,
    }
    json_path = out_dir / f"data_audit_{site}_{year}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        f"Step 0 data audit - {site} {year}",
        "=" * 64,
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
    ]
    for c in audit.checks:
        mark = {PASS: "ok ", FAIL: "FAIL", REPORT: "note", DEFERRED: "----"}[c["status"]]
        flag = " [BLOCKER]" if c["blocker"] and c["status"] == FAIL else ""
        lines.append(f"  {mark} #{c['check']:<4} {c['name']}{flag}")
        if c["status"] == DEFERRED:
            lines.append(f"           deferred: {c['observed']['reason']}")
    txt_path = out_dir / f"data_audit_{site}_{year}.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    return json_path, txt_path, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="site config json, e.g. config/srer_2022.json")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    site_dir = resolve(config["data_root"], config["site_name"])
    out_dir = resolve(config["results_root"], "stage1_data_and_features", "qa")

    if not site_dir.is_dir():
        sys.exit(f"data directory not found: {site_dir}")
    print(f"site dir : {site_dir}")
    print(f"output   : {out_dir}\n")

    audit = Audit()
    check_inventory(audit, config, site_dir)
    check_georeferencing(audit, config, site_dir)
    check_radiometry(audit, config, site_dir)
    check_phenocam(audit, config, site_dir)
    check_deferred(audit)

    json_path, txt_path, counts = write_report(audit, config, out_dir, site, year)
    print(txt_path.read_text())
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")

    if audit.blocking_failures:
        sys.exit("STEP 0 FAILED - blocking checks: " + ", ".join(f"#{c['check']}" for c in audit.blocking_failures))
    print("Step 0 passed for every check that can run now.")


if __name__ == "__main__":
    main()
