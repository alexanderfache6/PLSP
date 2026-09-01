"""Step 1a - feature construction (instructions5.md section 5 Step 1a).

Builds the per-tile 1 m feature stack.

The order of operations is fixed by R2 and is the whole point of this step:

    RGB 10 cm -> resample to TEXTURE_SCALE (0.6 m)
               -> compute chromatic coordinates, indices, texture THERE
               -> area-average onto the canonical 1 m analysis grid

Indices are nonlinear ratios, so computing at 0.6 m then averaging is not the
same number as averaging then computing. 0.6 m is the finest resolution NAIP
can supply, so anything computed at native 10 cm would be a NEON feature that
NAIP can never reproduce - which silently breaks every model-transfer claim.

NIR indices (SAVI/NDVI/EVI) and CHM are already on the native 1 m grid and are
carried through unchanged, with the CHM man-made-structure mask applied.

Outputs per tile:
    stage1_data_and_features/features/features_{SITE}_{tile}_{YEAR}.tif
    stage1_data_and_features/features/features_{SITE}_{YEAR}_bands.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from helpers import read_rgb_at_scale, resolve_config_path
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
EPS = 1e-6


def tile_paths(config, site_dir, tile):
    p = config["products"]
    out = {
        "rgb": site_dir / p["rgb"]["folder"] / p["rgb"]["pattern"].format(tile=tile),
        "chm": site_dir / p["chm"]["folder"] / p["chm"]["pattern"].format(tile=tile),
    }
    for index in p["vi"]["indices"]:
        out[f"vi_{index}"] = site_dir / p["vi"]["folder"] / p["vi"]["pattern"].format(tile=tile, index=index)
    return out


# ------------------------------------------------------------------ RGB at 0.6 m


def chromatic(rgb):
    """r, g, b normalized by total brightness - suppresses illumination/shadow."""
    total = rgb.sum(axis=0) + EPS
    return rgb[0] / total, rgb[1] / total, rgb[2] / total


def rgb_indices(rgb):
    """The five visible-band indices in the spec, all built on chromatic coords."""
    r, g, b = chromatic(rgb)
    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    out = {
        "r": r,
        "g": g,
        "b": b,
        "ExG": exg,
        "ExR": exr,
        "ExGR": exg - exr,
        "VARI": (g - r) / (g + r - b + EPS),
        "GLI": (2.0 * rgb[1] - rgb[0] - rgb[2]) / (2.0 * rgb[1] + rgb[0] + rgb[2] + EPS),
    }
    # Rec. 709 luma, shared with Step 1c shadow detection
    out["luma"] = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    mx = rgb.max(axis=0)
    # np.where(mx > EPS, ..., 0.0) would hand back 0.0 for a NaN pixel, because
    # NaN > EPS is False - so unflown ground (NaN in RGB) would silently report
    # a real saturation of zero while every other band correctly reports NaN.
    # The NaN has to be reinstated explicitly.
    saturation = np.where(mx > EPS, (mx - rgb.min(axis=0)) / (mx + EPS), 0.0)
    out["saturation"] = np.where(np.isnan(mx), np.nan, saturation)
    return out


# --------------------------------------------------------------------- texture


def glcm_features(img, window, levels, offset):
    """GLCM contrast, homogeneity and correlation as exact moving-window statistics.

    For a fixed offset d, the GLCM over a window is the joint distribution of
    (I(x), I(x+d)) inside that window, so the three measures below reduce to
    means of per-pixel-pair quantities and are exactly computable with a uniform
    filter - no per-window graycomatrix loop:

        contrast = E[(i-j)^2]
        homogeneity = E[1/(1+(i-j)^2)]
        correlation = (E[ij] - mu_i mu_j) / (sigma_i sigma_j)

    Entropy is NOT reducible this way (it needs the full joint distribution) and
    is handled separately by local_entropy().
    """
    lo, hi = np.nanpercentile(img, [1, 99])
    q = np.clip((img - lo) / (hi - lo + EPS), 0, 1) * (levels - 1)
    q = np.rint(q)

    dy, dx = offset
    a = q[: q.shape[0] - dy, : q.shape[1] - dx]
    b = q[dy:, dx:]

    diff2 = (a - b) ** 2
    contrast = uniform_filter(diff2, window)
    homogeneity = uniform_filter(1.0 / (1.0 + diff2), window)

    mu_a = uniform_filter(a, window)
    mu_b = uniform_filter(b, window)
    var_a = np.maximum(uniform_filter(a * a, window) - mu_a**2, 0.0)
    var_b = np.maximum(uniform_filter(b * b, window) - mu_b**2, 0.0)
    cov = uniform_filter(a * b, window) - mu_a * mu_b
    correlation = cov / (np.sqrt(var_a * var_b) + EPS)

    # pad back to the input shape so every band shares one grid
    def pad(arr):
        return np.pad(arr, ((0, dy), (0, dx)), mode="edge")

    return {
        "glcm_contrast": pad(contrast),
        "glcm_homogeneity": pad(homogeneity),
        "glcm_correlation": pad(correlation),
    }


def local_entropy(img, window, levels):
    """Shannon entropy of the local grey-level histogram.

    This is grey-level entropy over the window, not GLCM joint entropy - the
    joint form needs the full co-occurrence distribution and cannot be reduced
    to moving-window moments. Recorded as a deviation from the spec wording.
    """
    lo, hi = np.nanpercentile(img, [1, 99])
    q = np.clip((img - lo) / (hi - lo + EPS), 0, 1) * (levels - 1)
    q = np.rint(q).astype(np.int16)
    ent = np.zeros(img.shape, dtype=np.float32)
    for level in range(levels):
        p = uniform_filter((q == level).astype(np.float32), window)
        np.subtract(ent, np.where(p > 0, p * np.log(p + EPS), 0.0), out=ent)
    return ent


def lbp_features(img, window, points, radius):
    """Fraction of non-uniform LBP codes in the window - a local edge/texture rate."""
    from skimage.feature import local_binary_pattern

    lo, hi = np.nanpercentile(img, [1, 99])
    # quantize to uint8: LBP compares neighbours by magnitude, and on float input
    # tiny numerical differences flip comparisons arbitrarily
    norm = (np.clip((img - lo) / (hi - lo + EPS), 0, 1) * 255).astype(np.uint8)
    codes = local_binary_pattern(norm, points, radius, method="uniform")
    non_uniform = (codes == points + 1).astype(np.float32)
    return {"lbp_nonuniform": uniform_filter(non_uniform, window)}


def local_std(img, window):
    mu = uniform_filter(img, window)
    var = np.maximum(uniform_filter(img * img, window) - mu * mu, 0.0)
    return np.sqrt(var)


# ---------------------------------------------------------------- 0.6 m -> 1 m


def to_analysis_grid(arr, src_transform, src_crs, dst_profile):
    """Area-weighted average from the 0.6 m grid onto the canonical 1 m grid."""
    out = np.empty((dst_profile["height"], dst_profile["width"]), dtype="float32")
    reproject(
        source=np.ascontiguousarray(arr, dtype="float32"),
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=Resampling.average,
    )
    return out


def build_tile(config, site_dir, tile):
    paths = tile_paths(config, site_dir, tile)
    s1 = config["stage1_5_generate_features"]
    scale = s1["texture_scale_m"]
    window = s1["texture_window_px"]

    # canonical 1 m grid comes from the CHM (check 7 proved VI shares its origin)
    with rasterio.open(paths["chm"]) as ds:
        grid = {
            "transform": ds.transform,
            "crs": ds.crs,
            "width": ds.width,
            "height": ds.height,
        }
        chm = ds.read(1, masked=True)

    rgb, rgb_transform, rgb_crs, _ = read_rgb_at_scale(paths["rgb"], scale)

    bands, meta = {}, []

    # --- computed at 0.6 m, then averaged to 1 m
    for name, arr in rgb_indices(rgb).items():
        bands[name] = to_analysis_grid(arr, rgb_transform, rgb_crs, grid)
        group = "chromatic" if name in ("r", "g", "b") else ("brightness" if name in ("luma", "saturation") else "rgb_index")
        meta.append({"name": name, "group": group, "computed_at_m": scale})

    luma = rgb_indices(rgb)["luma"]

    # TEXTURE IS COMPUTED ON A FILLED COPY, THEN RE-MASKED. Every texture
    # function quantizes against a GLOBAL statistic - nanpercentile bounds, an
    # int16 or uint8 cast, a windowed mean - and a single NaN anywhere in the
    # input poisons the whole array, not just its own neighbourhood. Feeding
    # unflown ground straight in turned glcm_contrast, glcm_homogeneity,
    # glcm_correlation and std 100% NaN on 517000_3531000 and made the entire
    # tile unusable, from 1.1% genuinely missing ground.
    #
    # Filling with the median keeps the quantization bounds where the real data
    # puts them; the mask is reinstated afterwards so no unflown pixel escapes
    # with a fabricated texture value.
    unflown = np.isnan(luma)
    luma_filled = np.where(unflown, np.nanmedian(luma), luma).astype("float32")
    texture = {}
    texture.update(glcm_features(luma_filled, window, s1["glcm_levels"], tuple(s1["glcm_offset"])))
    texture["glcm_entropy"] = local_entropy(luma_filled, window, s1["glcm_levels"])
    texture.update(lbp_features(luma_filled, window, s1["lbp_points"], s1["lbp_radius"]))
    texture["std"] = local_std(luma_filled, window)
    for name, arr in texture.items():
        arr = np.where(unflown, np.nan, arr)
        bands[name] = to_analysis_grid(arr, rgb_transform, rgb_crs, grid)
        meta.append({"name": name, "group": "texture", "computed_at_m": scale})

    # --- already native 1 m, carried through unchanged
    for index in config["products"]["vi"]["indices"]:
        with rasterio.open(paths[f"vi_{index}"]) as ds:
            arr = ds.read(1, masked=True)
        bands[index] = arr.filled(np.nan).astype("float32")
        meta.append({"name": index, "group": "nir_index", "computed_at_m": 1.0})

    mask_limit = config["thresholds"]["chm_max_valid_m"]
    chm_masked = np.ma.masked_where(chm > mask_limit, chm)
    bands["CHM"] = chm_masked.filled(np.nan).astype("float32")
    meta.append(
        {
            "name": "CHM",
            "group": "chm",
            "computed_at_m": 1.0,
            "note": f"values > {mask_limit} m masked as man-made structures",
        }
    )

    # --- section 11.7 edge margin, in 1 m cells
    margin = int(np.ceil((window // 2) * scale / config["stage1_5_generate_features"]["analysis_grid_m"]))
    if margin:
        for name in bands:
            bands[name][:margin, :] = np.nan
            bands[name][-margin:, :] = np.nan
            bands[name][:, :margin] = np.nan
            bands[name][:, -margin:] = np.nan

    return bands, meta, grid, margin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--tiles", nargs="+", help="subset of tiles (default: all in config)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    site_dir = resolve_config_path(config["data_root"], config["site_name"])
    out_dir = resolve_config_path(config["results_root"], "stage1_data_and_features", "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = args.tiles or list(config["tiles"])
    band_meta = None
    for tile in tiles:
        print(f"[{tile}] building features...", flush=True)
        bands, meta, grid, margin = build_tile(config, site_dir, tile)
        band_meta = meta

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": len(bands),
            "width": grid["width"],
            "height": grid["height"],
            "crs": grid["crs"],
            "transform": grid["transform"],
            "nodata": np.nan,
            "compress": "deflate",
            "tiled": True,
        }
        path = out_dir / f"features_{site}_{tile}_{year}.tif"
        with rasterio.open(path, "w", **profile) as ds:
            for i, (name, arr) in enumerate(bands.items(), start=1):
                ds.write(arr.astype("float32"), i)
                ds.set_band_description(i, name)
        valid = {n: float(np.isfinite(a).mean()) for n, a in bands.items()}
        print(f"[{tile}] wrote {path.name} bands={len(bands)} edge margin={margin} m min valid frac={min(valid.values()):.3f}")

    if band_meta:
        # section 4.1 is authoritative (resolved). Each framework adds exactly one input
        # family to the one before it, which makes A-E a clean deconfounding of inputs.
        groups = {
            "A": ["chromatic", "rgb_index", "brightness"],
            "B": ["chromatic", "rgb_index", "brightness", "nir_index"],
            "C": ["chromatic", "rgb_index", "brightness", "nir_index", "texture"],
            "D": [
                "chromatic",
                "rgb_index",
                "brightness",
                "nir_index",
                "texture",
                "chm",
            ],
            "E": [
                "chromatic",
                "rgb_index",
                "brightness",
                "nir_index",
                "texture",
                "chm",
            ],
        }
        note = "Resolved in favour of the section 4.1 table: A = RGB, B = RGB + vegetation indices, C = RGB + vegetation indices + texture. NIR is permitted in A-C because NAIP is 4-band, so a NIR-bearing feature is reproducible at the transfer site. The rgb_index group (ExG/ExR/ExGR/VARI/GLI) stays in A: it is a deterministic function of RGB and adds no input layer."
        caveat = "B and C inherit a cross-sensor mismatch that A does not. NEON NIR indices come from the imaging spectrometer (narrowband, BRDF- and atmospherically corrected, native 1 m); NAIP NIR comes from an uncorrected broadband 4-band camera at 0.6 m. Same index name, different measurement - so R3 (percentile or learned boundaries, never absolute thresholds) is binding for B-E."
        spec = {
            "site": site,
            "year": year,
            "texture_scale_m": config["stage1_5_generate_features"]["texture_scale_m"],
            "analysis_grid_m": config["stage1_5_generate_features"]["analysis_grid_m"],
            "edge_margin_m": margin,
            "band_order": [b["name"] for b in band_meta],
            "bands": band_meta,
            "framework_groups": groups,
            "framework_note": note,
            "framework_caveat": caveat,
        }
        (out_dir / f"features_{site}_{year}_bands.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(f"\nwrote band spec: {out_dir / f'features_{site}_{year}_bands.json'}")
        print(f"bands ({len(band_meta)}): {', '.join(b['name'] for b in band_meta)}")


if __name__ == "__main__":
    main()
