"""Section 4.2 Stage 1b helper - CHM-derived shrub candidate proposals.

Shrub is at once the hardest class to hand-label (small, isolated, numerous)
and the weakest in every prior result - instructions1.md section 6 found it the
most confused class at 40-70% purity. Drawing every crown by hand is the
bottleneck, and a uniform minimum polygon area made it worse by filtering the
class by size.

Shrub is already defined by CHM in [H_GRASS_MAX, H_TREE_MIN) in section 3, so
the delineation criterion exists. This script applies it, emits one candidate
per connected component, and leaves the analyst to accept, edit, or reject each
one - a single click for the common case instead of an outline.

Candidates are PROPOSALS, NEVER LABELS. An unreviewed candidate must not enter
training. The CHM band rule is the reference-path definition and inherits every
CHM error; hand validation is what turns a candidate into ground truth.

NEON sites only. This is a labeling accelerator for sites with reliable CHM. It
creates no dependency in frameworks A-C: what it produces is ordinary
hand-validated polygons, indistinguishable downstream.

Per-component CHM statistics are persisted so the shrub/tree threshold can be
re-cut analytically without re-running anything (section 3 safeguard 1).

The full candidate set runs to tens of thousands per tile, which is more work
than drawing shrubs by hand and defeats the purpose. So the full set is written
as provenance, and a stratified REVIEW SUBSET is drawn from it - stratified by
area so small, medium and large crowns are all represented, because size-biased
shrub sampling is the exact failure this whole mechanism exists to prevent.
Hand review works from the subset; it is sized to clear the 50-per-class-per-role
gate with margin.

Usage: python run_stage2_2_find_chm_derived_shrub_candidates.py config/srer_2022.json
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from constants import SEVENTY
from helpers import resolve_config_path
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.validation import make_valid

SHRUB_CODE = 2
CONNECTIVITY = 8


def chm_path(config, site_dir, tile):
    """Path to a tile's canopy height model, from the config product block.

    Inputs: config - parsed site config; site_dir - Path to the site data dir;
             tile - tile key string
    Outputs: Path to the CHM GeoTIFF
    """
    product = config["products"]["chm"]
    return site_dir / product["folder"] / product["pattern"].format(tile=tile)


def shrub_band_mask(chm, grass_max, tree_min, chm_max_valid):
    """Boolean mask of pixels inside the shrub height band.

    Applies the section 3 reference rule H_GRASS_MAX <= CHM < H_TREE_MIN, after
    dropping NaN and values above the plausible-height limit, which the config
    documents as man-made structures.

    Inputs: chm - float array of canopy height in m; grass_max, tree_min -
             band bounds in m; chm_max_valid - upper plausibility limit in m
    Outputs: boolean array, True inside the shrub band
    """
    finite = np.isfinite(chm)
    plausible = finite & (chm <= chm_max_valid)
    return plausible & (chm >= grass_max) & (chm < tree_min)


def component_labels(mask, erode_px):
    """Connected-component labelling of the shrub band mask.

    Isolated crowns are exactly the case connected-component labelling handles
    cleanly, with none of the merged-canopy ambiguity that complicates tree
    delineation. Optional erosion separates crowns that touch at a corner.

    Inputs: mask - boolean array; erode_px - erosion radius in pixels, 0 to skip
    Outputs: (labels int32 array, n_components int)
    """
    if erode_px > 0:
        mask = ndimage.binary_erosion(mask, iterations=int(erode_px))
    structure = ndimage.generate_binary_structure(2, 2) if CONNECTIVITY == 8 else None
    labels, n = ndimage.label(mask, structure=structure)
    return labels.astype("int32"), int(n)


def component_stats(labels, n, chm):
    """Per-component CHM statistics, persisted so thresholds can be re-cut later.

    Section 3 safeguard 1 requires per-crown CHM statistics on the vector
    outputs so the shrub/tree cut can be moved analytically rather than by
    re-running the pipeline.

    Inputs: labels - int32 label array; n - number of components; chm - float
             canopy height array
    Outputs: dict of {label: {chm_min, chm_mean, chm_max, chm_p90, n_pixels}}
    """
    index = np.arange(1, n + 1)
    heights = np.where(np.isfinite(chm), chm, 0.0)
    counts = ndimage.sum(np.ones_like(heights), labels, index)
    means = ndimage.mean(heights, labels, index)
    minima = ndimage.minimum(heights, labels, index)
    maxima = ndimage.maximum(heights, labels, index)
    p90 = ndimage.labeled_comprehension(heights, labels, index, lambda v: float(np.percentile(v, 90)), float, 0.0)
    out = {}
    for i, label in enumerate(index):
        out[int(label)] = {
            "chm_min": float(minima[i]),
            "chm_mean": float(means[i]),
            "chm_max": float(maxima[i]),
            "chm_p90": float(p90[i]),
            "n_pixels": int(counts[i]),
        }
    return out


def component_geometries(labels, transform):
    """Vectorize each connected component to a valid polygon.

    8-connected labelling lets two pixel blocks meet at a single corner, and
    vectorizing that produces a ring that touches itself - a self-intersection
    that is invalid under OGC rules even though the shape is what we want.
    make_valid splits the pinch into parts, which is the right reading: two
    lobes joined at a point are two lobes. Repairing here rather than
    downstream keeps every consumer from having to know about it.

    Inputs: labels - int32 label array; transform - the raster affine
    Outputs: dict of {label: shapely geometry}, all valid
    """
    out = {}
    for geom, value in shapes(labels, mask=labels > 0, transform=transform, connectivity=CONNECTIVITY):
        label = int(value)
        if label > 0:
            polygon = shape(geom)
            out[label] = polygon if polygon.is_valid else make_valid(polygon)
    return out


def build_candidates(geoms, stats, min_area, max_area, point_below_min, tile):
    """Assemble candidate records, splitting polygons from sub-minimum points.

    Components at or above the class minimum are emitted as polygons. Smaller
    ones are emitted as a single interior point rather than discarded, since
    they are the small-shrub population the minimum-area rule would otherwise
    remove - which is precisely the part of the distribution that matters.
    Components above max_area are dropped as almost certainly not single shrubs.

    Inputs: geoms - {label: geometry}; stats - {label: stats dict}; min_area,
             max_area - m2 bounds; point_below_min - bool, keep sub-minimum
             components as points; tile - tile key string
    Outputs: (list of record dicts, dict of counts by disposition)
    """
    records = []
    counts = {"polygon": 0, "point": 0, "too_large": 0, "dropped_small": 0}
    for label, geom in geoms.items():
        area = float(geom.area)
        stat = stats[label]
        if area > max_area:
            counts["too_large"] += 1
            continue
        if area >= min_area:
            geometry, kind = geom, "polygon"
            counts["polygon"] += 1
        elif point_below_min:
            geometry, kind = geom.representative_point(), "point"
            counts["point"] += 1
        else:
            counts["dropped_small"] += 1
            continue
        records.append(
            {
                "class_code": SHRUB_CODE,
                "source": "chm_candidate",
                "reviewed": 0,
                "rejected": 0,
                "label_geometry": kind,
                "area_m2": round(area, 2),
                "tile": tile,
                "chm_min": round(stat["chm_min"], 2),
                "chm_mean": round(stat["chm_mean"], 2),
                "chm_max": round(stat["chm_max"], 2),
                "chm_p90": round(stat["chm_p90"], 2),
                "n_pixels": stat["n_pixels"],
                "geometry": geometry,
            }
        )
    return records, counts


def mark_review_sample(records, n_sample, n_strata, rng):
    """Flag a stratified subset of candidates for hand review.

    Stratified by area so that small, medium and large crowns are all offered.
    An unstratified draw would be dominated by the smallest components, which
    reintroduces the size bias this mechanism exists to remove - only inverted.
    Polygons are sampled; sub-minimum points are excluded from review, since
    they are the population most likely to be CHM noise.

    Inputs: records - list of record dicts, modified in place; n_sample -
             target subset size; n_strata - number of equal-count area strata;
             rng - a seeded numpy Generator
    Outputs: int, how many records were flagged
    """
    for record in records:
        record["review"] = 0
    polygons = [r for r in records if r["label_geometry"] == "polygon"]
    if not polygons:
        return 0
    areas = np.array([r["area_m2"] for r in polygons])
    edges = np.quantile(areas, np.linspace(0.0, 1.0, n_strata + 1))
    per_stratum = max(1, n_sample // n_strata)
    flagged = 0
    for i in range(n_strata):
        low, high = edges[i], edges[i + 1]
        in_stratum = [r for r, a in zip(polygons, areas) if (a >= low and a <= high if i == n_strata - 1 else a >= low and a < high)]
        if not in_stratum:
            continue
        take = min(per_stratum, len(in_stratum))
        for index in rng.choice(len(in_stratum), size=take, replace=False):
            in_stratum[int(index)]["review"] = 1
            flagged += 1
    return flagged


def write_candidates(records, crs, out_path, review_path):
    """Write the full candidate set and the review subset to GeoPackages.

    Mixed polygon and point geometries go to separate layers, because a
    GeoPackage layer carries a single geometry type and QGIS edits it as such.
    The full set is provenance; the review file is what gets opened for hand
    validation.

    Inputs: records - list of record dicts; crs - the tile CRS; out_path -
             Path for the full set; review_path - Path for the review subset
    Outputs: dict of {layer_name: n_features} actually written
    """
    written = {}
    # MultiPolygon rather than Polygon: make_valid splits a pinched component
    # into parts, and the layer must be able to hold the result
    for kind, geometry_type in (("polygon", "MultiPolygon"), ("point", "Point")):
        subset = [r for r in records if r["label_geometry"] == kind]
        if not subset:
            continue
        gdf = gpd.GeoDataFrame(subset, crs=crs)
        gdf["class_code"] = gdf["class_code"].astype("Int64")
        gdf["rejected"] = gdf["rejected"].astype("Int64")
        gdf["reviewed"] = gdf["reviewed"].astype("Int64")
        gdf["review"] = gdf["review"].astype("Int64")
        layer = f"{out_path.stem}_{kind}"
        gdf.to_file(out_path, driver="GPKG", layer=layer, geometry_type=geometry_type)
        written[layer] = len(gdf)
        chosen = gdf[gdf["review"] == 1]
        if len(chosen):
            review_layer = f"{review_path.stem}_{kind}"
            chosen.to_file(
                review_path,
                driver="GPKG",
                layer=review_layer,
                geometry_type=geometry_type,
            )
            written[review_layer] = len(chosen)
    return written


def review_decision_count(path):
    """How many review decisions a shrub_review GeoPackage already holds.

    The full candidate set is regenerable and carries no analyst input, so it is
    always rewritten. The review subset does carry input once work has started,
    and rewriting it would discard every accept and reject silently.

    Inputs: path - Path to the shrub_review GeoPackage
    Outputs: int, number of features with reviewed or rejected set
    """
    if not path.exists():
        return 0
    total = 0
    for layer in ("polygon", "point"):
        try:
            gdf = gpd.read_file(path, layer=f"{path.stem}_{layer}")
        except Exception:
            continue
        if "reviewed" in gdf.columns:
            total += int((gdf["reviewed"].fillna(0).astype(int) > 0).sum())
        if "rejected" in gdf.columns:
            total += int((gdf["rejected"].fillna(0).astype(int) > 0).sum())
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite review files that already contain review decisions - THIS DESTROYS THEM",
    )
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    params = config["parameters"]
    settings = config["stage2_2_shrub_candidates"]
    grass_max, tree_min = params["H_GRASS_MAX"], params["H_TREE_MIN"]
    chm_max_valid = config["thresholds"]["chm_max_valid_m"]
    min_area = settings["min_component_area_m2"]
    max_area = settings["max_component_area_m2"]
    point_below_min = bool(settings["point_label_below_min"])
    erode_px = settings["erode_px"]
    n_sample = settings["review_sample_per_tile"]
    n_strata = settings["review_area_strata"]
    seed = config["BASE_SEED"] + year

    site_dir = resolve_config_path(config["data_root"], config["site_name"])
    out_dir = resolve_config_path(config["results_root"], "stage2_labeling")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CHM shrub candidates - {site} {year}")
    print("=" * SEVENTY)
    print(f"shrub band {grass_max} <= CHM < {tree_min} m")
    print(f"area bounds {min_area} to {max_area} m2, sub-minimum kept as points: {point_below_min}")
    print(f"erosion {erode_px} px")
    print(f"review {n_sample} per tile, stratified into {n_strata} area bands, seed {seed}")

    totals = {"polygon": 0, "point": 0, "too_large": 0, "dropped_small": 0, "review": 0}
    summary = []

    for tile, role in config["tiles"].items():
        path = chm_path(config, site_dir, tile)
        if not path.exists():
            print(f"[{tile}] CHM not found, skipping: {path}")
            continue
        with rasterio.open(path) as ds:
            chm = ds.read(1, masked=True).filled(np.nan).astype("float32")
            transform, crs = ds.transform, ds.crs

        mask = shrub_band_mask(chm, grass_max, tree_min, chm_max_valid)
        labels, n = component_labels(mask, erode_px)
        if n == 0:
            print(f"[{tile}] no components in the shrub band")
            continue
        stats = component_stats(labels, n, chm)
        geoms = component_geometries(labels, transform)
        records, counts = build_candidates(geoms, stats, min_area, max_area, point_below_min, tile)
        rng = np.random.default_rng(seed)
        counts["review"] = mark_review_sample(records, n_sample, n_strata, rng)

        out_path = out_dir / f"shrub_candidates_{site}_{tile}_{year}.gpkg"
        review_path = out_dir / f"shrub_review_{site}_{tile}_{year}.gpkg"
        decisions = review_decision_count(review_path)
        if decisions > 0 and not args.force:
            print(f"[{tile}] KEEPING {review_path.name} - it holds {decisions} review decision(s). Pass --force to overwrite and lose them.")
            continue
        written = write_candidates(records, crs, out_path, review_path)
        for key in totals:
            totals[key] += counts[key]
        summary.append({"tile": tile, "role": role, "components": n, **counts, "layers": written})
        print(f"[{tile}] {role:<5} {n:>6} components -> {counts['polygon']:>5} polygons, {counts['point']:>5} points, {counts['too_large']:>4} too large review subset {counts['review']:>4}")

    print("\n" + "=" * SEVENTY)
    print(f"total {totals['polygon']} polygon candidates, {totals['point']} point candidates")
    print(f" {totals['too_large']} above {max_area} m2 (unlikely to be single shrubs), {totals['dropped_small']} sub-minimum dropped")
    print(f" {totals['review']} flagged for hand review across all tiles")
    print("\nOpen the shrub_review_* files, not the full shrub_candidates_* set - the full")
    print("set is provenance. The review subset is stratified by area so small, medium and")
    print("large crowns are all offered; reviewing only the easy large ones would rebuild")
    print("the size bias this mechanism exists to remove.")
    print("\nCandidates are proposals, not labels. Review each one in QGIS:")
    print(" accept set reviewed = 1, leave class_code = 2")
    print(" edit adjust the geometry, set reviewed = 1, keep class_code = 2")
    print(" reject set reviewed = 1, rejected = 1, and clear class_code")
    print("reviewed exists so an accepted candidate is distinguishable from one not yet")
    print("looked at - without it, zero-click accept and untouched are identical, which is")
    print("unworkable at 150 per tile. Pending candidates render magenta, matching the")
    print("unlabelled convention used by the zone and polygon layers.")
    print("A rejected candidate is retained deliberately - a CHM-band object that is not")
    print("shrub is evidence about where the height rule fails (section 3 safeguards 1-2).")
    print("Track the accept/reject rate per tile: a low rate means H_GRASS_MAX or")
    print("H_TREE_MIN needs revisiting, not that the candidates were a bad idea.")

    report = {
        "site": site,
        "year": year,
        "shrub_band_m": [grass_max, tree_min],
        "min_component_area_m2": min_area,
        "max_component_area_m2": max_area,
        "erode_px": erode_px,
        "review_sample_per_tile": n_sample,
        "review_area_strata": n_strata,
        "seed": seed,
        "totals": totals,
        "per_tile": summary,
    }
    out = out_dir / f"shrub_candidates_summary_{site}_{year}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
