"""Section 4.2 Stage 1 - K-means labeling zones (instructions5.md section 4.2).

Clustering here is a TARGETING AID, not a label source. Unguided hand labeling
drifts toward the visually obvious, which starves the classifier of the
intermediate cases that decide accuracy. Clustering the feature space first
exposes the full range of what is present so labeling can be spread across it.

The cluster ID is never a label. It is carried into the GeoPackage as an
attribute for provenance only; the analyst supplies class_code.

Feature handling deviates from "use the full stack" for a forced reason. The
Step 1a stack has rank 16 of 20 - r+g+b=1, and ExG/ExR/ExGR are exact linear
combinations of r,g,b - so its covariance is singular and the Mahalanobis
whitening section 4.2 asks for amplifies numerical noise (measured: empty
clusters, cond 2.4e5 even after dropping the redundant bands). Dropping the four
collinear bands loses no information for a distance-based method, and PCA
whitening then removes the near-degenerate directions instead of damping them.
Euclidean K-means on whitened components IS Mahalanobis on the retained subspace.
Random Forest at Step 1d is unaffected by collinearity and keeps all 20 bands.

Outputs per tile:
    stage2_labeling/cluster_map_{SITE}_{tile}_{YEAR}.tif uint8, 1..k, 0 = nodata
    stage2_labeling/labeling_zones_{SITE}_{tile}_{YEAR}.gpkg candidate sites
    stage2_labeling/training_polygons_{SITE}_{tile}_{YEAR}.gpkg empty template to draw into
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from helpers import resolve_config_path
from scipy import ndimage as ndi
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def load_tile(feat_dir, site, tile, year, drop):
    with rasterio.open(feat_dir / f"features_{site}_{tile}_{year}.tif") as ds:
        names = list(ds.descriptions)
        arr = ds.read()
        profile = ds.profile
        transform = ds.transform
        crs = ds.crs
    keep = [i for i, n in enumerate(names) if n not in drop]
    kept_names = [names[i] for i in keep]
    flat = arr[keep].reshape(len(keep), -1).T
    valid = np.isfinite(flat).all(axis=1)
    return flat, valid, kept_names, arr, names, profile, transform, crs


def fit_kmeans_model(config, feat_dir, tiles, drop, seed):
    """Pooled scaler + PCA whitening + K-means over all tiles."""
    s = config["stage2_1_labeling_zones"]
    rng = np.random.default_rng(seed)
    parts = []
    for tile in tiles:
        flat, valid, kept_names, *_ = load_tile(feat_dir, config["site"], tile, config["year"], drop)
        good = flat[valid]
        n = min(s["fit_sample_per_tile"], len(good))
        parts.append(good[rng.choice(len(good), n, replace=False)])
    X = np.vstack(parts)

    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=s["pca_variance"], whiten=True, random_state=seed).fit(scaler.transform(X))
    km = KMeans(n_clusters=s["k"], n_init=s["n_init"], random_state=seed).fit(pca.transform(scaler.transform(X)))
    return scaler, pca, km, kept_names, X


def interior_mask(labels, cluster, window, min_fraction):
    """Pixels whose neighbourhood is dominated by their own cluster.

    A 5x5 majority test, per section 4.2 - it keeps candidate sites off segment
    boundaries and speckle, which are the least reliable pixels to label.
    """
    own = (labels == cluster).astype(np.float32)
    frac = ndi.uniform_filter(own, size=window)
    return (labels == cluster) & (frac >= min_fraction)


def interior_mask_relaxed(labels, cluster, levels):
    """Interior test with fallback, so a speckle-only cluster is still labelable.

    Some clusters never form a 5x5-dominant patch anywhere - at SRER, cluster 12
    is 2.8% of the train block yet has zero pixels passing 5x5 >= 0.8. Refusing
    to sample it would violate the section 4.2 rule that every cluster must
    receive labels, so the test is relaxed stepwise and the level used is
    recorded rather than hidden.
    """
    for window, min_fraction in levels:
        mask = interior_mask(labels, cluster, window, min_fraction)
        if mask.any():
            return mask, f"{window}x{window}>={min_fraction}"
    mask = labels == cluster
    return mask, "no interior test (speckle-only cluster)"


def sample_sites(mask, n_sites, min_sep_px, rng):
    """Spatially spread sample of interior pixels."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    order = rng.permutation(len(ys))
    chosen = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - cy) ** 2 + (x - cx) ** 2 >= min_sep_px**2 for cy, cx in chosen):
            chosen.append((y, x))
            if len(chosen) >= n_sites:
                break
    return chosen


def filled_zone_count(path):
    """How many candidate sites in a zones file already carry a class_code.

    Zones are regenerated from a fresh k-means fit, so re-running this script
    would otherwise silently discard the analyst's calls at those points. The
    polygons are the deliverable, but a filled zone records a judgement made at
    a location and is not reproducible.

    Inputs: path - Path to a labeling_zones GeoPackage
    Outputs: int, number of sites with class_code set
    """
    if not path.exists():
        return 0
    try:
        gdf = gpd.read_file(path)
        return int(gdf["class_code"].notna().sum()) if "class_code" in gdf.columns else 0
    except Exception:
        return 0


def drawn_polygon_count(path):
    """How many features a training_polygons GeoPackage already holds.

    Used to refuse overwriting hand-drawn work with the empty template. A
    missing or unreadable file counts as zero, so a first run is unaffected.

    Inputs: path - Path to the training_polygons GeoPackage
    Outputs: int, number of features present
    """
    if not path.exists():
        return 0
    try:
        return len(gpd.read_file(path))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    # deliberately TWO flags, not one. Zones and polygons hold different work at
    # very different cost: refitting clusters is a routine reason to regenerate
    # zones, while overwriting polygons destroys the deliverable. A single
    # --force would make the cheap action silently do the expensive one.
    ap.add_argument(
        "--overwrite-zones",
        action="store_true",
        help="regenerate labeling_zones even where candidate sites are already filled - loses those class_code values",
    )
    ap.add_argument(
        "--overwrite-polygons",
        action="store_true",
        help="overwrite training_polygons that already contain drawn polygons - THIS DESTROYS HAND LABELS",
    )
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    s = config["stage2_1_labeling_zones"]
    site, year = config["site"], config["year"]
    seed = config["BASE_SEED"] + year
    drop = set(s["drop_collinear_bands"])

    feat_dir = resolve_config_path(config["results_root"], "stage1_data_and_features", "features")
    out_dir = resolve_config_path(config["results_root"], "stage2_labeling")
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = list(config["tiles"])
    print(f"fitting k={s['k']} over {len(tiles)} tiles, seed={seed}")
    scaler, pca, km, kept_names, X = fit_kmeans_model(config, feat_dir, tiles, drop, seed)
    print(f"features kept ({len(kept_names)}): {', '.join(kept_names)}")
    print(f"PCA components at {s['pca_variance']:.1%} variance: {pca.n_components_}\n")

    rng = np.random.default_rng(seed)
    all_zones, cluster_px = [], np.zeros(s["k"] + 1, dtype=np.int64)
    cluster_sums = np.zeros((s["k"] + 1, len(kept_names)))

    for tile in tiles:
        flat, valid, _, arr, names, profile, transform, crs = load_tile(feat_dir, site, tile, year, drop)
        labels = np.zeros(flat.shape[0], dtype=np.uint8)
        Z = pca.transform(scaler.transform(flat[valid]))
        labels[valid] = km.predict(Z).astype(np.uint8) + 1  # 0 reserved for nodata
        lab2d = labels.reshape(profile["height"], profile["width"])

        lab_profile = dict(profile, dtype="uint8", count=1, nodata=0, compress="deflate")
        with rasterio.open(out_dir / f"cluster_map_{site}_{tile}_{year}.tif", "w", **lab_profile) as ds:
            ds.write(lab2d, 1)

        rows = []
        role = config["tiles"][tile]
        n_role_tiles = sum(1 for t, r in config["tiles"].items() if r == role)
        # budget is per cluster PER ROLE, split across that role's tiles: the
        # classifier only ever trains on train-block labels, so train coverage
        # must be guaranteed on its own rather than as a side effect of pooling
        per_tile_budget = max(1, round(s["sites_per_cluster_per_role"] / n_role_tiles))
        levels = [(lv["window"], lv["min_fraction"]) for lv in s["interior_levels"]]
        for c in range(1, s["k"] + 1):
            sel = lab2d == c
            cluster_px[c] += int(sel.sum())
            if sel.any():
                cluster_sums[c] += flat[sel.ravel() & valid].sum(axis=0)
            if not sel.any():
                continue
            inner, level_used = interior_mask_relaxed(lab2d, c, levels)
            for y, x in sample_sites(inner, per_tile_budget, s["min_separation_px"], rng):
                east, north = rasterio.transform.xy(transform, y, x)
                # class_code left empty: the analyst fills 0 bare, 1 grass, 2 shrub, 3 tree
                rows.append(
                    {
                        "cluster_id": c,
                        "tile": tile,
                        "tile_role": role,
                        "interior_level": level_used,
                        "row": y,
                        "col": x,
                        "class_code": None,
                        "geometry": Point(east, north),
                    }
                )
        gdf = gpd.GeoDataFrame(rows, crs=crs)
        gdf["class_code"] = gdf["class_code"].astype("Int64")
        # never overwrite zones that already carry analyst calls - a refit
        # renumbers clusters and regenerates sites, so the filled class_code
        # values would be lost with no way to recover them
        zone_path = out_dir / f"labeling_zones_{site}_{tile}_{year}.gpkg"
        filled = filled_zone_count(zone_path)
        if filled > 0 and not args.overwrite_zones:
            print(f"[{tile}] KEEPING {zone_path.name} - it holds {filled} filled candidate site(s). Pass --overwrite-zones to regenerate and lose them.")
        else:
            gdf.to_file(zone_path, driver="GPKG")

        # empty polygon layer with the right schema, so the analyst draws into a
        # file that already matches what Step 1d will read
        template = gpd.GeoDataFrame(
            {
                "class_code": np.array([], dtype="int64"),
                "cluster_id": np.array([], dtype="int64"),
                "tile": np.array([], dtype=object),
                "geometry": [],
            },
            crs=crs,
        )
        # geometry_type must be declared: an empty layer otherwise lands as
        # "Unknown", and QGIS will not start a polygon edit session on it
        # never overwrite drawn work: this template is empty by construction, so
        # writing it over a file that already holds polygons destroys hand labels
        # outright. Re-running 2a to refresh zones is a normal thing to want; losing
        # a week of labeling to it is not.
        poly_path = out_dir / f"training_polygons_{site}_{tile}_{year}.gpkg"
        if drawn_polygon_count(poly_path) > 0 and not args.overwrite_polygons:
            print(f"[{tile}] KEEPING {poly_path.name} - it holds {drawn_polygon_count(poly_path)} drawn polygon(s). Pass --overwrite-polygons to overwrite and lose them.")
        else:
            template.to_file(poly_path, driver="GPKG", geometry_type="Polygon")

        covered = gdf.cluster_id.nunique()
        print(f"[{tile}] {len(gdf):>4} candidate sites over {covered}/{s['k']} clusters ({config['tiles'][tile]})")
        all_zones.append(gdf)

    # per-cluster feature means, so the analyst can see what each cluster is
    profile_rows = []
    for c in range(1, s["k"] + 1):
        if cluster_px[c] == 0:
            continue
        means = cluster_sums[c] / cluster_px[c]
        entry = {
            "cluster_id": c,
            "pixels": int(cluster_px[c]),
            "pct_of_site": round(100.0 * cluster_px[c] / cluster_px.sum(), 3),
        }
        entry.update({n: round(float(v), 4) for n, v in zip(kept_names, means)})
        profile_rows.append(entry)

    spec = {
        "site": site,
        "year": year,
        "k": s["k"],
        "seed": seed,
        "features_used": kept_names,
        "dropped_collinear": sorted(drop),
        "pca_components": int(pca.n_components_),
        "pca_variance": s["pca_variance"],
        "distance": "Euclidean on PCA-whitened components == Mahalanobis on the retained subspace",
        "sites_per_cluster_per_role": s["sites_per_cluster_per_role"],
        "interior_levels": s["interior_levels"],
        "min_polygon_area_m2": s["min_polygon_area_m2"],
        "class_codes": {"0": "bare", "1": "grass", "2": "shrub", "3": "tree"},
        "analyst_note": ("Fill class_code in labeling_zones (0-3) or draw polygons into training_polygons. cluster_id is provenance only and must never be mapped to a class. Every cluster must receive labels, and the per-class minimum still applies: >= 50 polygons per class per tile role."),
        "cluster_profiles": profile_rows,
    }
    (out_dir / f"labeling_zones_{site}_{year}_spec.json").write_text(json.dumps(spec, indent=2) + "\n")

    total = sum(len(g) for g in all_zones)
    print(f"\ntotal candidate sites: {total}")
    print(f"wrote {out_dir / f'labeling_zones_{site}_{year}_spec.json'}")


if __name__ == "__main__":
    main()
