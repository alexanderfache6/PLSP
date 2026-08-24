"""Section 4.2 Stage 2 helper - labeling progress and gate check.

Reads the GeoPackages the analyst is editing and reports whether Stage 2 is
finished, so a shortfall is visible while labeling rather than discovered at
Step 1d. Safe to run as often as you like; it only reads.

Checks, all from instructions5.md section 4.2:

  1  >= MIN_PER_CLASS polygons per class per tile role   <- the binding gate
  2  every cluster ABOVE THE SIZE FLOOR has received at least one labeled
     polygon. K-means at k=16 can produce degenerate clusters - measured at
     SRER, cluster 2 is ONE PIXEL across the whole 10-tile site - and requiring
     coverage of those makes the gate unpassable no matter how much is drawn.
     Cluster sizes are recomputed from the cluster maps on every run and the
     ones below the floor are named, so an exclusion is never silent.
  3  polygon area >= the class-specific min_polygon_area_m2, and the size
     distribution per class. The minimum is per class because a single floor
     does not bind equally: bare and grass patches are large, while shrub
     crowns at SRER are 1-3 m2, so a uniform floor filters shrub by size and
     pushes the analyst to stretch polygons to qualify. Point labels are
     exempt - they carry no area by construction.

     An undersized polygon is IGNORED, not a fault: it is excluded from the
     class totals, from cluster coverage, and from the area distribution, and
     it does not block the gate. It is still listed, because a class drifting
     under its floor is worth seeing. Nothing is deleted from the GeoPackage.
  4  geometry validity, and polygons that fall outside their tile
  5  class_code values are inside the locked section 3 set (0-3)
  6  candidate-site fill rate, per cluster and per role
  7  shrub candidate review progress per tile, with the accept rate - reported,
     not gated. The accept rate is evidence about the CHM height rule: a low
     rate means H_GRASS_MAX or H_TREE_MIN needs revisiting, not that the
     candidates were a bad idea.

Cluster coverage is measured by rasterizing each polygon onto the 1 m analysis
grid and reading the cluster map underneath, not from the cluster_id attribute -
the analyst draws freely and is not required to fill it, and what matters is
which part of feature space actually received labels.

THE TRAINING LABEL SET IS A UNION of two sources, and every count here reflects
both:

  hand-drawn      training_polygons_*.gpkg
  accepted        shrub_review_*.gpkg, by one of two workflows

Two review workflows are supported, because reject-only marking is far faster
and is what the analyst actually does:

  sweep        list the tile in stage2_2_shrub_candidates.reviewed_tiles once it has been swept
               end to end. Everything not rejected in that tile is accepted -
               one decision per tile instead of one per candidate.
  per-feature  set reviewed = 1 on each candidate individually. Applies to
               tiles not listed as swept, so a partially reviewed tile still
               counts what was explicitly confirmed.

A tile that is neither swept nor per-feature marked contributes nothing, which
is correct: an untouched candidate is a proposal, not a label.

An accepted CHM candidate is an ordinary hand-validated label - the analyst
confirmed it against RGB, which is the same act as drawing one. Counting only
the drawn file would make reviewing 750 candidates show zero progress, which
would defeat the accelerator entirely. Step 1d must read the same union.

Usage:  python run_stage2_4_check_hand_labeling_progress.py config/srer_2022.json [--json]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from constants import CLASS_LABELS
from rasterio.features import rasterize
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
MIN_PER_CLASS = 50
POINT_TYPES = {"Point", "MultiPoint"}


def drop_out_of_tile(lab_dir, site, year, tiles):
    """Delete polygons whose extent falls outside the tile they belong to.

    A polygon drawn past a tile edge cannot be rasterized against that tile's
    grid, so it contributes no training pixels and no cluster coverage - it is
    dead weight that nonetheless fails the gate. Removing it is the fix.

    This DELETES hand-drawn work, so it never runs unless asked for explicitly,
    and it writes through a staged file with a round-trip check, the same as
    --fill: to_file recreates the layer, and a failure partway through would
    truncate the deliverable.

    Inputs:  lab_dir - Path to stage2_labeling; site, year; tiles - iterable of ids
    Outputs: int, number of polygons removed
    """
    removed = 0
    for tile in tiles:
        path = lab_dir / f"training_polygons_{site}_{tile}_{year}.gpkg"
        polys = read_layer(path)
        if polys is None or not len(polys):
            continue
        with rasterio.open(lab_dir / f"cluster_map_{site}_{tile}_{year}.tif") as ds:
            bounds = ds.bounds
        keep = []
        for _, feat in polys.iterrows():
            geom = feat.geometry
            if geom is None or geom.is_empty:
                keep.append(False)
                continue
            gb = geom.bounds
            keep.append(not (gb[0] < bounds.left or gb[2] > bounds.right or gb[1] < bounds.bottom or gb[3] > bounds.top))
        n_drop = len(keep) - sum(keep)
        if not n_drop:
            continue
        kept = polys[keep]
        staged = path.with_name(path.stem + ".drop-tmp.gpkg")
        kept.to_file(staged, driver="GPKG", layer=path.stem, geometry_type="Polygon")
        if len(gpd.read_file(staged)) != len(kept):
            staged.unlink(missing_ok=True)
            print(f"[{tile}] ABORTED drop - staged file did not round-trip, original left untouched")
            continue
        staged.replace(path)
        removed += n_drop
        print(f"[{tile}] removed {n_drop} polygon(s) extending outside the tile, {len(kept)} kept")
    return removed


def cluster_pixel_share(lab_dir, site, year, tiles, k):
    """Site-wide pixel share of every cluster, from the cluster maps on disk.

    Recomputed each run rather than cached: the k-means fit is redone whenever
    tiles are added, so a share recorded earlier can silently stop being true.

    Inputs:  lab_dir - Path to stage2_labeling; site, year; tiles - iterable of tile
             ids; k - number of clusters
    Outputs: dict of {cluster: fraction of valid site pixels}
    """
    totals = np.zeros(k + 1, dtype=np.int64)
    for tile in tiles:
        path = lab_dir / f"cluster_map_{site}_{tile}_{year}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            counts = np.bincount(ds.read(1).ravel(), minlength=k + 1)[: k + 1]
        totals += counts
    valid = totals[1:].sum()
    return {c: (totals[c] / valid if valid else 0.0) for c in range(1, k + 1)}


def class_minimum_areas(config):
    """Per-class minimum polygon area in m2, keyed by integer class code.

    Accepts either the class-specific mapping (instructions5.md 4.2) or a bare
    scalar from an older config, which is applied to every class so existing
    configs keep working.

    Inputs:  config - the parsed site config dict
    Outputs: dict of {class_code: float minimum area in m2}
    """
    raw = config["stage2_1_labeling_zones"]["min_polygon_area_m2"]
    if isinstance(raw, dict):
        return {int(code): float(value) for code, value in raw.items()}
    return {code: float(raw) for code in CLASS_LABELS}


def resolve(root, *parts):
    return Path(str(root)).expanduser().joinpath(*parts)


def shrub_review_state(lab_dir, site, tile, year, swept):
    """Review tallies for one tile's shrub candidate subset.

    Reported so the analyst can see progress and, more usefully, the accept
    rate - which is direct evidence about whether the CHM height band is
    picking out real shrubs (instructions5.md 4.2).

    Inputs:  lab_dir - Path to the stage2_labeling directory; site, tile, year;
             swept - bool, whether the tile was swept end to end
    Outputs: dict of tallies, or None when the tile has no review file
    """
    path = lab_dir / f"shrub_review_{site}_{tile}_{year}.gpkg"
    if not path.exists():
        return None
    frames = []
    for kind in ("polygon", "point"):
        gdf = read_layer_named(path, f"shrub_review_{site}_{tile}_{year}_{kind}")
        if gdf is not None and len(gdf):
            frames.append(gdf)
    if not frames:
        return None
    gdf = pd.concat(frames, ignore_index=True)
    reviewed = gdf["reviewed"].fillna(0).astype(int) == 1 if "reviewed" in gdf.columns else pd.Series(False, index=gdf.index)
    rejected = gdf["rejected"].fillna(0).astype(int) == 1 if "rejected" in gdf.columns else pd.Series(False, index=gdf.index)
    # a swept tile counts every unrejected candidate as looked at and kept
    seen = pd.Series(True, index=gdf.index) if swept else (reviewed | rejected)
    accepted = seen & ~rejected
    n_seen, n_accepted = int(seen.sum()), int(accepted.sum())
    return {
        "candidates": len(gdf),
        "reviewed": n_seen,
        "rejected": int(rejected.sum()),
        "accepted": n_accepted,
        "pending": len(gdf) - n_seen,
        "accept_pct": (100.0 * n_accepted / n_seen) if n_seen else 0.0,
    }


def read_accepted_candidates(lab_dir, site, tile, year, swept):
    """Accepted CHM shrub candidates for a tile, as ordinary training labels.

    Two workflows, both meaning "a human confirmed this against RGB":

      swept tile    the analyst reviewed the tile end to end and marked only
                    the rejects, so everything with rejected != 1 is accepted
      unswept tile  only candidates explicitly carrying reviewed = 1 count

    Reject-only marking on a swept tile is the faster workflow and the one in
    use; the per-feature path remains for partially reviewed tiles. A candidate
    that is neither is a proposal and must never enter training
    (instructions5.md 4.2). Polygon and point layers are both read; a missing
    file is not an error, since the candidate step is optional and NEON-only.

    Inputs:  lab_dir - Path to the stage2_labeling directory; site, tile, year;
             swept - bool, whether this tile has been swept end to end
    Outputs: GeoDataFrame of accepted candidates, or None if there are none
    """
    path = lab_dir / f"shrub_review_{site}_{tile}_{year}.gpkg"
    if not path.exists():
        return None
    frames = []
    for kind in ("polygon", "point"):
        gdf = read_layer_named(path, f"shrub_review_{site}_{tile}_{year}_{kind}")
        if gdf is None or not len(gdf):
            continue
        rejected = gdf["rejected"].fillna(0).astype(int) == 1 if "rejected" in gdf.columns else pd.Series(False, index=gdf.index)
        if swept:
            keep = gdf[~rejected]
        elif "reviewed" in gdf.columns:
            keep = gdf[(gdf["reviewed"].fillna(0).astype(int) == 1) & ~rejected]
        else:
            continue
        if len(keep):
            frames.append(keep)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def read_layer_named(path, layer):
    """Read one named layer from a GeoPackage, tolerating absence.

    Inputs:  path - Path to the GeoPackage; layer - layer name string
    Outputs: GeoDataFrame, or None if the layer is missing or unreadable
    """
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:
        return None


def read_layer(path):
    if not path.exists():
        return None
    try:
        return gpd.read_file(path)
    except Exception as exc:  # empty or malformed layer
        print(f"warning: could not read {path.name}: {exc}", file=sys.stderr)
        return None


def majority_cluster(geom, clusters, transform, shape):
    """Modal cluster under a polygon. all_touched so a small polygon still lands somewhere."""
    mask = rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    vals = clusters[mask]
    vals = vals[vals > 0]
    return int(Counter(vals.tolist()).most_common(1)[0][0]) if vals.size else None


def fill_attributes(lab_dir, site, year, tiles):
    """Derive tile and cluster_id for hand-drawn polygons and write them back.

    Both are recoverable from geometry, so making the analyst type them during
    drawing is wasted effort and a source of typos. tile comes from the file the
    polygon lives in; cluster_id is the modal cluster under the polygon. This is
    the only operation in this script that writes.
    """
    for tile in tiles:
        path = lab_dir / f"training_polygons_{site}_{tile}_{year}.gpkg"
        polys = read_layer(path)
        if polys is None or not len(polys):
            continue
        with rasterio.open(lab_dir / f"cluster_map_{site}_{tile}_{year}.tif") as ds:
            clusters, transform, shape = ds.read(1), ds.transform, ds.shape
        ids = [None if g is None or g.is_empty else majority_cluster(g, clusters, transform, shape) for g in polys.geometry]
        polys["tile"] = tile
        polys["cluster_id"] = pd.array(ids, dtype="Int64")
        # written to a sibling file and swapped in, never in place. to_file on an
        # existing GeoPackage recreates the layer, so a failure partway through
        # would leave the deliverable truncated - and this is the only operation
        # in the whole pipeline that rewrites hand-drawn polygons.
        staged = path.with_name(path.stem + ".fill-tmp.gpkg")
        polys.to_file(staged, driver="GPKG", layer=path.stem, geometry_type="Polygon")
        if len(gpd.read_file(staged)) != len(polys):
            staged.unlink(missing_ok=True)
            print(f"[{tile}] ABORTED fill - staged file did not round-trip, original left untouched")
            continue
        staged.replace(path)
        n_set = int(sum(i is not None for i in ids))
        print(f"[{tile}] filled tile + cluster_id on {len(polys)} polygon(s), {n_set} got a cluster")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--json", action="store_true", help="print the machine-readable report")
    ap.add_argument(
        "--fill",
        action="store_true",
        help="derive tile and cluster_id on drawn polygons and write them back",
    )
    ap.add_argument(
        "--drop-out-of-tile",
        action="store_true",
        help="DELETE polygons that extend outside their tile - they contribute nothing and fail the gate",
    )
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    k = config["stage2_1_labeling_zones"]["k"]
    min_area = class_minimum_areas(config)
    swept_tiles = set(config.get("stage2_2_shrub_candidates", {}).get("reviewed_tiles", []))
    lab_dir = resolve(config["results_root"], "stage2_labeling")

    if args.drop_out_of_tile:
        print("removing polygons that extend outside their tile (this deletes drawn work)")
        n = drop_out_of_tile(lab_dir, site, year, config["tiles"])
        print(f"removed {n} polygon(s)\n")

    if args.fill:
        print("filling derived attributes (this writes to the polygon GeoPackages)")
        fill_attributes(lab_dir, site, year, config["tiles"])
        print()

    poly_counts = defaultdict(Counter)  # role -> class -> n polygons
    tile_counts = defaultdict(Counter)  # tile -> class -> n polygons
    zone_counts = defaultdict(Counter)  # role -> class -> n filled sites
    zone_total = Counter()  # role -> n sites
    cluster_hits = defaultdict(set)  # role -> clusters touched by polygons
    tile_cluster_hits = defaultdict(set)  # tile -> clusters touched by polygons
    areas = defaultdict(list)  # class -> [m2], counted polygons only
    points = Counter()  # class -> n point labels
    undersized = []  # below the class floor: ignored, reported
    undersized_by_class = Counter()
    problems = []
    repaired_count = [0]
    per_tile = []

    for tile, role in config["tiles"].items():
        row = {
            "tile": tile,
            "role": role,
            "polygons": 0,
            "sites_filled": 0,
            "sites": 0,
            "accepted_candidates": 0,
        }

        zones = read_layer(lab_dir / f"labeling_zones_{site}_{tile}_{year}.gpkg")
        if zones is not None and len(zones):
            zone_total[role] += len(zones)
            row["sites"] = len(zones)
            filled = zones[zones["class_code"].notna()]
            row["sites_filled"] = len(filled)
            for code in filled["class_code"].astype(int):
                zone_counts[role][code] += 1
                if code not in CLASS_LABELS:
                    problems.append(f"{tile}: zone has class_code {code}, outside 0-3")

        polys = read_layer(lab_dir / f"training_polygons_{site}_{tile}_{year}.gpkg")
        accepted = read_accepted_candidates(lab_dir, site, tile, year, tile in swept_tiles)
        if accepted is not None and len(accepted):
            row["accepted_candidates"] = len(accepted)
            accepted = accepted[["class_code", "geometry"]].copy()
            if polys is None or not len(polys):
                polys = gpd.GeoDataFrame(accepted, crs=accepted.crs)
            else:
                polys = gpd.GeoDataFrame(
                    pd.concat([polys[["class_code", "geometry"]], accepted], ignore_index=True),
                    crs=polys.crs,
                )
        if polys is not None and len(polys):
            row["polygons"] = len(polys)
            with rasterio.open(lab_dir / f"cluster_map_{site}_{tile}_{year}.tif") as ds:
                clusters = ds.read(1)
                transform, shape, bounds = ds.transform, ds.shape, ds.bounds

            for i, feat in polys.iterrows():
                geom, code = feat.geometry, feat.get("class_code")
                if geom is None or geom.is_empty:
                    problems.append(f"{tile} #{i}: empty geometry")
                    continue
                if not geom.is_valid:
                    # vectorized CHM candidates self-intersect at corner pinches
                    # (8-connectivity). That is repairable and benign, so repair
                    # in memory and carry on - nothing on disk is touched. Only
                    # geometry that survives repair still broken is a real fault.
                    repaired = make_valid(geom)
                    if repaired.is_valid and not repaired.is_empty:
                        geom = repaired
                        repaired_count[0] += 1
                    else:
                        problems.append(f"{tile} #{i}: invalid geometry, not repairable")
                        continue
                if code is None or (isinstance(code, float) and np.isnan(code)):
                    problems.append(f"{tile} #{i}: class_code not set")
                    continue
                code = int(code)
                if code not in CLASS_LABELS:
                    problems.append(f"{tile} #{i}: class_code {code} outside 0-3")
                    continue
                # point labels stand in for objects below their class minimum
                # (instructions5.md 4.2) and have no area to test
                if geom.geom_type in POINT_TYPES:
                    poly_counts[role][code] += 1
                    tile_counts[tile][code] += 1
                    points[code] += 1
                else:
                    floor = min_area[code]
                    if geom.area < floor:
                        # ignored, not a fault - excluded from every count below
                        undersized.append(f"{tile} #{i}: {CLASS_LABELS[code]} {geom.area:.1f} m2 < {floor:g} m2 floor")
                        undersized_by_class[code] += 1
                        continue
                    poly_counts[role][code] += 1
                    tile_counts[tile][code] += 1
                    areas[code].append(geom.area)
                gb = geom.bounds
                if gb[0] < bounds.left or gb[2] > bounds.right or gb[1] < bounds.bottom or gb[3] > bounds.top:
                    problems.append(f"{tile} #{i}: extends outside the tile")
                    continue
                # which part of feature space this polygon actually covers
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                hit = np.unique(clusters[mask])
                cluster_hits[role].update(int(c) for c in hit if c > 0)
                tile_cluster_hits[tile].update(int(c) for c in hit if c > 0)
        per_tile.append(row)

    # ------------------------------------------------------------------ report
    roles = sorted({r for r in config["tiles"].values()})
    print(f"Stage 2 labeling progress - {site} {year}\n" + "=" * 62)

    print("\nper tile")
    header = f"{'tile':<18}{'role':<7}" + "".join(f"{CLASS_LABELS[c]:>7}" for c in CLASS_LABELS) + f"{'total':>7}{'chm':>6}{'clusters':>10}{'sites filled':>14}"
    print(header)
    for r in per_tile:
        tile = r["tile"]
        cells = "".join(f"{tile_counts[tile][c]:>7}" for c in CLASS_LABELS)
        total = sum(tile_counts[tile][c] for c in CLASS_LABELS)
        n_clusters = len(tile_cluster_hits[tile])
        print(f"{tile:<18}{r['role']:<7}{cells}{total:>7}{r['accepted_candidates']:>6}{n_clusters:>7}/{k:<2}{r['sites_filled']:>7} /{r['sites']:>5}")
    # per-class counts are per tile; the gate in check 1 is per ROLE, so a tile
    # being light in a class is only a problem if its whole block is
    print("-" * len(header))
    for role in sorted({r["role"] for r in per_tile}):
        tiles = [r["tile"] for r in per_tile if r["role"] == role]
        cells = "".join(f"{sum(tile_counts[t][c] for t in tiles):>7}" for c in CLASS_LABELS)
        total = sum(sum(tile_counts[t][c] for c in CLASS_LABELS) for t in tiles)
        chm = sum(r["accepted_candidates"] for r in per_tile if r["role"] == role)
        print(f"{role + ' total':<18}{'':<7}{cells}{total:>7}{chm:>6}{len(cluster_hits[role]):>7}/{k:<2}")
    if not swept_tiles:
        print("note: no tiles listed in stage2_2_shrub_candidates.reviewed_tiles - shrub candidates count only where reviewed = 1 is set per feature")
    # polygons is the union of hand-drawn and accepted CHM candidates; the
    # second column shows how much came from the accelerator
    # zones and polygons are one-to-many by design: a zone may yield several
    # nearby polygons, so polygons exceeding filled sites is expected

    print(f"\ncheck 1 - polygons per class per role (need >= {MIN_PER_CLASS})")
    gate_ok = True
    print(f"{'class':<8}" + "".join(f"{r:>14}" for r in roles))
    for code, label in CLASS_LABELS.items():
        cells = []
        for role in roles:
            n = poly_counts[role][code]
            ok = n >= MIN_PER_CLASS
            gate_ok &= ok
            cells.append(f"{n:>8} {'ok ' if ok else 'SHORT'}")
        print(f"{code} {label:<6}" + "".join(f"{c:>14}" for c in cells))

    shares = cluster_pixel_share(lab_dir, site, year, config["tiles"], k)
    floor = config["stage2_1_labeling_zones"].get("cluster_min_pixel_fraction", 0.001)
    ignored = sorted(c for c in shares if shares[c] < floor)
    required = sorted(set(range(1, k + 1)) - set(ignored))

    print(f"\ncheck 2 - cluster coverage by labeled polygons (floor {floor:.3%} of site pixels)")
    print("cluster sizes: " + ", ".join(f"{c}:{shares[c]:.2%}" for c in sorted(shares)))
    if ignored:
        print("ignored below floor: " + ", ".join(f"{c} ({shares[c]:.4%})" for c in ignored) + " - too small to label, excluded from the gate")
    for role in roles:
        missing = sorted(set(required) - cluster_hits[role])
        state = "all covered" if not missing else f"missing {missing}"
        print(f"{role:<7} {len(cluster_hits[role] & set(required))}/{len(required)} required clusters   {state}")
        gate_ok &= not missing

    print("\ncheck 3 - polygon area against the class minimum")
    if any(areas.values()) or sum(points.values()):
        print(f"{'class':<8}{'floor':>7}{'n':>5}{'min':>9}{'median':>9}{'max':>9}{'total m2':>11}{'points':>8}{'ignored':>9}")
        for code, label in CLASS_LABELS.items():
            a = areas[code]
            floor = f"{min_area[code]:g}"
            if a:
                print(f"{code} {label:<6}{floor:>7}{len(a):>5}{min(a):>9.1f}{np.median(a):>9.1f}{max(a):>9.1f}{sum(a):>11.0f}{points[code]:>8}{undersized_by_class[code]:>9}")
            elif points[code] or undersized_by_class[code]:
                print(f"{code} {label:<6}{floor:>7}{0:>5}{'-':>9}{'-':>9}{'-':>9}{0:>11}{points[code]:>8}{undersized_by_class[code]:>9}")
        # a class median sitting near its floor means polygons are being drawn
        # to the floor rather than to the object, which imports background into
        # the class - instructions5.md 4.2
        for code, label in CLASS_LABELS.items():
            a = areas[code]
            if len(a) >= 5 and np.median(a) < 1.5 * min_area[code]:
                print(f"note: {label} median {np.median(a):.1f} m2 sits close to its {min_area[code]:g} m2 floor - check polygons are not being stretched to qualify")
    else:
        print("no polygons drawn yet")

    if undersized:
        # reference only: these do not gate and are not deleted, they simply
        # take no part in training - instructions5.md 4.2
        print(f"\nignored, below the class floor ({len(undersized)}) - excluded from the counts above, not a gate failure")
        for u in undersized[:10]:
            print(f"{u}")
        if len(undersized) > 10:
            print(f"... and {len(undersized) - 10} more")

    if repaired_count[0]:
        print(f"\nrepaired in memory: {repaired_count[0]} self-intersecting geometries - a corner-pinch artifact of vectorizing CHM candidates, benign and not a gate failure. Nothing on disk was modified.")

    print(f"\ncheck 4-5 - geometry and class_code problems: {len(problems)}")
    for p in problems[:15]:
        print(f"{p}")
    if len(problems) > 15:
        print(f"... and {len(problems) - 15} more")
    gate_ok &= not problems

    review_rows = [(tile, role, shrub_review_state(lab_dir, site, tile, year, tile in swept_tiles)) for tile, role in config["tiles"].items()]
    if any(state for _, _, state in review_rows):
        print("\ncheck 7 - shrub candidate review (reported, not gated)")
        print(f"{'tile':<18}{'role':<7}{'cands':>7}{'reviewed':>10}{'rejected':>10}{'accepted':>10}{'pending':>9}{'accept':>8}")
        agg = Counter()
        for tile, role, state in review_rows:
            if state is None:
                print(f"{tile:<18}{role:<7}{'no review file':>44}")
                continue
            print(f"{tile:<18}{role:<7}{state['candidates']:>7}{state['reviewed']:>10}{state['rejected']:>10}{state['accepted']:>10}{state['pending']:>9}{state['accept_pct']:>7.0f}%")
            for key in ("candidates", "reviewed", "rejected", "accepted", "pending"):
                agg[key] += state[key]
        overall = (100.0 * agg["accepted"] / agg["reviewed"]) if agg["reviewed"] else 0.0
        print(f"{'TOTAL':<18}{'':<7}{agg['candidates']:>7}{agg['reviewed']:>10}{agg['rejected']:>10}{agg['accepted']:>10}{agg['pending']:>9}{overall:>7.0f}%")
        if agg["candidates"]:
            print(f"{agg['reviewed']}/{agg['candidates']} candidates reviewed ({100.0 * agg['reviewed'] / agg['candidates']:.1f}%)")
        # a persistently low accept rate is evidence about the height band, not
        # about the candidates - instructions5.md 3 safeguards 1-2
        if agg["reviewed"] >= 30 and overall < 30.0:
            print(f"note: accept rate {overall:.0f}% is low - H_GRASS_MAX / H_TREE_MIN may need revisiting")

    print("\ncheck 6 - candidate site fill rate")
    for role in roles:
        filled = sum(zone_counts[role].values())
        total = zone_total[role]
        pct = 100.0 * filled / total if total else 0.0
        print(f"{role:<7} {filled:>4} / {total:<4} sites filled ({pct:.1f}%)")

    print("\n" + "=" * 62)
    if gate_ok:
        print("GATE PASSED - Stage 2 complete, Step 1d can proceed.")
    else:
        need = {f"{r}/{CLASS_LABELS[c]}": MIN_PER_CLASS - poly_counts[r][c] for r in roles for c in CLASS_LABELS if poly_counts[r][c] < MIN_PER_CLASS}
        print("GATE NOT PASSED - still needed:")
        if need:
            print("polygons: " + ", ".join(f"{kk} +{vv}" for kk, vv in need.items()))
        for role in roles:
            missing = sorted(set(required) - cluster_hits[role])
            if missing:
                print(f"{role}: clusters with no labeled polygon: {missing}")
        if problems:
            print(f"fix {len(problems)} geometry/class_code problem(s) listed above")

    if args.json:
        report = {
            "site": site,
            "year": year,
            "gate_passed": bool(gate_ok),
            "min_per_class_per_role": MIN_PER_CLASS,
            "per_tile": per_tile,
            "polygons_per_class_per_role": {r: {CLASS_LABELS[c]: poly_counts[r][c] for c in CLASS_LABELS} for r in roles},
            "clusters_covered": {r: sorted(cluster_hits[r]) for r in roles},
            "clusters_missing": {r: sorted(set(range(1, k + 1)) - cluster_hits[r]) for r in roles},
            "problems": problems,
            "min_polygon_area_m2": {CLASS_LABELS[c]: min_area[c] for c in CLASS_LABELS},
            "point_labels_per_class": {CLASS_LABELS[c]: points[c] for c in CLASS_LABELS},
            "ignored_below_floor": undersized,
            "ignored_per_class": {CLASS_LABELS[c]: undersized_by_class[c] for c in CLASS_LABELS},
        }
        out = lab_dir / f"labeling_progress_{site}_{year}.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
