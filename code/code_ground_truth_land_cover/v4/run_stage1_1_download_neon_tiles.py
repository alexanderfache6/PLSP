#!/usr/bin/env python3
"""Download NEON AOP tiles for a site and place them where the pipeline expects.

Wraps neonutilities.by_tile_aop, which fetches individual 1 km AOP tiles rather
than a whole site - the difference between a few hundred MB and a few hundred GB.

    https://www.neonscience.org/resources/learning-hub/tutorials/download-explore-neon-data
    https://www.neonscience.org/resources/learning-hub/tutorials/neon-api-tokens-tutorial

TWO DETAILS THIS SCRIPT EXISTS TO HANDLE.

1. by_tile_aop takes COORDINATES, not tile ids. A tile id such as
   511000_3527000 names the tile's southwest CORNER, and a corner is shared by
   four tiles - passing it exactly is ambiguous. This script passes the tile
   CENTRE (corner + 500 m), which lands unambiguously inside the intended tile.

2. by_tile_aop writes its own directory tree under savepath
   (dpid/neon-aop-products/year/FullSite/...). The pipeline config expects the
   Data Portal bulk-download layout instead - NEON_struct-ecosystem/
   NEON.D14.SRER.DP3.30015.001.2022-08.basic/... - so downloaded files are
   located by name and placed into the config layout afterwards. Placement is
   done by globbing for the expected filenames rather than by assuming the
   package's tree, so a change in neonutilities' layout does not break this.

THREE STAGES, run in order by a single invocation:

    1  LIST     every tile actually flown at the site, from the NEON API. The
                AOP footprint follows flight lines rather than a rectangle, so a
                tile id inside the bounding box may never have been flown.
    2  FILTER   intersect those tiles against a site boundary (--aoi), report
                the count and the ids, and drop tiles whose overlap with the
                boundary is below --min-coverage. Edge tiles clipped by the
                boundary carry partial data and are usually not wanted.
    3  DOWNLOAD fetch the surviving tiles and place them in the config layout.

Usage:

# 1 to list all tiles
python run_stage1_1_download_neon_tiles.py --site SRER --year 2022 --config config/srer_2022.json --list-available-tiles
# 2 to list tile that intersect with site
python run_stage1_1_download_neon_tiles.py --site SRER --year 2022 --config config/srer_2022.json --tiles-from config/srer_2022.json --aoi-filter ~/Dropbox/planet/data/planet/geojson/Santa_Rita_Experimental_Range_NEON.geojson
# 2b add tiles to config/srer_2022.json
# 3 to download all tiles that intersect with site
python run_stage1_1_download_neon_tiles.py --site SRER --year 2022 --config config/srer_2022.json --tiles-from config/srer_2022.json --products chm rgb vi
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# the three products the pipeline uses, restricted deliberately to what a
# non-NEON site can mimic - see instructions5.md section 2A
PRODUCTS = {
    "rgb": {"dpid": "DP3.30010.001", "label": "camera ortho-mosaic, 10 cm"},
    "vi": {"dpid": "DP3.30026.002", "label": "spectrometer vegetation indices, 1 m"},
    "chm": {"dpid": "DP3.30015.001", "label": "ecosystem structure / CHM, 1 m"},
}
TILE_SIZE_M = 1000
API = "https://data.neonscience.org/api/v0"


def list_available_tiles(site, year, token):
    """Every AOP tile id actually flown at a site, from the CHM product listing.

    The AOP footprint is not a full rectangle - flight lines follow the site
    boundary - so a tile id cannot be assumed to exist just because it falls
    inside the bounding box. Listing before selecting avoids requesting tiles
    that were never flown.

    CHM is used as the index product because it is the smallest of the three and
    is tiled on the same 1 km grid as the others.

    Inputs: site - NEON site code; year - acquisition year; token - API token
    Outputs: (sorted list of tile id strings, yearMonth string used)
    """
    import re
    import urllib.request

    product = PRODUCTS["chm"]["dpid"]
    months = []
    with urllib.request.urlopen(f"{API}/products/{product}", timeout=60) as response:
        for entry in json.load(response)["data"]["siteCodes"]:
            if entry["siteCode"] == site:
                months = [m for m in entry["availableMonths"] if m.startswith(str(year))]
    if not months:
        return [], None

    tiles = set()
    for month in months:
        request = urllib.request.Request(f"{API}/data/{product}/{site}/{month}", headers={"X-API-Token": token})
        with urllib.request.urlopen(request, timeout=120) as response:
            for entry in json.load(response)["data"]["files"]:
                found = re.search(r"_(\d{6})_(\d{7})_CHM\.tif$", entry["name"])
                if found:
                    tiles.add(f"{found.group(1)}_{found.group(2)}")
    return sorted(tiles), months[0]


def filter_tiles_by_aoi(tiles, aoi_path, tile_crs, min_coverage=0.5):
    """Keep only tiles that overlap a site boundary, with their coverage fraction.

    Each NEON tile id names its southwest corner, so the tile is the 1 km square
    with that corner. The boundary is reprojected to the tile CRS rather than
    the reverse: tile geometry is exact in projected metres and would be
    distorted by reprojection to geographic coordinates.

    Coverage is reported because edge tiles clipped by the boundary carry
    partial data - a tile 8% inside the site is mostly nodata and is rarely
    worth downloading or labelling.

    Inputs: tiles - list of tile id strings; aoi_path - Path to a vector file;
            tile_crs - CRS of the tile grid, e.g. "EPSG:32612"; min_coverage -
            float 0-1, the minimum share of a tile inside the boundary
    Outputs: (list of (tile_id, coverage) kept, list of (tile_id, coverage) dropped)
    """
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.validation import make_valid

    aoi = gpd.read_file(aoi_path).to_crs(tile_crs)
    boundary = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    # site boundaries are often hand-digitised and self-intersecting; an invalid
    # polygon makes shapely return NaN areas rather than raising, which would
    # silently drop tiles
    if not boundary.is_valid:
        boundary = make_valid(boundary)

        # cheap bounding-box reject first, purely for speed: most candidates in a
        # generated grid are nowhere near the site, and a bounds comparison is far
        # cheaper than a GEOS predicate.
        #
        # Note: GEOS emits "invalid value encountered in intersection" RuntimeWarnings
        # on boundary tiles for rotated site polygons like SRER's. Checked against
        # computed areas - the values are CORRECT (verified 2026-08), so the warning
        # is noise, not a wrong coverage fraction. Do not "fix" it by dropping tiles
        # that warn.
    left, bottom, right, top = boundary.bounds

    kept, dropped = [], []
    for tile_id in tiles:
        easting, northing = (int(v) for v in tile_id.split("_"))
        if easting + TILE_SIZE_M <= left or easting >= right or northing + TILE_SIZE_M <= bottom or northing >= top:
            continue
        square = box(easting, northing, easting + TILE_SIZE_M, northing + TILE_SIZE_M)
        if not square.intersects(boundary):
            continue
        coverage = square.intersection(boundary).area / square.area
        (kept if coverage >= min_coverage else dropped).append((tile_id, coverage))
    return kept, dropped


def product_files(dpid, site, year, token):
    """Every file NEON has for a product at a site in a year, with signed URLs.

    Inputs: dpid - product code; site; year; token - API token
    Outputs: (list of file dicts with name/url/size, yearMonth string)
    """
    import urllib.request

    months = []
    with urllib.request.urlopen(f"{API}/products/{dpid}", timeout=60) as response:
        for entry in json.load(response)["data"]["siteCodes"]:
            if entry["siteCode"] == site:
                months = [m for m in entry["availableMonths"] if m.startswith(str(year))]
    files = []
    for month in months:
        request = urllib.request.Request(f"{API}/data/{dpid}/{site}/{month}", headers={"X-API-Token": token})
        with urllib.request.urlopen(request, timeout=120) as response:
            files.extend(json.load(response)["data"]["files"])
    return files, (months[0] if months else None)


def fetch(url, target, expected_size):
    """Stream one file to its final path, atomically.

    Written to a sibling temp file and renamed on success, so an interrupted
    transfer never leaves a partial file where the pipeline's existence checks
    would count it as present.

    NOTE: this replaces neonutilities.by_tile_aop, which cannot write these
    files. NEON now returns Google-signed URLs carrying ~900-character query
    strings, and by_tile_aop derives the local filename from the whole URL - so
    every write exceeds the 255-byte filename limit and fails. Verified
    2026-08: 66/66 files failed that way while reporting only "could not be
    downloaded". The URLs themselves are fine; only that filename derivation is
    broken, so fetching them directly works.

    Inputs: url - signed URL; target - Path to write; expected_size - int bytes
            from the API listing, or None
    Outputs: True when the file landed at its expected size
    """
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=600) as response,
        open(temporary, "wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1 << 20)
    if expected_size and temporary.stat().st_size != expected_size:
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(target)
    return True


def fetch_tile_bundle(listing, tile, dest_dir):
    """Fetch every file a product has for one tile into a destination directory.

    Used for the vegetation-index product, which does not expose its per-index
    GeoTIFFs individually the way RGB and CHM do - the tile arrives as a bundle,
    and may be a zip. Matching on the tile id rather than on an exact filename
    means both shapes work without the caller knowing which.

    A zip is extracted flat into dest_dir: NEON nests the indices inside a
    folder that repeats the directory name, and the pipeline config expects the
    tifs directly under the VegIndices directory, not one level deeper.

    Inputs: listing - the product's file dicts; tile - tile id; dest_dir - Path
    Outputs: (placed int, failed int)
    """
    import tempfile
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = [entry for entry in listing if tile in entry["name"]]
    placed = failed = 0
    for entry in entries:
        name = entry["name"]
        if name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as staging:
                archive = Path(staging) / "bundle.zip"
                try:
                    if not fetch(entry["url"], archive, entry.get("size")):
                        failed += 1
                        continue
                    with zipfile.ZipFile(archive) as zf:
                        for member in zf.namelist():
                            if member.endswith("/"):
                                continue
                            target = dest_dir / Path(member).name
                            if target.exists():
                                continue
                            with (
                                zf.open(member) as source,
                                open(target, "wb") as handle,
                            ):
                                shutil.copyfileobj(source, handle, length=1 << 20)
                            placed += 1
                except Exception as exc:
                    print(f"FAILED {name}: {exc}")
                    failed += 1
            continue
        target = dest_dir / name
        if target.exists():
            continue
        try:
            placed += 1 if fetch(entry["url"], target, entry.get("size")) else 0
        except Exception as exc:
            print(f"FAILED {name}: {exc}")
            failed += 1
    return placed, failed


def prune_unused_indices(site_config, site_dir, tiles):
    """Delete vegetation indices the pipeline never reads.

    The NEON VI zip ships five indices; the pipeline uses SAVI, NDVI and EVI
    (instructions5.md 2A). ARVI and PRI are dead weight - across ~70 tiles they
    are gigabytes of files nothing opens.

    Matching is anchored to the exact index name with an optional _error suffix
    and an optional .aux.xml companion, so a partial name can never take a file
    that is still in use. Runs on every invocation, not only after a download,
    so tiles fetched before this existed are cleaned up too.

    Inputs: site_config - parsed config; site_dir - Path to the site data root;
            tiles - iterable of tile ids
    Outputs: (removed int, bytes_freed int)
    """
    import re

    product = site_config["products"]["vi"]
    drop = product.get("drop_indices") or []
    if not drop:
        return 0, 0
    pattern = re.compile(r"_(?:" + "|".join(re.escape(name) for name in drop) + r")(?:_error)?\.tif(?:\.aux\.xml)?$")

    removed = freed = 0
    for tile in tiles:
        folder = (site_dir / product["folder"] / product["pattern"].format(tile=tile, index="SAVI")).parent
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and pattern.search(path.name):
                freed += path.stat().st_size
                path.unlink()
                removed += 1
    return removed, freed


def parse_tile(tile_id):
    """Tile centre coordinates from a NEON tile id.

    The id names the tile's southwest corner, and a corner is shared by four
    tiles, so the corner alone does not identify one. The centre does.

    Inputs: tile_id - string "{easting}_{northing}"
    Outputs: (easting int, northing int) of the tile centre
    """
    easting, northing = tile_id.split("_")
    return int(easting) + TILE_SIZE_M // 2, int(northing) + TILE_SIZE_M // 2


def expected_paths(site_config, site_dir, tile_id, product_key):
    """Where the pipeline expects a product's files for one tile.

    Read from the config's product block so this script and the pipeline can
    never disagree about a path. The vegetation-index entry names a directory of
    per-index files, so its pattern is expanded once per index.

    Inputs: site_config - parsed site config; site_dir - Path to the site data root;
            tile_id - tile key; product_key - "rgb" | "vi" | "chm"
    Outputs: list of Paths
    """
    product = site_config["products"][product_key]
    folder = site_dir / product["folder"]
    if product_key == "vi":
        return [folder / product["pattern"].format(tile=tile_id, index=index) for index in product["indices"]]
    return [folder / product["pattern"].format(tile=tile_id)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="NEON site code, e.g. SRER")
    ap.add_argument("--year", required=True, help="acquisition year, e.g. 2022")
    ap.add_argument("--tiles-to-download", nargs="+", help="tile ids, e.g. 511000_3527000")
    ap.add_argument(
        "--tiles-from",
        type=Path,
        help="read the tile list from a site config instead of --tiles",
    )
    ap.add_argument(
        "--products",
        nargs="+",
        default=["rgb", "vi", "chm"],
        choices=sorted(PRODUCTS),
        help="which products to fetch",
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="site config, used for the data root and the expected file layout; not needed with --list-tiles",
    )
    ap.add_argument(
        "--list-available-tiles",
        action="store_true",
        help="stage 1 only: list every tile flown at the site and exit",
    )
    ap.add_argument(
        "--aoi-filter",
        type=Path,
        help="stage 2: site boundary (geojson/gpkg). Tiles are derived by intersecting the flown tiles with it, instead of being passed with --tiles",
    )
    args = ap.parse_args()

    if not args.config and not args.list_available_tiles:
        print("--config is required unless you are using --list-tiles")
        return 1

    token = os.environ.get("NEON_DATA_API_TOKEN")

    if args.list_available_tiles:
        tiles, month = list_available_tiles(args.site, args.year, token)
        eastings = sorted({int(t.split("_")[0]) for t in tiles})
        northings = sorted({int(t.split("_")[1]) for t in tiles})
        print(f"{args.site} {args.year} ({month}): {len(tiles)} tiles flown")
        print(f"easting {eastings[0]} to {eastings[-1]} ({len(eastings)} columns)")
        print(f"northing {northings[0]} to {northings[-1]} ({len(northings)} rows)")
        print()
        for tile in tiles:
            print(tile)
        return 0

    site_config = json.loads(args.config.read_text())
    site_dir = Path(str(site_config["data_root"])).expanduser() / site_config["site_name"]
    print(f"NEON AOP tiles - {args.site} {args.year}")
    print("=" * 70)

    tiles = args.tiles_to_download
    if not tiles and args.tiles_from:
        tiles = list(json.loads(args.tiles_from.read_text())["tiles"])

    if args.aoi_filter:
        print(f"\nstage 1 - listing tiles flown at {args.site} {args.year}")
        flown, month = list_available_tiles(args.site, args.year, token)
        if not flown:
            print(f"no {args.site} AOP tiles found for {args.year}")
            return 1
        print(f"{len(flown)} tiles flown ({month})")

        tile_crs = site_config.get("expected_crs")
        if not tile_crs:
            print("config has no expected_crs - needed to place the tile grid against the boundary")
            return 1
        print(f"stage 2 - intersecting with {args.aoi_filter.name} in {tile_crs}")
        kept, dropped = filter_tiles_by_aoi(flown, args.aoi_filter, tile_crs)
        full = [t for t, c in kept if c >= 0.999]
        print(f"{len(kept) + len(dropped)} tiles intersect the boundary")
        print(f"{len(kept)} kept ({len(full)} fully inside, {len(kept) - len(full)} partial), {len(dropped)} dropped")
        for tile_id, coverage in sorted(kept):
            marker = "" if coverage >= 0.999 else f"{coverage:.0%} inside"
            already = " [have]" if all(p.exists() for key in args.products for p in expected_paths(site_config, site_dir, tile_id, key)) else ""
            print(f"  {tile_id}{marker}{already}")
        if dropped:
            print(f"dropped: {', '.join(f'{t} ({c:.0%})' for t, c in sorted(dropped))}")
        aoi_report = {
            "flown": len(flown),
            "intersecting": len(kept) + len(dropped),
            "kept": len(kept),
            "dropped": len(dropped),
        }
        tiles = [t for t, _ in sorted(kept)]

        print(aoi_report)

    if not tiles:
        print("no tiles given - pass --tiles, --tiles-from, or --aoi")
        return 1

    print("\n" + "=" * 70)
    print(f"stage 3 - download    {args.site} {args.year}")
    print(f"tiles     {len(tiles)}")
    print(f"target    {site_dir}")

    # only fetch what is actually missing - re-running should be cheap and safe
    todo = {}
    for key in args.products:
        missing_tiles = []
        for tile in tiles:
            if not all(path.exists() for path in expected_paths(site_config, site_dir, tile, key)):
                missing_tiles.append(tile)
        todo[key] = missing_tiles
        have = len(tiles) - len(missing_tiles)
        print(f"{key:<4} {PRODUCTS[key]['dpid']}  {have}/{len(tiles)} already present, {len(missing_tiles)} to fetch")

    if not any(todo.values()):
        print("\nnothing to do - every requested tile is already in place")
    else:
        total_placed = 0
        for key in args.products:
            missing_tiles = todo[key]
            if not missing_tiles:
                continue
            dpid = PRODUCTS[key]["dpid"]
            print(f"\n--- {key}  {dpid}  ({PRODUCTS[key]['label']}) ---")

            listing, month = product_files(dpid, args.site, args.year, token)
            if not listing:
                print(f"no files returned for {dpid} {args.site} {args.year}")
                continue
            by_name = {entry["name"]: entry for entry in listing}
            print(f"{len(listing)} files available ({month}), fetching {len(missing_tiles)} tile(s)")

            placed = failed = absent = 0
            for tile in missing_tiles:
                # the vegetation-index product arrives as a per-tile bundle rather
                # than individually listed tifs, so it is fetched by tile
                if key == "vi":
                    got, bad = fetch_tile_bundle(
                        listing,
                        tile,
                        expected_paths(site_config, site_dir, tile, key)[0].parent,
                    )
                    placed += got
                    failed += bad
                    if got == 0 and bad == 0:
                        print(f"NOTHING MATCHED tile {tile} in the {dpid} listing")
                        absent += 1
                    continue
                for target in expected_paths(site_config, site_dir, tile, key):
                    if target.exists():
                        continue
                    entry = by_name.get(target.name)
                    if entry is None:
                        print(f"NOT IN LISTING: {target.name}")
                        absent += 1
                        continue
                    try:
                        ok = fetch(entry["url"], target, entry.get("size"))
                    except Exception as exc:
                        print(f"FAILED {target.name}: {exc}")
                        failed += 1
                        continue
                    if ok:
                        placed += 1
                    else:
                        print(f"SIZE MISMATCH {target.name}")
                        failed += 1
                print(f"{tile}: {placed} placed so far", end="\r")
            total_placed += placed
            print(f"placed {placed}, failed {failed}, not in listing {absent}" + " " * 20)

            # always, not only after a download: tiles fetched before this existed still
            # carry the unused indices
    if "vi" in args.products:
        print("pruning")
        removed, freed = prune_unused_indices(site_config, site_dir, tiles)
        if removed:
            print(f"\npruned {removed} unused index file(s) ({', '.join(site_config['products']['vi'].get('drop_indices', []))}), freed {freed / 1e6:.0f} MB")

    print("\n" + "=" * 70)
    print("verifying against the expected layout")
    incomplete = 0
    for key in args.products:
        for tile in tiles:
            absent = [p for p in expected_paths(site_config, site_dir, tile, key) if not p.exists()]
            if absent:
                incomplete += 1
                print(f"{key} {tile}: {len(absent)} file(s) still missing, first is {absent[0].name}")
    if incomplete:
        print(f"{incomplete} product/tile combination(s) incomplete - check the names above against the config patterns")
        print("a name mismatch here usually means the config pattern is wrong, not that the download failed")
        return 1
    print("all requested tiles present in the expected layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
