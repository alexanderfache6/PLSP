#!/usr/bin/env python3
"""Find 3DEP lidar point cloud (LPC) tiles covering each selected site.

Sites come from results/01_selected_sites_short_2.csv; the bounding box of the
`coordinates` column (already WGS84 decimal lat/lon) is the area of interest.

This is the programmatic equivalent of https://apps.nationalmap.gov/downloader/
with
  Datasets -> Data -> Elevation Source Data (3DEP) - Lidar, IfSAR
           -> Subcategories -> Lidar Point Cloud (LPC)
           -> File Formats -> LAS,LAZ
  Area of Interest -> Enter Coords -> xmin / ymin / xmax / ymax
It queries the TNM Access API, which backs the downloader and needs no login.

The products API only reports a publication date, which lags collection by a
year or more, so the actual collection window is read from each project's
ScienceBase record (one lookup per project, not per tile).
"""

import argparse
import ast
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SITES_CSV = RESULTS_DIR / "01_selected_sites_short_2.csv"
TILES_CSV = RESULTS_DIR / "03_3dep_metadata.csv"
DATES_CSV = RESULTS_DIR / "03_3dep_data_availability.csv"

PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
SCIENCEBASE_URL = "https://www.sciencebase.gov/catalog/item/{item_id}"

# "Elevation Source Data (3DEP) - Lidar, IfSAR" -> "Lidar Point Cloud (LPC)"
DATASET = "Lidar Point Cloud (LPC)"
PROD_FORMATS = "LAS,LAZ"
PAGE_SIZE = 1000

# "USGS Lidar Point Cloud AZ_CochiseCounty_2020_B20 12R_WA_1025" -> project, tile
TITLE_RE = re.compile(r"^USGS Lidar Point Cloud\s+(.*?)\s+(\S+)$")


def get_json(url, params=None):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        sys.exit(f"request failed: HTTP {exc.code} {url}")
    except urllib.error.URLError as exc:
        sys.exit(f"request failed: {exc} {url}")


def read_sites(path):
    """Return [(site_full, site_id, (xmin, ymin, xmax, ymax))]."""
    sites = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("coordinates") or "").strip()
            if not raw:
                print(
                    f"warning: no coordinates for {row.get('site_full')}, skipping",
                    file=sys.stderr,
                )
                continue
            points = ast.literal_eval(raw)  # [[lat, lon], ...]
            lats = [float(lat) for lat, _ in points]
            lons = [float(lon) for _, lon in points]
            sites.append(
                (
                    (row.get("site_full") or "").strip(),
                    (row.get("site_id") or "").strip(),
                    (min(lons), min(lats), max(lons), max(lats)),
                )
            )
    return sites


def search_lpc(bbox, delay=0.2):
    """Return all LPC LAS/LAZ products intersecting the bounding box."""
    items = []
    offset = 0
    while True:
        data = get_json(
            PRODUCTS_URL,
            {
                "datasets": DATASET,
                "prodFormats": PROD_FORMATS,
                "bbox": ",".join(f"{v:.10f}" for v in bbox),
                "max": PAGE_SIZE,
                "offset": offset,
            },
        )
        page = data.get("items") or []
        items.extend(page)
        total = data.get("total") or 0
        if not page or len(items) >= total:
            break
        offset += len(page)
        time.sleep(delay)
    return items


def split_title(title):
    """Split a product title into (project, tile) names."""
    match = TITLE_RE.match(title or "")
    return (match.group(1), match.group(2)) if match else (title or "", "")


def collection_window(item_id, cache, delay=0.2):
    """Return (start, end) collection dates from the ScienceBase record."""
    if item_id in cache:
        return cache[item_id]
    data = get_json(
        SCIENCEBASE_URL.format(item_id=item_id), {"format": "json", "fields": "dates"}
    )
    start = end = ""
    for entry in data.get("dates") or []:
        if entry.get("type") == "Start":
            start = entry.get("dateString") or ""
        elif entry.get("type") == "End":
            end = entry.get("dateString") or ""
    cache[item_id] = (start, end)
    time.sleep(delay)
    return start, end


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    ap.add_argument("--tiles-csv", type=Path, default=TILES_CSV)
    ap.add_argument("--dates-csv", type=Path, default=DATES_CSV)
    ap.add_argument("--delay", type=float, default=0.2, help="pause between requests")
    args = ap.parse_args()

    sites = read_sites(args.sites_csv)
    if not sites:
        sys.exit(f"No sites with coordinates found in {args.sites_csv}")

    sb_cache = {}
    tile_rows = []
    date_rows = []

    for site_full, site_id, bbox in sites:
        items = search_lpc(bbox, args.delay)
        print(f"=== {site_id} ({site_full}) ===")
        print(f"  bbox xmin={bbox[0]:.6f} ymin={bbox[1]:.6f} xmax={bbox[2]:.6f} ymax={bbox[3]:.6f}")
        print(f"  {len(items)} LPC tiles")

        # one ScienceBase lookup per project, keyed off a representative tile
        projects = OrderedDict()
        for item in items:
            project, _ = split_title(item.get("title"))
            projects.setdefault(project, item.get("sourceId"))

        windows = {
            project: collection_window(source_id, sb_cache, args.delay)
            for project, source_id in projects.items()
            if source_id
        }

        for project in projects:
            start, end = windows.get(project, ("", ""))
            count = sum(1 for i in items if split_title(i.get("title"))[0] == project)
            print(f"  {project}: {count} tiles, collected {start or '?'} to {end or '?'}")
        print()

        for item in items:
            project, tile = split_title(item.get("title"))
            start, end = windows.get(project, ("", ""))
            box = item.get("boundingBox") or {}
            tile_rows.append(
                {
                    "site_id": site_id,
                    "project": project,
                    "tile": tile,
                    "title": item.get("title") or "",
                    "format": item.get("format") or "",
                    "collection_start": start,
                    "collection_end": end,
                    "publication_date": item.get("publicationDate") or "",
                    "size_bytes": item.get("sizeInBytes") or "",
                    "min_lon": box.get("minX", ""),
                    "min_lat": box.get("minY", ""),
                    "max_lon": box.get("maxX", ""),
                    "max_lat": box.get("maxY", ""),
                    "source_id": item.get("sourceId") or "",
                    "meta_url": item.get("metaUrl") or "",
                    "download_url": item.get("downloadURL") or "",
                }
            )

        dates = sorted(
            {w[0] for w in windows.values() if w[0]},
            reverse=True,
        )
        date_rows.append(
            {
                "site_id": site_id,
                "lidar_dates": "[" + ", ".join(dates) + "]",
                "lidar_projects": "[" + ", ".join(projects) + "]",
            }
        )

    tile_fields = [
        "site_id",
        "project",
        "tile",
        "title",
        "format",
        "collection_start",
        "collection_end",
        "publication_date",
        "size_bytes",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "source_id",
        "meta_url",
        "download_url",
    ]
    with open(args.tiles_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tile_fields)
        writer.writeheader()
        writer.writerows(tile_rows)
    print(f"Wrote {args.tiles_csv}")

    with open(args.dates_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["site_id", "lidar_dates", "lidar_projects"]
        )
        writer.writeheader()
        writer.writerows(date_rows)
    print(f"Wrote {args.dates_csv}")


if __name__ == "__main__":
    main()
