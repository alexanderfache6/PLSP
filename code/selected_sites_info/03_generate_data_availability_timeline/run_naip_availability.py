#!/usr/bin/env python3
"""Find NAIP imagery covering each selected site, with per-scene metadata.

Sites come from results/01_selected_sites_short_2.csv; the `coordinates` column
(list of [lat, lon] points) is the site boundary box and is used as the search
polygon.

This drives https://earthexplorer.usgs.gov/ exactly as the web UI does, which
needs no login (an account is only required to *download* imagery):

  1. GET  /                        -> session cookie
  2. POST /tabs/save               -> Search Criteria -> Polygon -> Decimal
  3. POST /dataset/select          -> Data Sets -> Aerial Imagery -> NAIP
  4. POST /scene/search            -> results rows (Entity ID, Acquisition Date)
  5. GET  /scene/metadata/full/... -> the per-scene metadata table

Outputs one row per scene plus a per-site list of acquisition dates.
"""

import argparse
import ast
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from http.cookiejar import CookieJar
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SITES_CSV = RESULTS_DIR / "01_selected_sites_short_2.csv"
SCENES_CSV = RESULTS_DIR / "03_naip_metadata.csv"
DATES_CSV = RESULTS_DIR / "03_naip_data_availability.csv"

BASE_URL = "https://earthexplorer.usgs.gov/"
# "Aerial Imagery -> NAIP" in the EarthExplorer data set tree
NAIP_DATASET_ID = "5e83a340bf820c39"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
RESULTS_PER_PAGE = 100

# metadata fields to keep, in report order; the values are the labels used in
# the EarthExplorer metadata table, matched case/punctuation-insensitively.
METADATA_FIELDS = OrderedDict(
    [
        ("map_projection", ["Map Projection"]),
        ("projection_zone", ["Projection Zone"]),
        ("datum", ["Datum"]),
        ("resolution", ["Resolution"]),
        ("units", ["Units"]),
        ("number_of_bands", ["Number of Bands"]),
        ("sensor_type", ["Sensor Type"]),
    ]
)

ROW_RE = re.compile(r'<tr id="resultRow_[^"]*".*?</tr>', re.S)
ENTITY_RE = re.compile(r'data-entityId="([^"]+)"')
DISPLAY_RE = re.compile(r'data-displayId="([^"]+)"')
ACQUIRED_RE = re.compile(
    r"Acquisition Date:\s*</strong>\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"
)
DISPLAYING_RE = re.compile(r"Displaying\s+([\d,]+)\s*-\s*([\d,]+)\s+of\s+([\d,]+)")
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def normalize(name):
    """Collapse a metadata label so spelling variants compare equal."""
    return "".join(c for c in str(name).lower() if c.isalnum())


def strip_tags(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


class EarthExplorer:
    """Minimal unauthenticated EarthExplorer session."""

    def __init__(self, delay=0.2):
        self.delay = delay
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        self.opener.addheaders = [
            ("User-Agent", USER_AGENT),
            ("Referer", BASE_URL),
            ("X-Requested-With", "XMLHttpRequest"),
        ]
        self.request("")  # pick up the session cookie

    def request(self, path, data=None):
        url = urllib.parse.urljoin(BASE_URL, path)
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        try:
            with self.opener.open(urllib.request.Request(url, data=body), timeout=300) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            sys.exit(f"{path} failed: HTTP {exc.code}")
        except urllib.error.URLError as exc:
            sys.exit(f"{path} failed: {exc}")
        time.sleep(self.delay)
        return text

    def set_polygon(self, points):
        """Set Search Criteria -> Polygon, entered as decimal lat/lon."""
        coordinates = [
            {"c": i, "a": f"{lat:.6f}", "o": f"{lon:.6f}"}
            for i, (lat, lon) in enumerate(points)
        ]
        criteria = {
            "tab": 1,
            "destination": 2,
            "coordinates": coordinates,
            "format": "dec",
            "dStart": "",
            "dEnd": "",
            "searchType": "Std",
            "includeUnknownCC": "1",
            "maxCC": 100,
            "minCC": 0,
            "months": [],
            "pType": "polygon",
        }
        self.request("tabs/save", {"data": json.dumps(criteria)})

    def select_dataset(self, dataset_id=NAIP_DATASET_ID):
        response = self.request("dataset/select", {"datasetId": dataset_id})
        if '"success":true' not in response:
            sys.exit(f"dataset/select failed: {response[:200]}")

    def search(self, dataset_id=NAIP_DATASET_ID):
        """Return [(entity_id, display_id, acquisition_date)] for the current AOI."""
        scenes = []
        page = 1
        while True:
            payload = {"datasetId": dataset_id, "resultsPerPage": RESULTS_PER_PAGE}
            if page > 1:
                payload["pageNum"] = page
            page_html = self.request("scene/search", payload)

            for row in ROW_RE.findall(page_html):
                entity = ENTITY_RE.search(row)
                display = DISPLAY_RE.search(row)
                acquired = ACQUIRED_RE.search(row)
                if not entity:
                    continue
                scenes.append(
                    (
                        entity.group(1),
                        display.group(1) if display else "",
                        acquired.group(1) if acquired else "",
                    )
                )

            shown = DISPLAYING_RE.search(page_html)
            if not shown:
                break
            last, total = int(shown.group(2).replace(",", "")), int(shown.group(3).replace(",", ""))
            if last >= total:
                break
            page += 1
        return scenes

    def metadata(self, entity_id, dataset_id=NAIP_DATASET_ID):
        """Return the scene's full metadata table as {label: value}."""
        page_html = self.request(f"scene/metadata/full/{dataset_id}/{entity_id}/")
        cells = [strip_tags(c) for c in CELL_RE.findall(page_html)]
        return {
            normalize(cells[i]): cells[i + 1]
            for i in range(0, len(cells) - 1, 2)
            if cells[i]
        }


def read_sites(path):
    """Return [(site_full, site_id, polygon)] where polygon is [(lat, lon), ...]."""
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
            points = [(float(lat), float(lon)) for lat, lon in ast.literal_eval(raw)]
            # the stored ring repeats its first point; EarthExplorer closes it itself
            if len(points) > 1 and points[0] == points[-1]:
                points = points[:-1]
            sites.append(
                (
                    (row.get("site_full") or "").strip(),
                    (row.get("site_id") or "").strip(),
                    points,
                )
            )
    return sites


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    ap.add_argument("--scenes-csv", type=Path, default=SCENES_CSV)
    ap.add_argument("--dates-csv", type=Path, default=DATES_CSV)
    ap.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="seconds to pause between EarthExplorer requests",
    )
    args = ap.parse_args()

    sites = read_sites(args.sites_csv)
    if not sites:
        sys.exit(f"No sites with coordinates found in {args.sites_csv}")

    ee = EarthExplorer(delay=args.delay)
    scene_rows = []
    date_rows = []

    for site_full, site_id, polygon in sites:
        ee.set_polygon(polygon)
        ee.select_dataset()
        scenes = ee.search()

        print(f"=== {site_id} ({site_full}) ===")
        print(f"  {len(scenes)} NAIP scenes; fetching metadata...")

        rows = []
        for entity_id, display_id, acquired in scenes:
            meta = ee.metadata(entity_id)
            row = {
                "entity_id": display_id or meta.get(normalize("NAIP Entity ID"), entity_id),
                "ee_internal_id": entity_id,
                "acquisition_date": acquired
                or meta.get(normalize("Acquisition Date"), "").replace("/", "-"),
            }
            for key, labels in METADATA_FIELDS.items():
                row[key] = next((meta[normalize(label)] for label in labels if meta.get(normalize(label))), "")
            rows.append(row)

        rows.sort(key=lambda r: (r["acquisition_date"], r["entity_id"]), reverse=True)
        for r in rows:
            meta_text = " | ".join(f"{k}={r[k]}" for k in METADATA_FIELDS if r[k])
            print(f"  {r['acquisition_date']}  {r['entity_id']}  {meta_text}")
        print()

        for r in rows:
            scene_rows.append({"site_id": site_id, **r})

        dates = sorted({r["acquisition_date"] for r in rows if r["acquisition_date"]}, reverse=True)
        date_rows.append(
            {
                "site_id": site_id,
                "naip_dates": "[" + ", ".join(dates) + "]",
            }
        )

    scene_fields = [
        "site_id",
        "entity_id",
        "ee_internal_id",
        "acquisition_date",
        *METADATA_FIELDS,
    ]
    with open(args.scenes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scene_fields)
        writer.writeheader()
        writer.writerows(scene_rows)
    print(f"Wrote {args.scenes_csv}")

    with open(args.dates_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["site_id", "naip_dates"])
        writer.writeheader()
        writer.writerows(date_rows)
    print(f"Wrote {args.dates_csv}")


if __name__ == "__main__":
    main()
