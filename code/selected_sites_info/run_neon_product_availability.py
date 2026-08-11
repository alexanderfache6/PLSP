#!/usr/bin/env python3
"""Check which years all NEON AOP data products are available for the selected NEON sites.

Sites come from info/01_selected_sites_raw2.csv (rows with is_neon == 1).
Products come from info/neon_data_products.csv.

Availability is read from the NEON data portal's public API, which is the same
source that backs https://data.neonscience.org/data-products/{product_id}.
"""

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

INFO_DIR = Path(__file__).resolve().parent / "info"
SITES_CSV = INFO_DIR / "01_selected_sites_raw2.csv"
PRODUCTS_CSV = INFO_DIR / "neon_products.csv"
OUT_CSV = INFO_DIR / "neon_product_availability.csv"

API_URL = "https://data.neonscience.org/api/v0/products/{product_id}"
PORTAL_URL = "https://data.neonscience.org/data-products/{product_id}"


def read_neon_sites(path):
    """Return [(neon_id, site_name)] for rows flagged is_neon == 1."""
    sites = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_neon", "")).strip() != "1":
                continue
            neon_id = (row.get("neon_id") or "").strip()
            if not neon_id:
                print(
                    f"warning: skipping is_neon row with no neon_id: "
                    f"{row.get('plsp_product_id')}",
                    file=sys.stderr,
                )
                continue
            sites.append((neon_id, (row.get("site_name") or "").strip()))
    return sites


def read_products(path):
    """Return OrderedDict of product_id -> description."""
    products = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("product_id") or "").strip()
            if pid:
                products[pid] = (row.get("description") or "").strip()
    return products


def fetch_product(product_id):
    """Fetch the product record from the NEON API."""
    url = API_URL.format(product_id=product_id)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["data"]


def years_by_site(product_data):
    """Return {site_code: set(years)} from a product's availableMonths."""
    out = {}
    for entry in product_data.get("siteCodes") or []:
        months = entry.get("availableMonths") or []
        out[entry["siteCode"]] = {m.split("-")[0] for m in months}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    ap.add_argument("--products-csv", type=Path, default=PRODUCTS_CSV)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    sites = read_neon_sites(args.sites_csv)
    products = read_products(args.products_csv)

    if not sites:
        sys.exit(f"No is_neon sites found in {args.sites_csv}")
    if not products:
        sys.exit(f"No products found in {args.products_csv}")

    print(f"NEON sites: {', '.join(s for s, _ in sites)}")
    print(f"Products: {', '.join(p for p in products)}")
    availability = {}
    for pid, desc in products.items():
        print(f"{pid} {desc}")
        print(f"{PORTAL_URL.format(product_id=pid)}")
        try:
            availability[pid] = years_by_site(fetch_product(pid))
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as exc:
            sys.exit(f"Failed to fetch {pid}: {exc}")

    rows = []
    print()
    for site, name in sites:
        per_product = {pid: sorted(availability[pid].get(site, set())) for pid in products}
        common = set.intersection(
            *(set(availability[pid].get(site, set())) for pid in products)
        )
        common = sorted(common)

        print(f"=== {site} ({name}) ===")
        for pid in products:
            yrs = per_product[pid]
            print(f"  {pid}: {', '.join(yrs) if yrs else '(none)'}")
        label = ", ".join(common) if common else "(none)"
        print(f"  ALL {len(products)} PRODUCTS: {label}")
        print()

        all_years = sorted(set().union(*(set(v) for v in per_product.values())))
        for year in all_years:
            rows.append(
                {
                    "neon_id": site,
                    "year": year,
                    **{pid: int(year in per_product[pid]) for pid in products},
                    "all_products": int(year in common),
                }
            )

    fieldnames = ["neon_id", "year"] + list(products) + ["all_products"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
