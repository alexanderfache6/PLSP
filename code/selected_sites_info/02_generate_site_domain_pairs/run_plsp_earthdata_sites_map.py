#!/usr/bin/env python3
"""Interactive map of the PLSP sites, over the NEON domain boundaries.

Sites come from data/plsp_earthdata_sites.csv; coordinates and site attributes are joined
from the AmeriFlux site table in data/, and each site is assigned to a NEON
domain by point-in-polygon against the NEON domain boundaries:
https://www.neonscience.org/field-site-map-and-info

Writes an interactive Plotly HTML map plus the joined site/domain table.
"""

import argparse
import csv
import io
import sys
import textwrap
import urllib.request
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import Point, box

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

PLSP_CSV = DATA_DIR / "plsp_earthdata_sites.csv"
FLUX_TSV = DATA_DIR / "AmeriFlux-site-search-results-202607161927.tsv"
DOMAIN_DIR = DATA_DIR / "neon_domains"
OUT_HTML = RESULTS_DIR / "02_plsp_earthdata_sites_map.html"
OUT_CSV = RESULTS_DIR / "02_plsp_earthdata_sites_domains.csv"

DOMAIN_ZIP_URL = "https://www.neonscience.org/sites/default/files/NEONDomains_2024.zip"

NEON_COLOR = "#008300"  # green
FLUX_COLOR = "#2a78d6"  # blue
DOMAIN_LINE = "#52514e"
DOMAIN_FILL = "rgba(27, 175, 122, 0.10)"
# domains that cannot supply a NEON/AmeriFlux pair are flagged in light red
DOMAIN_FILL_UNPAIRABLE = "rgba(227, 73, 72, 0.18)"

# lower 48 bounding box; latitude alone excludes Alaska, Hawaii and Puerto Rico
CONUS_BOUNDS = dict(min_lat=24.0, max_lat=49.6, min_lon=-125.0, max_lon=-66.9)

HOVER_WRAP = 80  # characters before the IGBP row wraps


def load_domains(domain_dir, url=DOMAIN_ZIP_URL):
    """Load NEON domain polygons, downloading and caching them on first use."""
    if not domain_dir.exists():
        print(f"downloading NEON domains from {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=300) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(domain_dir)
    shapefiles = sorted(domain_dir.rglob("*.shp"))
    if not shapefiles:
        sys.exit(f"no shapefile found under {domain_dir}")
    return gpd.read_file(shapefiles[0]).to_crs(4326)


def load_sites(plsp_csv, flux_tsv):
    """Join PLSP site codes to AmeriFlux coordinates and attributes."""
    flux = {}
    with open(flux_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            flux[row["Site ID"].upper()] = row

    records = []
    for row in csv.DictReader(open(plsp_csv, newline="")):
        code = row["Site_Code"].strip()
        meta = flux.get(code.upper())
        if not meta:
            print(f"warning: {code} not in the AmeriFlux table, skipping", file=sys.stderr)
            continue
        records.append(
            {
                "site_code": code,
                "name": row["Site_Full_Name"].strip(),
                "is_neon": row["Is_NEON"].strip().upper() == "TRUE",
                "lat": float(meta["Latitude (degrees)"]),
                "lon": float(meta["Longitude (degrees)"]),
                "igbp": meta["Vegetation Abbreviation (IGBP)"] or "",
                "veg": meta["Vegetation Description (IGBP)"] or "",
                "koeppen": meta["Climate Class Abbreviation (Koeppen)"] or "",
                "map_mm": meta["Mean Average Precipitation (mm)"] or "",
                "mat_c": meta["Mean Average Temperature (degrees C)"] or "",
                "data_start": meta["AmeriFlux BASE Data Start"] or "",
                "data_end": meta["AmeriFlux BASE Data End"] or "",
            }
        )
    return gpd.GeoDataFrame(
        records,
        geometry=[Point(r["lon"], r["lat"]) for r in records],
        crs=4326,
    )


def hover_text(row):
    kind = "NEON" if row.is_neon else "AmeriFlux"
    domain = (
        f"{row.domainID} {row.domainName}"
        if isinstance(row.domainID, str)
        else "outside any NEON domain"
    )
    igbp = "<br>".join(textwrap.wrap(f"IGBP: {row.igbp} ({row.veg})", HOVER_WRAP))
    return (
        f"<b>{row.site_code}</b> — {row['name']}<br>"
        f"{kind}<br>"
        f"Domain: {domain}<br>"
        f"{igbp}<br>"
        f"Koppen: {row.koeppen} · MAP {row.map_mm} mm · MAT {row.mat_c} C<br>"
        f"AmeriFlux BASE: {row.data_start}-{row.data_end}<br>"
        f"{row.lat:.4f}, {row.lon:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plsp-csv", type=Path, default=PLSP_CSV)
    ap.add_argument("--flux-tsv", type=Path, default=FLUX_TSV)
    ap.add_argument("--domain-dir", type=Path, default=DOMAIN_DIR)
    ap.add_argument("--out-html", type=Path, default=OUT_HTML)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    domains = load_domains(args.domain_dir)
    sites = load_sites(args.plsp_csv, args.flux_tsv)

    # restrict to the lower 48: drop Alaska, Hawaii and Puerto Rico sites, then
    # keep only the domains that still reach into that window
    conus = box(
        CONUS_BOUNDS["min_lon"],
        CONUS_BOUNDS["min_lat"],
        CONUS_BOUNDS["max_lon"],
        CONUS_BOUNDS["max_lat"],
    )
    outside = sites[~sites.geometry.within(conus)]
    if len(outside):
        print(
            f"dropped {len(outside)} site(s) outside the lower 48: "
            + ", ".join(outside.site_code),
            file=sys.stderr,
        )
    sites = sites[sites.geometry.within(conus)].reset_index(drop=True)
    domains = domains[domains.geometry.intersects(conus)].reset_index(drop=True)

    joined = gpd.sjoin(
        sites,
        domains[["domainID", "domainName", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns="index_right")
    joined.drop(columns="geometry").to_csv(args.out_csv, index=False)

    orphans = joined[joined.domainID.isna()]
    if len(orphans):
        print(
            f"note: {len(orphans)} site(s) fall outside every domain polygon: "
            + ", ".join(orphans.site_code),
            file=sys.stderr,
        )

    fig = go.Figure()

    # domain polygons, drawn beneath the sites
    simplified = domains.copy()
    with warnings.catch_warnings():
        # GEOS raises a benign float warning here; the simplified polygons come
        # back complete and valid, so the noise is not worth surfacing
        warnings.filterwarnings("ignore", message="invalid value encountered")
        simplified["geometry"] = simplified.geometry.simplify(0.01)

    # a domain is pairable only if it holds at least one NEON and one AmeriFlux
    # site; the rest are flagged so unusable domains are visible at a glance
    counts = joined.dropna(subset=["domainID"]).groupby("domainID")["is_neon"]
    pairable = {
        domain_id
        for domain_id, flags in counts
        if flags.any() and (~flags).any()
    }
    simplified["pairable"] = simplified.domainID.isin(pairable)

    for is_pairable, fill, label in (
        (True, DOMAIN_FILL, "Domain with a NEON + AmeriFlux pair"),
        (False, DOMAIN_FILL_UNPAIRABLE, "Domain missing a NEON or AmeriFlux site"),
    ):
        group = simplified[simplified.pairable == is_pairable]
        if not len(group):
            continue
        fig.add_trace(
            go.Choroplethmap(
                geojson=group.__geo_interface__,
                locations=group.index,
                z=[0] * len(group),
                featureidkey="id",
                colorscale=[[0, fill], [1, fill]],
                showscale=False,
                marker=dict(line=dict(color=DOMAIN_LINE, width=1)),
                customdata=group[["domainID", "domainName"]].values,
                hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<extra></extra>",
                name=f"{label} ({len(group)})",
                showlegend=True,
            )
        )

    for is_neon, label, color in (
        (True, "NEON", NEON_COLOR),
        (False, "AmeriFlux", FLUX_COLOR),
    ):
        group = joined[joined.is_neon == is_neon]
        fig.add_trace(
            go.Scattermap(
                lat=group.lat,
                lon=group.lon,
                mode="markers",
                marker=dict(size=10, color=color),
                name=f"{label} ({len(group)})",
                text=[hover_text(r) for _, r in group.iterrows()],
                hovertemplate="%{text}<extra></extra>",
                customdata=group.site_code,
            )
        )

    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=42, lon=-100), zoom=2.7),
        margin=dict(l=0, r=0, t=56, b=0),
        title=dict(
            text=f"PLSP sites over NEON domains — {len(joined)} sites",
            x=0.01,
            font=dict(size=17),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1
        ),
        height=760,
    )

    fig.write_html(args.out_html, include_plotlyjs="cdn")
    print(f"Wrote {args.out_html}")
    print(f"Wrote {args.out_csv}")

    summary = (
        joined.dropna(subset=["domainID"])
        .groupby(["domainID", "domainName"])
        .agg(neon=("is_neon", "sum"), flux=("is_neon", lambda s: (~s).sum()))
        .reset_index()
        .sort_values("domainID")
    )
    print("\ndomain            NEON  flux  pairable")
    for _, r in summary.iterrows():
        mark = "yes" if r.neon and r.flux else ""
        print(f"{r.domainID} {r.domainName[:26]:<28}{r.neon:>3}{r.flux:>6}   {mark}")


if __name__ == "__main__":
    main()
