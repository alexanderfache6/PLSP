#!/usr/bin/env python3
"""Static map of the PLSP sites over NEON domain boundaries, with the 8 selected sites picked out."""

import ast
import sys

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from aquarel import load_theme
from run_plsp_earthdata_sites_map import CONUS_BOUNDS, DOMAIN_DIR, FLUX_TSV, PLSP_CSV, RESULTS_DIR, load_domains, load_sites
from shapely.geometry import Polygon, box

SELECTED_SITES_CSV = RESULTS_DIR / "01_selected_sites_short_2.csv"
OUT_PNG = RESULTS_DIR / "02_selected_sites_detail_map.png"

DRYLAND_DOMAIN_IDS = {"D10", "D13", "D14", "D15", "D17"}

DOMAIN_FILL_PAIRABLE = "#1baf7a"
DOMAIN_FILL_UNPAIRABLE = "#e34948"
DOMAIN_LINE = "#52514e"
DRYLAND_LINE = "#1a1a1a"
NEON_COLOR = "#2a78d6"
FLUX_COLOR = "#1baf7a"
SELECTED_RING_COLOR = "#e6007e"

DOMAIN_FILL_ALPHA = 0.35
DOMAIN_LINE_WIDTH = 0.5
DRYLAND_LINE_WIDTH = 1

SITE_MARKER = "o"
SITE_MARKER_SIZE = 25
SELECTED_MARKER_SIZE = 25

SELECTED_RING_RADIUS = 0.5
SELECTED_RING_WIDTH = 1.5
SELECTED_RING_DASHES = "-"

# drawing order, bottom to top
ZORDER_DOMAIN_FILL = 1
ZORDER_DOMAIN_LINE = 2
ZORDER_SITE = 4
ZORDER_SELECTED_RING = 5

FIGURE_SIZE = (8, 8.0)
FIGURE_DPI = 300
FIGURE_FACECOLOR = "white"
FIGURE_THEME = "boxy_light"
FIGURE_FONT = "DejaVu Sans"
TITLE = "Available NEON Domain Pairs from PLSP NEON and AmeriFlux Sites"
TITLE_FONTSIZE = 14
LEGEND_LOCATION = "lower left"
LEGEND_FONTSIZE = 10
LEGEND_MARKER_SIZE = 8
LEGEND_FRAME_ALPHA = 0.9
CRS_LEGEND_LOCATION = "lower right"
CRS_LEGEND_FONTSIZE = 8

EAST_LIMIT_LON = -90
WEST_LIMIT_LON = -125
SOUTH_LIMIT_LAT = 25
NORTH_LIMIT_LAT = 50


def load_selected_sites(csv_path):
    """The 8 PLSP-selected sites, with each site's precomputed 10km box.

    THE BOXES ARE NO LONGER DRAWN: 10 km is about a tenth of a degree, which is
    smaller than a site marker at CONUS scale, so the outlines read as noise on
    top of the points. The frame is still loaded, because the site_id column is
    what marks a site as selected and its CRS is one of the three checked for
    the projection legend.

    The coordinates column is a Python list literal of [lat, lon] pairs
    already computed elsewhere; parsed here and reused as-is rather than
    recomputed.

    Inputs: csv_path - Path to 01_selected_sites_short_2.csv
    Outputs: GeoDataFrame with site_id, site_name and a box geometry
    """
    sites = pd.read_csv(csv_path)
    records = []
    for _, row in sites.iterrows():
        ring = ast.literal_eval(row["coordinates"])
        records.append({"site_id": row["site_id"], "site_name": row["site_name"], "geometry": Polygon([(lon, lat) for lat, lon in ring])})
    return gpd.GeoDataFrame(records, crs=4326)


def main():
    # NO PATH ARGUMENTS, as in run_plsp_earthdata_sites_map.py: the inputs are
    # constants here and imported from that script, so both maps read one set
    # of files.
    domains = load_domains(DOMAIN_DIR)
    sites = load_sites(PLSP_CSV, FLUX_TSV)
    selected = load_selected_sites(SELECTED_SITES_CSV)

    conus = box(CONUS_BOUNDS["min_lon"], CONUS_BOUNDS["min_lat"], CONUS_BOUNDS["max_lon"], CONUS_BOUNDS["max_lat"])
    outside = sites[~sites.geometry.within(conus)]
    if len(outside):
        print(f"dropped {len(outside)} site(s) outside the lower 48: " + ", ".join(outside.site_code), file=sys.stderr)
    sites = sites[sites.geometry.within(conus)].reset_index(drop=True)
    domains = domains[domains.geometry.intersects(conus)].reset_index(drop=True)

    joined = gpd.sjoin(sites, domains[["domainID", "domainName", "geometry"]], how="left", predicate="within").drop(columns="index_right")

    counts = joined.dropna(subset=["domainID"]).groupby("domainID")["is_neon"]
    pairable = {domain_id for domain_id, flags in counts if flags.any() and (~flags).any()}
    domains["pairable"] = domains.domainID.isin(pairable)
    domains["dryland"] = domains.domainID.isin(DRYLAND_DOMAIN_IDS)

    joined["selected"] = joined.site_code.isin(selected.site_id)

    theme = load_theme(FIGURE_THEME)
    theme.apply()
    plt.rcParams["font.family"] = FIGURE_FONT

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for is_pairable, fill in ((True, DOMAIN_FILL_PAIRABLE), (False, DOMAIN_FILL_UNPAIRABLE)):
        group = domains[domains.pairable == is_pairable]
        if len(group):
            group.plot(ax=ax, facecolor=fill, alpha=DOMAIN_FILL_ALPHA, edgecolor="none", zorder=ZORDER_DOMAIN_FILL)

    for is_dryland, color, width in ((False, DOMAIN_LINE, DOMAIN_LINE_WIDTH), (True, DRYLAND_LINE, DRYLAND_LINE_WIDTH)):
        group = domains[domains.dryland == is_dryland]
        if len(group):
            group.boundary.plot(ax=ax, color=color, linewidth=width, zorder=ZORDER_DOMAIN_LINE)

    for is_neon, color, label in ((True, NEON_COLOR, "NEON"), (False, FLUX_COLOR, "AmeriFlux")):
        for is_selected in (False, True):
            group = joined[(joined.is_neon == is_neon) & (joined.selected == is_selected)]
            if len(group):
                group.plot(ax=ax, color=color, marker=SITE_MARKER, markersize=SELECTED_MARKER_SIZE if is_selected else SITE_MARKER_SIZE, zorder=ZORDER_SITE)

    for point in joined[joined.selected].geometry:
        ax.add_patch(mpatches.Circle((point.x, point.y), radius=SELECTED_RING_RADIUS, facecolor="none", edgecolor=SELECTED_RING_COLOR, linewidth=SELECTED_RING_WIDTH, linestyle=SELECTED_RING_DASHES, zorder=ZORDER_SELECTED_RING))

    ax.set_xlim(WEST_LIMIT_LON, EAST_LIMIT_LON)
    ax.set_ylim(SOUTH_LIMIT_LAT, NORTH_LIMIT_LAT)
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    handles = [
        mpatches.Patch(facecolor=DOMAIN_FILL_PAIRABLE, alpha=DOMAIN_FILL_ALPHA, label="Domain: NEON + AmeriFlux pair available"),
        mpatches.Patch(facecolor=DOMAIN_FILL_UNPAIRABLE, alpha=DOMAIN_FILL_ALPHA, label="Domain: NEON + AmeriFlux pair unavailable"),
        plt.Line2D([0], [0], color=DRYLAND_LINE, linewidth=DRYLAND_LINE_WIDTH, label="Dryland domain"),
        plt.Line2D([0], [0], marker=SITE_MARKER, color="none", markerfacecolor=NEON_COLOR, markersize=LEGEND_MARKER_SIZE, label="NEON site"),
        plt.Line2D([0], [0], marker=SITE_MARKER, color="none", markerfacecolor=FLUX_COLOR, markersize=LEGEND_MARKER_SIZE, label="AmeriFlux site"),
        plt.Line2D([0], [0], color=SELECTED_RING_COLOR, linewidth=SELECTED_RING_WIDTH, linestyle=SELECTED_RING_DASHES, label="Selected site"),
    ]
    site_legend = ax.legend(handles=handles, loc=LEGEND_LOCATION, fontsize=LEGEND_FONTSIZE, framealpha=LEGEND_FRAME_ALPHA)
    # ax.legend REPLACES any legend already on the axes, so the first one is
    # re-added as an artist before the second is created.
    ax.add_artist(site_legend)

    # THE PROJECTION IS READ FROM THE DATA, NEVER TYPED. Every layer is plotted
    # in whatever CRS its GeoDataFrame carries, so a label written by hand would
    # keep saying WGS 84 after someone reprojected a layer. All three frames are
    # checked, and a disagreement is shown rather than hidden behind one name.
    crs_labels = sorted({f"{frame.crs.to_string()}" for frame in (domains, joined, selected) if frame.crs is not None})
    crs_handles = [plt.Line2D([0], [0], color="none", label=crs_label) for crs_label in crs_labels]
    ax.legend(handles=crs_handles, loc=CRS_LEGEND_LOCATION, fontsize=CRS_LEGEND_FONTSIZE, framealpha=LEGEND_FRAME_ALPHA, handlelength=0, handletextpad=0)

    theme.apply_transforms()
    fig.tight_layout()
    ax.grid(False)
    # plt.axis("square")
    fig.savefig(OUT_PNG, dpi=FIGURE_DPI, facecolor=FIGURE_FACECOLOR)
    print(f"Wrote {OUT_PNG}")

    summary = joined.dropna(subset=["domainID"]).groupby(["domainID", "domainName"]).agg(neon=("is_neon", "sum"), flux=("is_neon", lambda s: (~s).sum())).reset_index().sort_values("domainID")
    print("\ndomain             NEON  flux  pairable  dryland")
    for _, r in summary.iterrows():
        mark = "yes" if r.neon and r.flux else ""
        dry = "yes" if r.domainID in DRYLAND_DOMAIN_IDS else ""
        print(f"{r.domainID} {r.domainName[:26]:<28}{r.neon:>3}{r.flux:>6}   {mark:<8}  {dry}")


if __name__ == "__main__":
    main()
