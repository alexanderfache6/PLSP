#!/usr/bin/env python3
"""One zoomed panel per dryland domain, showing its selected NEON and AmeriFlux pair."""

import ast
import sys

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from aquarel import load_theme
from FIG_create_site_pairs_map import FLUX_COLOR, NEON_COLOR, SELECTED_SITES_CSV
from run_plsp_earthdata_sites_map import DOMAIN_DIR, RESULTS_DIR, load_domains
from shapely.geometry import Polygon

OUT_PNG = RESULTS_DIR / "02_domain_zoom_selected_pairs.png"

# THE FOUR DRYLAND DOMAINS THAT HAVE A PAIR. Checked, not assumed: the script
# stops if the selected-sites table does not hold exactly these four domains
# with two sites each, because a silently missing site would leave a panel that
# looks complete and shows half a pair.
EXPECTED_DOMAINS = ("D13", "D14", "D15", "D17")
SITES_PER_DOMAIN = 2

# colours
DOMAIN_LINE_COLOR = "#52514e"
DOMAIN_FILL_COLOR = "#c8a97e"
DOMAIN_FILL_ALPHA = 0.45
PHENOCAM_COLOR = "#1a1a1a"

# marks
SITE_MARKER = "o"
SITE_MARKER_SIZE = 10
SITE_EDGE_COLOR = "#ffffff"
SITE_EDGE_WIDTH = 0.8
PHENOCAM_MARKER = "o"
PHENOCAM_MARKER_SIZE = 4
DOMAIN_LINE_WIDTH = 0.8
LABEL_FONTSIZE = 10

# drawing order, bottom to top
ZORDER_DOMAIN_FILL = 0
ZORDER_DOMAIN = 1
ZORDER_SITE = 3
ZORDER_PHENOCAM = 4

# EVERY PANEL GETS THE SAME SPAN IN DEGREES, centred on its own pair. The pairs
# are 0.9 to 4 degrees apart, and letting each panel choose its own extent made
# four maps at four scales sitting at four heights, which invites the eye to
# compare distances that are not comparable. The shared span is whatever the
# widest pair needs, so one scale fits all four and the panels line up.
EXTENT_PAD_SHARE = 0.18
MINIMUM_EXTENT_DEG = 0.35

# figure, legend and text
FIGURE_HEIGHT = 5.2
FIGURE_WIDTH_PER_PANEL = 4.2
FIGURE_DPI = 300
FIGURE_FACECOLOR = "white"
FIGURE_THEME = "boxy_light"
FIGURE_FONT = "DejaVu Sans"
TITLE = "Domains with Selected NEON and AmeriFlux Site Pairs"
TITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 9
TICK_LABEL_FONTSIZE = 8
LEGEND_FONTSIZE = 10
LEGEND_LOCATION = "lower center"
LEGEND_ANCHOR = (0.5, 0.0)
LEGEND_COLUMNS = 4
TITLE_Y = 0.97
TOP_MARGIN = 0.84  # room between the figure title and the panel titles


def load_selected_pairs(csv_path):
    """The selected sites with their 10 km box, phenocams and domain.

    ONE ROW PER SITE, carrying everything a panel needs, so the panel code never
    reaches back into the CSV. The coordinates column is a Python list literal
    of [lat, lon] pairs computed elsewhere and reused as-is; the phenocam
    columns are blank where a site has no camera, which is the case at SJER.

    Inputs: csv_path - Path to 01_selected_sites_short_2.csv
    Outputs: GeoDataFrame of site_id, site_name, domain, is_neon, the box
             geometry, a list of (lon, lat) phenocam positions and the matching
             list of phenocam ids, in camera order
    """
    sites = pd.read_csv(csv_path)
    records = []
    for _, row in sites.iterrows():
        ring = ast.literal_eval(row["coordinates"])
        phenocams, phenocam_ids = [], []
        for camera_number in (1, 2):
            latitude, longitude = row[f"(p{camera_number})latitude"], row[f"(p{camera_number})longitude"]
            camera_id = row[f"phenocam{camera_number}"]
            if pd.notna(latitude) and pd.notna(longitude):
                phenocams.append((float(longitude), float(latitude)))
                phenocam_ids.append(str(camera_id).strip() if pd.notna(camera_id) else "")
        records.append(
            {
                "site_id": row["site_id"],
                "site_name": row["site_name"],
                "domain": row["neon_domain"],
                "is_neon": bool(row["is_neon"]),
                "phenocams": phenocams,
                "phenocam_ids": phenocam_ids,
                "geometry": Polygon([(lon, lat) for lat, lon in ring]),
            }
        )
    return gpd.GeoDataFrame(records, crs=4326)


def check_pairs(selected):
    """Stop unless every expected domain holds exactly one NEON and one AmeriFlux site.

    Inputs: selected - from load_selected_pairs
    Outputs: None, raises SystemExit on anything unexpected
    """
    found = sorted(selected.domain.unique())
    if found != sorted(EXPECTED_DOMAINS):
        raise SystemExit(f"FAIL - expected domains {sorted(EXPECTED_DOMAINS)} in {SELECTED_SITES_CSV.name}, found {found}")
    for domain, group in selected.groupby("domain"):
        if len(group) != SITES_PER_DOMAIN or group.is_neon.sum() != 1:
            raise SystemExit(f"FAIL - {domain} holds {len(group)} site(s), {int(group.is_neon.sum())} of them NEON; each domain needs one NEON and one AmeriFlux site")


def required_span(group):
    """The span in degrees one domain's pair needs, padded, square.

    Inputs: group - the two rows for one domain
    Outputs: float
    """
    left, bottom, right, top = group.total_bounds
    span = max(right - left, top - bottom) * (1.0 + 2.0 * EXTENT_PAD_SHARE)
    # A pair that sits almost on top of itself would otherwise ask for a panel
    # a few hundred metres wide, where the 10 km boxes fill the whole frame.
    return max(span, MINIMUM_EXTENT_DEG)


def panel_extent(group, span):
    """One domain's limits: the shared span, centred on this pair.

    Inputs: group - the two rows for one domain; span - the shared span in
            degrees, from required_span across every domain
    Outputs: (left, right, bottom, top)
    """
    left, bottom, right, top = group.total_bounds
    centre_x, centre_y = (left + right) / 2, (bottom + top) / 2
    return centre_x - span / 2, centre_x + span / 2, centre_y - span / 2, centre_y + span / 2


def draw_panel(ax, domain, group, domains, span):
    """One domain: its boundary, the two 10 km boxes, the sites and their phenocams.

    THE SITE MARKER IS PLACED FROM THE BOX CENTRE, not from the flux tower
    coordinate, so the dot and the box it belongs to cannot drift apart. The
    tower position is what built the box in the first place.

    Inputs: ax; domain - e.g. "D14"; group - the two rows for it; domains;
            span - the shared panel span in degrees
    Outputs: None
    """
    boundary = domains[domains.domainID == domain]
    if len(boundary):
        # ONLY THIS PANEL'S OWN DOMAIN IS FILLED. A neighbouring domain crossing
        # the frame keeps its outline but no fill, so the shading always means
        # "this is the domain named in the title" and never something else.
        boundary.plot(ax=ax, facecolor=DOMAIN_FILL_COLOR, alpha=DOMAIN_FILL_ALPHA, edgecolor="none", zorder=ZORDER_DOMAIN_FILL)
        boundary.boundary.plot(ax=ax, color=DOMAIN_LINE_COLOR, linewidth=DOMAIN_LINE_WIDTH, zorder=ZORDER_DOMAIN)
    # THE NAME COMES FROM THE DOMAIN SHAPEFILE, never a typed lookup table, so
    # it cannot disagree with the boundary drawn beside it.
    domain_name = boundary.domainName.iloc[0] if len(boundary) else ""

    for _, row in group.iterrows():
        centre = row.geometry.centroid
        colour = NEON_COLOR if row.is_neon else FLUX_COLOR
        ax.plot(centre.x, centre.y, marker=SITE_MARKER, markersize=SITE_MARKER_SIZE, color=colour, linestyle="none", zorder=ZORDER_SITE)
        ax.annotate(row.site_id, (centre.x, centre.y), xytext=(0, SITE_MARKER_SIZE), textcoords="offset points", ha="center", fontsize=LABEL_FONTSIZE, color=colour, zorder=ZORDER_SITE)
        for longitude, latitude in row.phenocams:
            ax.plot(longitude, latitude, marker=PHENOCAM_MARKER, markersize=PHENOCAM_MARKER_SIZE, color=PHENOCAM_COLOR, linestyle="none", zorder=ZORDER_PHENOCAM)

    left, right, bottom, top = panel_extent(group, span)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{domain} - {domain_name}", fontsize=PANEL_TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)


def main():
    domains = load_domains(DOMAIN_DIR)
    selected = load_selected_pairs(SELECTED_SITES_CSV)
    check_pairs(selected)

    ordered_domains = sorted(selected.domain.unique())
    print(f"{len(ordered_domains)} domains with a pair: {', '.join(ordered_domains)}")
    for domain in ordered_domains:
        group = selected[selected.domain == domain]
        names = ", ".join(f"{row.site_id} ({'NEON' if row.is_neon else 'AmeriFlux'}, {len(row.phenocams)} phenocam)" for _, row in group.iterrows())
        print(f"{domain}: {names}")

    theme = load_theme(FIGURE_THEME)
    theme.apply()
    plt.rcParams["font.family"] = FIGURE_FONT

    shared_span = max(required_span(selected[selected.domain == domain]) for domain in ordered_domains)
    print(f"panel span {shared_span:.2f} degrees, set by the widest pair, applied to all {len(ordered_domains)} panels")

    fig, axes = plt.subplots(1, len(ordered_domains), figsize=(FIGURE_WIDTH_PER_PANEL * len(ordered_domains), FIGURE_HEIGHT))
    axes = [axes] if len(ordered_domains) == 1 else list(axes)
    for ax, domain in zip(axes, ordered_domains):
        draw_panel(ax, domain, selected[selected.domain == domain], domains, shared_span)
    # ONLY THE LEFTMOST PANEL IS LABELLED. Every panel now spans the same number
    # of degrees, so one pair of axis labels describes all four, and repeating
    # them four times only adds ink.
    axes[0].set_xlabel("Longitude (°)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel("Latitude (°)", fontsize=AXIS_LABEL_FONTSIZE)

    handles = [
        plt.Line2D([0], [0], marker=SITE_MARKER, color="none", markerfacecolor=NEON_COLOR, markeredgecolor=SITE_EDGE_COLOR, markersize=SITE_MARKER_SIZE, label="NEON site"),
        plt.Line2D([0], [0], marker=SITE_MARKER, color="none", markerfacecolor=FLUX_COLOR, markeredgecolor=SITE_EDGE_COLOR, markersize=SITE_MARKER_SIZE, label="AmeriFlux site"),
        plt.Line2D([0], [0], marker=PHENOCAM_MARKER, color="none", markerfacecolor=PHENOCAM_COLOR, markersize=PHENOCAM_MARKER_SIZE + 2, label="Phenocam Available"),
        mpatches.Patch(facecolor=DOMAIN_FILL_COLOR, alpha=DOMAIN_FILL_ALPHA, label="Domain"),
    ]

    fig.suptitle(TITLE, fontsize=TITLE_FONTSIZE, y=TITLE_Y)
    theme.apply_transforms()
    fig.tight_layout()
    # The legend and the figure title are both added AFTER tight_layout, which
    # accounts for neither, so the panels are pulled in from the top and bottom
    # to leave room for them.
    fig.subplots_adjust(bottom=0.22, top=TOP_MARGIN)
    fig.legend(handles=handles, loc=LEGEND_LOCATION, bbox_to_anchor=LEGEND_ANCHOR, ncol=LEGEND_COLUMNS, fontsize=LEGEND_FONTSIZE, frameon=False)
    fig.savefig(OUT_PNG, dpi=FIGURE_DPI, facecolor=FIGURE_FACECOLOR)
    print(f"Wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
