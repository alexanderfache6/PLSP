#!/usr/bin/env python3
"""Plot a data availability timeline for the selected sites.

Reads the three availability CSVs produced by the run_*_availability.py scripts
and draws one dot per acquisition, on a month-resolution time axis:

  3DEP  blue   from lidar_dates      (collection_start of each lidar project)
  NAIP  red    from naip_dates       (acquisition_date of each scene)
  NEON  green  from year_month       (only rows where all_products == 1)

Each site is a band of three rows, one per data source, so the source is
encoded by row position as well as by color.
"""

import argparse
import csv
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INFO_DIR = Path(__file__).resolve().parent / "info"
SITES_CSV = INFO_DIR / "02_selected_sites_short_2.csv"
DEP_CSV = INFO_DIR / "3dep_data_availability.csv"
NAIP_CSV = INFO_DIR / "naip_data_availability.csv"
NEON_CSV = INFO_DIR / "neon_data_availability.csv"
OUT_PNG = INFO_DIR / "data_availability_timeline.png"

# PlanetScope coverage starts in 2017, so earlier acquisitions are not plotted
MIN_YEAR = 2017

# categorical slots 1 / 6 / 8 of the validated reference palette
SOURCES = OrderedDict(
    [
        ("3DEP", "#2a78d6"),  # blue
        ("NAIP", "#e34948"),  # red
        ("NEON", "#008300"),  # green
    ]
)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"

ROW_STEP = 1.0  # vertical spacing between source rows
BAND_GAP = 0.8  # extra spacing between site bands


def parse_date_list(cell):
    """Pull YYYY-MM-DD dates out of a '[d1, d2, ...]' cell."""
    return [
        date(int(y), int(m), int(d))
        for y, m, d in re.findall(r"(\d{4})-(\d{2})-(\d{2})", cell or "")
    ]


def read_site_order(path):
    """Return OrderedDict site_id -> neon_id (blank when not a NEON site)."""
    sites = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            site_id = (row.get("site_id") or "").strip()
            if site_id:
                sites[site_id] = (row.get("neon_id") or "").strip()
    return sites


def read_simple_dates(path, column):
    """Return {site_id: [date, ...]} from a '[...]' list column."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            site_id = (row.get("site_id") or "").strip()
            if site_id:
                out[site_id] = parse_date_list(row.get(column))
    return out


def read_neon_dates(path):
    """Return {neon_id: [date, ...]} for months where all_products is true."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "year_month" not in (reader.fieldnames or []):
            sys.exit(
                f"{path} has no year_month column - re-run run_neon_availability.py"
            )
        for row in reader:
            if str(row.get("all_products", "")).strip() != "1":
                continue
            neon_id = (row.get("neon_id") or "").strip()
            year, month = (row.get("year_month") or "").split("-")
            out.setdefault(neon_id, []).append(date(int(year), int(month), 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    ap.add_argument("--dep-csv", type=Path, default=DEP_CSV)
    ap.add_argument("--naip-csv", type=Path, default=NAIP_CSV)
    ap.add_argument("--neon-csv", type=Path, default=NEON_CSV)
    ap.add_argument("--out-png", type=Path, default=OUT_PNG)
    ap.add_argument(
        "--min-year",
        type=int,
        default=MIN_YEAR,
        help="drop acquisitions before this year (default matches PlanetScope)",
    )
    args = ap.parse_args()

    sites = read_site_order(args.sites_csv)
    dep = read_simple_dates(args.dep_csv, "lidar_dates")
    naip = read_simple_dates(args.naip_csv, "naip_dates")
    neon = read_neon_dates(args.neon_csv)

    # build the row layout: one band per site, one row per source within it
    rows = []  # (y, site_id, source, [dates])
    y = 0.0
    band_centers = OrderedDict()
    for site_id, neon_id in sites.items():
        band_top = y
        for source in SOURCES:
            if source == "3DEP":
                dates = dep.get(site_id, [])
            elif source == "NAIP":
                dates = naip.get(site_id, [])
            else:
                dates = neon.get(neon_id, []) if neon_id else []
            dates = [d for d in dates if d.year >= args.min_year]
            rows.append((y, site_id, source, dates))
            y -= ROW_STEP
        band_centers[site_id] = (band_top + (y + ROW_STEP)) / 2
        y -= BAND_GAP

    all_dates = [d for _, _, _, dates in rows for d in dates]
    if not all_dates:
        sys.exit("No dates found in any availability CSV")
    x_min = date(args.min_year, 1, 1)
    x_max = date(max(all_dates).year + 1, 1, 1)

    height = 1.6 + 0.42 * len(rows) + 0.3 * len(sites)
    fig, ax = plt.subplots(figsize=(14, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for row_y, _, source, dates in rows:
        # recessive guide line so empty rows still read as a row
        ax.plot(
            [x_min, x_max],
            [row_y, row_y],
            color=GRID,
            linewidth=1,
            zorder=1,
            solid_capstyle="butt",
        )
        if dates:
            ax.scatter(
                dates,
                [row_y] * len(dates),
                s=70,
                color=SOURCES[source],
                edgecolors=SURFACE,  # 2px surface ring so overlaps stay legible
                linewidths=1.6,
                zorder=3,
                clip_on=False,
            )

    # site band labels and separators
    for index, (site_id, center) in enumerate(band_centers.items()):
        ax.text(
            -0.085,
            center,
            site_id,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT_PRIMARY,
        )
        if index:
            ax.axhline(
                center + (len(SOURCES) * ROW_STEP + BAND_GAP) / 2,
                color=GRID,
                linewidth=1,
                zorder=0,
            )

    ax.set_yticks([row_y for row_y, _, _, _ in rows])
    ax.set_yticklabels([source for _, _, source, _ in rows], fontsize=9.5)
    ax.tick_params(axis="y", length=0, colors=TEXT_SECONDARY, pad=6)
    last_row_y = rows[-1][0]
    ax.set_ylim(last_row_y - 0.7, 0.7)

    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="major", colors=TEXT_SECONDARY, labelsize=9.5, length=5)
    # month ticks stay as tick marks only - no vertical gridlines
    ax.tick_params(axis="x", which="minor", colors=GRID, length=3)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="x", which="minor", visible=False)
    ax.set_axisbelow(True)

    for name, spine in ax.spines.items():
        # keep a light baseline so the month tick marks read as ticks
        spine.set_visible(name == "bottom")
        spine.set_color(GRID)

    ax.set_title(
        "Ground truth data availability",
        loc="left",
        fontsize=14,
        color=TEXT_PRIMARY,
        pad=42,
    )
    ax.annotate(
        f"{args.min_year}-present (PlanetScope coverage), one dot per acquisition, NEON shown only where all AOP products exist",
        xy=(0, 1.0),
        xytext=(0, 12),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            label=source,
        )
        for source, color in SOURCES.items()
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1, 1.0),
        ncol=len(SOURCES),
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out_png}")

    for _, site_id, source, dates in rows:
        label = ", ".join(d.isoformat() for d in sorted(dates, reverse=True))
        print(f"{site_id:<8} {source:<5} {len(dates):>3}  {label or '(none)'}")


if __name__ == "__main__":
    main()
