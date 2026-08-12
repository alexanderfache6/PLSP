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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SITES_CSV = RESULTS_DIR / "01_selected_sites_short_2.csv"
DEP_CSV = RESULTS_DIR / "03_3dep_data_availability.csv"
NAIP_CSV = RESULTS_DIR / "03_naip_data_availability.csv"
NEON_CSV = RESULTS_DIR / "03_neon_data_availability.csv"
LONG_CSV = RESULTS_DIR / "01_selected_sites_long_2.csv"
OUT_PNG = RESULTS_DIR / "03_data_availability_timeline.png"

# PlanetScope coverage starts in 2017, so earlier acquisitions are not plotted
MIN_YEAR = 2017

# categorical slots 1 / 8 / 6 / 3 of the validated reference palette
SOURCES = OrderedDict(
    [
        ("3DEP", "#2a78d6"),  # blue
        ("NAIP", "#e34948"),  # red
        ("NEON", "#008300"),  # green
        ("PhenoCam", "#1baf7a"),  # aqua-green, distinct from the NEON green
    ]
)
DOT_SOURCES = ("3DEP", "NAIP", "NEON")

# vertical offsets for the two phenocam spans inside the PhenoCam row
CAM_OFFSETS = (0.14, -0.14)

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


def parse_day(value):
    """Parse a YYYY-MM-DD cell, returning None when blank or malformed."""
    match = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", value or "")
    return date(*(int(g) for g in match.groups())) if match else None


def read_phenocam_spans(path, min_year):
    """Return {site_id: [(name, start, end), ...]} for phenocam 1 and 2.

    Spans starting before min_year are clipped to it; a missing date_last is
    treated as still running, so it extends to today.
    """
    floor = date(min_year, 1, 1)
    today = date.today()
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            site_id = (row.get("site_id") or "").strip()
            if not site_id:
                continue
            spans = []
            for n in (1, 2):
                name = (row.get(f"phenocam{n}") or "").strip()
                if not name:
                    continue
                start = parse_day(row.get(f"(p{n})date_first"))
                end = parse_day(row.get(f"(p{n})date_last")) or today
                if not start:
                    print(
                        f"warning: {site_id} phenocam{n} has no date_first, skipping",
                        file=sys.stderr,
                    )
                    continue
                start = max(start, floor)
                if end > start:
                    spans.append((name, start, end))
            out[site_id] = spans
    return out


def read_neon_dates(path):
    """Return ({neon_id: [date, ...]}, [product_id, ...]).

    Dates are the months where all_products is true; the product ids are the
    per-product columns those months were intersected from.
    """
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "year_month" not in fields:
            sys.exit(
                f"{path} has no year_month column - re-run run_neon_availability.py"
            )
        products = [
            c
            for c in fields
            if c not in ("neon_id", "year", "month", "year_month", "all_products")
        ]
        for row in reader:
            if str(row.get("all_products", "")).strip() != "1":
                continue
            neon_id = (row.get("neon_id") or "").strip()
            year, month = (row.get("year_month") or "").split("-")
            out.setdefault(neon_id, []).append(date(int(year), int(month), 1))
    return out, products


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites-csv", type=Path, default=SITES_CSV)
    ap.add_argument("--dep-csv", type=Path, default=DEP_CSV)
    ap.add_argument("--naip-csv", type=Path, default=NAIP_CSV)
    ap.add_argument("--neon-csv", type=Path, default=NEON_CSV)
    ap.add_argument("--long-csv", type=Path, default=LONG_CSV)
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
    neon, neon_products = read_neon_dates(args.neon_csv)
    phenocam = read_phenocam_spans(args.long_csv, args.min_year)

    # build the row layout: one band per site, one row per source within it
    rows = []  # (y, site_id, source, [dates], [(name, start, end), ...])
    y = 0.0
    band_centers = OrderedDict()
    for site_id, neon_id in sites.items():
        band_top = y
        for source in SOURCES:
            dates, spans = [], []
            if source == "3DEP":
                dates = dep.get(site_id, [])
            elif source == "NAIP":
                dates = naip.get(site_id, [])
            elif source == "NEON":
                dates = neon.get(neon_id, []) if neon_id else []
            else:
                spans = phenocam.get(site_id, [])
            dates = [d for d in dates if d.year >= args.min_year]
            rows.append((y, site_id, source, dates, spans))
            y -= ROW_STEP
        band_centers[site_id] = (band_top + (y + ROW_STEP)) / 2
        y -= BAND_GAP

    all_dates = [d for _, _, _, dates, _ in rows for d in dates]
    all_dates += [end for _, _, _, _, spans in rows for _, _, end in spans]
    if not all_dates:
        sys.exit("No dates found in any availability CSV")
    x_min = date(args.min_year, 1, 1)
    # stop at the month after the last acquisition rather than the next new year
    latest = max(all_dates)
    x_max = (
        date(latest.year + 1, 1, 1)
        if latest.month == 12
        else date(latest.year, latest.month + 1, 1)
    )

    height = 1.6 + 0.42 * len(rows) + 0.3 * len(sites)
    fig, ax = plt.subplots(figsize=(14, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for row_y, _, source, dates, spans in rows:
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
        # phenocam operating periods: one filled line per camera, offset so two
        # cameras with the same range stay individually visible
        for index, (_, start, end) in enumerate(spans):
            offset = CAM_OFFSETS[index] if len(spans) > 1 else 0.0
            ax.plot(
                [start, end],
                [row_y + offset] * 2,
                color=SOURCES[source],
                linewidth=4,
                solid_capstyle="butt",
                zorder=3,
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

    ax.set_yticks([row_y for row_y, _, _, _, _ in rows])
    ax.set_yticklabels([source for _, _, source, _, _ in rows], fontsize=9.5)
    ax.tick_params(axis="y", length=0, colors=TEXT_SECONDARY, pad=6)
    last_row_y = rows[-1][0]
    ax.set_ylim(last_row_y - 0.7, 0.7)

    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(
        axis="x",
        which="major",
        colors=TEXT_SECONDARY,
        labelsize=9.5,
        length=7,
        width=1.1,
    )
    # month ticks stay as tick marks only - no vertical gridlines
    ax.tick_params(axis="x", which="minor", color=TEXT_SECONDARY, length=4, width=0.7)
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
        f"{args.min_year}-present (PlanetScope coverage), one dot per acquisition, "
        f"NEON shown only where all AOP ({', '.join(neon_products)}) products exist",
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
        if source in DOT_SOURCES
        # phenocam is an operating period, so its key is a line, not a dot
        else plt.Line2D([], [], color=color, linewidth=4, label=f"{source} (operating)")
        for source, color in SOURCES.items()
    ]
    # legend sits below the axis so it never collides with the subtitle
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, -0.09),
        ncol=len(SOURCES),
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=2.2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out_png}")

    for _, site_id, source, dates, spans in rows:
        if spans:
            label = "; ".join(
                f"{name} {start.isoformat()}..{end.isoformat()}"
                for name, start, end in spans
            )
            print(f"{site_id:<8} {source:<9} {len(spans):>3}  {label}")
        else:
            label = ", ".join(d.isoformat() for d in sorted(dates, reverse=True))
            print(f"{site_id:<8} {source:<9} {len(dates):>3}  {label or '(none)'}")


if __name__ == "__main__":
    main()
