#!/usr/bin/env python3
"""One PhenoCam snapshot per selected site for a given date, arranged by domain.

WHAT IT DOES. For each selected site with a phenocam, it finds that camera's
images for the date, picks the one closest to 12:00 LOCAL, downloads it once and
draws it. The panel is one column per domain, NEON on top and AmeriFlux below.

HOW AN IMAGE IS FOUND, and why it is not a formula. A PhenoCam filename ends in
a timestamp, kendall_2025_09_17_120805.jpg, and that time DIFFERS BY SITE AND BY
DAY, so the URL cannot be built from the date alone. The day's browse page is
read instead and every image it lists for that camera is a candidate.

RGB ONLY. NEON cameras publish an infrared twin of every frame, named with an
_IR_ segment: NEON.D14.SRER.DP1.00042_IR_2017_02_24_120006.jpg beside
NEON.D14.SRER.DP1.00042_2017_02_24_120006.jpg. Both are valid images of the same
moment, so a naive "first match" picks the infrared one about half the time and
the panel silently shows a greyscale frame. Any path carrying _IR_ is dropped.

CLOSEST TO 12:00 LOCAL. PhenoCam timestamps are the camera's own local standard
time, so the comparison is made directly against noon in that filename's terms.
The camera's utc_offset is read from the API and reported, so a reader can see
that Tonzi's noon is an hour behind Kendall's rather than having to know it.

A MISSING IMAGE IS A LABELLED BLANK, NOT A CRASH. Cameras go offline, and a
whole panel of good sites should not be lost to one of them. Sites with no
phenocam at all, which is SJER, are blank for the same reason.

INPUTS

    a date, yyyy-mm-dd, positional
    results/01_selected_sites_short_2.csv, for the sites and their camera ids
    the PhenoCam API and archive at phenocam.nau.edu

OUTPUTS

    results/02_phenocam_snapshots_{date}.png
    data/phenocam_snapshots/{camera}/{filename}.jpg, cached, so re-running a
        date costs no network

EXAMPLE COMMANDS

    conda activate LCSC
    python FIG_live_phenocam_snapshots_per_domain.py 2025-09-17
"""

import argparse
import datetime
import json
import re
import sys
import urllib.request

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from aquarel import load_theme
from FIG_create_site_pairs_map import SELECTED_SITES_CSV
from FIG_domain_zoom_in_of_pairs import load_selected_pairs
from run_plsp_earthdata_sites_map import DATA_DIR, RESULTS_DIR

PHENOCAM_HOST = "https://phenocam.nau.edu"
BROWSE_URL = PHENOCAM_HOST + "/webcam/browse/{camera}/{year}/{month}/{day}/"
CAMERA_API_URL = PHENOCAM_HOST + "/api/cameras/{camera}/?format=json"
CACHE_DIR = DATA_DIR / "phenocam_snapshots"
REQUEST_TIMEOUT_S = 45

# The archive path as the browse page writes it, with the timestamp captured.
# _IR_ is excluded HERE rather than filtered afterwards, so an infrared frame
# can never reach the selection step.
IMAGE_PATH = re.compile(r"/data/archive/(?P<camera>[^/\"]+)/(?P<year>\d{4})/(?P<month>\d{2})/(?P<name>[^/\"]+?_(?P<stamp>\d{4}_\d{2}_\d{2}_\d{6})\.jpg)")
INFRARED_MARKER = "_IR_"
TARGET_LOCAL_TIME = datetime.time(12, 0, 0)

# WHICH CAMERA A SITE USES. Phenocam 1 everywhere, except where a site's second
# camera is the one worth showing. SRER's DP1.00033 looks over the grassland
# the end members were drawn on, while DP1.00042 points elsewhere. Keyed by
# AmeriFlux site id and stated as a camera NUMBER, 1 or 2, as the selected-sites
# table names them.
CAMERA_NUMBER_BY_SITE = {"US-xSR": 2}
DEFAULT_CAMERA_NUMBER = 1

# layout: one column per domain, NEON on top, AmeriFlux below
ROWS = 2
PANEL_TITLE_FONTSIZE = 11
PANEL_SUBTITLE_FONTSIZE = 8
MISSING_FONTSIZE = 10
MISSING_COLOR = "#6b6b6b"

# figure
FIGURE_WIDTH_PER_PANEL = 4.2
FIGURE_HEIGHT_PER_ROW = 3.6
FIGURE_DPI = 200
FIGURE_FACECOLOR = "white"
FIGURE_THEME = "boxy_light"
FIGURE_FONT = "DejaVu Sans"
TITLE = "PhenoCam Snapshots at Selected NEON and AmeriFlux Sites"
TITLE_FONTSIZE = 14
TITLE_Y = 0.98
TOP_MARGIN = 0.88
ROW_SPACE = 0.15  # room for the lower row's title under the upper row's timestamp


def read_url(url):
    """One GET, returning bytes, with a timeout so a hung host cannot stall the figure.

    Inputs: url - str
    Outputs: bytes
    """
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
        return response.read()


def camera_utc_offset(camera):
    """The camera's UTC offset, from the API, or None when it cannot be read.

    REPORTED, NOT USED FOR SELECTION. The filename timestamps are already local
    standard time, so noon is compared against them directly. The offset is
    carried so the console can say WHICH noon each panel is showing.

    Inputs: camera - the phenocam id
    Outputs: int or None
    """
    try:
        return json.loads(read_url(CAMERA_API_URL.format(camera=camera))).get("utc_offset")
    except Exception:
        return None


def candidate_images(camera, date):
    """Every RGB image the browse page lists for one camera on one date.

    Inputs: camera - the phenocam id; date - datetime.date
    Outputs: list of (local datetime, archive path, filename), oldest first
    """
    url = BROWSE_URL.format(camera=camera, year=date.year, month=f"{date.month:02d}", day=f"{date.day:02d}")
    try:
        page = read_url(url).decode("utf-8", "ignore")
    except Exception as error:
        print(f"{camera}: browse page unavailable ({type(error).__name__})")
        return []
    found = {}
    for match in IMAGE_PATH.finditer(page):
        if match["camera"] != camera or INFRARED_MARKER in match["name"]:
            continue
        stamp = datetime.datetime.strptime(match["stamp"], "%Y_%m_%d_%H%M%S")
        if stamp.date() != date:
            continue
        found[match.group(0)] = (stamp, match.group(0), match["name"])
    return sorted(found.values())


def closest_to_local_noon(images, date):
    """The image whose local time is nearest 12:00 on that date.

    Inputs: images - from candidate_images; date - datetime.date
    Outputs: (local datetime, archive path, filename) or None
    """
    if not images:
        return None
    noon = datetime.datetime.combine(date, TARGET_LOCAL_TIME)
    return min(images, key=lambda image: abs(image[0] - noon))


def fetch_image(camera, archive_path, file_name):
    """The JPEG on disk, downloaded only if it is not cached already.

    Inputs: camera; archive_path - the /data/archive/... path; file_name
    Outputs: Path to the cached file
    """
    destination = CACHE_DIR / camera / file_name
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(read_url(PHENOCAM_HOST + archive_path))
    return destination


def snapshot_for_site(row, date):
    """Everything one panel needs for one site, or a reason there is nothing.

    Inputs: row - a selected-sites row; date - datetime.date
    Outputs: dict with camera, image path, local time, utc offset and note
    """
    if not row.phenocam_ids:
        return {"camera": None, "path": None, "note": "no phenocam"}
    camera_number = CAMERA_NUMBER_BY_SITE.get(row.site_id, DEFAULT_CAMERA_NUMBER)
    if camera_number > len(row.phenocam_ids):
        return {"camera": None, "path": None, "note": f"no phenocam {camera_number}"}
    camera = row.phenocam_ids[camera_number - 1]
    chosen = closest_to_local_noon(candidate_images(camera, date), date)
    if chosen is None:
        return {"camera": camera, "path": None, "note": f"no RGB image on {date}"}
    stamp, archive_path, file_name = chosen
    try:
        path = fetch_image(camera, archive_path, file_name)
    except Exception as error:
        return {"camera": camera, "path": None, "note": f"download failed ({type(error).__name__})"}
    return {"camera": camera, "path": path, "local_time": stamp, "utc_offset": camera_utc_offset(camera), "note": None}


def draw_panel(ax, row, snapshot):
    """One site's snapshot, or a labelled blank where there is none.

    Inputs: ax; row - the selected-sites row; snapshot - from snapshot_for_site
    Outputs: None
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    camera_label = snapshot["camera"] or ""
    ax.set_title(f"{row.site_id}{' - ' + camera_label if camera_label else ''}", fontsize=PANEL_TITLE_FONTSIZE)
    if snapshot["path"] is None:
        ax.text(0.5, 0.5, snapshot["note"], transform=ax.transAxes, ha="center", va="center", fontsize=MISSING_FONTSIZE, color=MISSING_COLOR)
        return
    ax.imshow(mpimg.imread(snapshot["path"]))


def main():
    parser = argparse.ArgumentParser(description="PhenoCam Snapshots Per Site")
    parser.add_argument("date", help="the date to show, yyyy-mm-dd")
    args = parser.parse_args()
    try:
        date = datetime.date.fromisoformat(args.date)
    except ValueError:
        raise SystemExit(f"FAIL - {args.date!r} is not a date in yyyy-mm-dd form")

    selected = load_selected_pairs(SELECTED_SITES_CSV)
    # ONE COLUMN PER DOMAIN, NEON ON TOP. is_neon sorts descending so True, the
    # NEON site, lands in row 0 of its column.
    ordered = selected.sort_values(["domain", "is_neon"], ascending=[True, False])
    domains = sorted(selected.domain.unique())

    print(f"PhenoCam snapshots for {date}, closest to {TARGET_LOCAL_TIME:%H:%M:%S} local")
    snapshots = {}
    for _, row in ordered.iterrows():
        snapshot = snapshot_for_site(row, date)
        snapshots[row.site_id] = snapshot
        if snapshot["path"] is None:
            print(f"{row.domain} {row.site_id}: {snapshot['note']}")
        else:
            print(f"{row.domain} {row.site_id} ({snapshot['camera']}): {snapshot['local_time']:%H:%M:%S} local, {snapshot['path'].name}")

    theme = load_theme(FIGURE_THEME)
    theme.apply()
    plt.rcParams["font.family"] = FIGURE_FONT
    fig, axes = plt.subplots(ROWS, len(domains), figsize=(FIGURE_WIDTH_PER_PANEL * len(domains), FIGURE_HEIGHT_PER_ROW * ROWS))
    for column, domain in enumerate(domains):
        for panel_row, is_neon in enumerate((True, False)):
            site_row = ordered[(ordered.domain == domain) & (ordered.is_neon == is_neon)].iloc[0]
            draw_panel(axes[panel_row][column], site_row, snapshots[site_row.site_id])

    fig.suptitle(f"{TITLE}", fontsize=TITLE_FONTSIZE, y=TITLE_Y)
    theme.apply_transforms()
    fig.tight_layout()
    fig.subplots_adjust(top=TOP_MARGIN, hspace=ROW_SPACE)
    out_png = RESULTS_DIR / f"02_phenocam_snapshots_{date}.png"
    fig.savefig(out_png, dpi=FIGURE_DPI, facecolor=FIGURE_FACECOLOR)
    missing = [site_id for site_id, snapshot in snapshots.items() if snapshot["path"] is None]
    print(f"{len(snapshots) - len(missing)} of {len(snapshots)} panels have an image" + (f"; missing: {', '.join(missing)}" if missing else ""))
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
