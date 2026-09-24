"""Stage 4_3 - write the end member config for every NEON and AmeriFlux site.

Each site used for end members gets its own small config in config/endmembers/,
read by stage 4_4 onward.

TWO STEPS, IN THIS ORDER

    0. THE SRER TEMPLATE. config/endmembers/SRER_2022_endmembers.json is built
       first, from two sources:
           shared values - data, results and PLSP roots, the two CSV paths and
               the height thresholds - from config/srer_2022.json, the stage 1
               to 5 site config where they are already defined
           SRER's own values - domain, CRS, flight month, filenames - from the
               same lookups used for every other site
       If the file does not exist it is written. If it exists it is NEVER
       overwritten; instead the freshly built version is compared with it, field
       by field, and the script STOPS on any difference. The existing SRER file
       was already proven by a real stage 4_4 and 4_5 run, so this comparison
       is the check that the lookups are right before they are trusted for any
       other site.

    1. EVERY OTHER NEON SITE is built from that template: shared keys copied,
       site-specific keys looked up. A site whose config exists is skipped.

NOTHING SITE-SPECIFIC IS GUESSED. Four things differ between NEON sites, and
each comes from the source that defines it:

    site_id, domain - the selected-sites table
    expected_crs - the central meridian in the site's production PLSP file,
        because UTM zones differ between sites
    flight month - the NEON API's list of months with CHM for that site and
        year; a site flown in more than one month uses the month with the most
        CHM tiles, and every month found is recorded
    RGB and CHM filenames - real filenames from the NEON API listing for that
        month, with the tile id replaced by {tile}. The RGB name carries a
        flight number, 5 at SRER, that differs by site and cannot be assumed

The Data Portal folder names in the products block follow SRER's layout, built
from the domain, site and flight month.

NEON SITES AND IMAGERY YEARS, from steps/stage4_steps.md section 2, the years
closest to 2022 with NEON AOP imagery.

    Santa_Rita_Experimental_Range_NEON, SRER, 2022 - the template
    Onaqui_NEON, ONAQ, 2022
    Moab_NEON, MOAB, 2022
    San_Joaquin_Experimental_Range_NEON, SJER, 2023

AMERIFLUX SITES AND NAIP YEARS, one per domain, paired with the NEON site above
it. NAIP flies a state every two years, so the year is the available one nearest
2022 rather than 2022 itself.

    Walnut_Gulch_Kendall_Grasslands, WKG, 2023 - D14, with SRER
    Reynolds_Creek_Wyoming_big_sagebrush, RWS, 2021 - D15, with ONAQ
    Willard_Juniper_Savannah, WJS, 2022 - D13, with MOAB
    Tonzi_Ranch, TON, 2022 - D17, with SJER

AN AMERIFLUX CONFIG CARRIES NO products BLOCK. That block is NEON Data Portal
folders and filename patterns, and there is no equivalent for NAIP: stage 4_4
finds NAIP by a STAC search, not by a path. It carries naip_root instead of
data_root, and no NEON API lookup is made for it.

OUTPUTS

    config/endmembers/{SITE}_{imagery year}_endmembers.json, one per site

ARGUMENTS

    none. The template is always config/endmembers/SRER_2022_endmembers.json,
    and its shared values always come from config/srer_2022.json.

EXAMPLE COMMANDS

    conda activate LCSC
    python run_stage4_3_create_site_endmember_configs.py

The NEON API token is read from NEON_DATA_API_TOKEN in the environment.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import netCDF4
import run_stage1_1_download_neon_tiles as neon_download
from constants import SEVENTY
from helpers import read_selected_site_row, resolve_config_path, resolve_script_relative

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SITE_CONFIG_PATH = SCRIPT_DIRECTORY / "config" / "srer_2022.json"
CONFIG_DIRECTORY = SCRIPT_DIRECTORY / "config" / "endmembers"
TEMPLATE_PATH = CONFIG_DIRECTORY / "SRER_2022_endmembers.json"
TEMPLATE_SITE_NAME = "Santa_Rita_Experimental_Range_NEON"
NEON_SITE_IMAGERY_YEARS = {"Santa_Rita_Experimental_Range_NEON": 2022, "Onaqui_NEON": 2022, "Moab_NEON": 2022, "San_Joaquin_Experimental_Range_NEON": 2023}
AMERIFLUX_SITE_IMAGERY_YEARS = {"Walnut_Gulch_Kendall_Grasslands": 2023, "Reynolds_Creek_Wyoming_big_sagebrush": 2021, "Willard_Juniper_Savannah": 2022, "Tonzi_Ranch": 2022}
DEFAULT_NAIP_ROOT = "~/Dropbox/planet/data/NAIP"
DEFAULT_STAGE4_4_SETTINGS = {"min_flown_share": 0.95, "naip_window_m": 2000}
TEMPLATE_CHECK_FIELDS = ["site", "site_name", "site_id", "domain", "year", "expected_crs", "data_root", "results_root", "planet_data_root", "phenocam_csv", "plsp_layers_csv", "parameters", "stage4_4_endmember_tiles", "products.rgb.folder", "products.rgb.pattern", "products.chm.folder", "products.chm.pattern"]
RGB_NAME = re.compile(r"^(?P<year>\d{4})_(?P<site>[A-Z]{4})_(?P<flight>\d+)_(?P<easting>\d{6})_(?P<northing>\d{7})_image\.tif$")
CHM_NAME = re.compile(r"^NEON_(?P<domain>D\d{2})_(?P<site>[A-Z]{4})_DP3_(?P<easting>\d{6})_(?P<northing>\d{7})_CHM\.tif$")


def utm_epsg_from_production(planet_data_root, site_name):
    """The site's UTM zone as an EPSG code, from its production PLSP file.

    Inputs: planet_data_root; site_name
    Outputs: (EPSG string, production file name)
    """
    directory = resolve_config_path(planet_data_root, "PLSP_production_nc", site_name)
    candidates = sorted(glob.glob(str(directory / "*PLSP_*.nc")))
    if not candidates:
        raise SystemExit(f"FAIL - no production PLSP file in {directory}")
    with netCDF4.Dataset(candidates[0]) as dataset:
        central_meridian = float(dataset.variables["crs"].getncattr("longitude_of_central_meridian"))
    return f"EPSG:326{int((central_meridian + 183) / 6):02d}", Path(candidates[0]).name


def months_with_product(dpid, site_code, year):
    """Every month in a year for which NEON lists a product at a site.

    Inputs: dpid; site_code; year
    Outputs: sorted list of "YYYY-MM" strings
    """
    with urllib.request.urlopen(f"{neon_download.API}/products/{dpid}", timeout=60) as response:
        for entry in json.load(response)["data"]["siteCodes"]:
            if entry["siteCode"] == site_code:
                return sorted(month for month in entry["availableMonths"] if month.startswith(str(year)))
    return []


def file_names_for_month(dpid, site_code, month, token):
    """Every filename NEON lists for a product, site and month.

    Inputs: dpid; site_code; month; token
    Outputs: list of names
    """
    request = urllib.request.Request(f"{neon_download.API}/data/{dpid}/{site_code}/{month}", headers={"X-API-Token": token})
    with urllib.request.urlopen(request, timeout=120) as response:
        return [entry["name"] for entry in json.load(response)["data"]["files"]]


def filename_pattern(names, name_expression, build_pattern):
    """The one filename pattern a product uses, with the tile id as {tile}.

    Every matching name is turned into a pattern. A site that mixes patterns,
    for example two flight numbers in one month, is reported rather than
    silently reduced to one.

    Inputs: names; name_expression - compiled regex; build_pattern - function
            from a match to a pattern string
    Outputs: (most common pattern or None, Counter of all patterns found)
    """
    patterns = collections.Counter(build_pattern(match) for match in (name_expression.match(name) for name in names) if match)
    if not patterns:
        return None, patterns
    return patterns.most_common(1)[0][0], patterns


def shared_values_from_site_config():
    """The keys every end member config shares, taken from config/srer_2022.json.

    Site-specific keys are present as None so every config keeps the same key
    order; build_site_config fills them in.

    Inputs: none
    Outputs: dict
    """
    site_config = json.loads(SITE_CONFIG_PATH.read_text())
    return {
        "config_note": "Config created by running code/code_ground_truth_land_cover/v4/run_stage4_3_create_site_endmember_configs.py.",
        "site": None,
        "site_name": None,
        "site_id": None,
        "domain": None,
        "year": None,
        "year_note": "The year the end member polygons are drawn in: the NEON AOP or NAIP imagery year closest to 2022, and the PLSP year used for labelling and end member statistics.",
        "data_root": site_config["data_root"],
        "results_root": site_config["results_root"],
        "planet_data_root": site_config["stage1_3_planet_grid"]["planet_data_root"],
        "expected_crs": None,
        "expected_crs_note": "Must equal the UTM zone of the site's production PLSP file; 4_4 and 4_5 stop otherwise.",
        "phenocam_csv": site_config["phenocam_csv"],
        "plsp_layers_csv": site_config["plsp_layers_csv"],
        "parameters": {"H_GRASS_MAX": site_config["parameters"]["H_GRASS_MAX"], "H_TREE_MIN": site_config["parameters"]["H_TREE_MIN"]},
        "parameters_note": "Height rule for tile choice: CHM below H_GRASS_MAX is bare or grass, H_GRASS_MAX up to H_TREE_MIN is shrub, H_TREE_MIN and above is tree. woody_share counts pixels at or above H_GRASS_MAX. Calibrated at SRER and used unchanged at every site.",
        "products": {"rgb": {key: value for key, value in site_config["products"]["rgb"].items() if key not in ("folder", "pattern")}, "chm": {key: value for key, value in site_config["products"]["chm"].items() if key not in ("folder", "pattern")}},
        "stage4_4_endmember_tiles": dict(DEFAULT_STAGE4_4_SETTINGS),
    }


def build_site_config(shared, site_row, site_name, imagery_year, token, source_label):
    """One site's end member config: shared values plus looked-up values.

    Inputs: shared - dict of the keys every site shares; site_row; site_name;
            imagery_year; token; source_label - where the shared values came
            from, for the provenance note
    Outputs: (config dict, list of notes for the console)
    """
    notes = []
    site_code = (site_row.get("neon_id") or "").strip()
    if not site_code:
        raise SystemExit(f"FAIL - {site_name} has no neon_id in the selected-sites table")
    domain = site_row["neon_domain"].strip()
    expected_crs, production_file = utm_epsg_from_production(shared["planet_data_root"], site_name)

    chm_dpid = neon_download.PRODUCTS["chm"]["dpid"]
    rgb_dpid = neon_download.PRODUCTS["rgb"]["dpid"]
    months = months_with_product(chm_dpid, site_code, imagery_year)
    if not months:
        raise SystemExit(f"FAIL - NEON lists no CHM for {site_code} in {imagery_year}")
    chm_names_by_month = {month: file_names_for_month(chm_dpid, site_code, month, token) for month in months}
    tile_count_by_month = {month: sum(bool(CHM_NAME.match(name)) for name in names) for month, names in chm_names_by_month.items()}
    month = max(months, key=lambda candidate: tile_count_by_month[candidate])
    if len(months) > 1:
        notes.append(f"flown in {len(months)} months {tile_count_by_month}; using {month}, the one with the most CHM tiles")

    chm_pattern, chm_patterns = filename_pattern(chm_names_by_month[month], CHM_NAME, lambda match: f"NEON_{match['domain']}_{match['site']}_DP3_{{tile}}_CHM.tif")
    rgb_names = file_names_for_month(rgb_dpid, site_code, month, token)
    rgb_pattern, rgb_patterns = filename_pattern(rgb_names, RGB_NAME, lambda match: f"{match['year']}_{match['site']}_{match['flight']}_{{tile}}_image.tif")
    if chm_pattern is None or rgb_pattern is None:
        raise SystemExit(f"FAIL - {site_code} {month}: no {'CHM' if chm_pattern is None else 'RGB'} filenames matched the expected form")
    for label, patterns in (("CHM", chm_patterns), ("RGB", rgb_patterns)):
        if len(patterns) > 1:
            notes.append(f"{label} names follow {len(patterns)} patterns {dict(patterns)}; using the most common")

    config = json.loads(json.dumps(shared))
    config.update(
        {
            "site": site_code,
            "site_name": site_name,
            "site_id": site_row["site_id"].strip(),
            "domain": domain,
            "year": imagery_year,
            "expected_crs": expected_crs,
        }
    )
    config["products"]["rgb"].update({"folder": f"NEON_images-camera-ortho-mosaic/NEON.{domain}.{site_code}.DP3.30010.001.{month}.basic", "pattern": rgb_pattern})
    config["products"]["chm"].update({"folder": f"NEON_struct-ecosystem/NEON.{domain}.{site_code}.DP3.30015.001.{month}.basic", "pattern": chm_pattern})
    config["stage4_4_endmember_tiles"]["imagery_year"] = imagery_year
    config["neon_flight_month"] = month
    config["neon_flight_months_found"] = months
    config["provenance_note"] = f"Shared values from {source_label}. expected_crs from {production_file}; flight month and filename patterns from the NEON API for {site_code} {month}."
    return config, notes


def build_ameriflux_config(shared, site_row, site_name, imagery_year, source_label):
    """One AmeriFlux site's end member config, for a NAIP window.

    NO NEON LOOKUP RUNS HERE. An AmeriFlux site has no AOP flight month and no
    Data Portal folders, so the products block is dropped rather than filled
    with values that would never be read, and naip_root replaces data_root.
    expected_crs still comes from the site's own production PLSP file, because
    UTM zones differ between sites and the NAIP window is snapped to that grid.

    Inputs: shared; site_row; site_name; imagery_year; source_label
    Outputs: (config dict, list of notes for the console)
    """
    site_code = site_row["site_id"].strip().replace("US-", "").upper()
    expected_crs, production_file = utm_epsg_from_production(shared["planet_data_root"], site_name)
    config = json.loads(json.dumps(shared))
    config.pop("products")
    config.pop("data_root")
    config.update(
        {
            "site": site_code,
            "site_name": site_name,
            "site_id": site_row["site_id"].strip(),
            "domain": site_row["neon_domain"].strip(),
            "year": imagery_year,
            "expected_crs": expected_crs,
        }
    )
    config["naip_root"] = DEFAULT_NAIP_ROOT
    config["stage4_4_endmember_tiles"]["imagery_year"] = imagery_year
    config["imagery_note"] = "NAIP, one window centred on phenocam 1 and snapped to the PLSP grid. Stage 4_4 finds it by a STAC search of the Planetary Computer naip collection, so this config carries no products block."
    config["provenance_note"] = f"Shared values from {source_label}. expected_crs from {production_file}; NAIP year chosen as the available flight nearest 2022."
    return config, []


def field_value(config, dotted_name):
    """A possibly nested config value, addressed as "a.b.c".

    Inputs: config; dotted_name
    Outputs: the value, or the string "<absent>"
    """
    value = config
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return "<absent>"
        value = value[part]
    return value


def ensure_srer_template(token, selected_sites_csv):
    """Step 0: write the SRER template if absent, otherwise prove it matches.

    Inputs: token; selected_sites_csv - Path
    Outputs: None, stops on any mismatch
    """
    site_row = read_selected_site_row(selected_sites_csv, TEMPLATE_SITE_NAME)
    fresh, notes = build_site_config(shared_values_from_site_config(), site_row, TEMPLATE_SITE_NAME, NEON_SITE_IMAGERY_YEARS[TEMPLATE_SITE_NAME], token, SITE_CONFIG_PATH.name)
    print(f"\n0 SRER template, shared values from {SITE_CONFIG_PATH.name}")
    for note in notes:
        print(f"NOTE {note}")
    if not TEMPLATE_PATH.exists():
        TEMPLATE_PATH.write_text(json.dumps(fresh, indent=2) + "\n")
        print(f"wrote {TEMPLATE_PATH.name}")
        return
    existing = json.loads(TEMPLATE_PATH.read_text())
    mismatches = []
    for field in TEMPLATE_CHECK_FIELDS:
        built, stored = field_value(fresh, field), field_value(existing, field)
        agrees = built == stored
        if agrees:
            print(f"match {field}: {built}")
        else:
            print(f"DIFFERENT {field}: built {built} stored {stored}")
        if not agrees:
            mismatches.append(field)
    if mismatches:
        raise SystemExit(f"FAIL - {TEMPLATE_PATH.name} exists and was left untouched, but the lookups disagree with it on {mismatches}. No other site was written.")
    print(f"{TEMPLATE_PATH.name} exists, left untouched, and every checked field matches the lookups")


def main():
    argparse.ArgumentParser(description="Write the end member config for every NEON and AmeriFlux site, SRER template first.").parse_args()
    token = os.environ.get("NEON_DATA_API_TOKEN")
    if not token:
        raise SystemExit("FAIL - NEON_DATA_API_TOKEN is not set in the environment")
    CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    selected_sites_csv = resolve_script_relative(json.loads(SITE_CONFIG_PATH.read_text())["phenocam_csv"])

    print("Stage 4_3 - end member configs for the NEON and AmeriFlux sites")
    print("=" * SEVENTY)
    ensure_srer_template(token, selected_sites_csv)
    template = json.loads(TEMPLATE_PATH.read_text())
    shared = {key: value for key, value in template.items() if key not in ("neon_flight_month", "neon_flight_months_found", "provenance_note")}

    written = skipped = 0
    for site_name, imagery_year in NEON_SITE_IMAGERY_YEARS.items():
        if site_name == TEMPLATE_SITE_NAME:
            continue
        site_row = read_selected_site_row(selected_sites_csv, site_name)
        site_code = site_row["neon_id"].strip()
        target = CONFIG_DIRECTORY / f"{site_code}_{imagery_year}_endmembers.json"
        print(f"\n{site_code} {site_name}, NEON imagery taken from {imagery_year}")
        if target.exists():
            print(f"exists, left untouched: {target.name}")
            skipped += 1
            continue
        config, notes = build_site_config(shared, site_row, site_name, imagery_year, token, TEMPLATE_PATH.name)
        target.write_text(json.dumps(config, indent=2) + "\n")
        written += 1
        print(f"wrote {target.name}")
        print(f"domain {config['domain']}, {config['expected_crs']}, flight month {config['neon_flight_month']}")
        print(f"rgb {config['products']['rgb']['folder']} / {config['products']['rgb']['pattern']}")
        print(f"chm {config['products']['chm']['folder']} / {config['products']['chm']['pattern']}")
        for note in notes:
            print(f"NOTE {note}")
    for site_name, imagery_year in AMERIFLUX_SITE_IMAGERY_YEARS.items():
        site_row = read_selected_site_row(selected_sites_csv, site_name)
        site_code = site_row["site_id"].strip().replace("US-", "").upper()
        target = CONFIG_DIRECTORY / f"{site_code}_{imagery_year}_endmembers.json"
        print(f"\n{site_code} {site_name}, NAIP imagery taken from {imagery_year}")
        if target.exists():
            print(f"exists, left untouched: {target.name}")
            skipped += 1
            continue
        config, notes = build_ameriflux_config(shared, site_row, site_name, imagery_year, TEMPLATE_PATH.name)
        target.write_text(json.dumps(config, indent=2) + "\n")
        written += 1
        print(f"wrote {target.name}")
        print(f"domain {config['domain']}, {config['expected_crs']}, NAIP window {config['stage4_4_endmember_tiles']['naip_window_m']:.0f} m into {config['naip_root']}/{site_name}")
        for note in notes:
            print(f"NOTE {note}")
    print("\n" + "=" * SEVENTY)
    print(f"{written} written, {skipped} already existed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
