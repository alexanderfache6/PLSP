"""Stage 5_1 - RF-B, unmixing phenology characteristics into fractional cover (instructions5.md section 5 Step 5).

Maps 13 PLSP phenology characteristics to four class fractions, one prediction
per 3 m Planet pixel. This is a REGRESSOR, not a classifier: RF-A classifies 1 m
pixels and stage 4 counts them into fractions; this step predicts those
fractions directly from phenology (section 4.3).

THE FEATURE SET IS PHENOLOGY CHARACTERISTICS, NOT TIMING ALONE. Three groups,
all addressed by product_lyr number out of PLSP_Layers.csv:

    timing, layers 2 to 8 - OGI, 50PCGI, OGMx, Peak, OGD, 50PCGD, OGMn
    derived durations - DurGU, DurGD, LOS, differences of those dates
    greening, layers 9 to 11 - EVImax, EVIamp, EVIarea

TIMING AND GREENING ANSWER DIFFERENT QUESTIONS, which is why both are needed.
Timing says when a pixel greened up; greening says how green it got. At SRER the
class medians for dates differ by only 2 to 6 days, while EVImax separates bare
at 0.241 from tree at 0.341. Class identity lives mostly in greening, and site
and year live mostly in timing.

THIS SCRIPT REPLACES AN EARLIER SPLIT INTO TWO. Stage 5_1 once fitted timing
only and stage 5_1b added greening, so that the value of the greening layers
could be measured against a control. That comparison is settled and recorded in
stage5_results.md sections 9 and 11: timing alone did not beat a constant-mean
baseline on held-out tiles at all, and adding greening moved it to about 8 per
cent better. Both retired scripts are kept in `unused/`. There is now ONE
feature set and one script, and the work is unmixing of phenology
characteristics rather than purely temporal unmixing.

TRAINING USES EVERY RETAINED BLOCK, MIXED AND PURE ALIKE, AND UNWEIGHTED.
Restricting to pure blocks is the classic unmixing error: a model that has only
seen fractions of 0 and 1 measures distance to a decision boundary rather than
mixing proportion, and returns 0.85/0.15 for a genuinely half-and-half block.
Weighting by block_prediction_quality would be almost as bad - quality
correlates with mixedness at -0.824 at SRER, so weighting downweights exactly
the mixed blocks this project exists to resolve. Quality is therefore carried as
a REPORTED COVARIATE, never as a weight.

TWO MODELS ARE FITTED. THE JOINT ONE IS THE PRODUCT:

    joint - one RandomForestRegressor with a four-column target. Every leaf
            stores a mean of rows that each sum to 1, and a forest prediction
            averages leaf means, so the four fractions sum to 1 exactly.
    independent - four separate RandomForestRegressors, kept as a DIAGNOSTIC
            rather than a product. Its raw sum ranges from 0.45 to 1.83 site
            wide, which is unusable as a cover map but is a useful warning
            layer: the joint sum is always 1 and so can never signal that the
            model is confused, while a region where the independent sum drifts
            far from 1 marks where the model, or a transfer, is failing.

The decision and its numbers are recorded in stage5_results.md section 13.

NEITHER MODEL RENORMALISES, AND THE RAW SUM IS WRITTEN AS ITS OWN LAYER.
Whether four fractions reconstruct full coverage is a property worth seeing as a
map, and renormalising first would hide it.

SPATIAL HOLDOUT ONLY. Blocks are 3 m apart and strongly autocorrelated, so a
random split puts immediate neighbours on both sides of it and inflates every
score. Leave-one-tile-out over the train tiles, then a fit on all train tiles
evaluated against the four held-out test tiles, mirroring stage 3.

NO GRID TRANSFORMATION IS NEEDED OR PERFORMED. Stage 4_1 wrote the fractions on
the measured Planet grid - 3333 x 3334, origin 510555.0 / 3535548.0, EPSG:32612 -
precisely so the arrays align 1:1 with the LSP netCDF. Array index [row, col] in
a fraction raster IS index [y, x] in the netCDF. The alignment is asserted at
startup rather than assumed.

TWO MODES, DECIDED BY THE YEAR AND NOTHING ELSE:

    train, phenology year == ground truth year - fit, cross-validate, score
        against the held-out test tiles, map the site, and SAVE THE FITTED
        MODELS AS A PICKLE.
    predict, any other phenology year - load that pickle and map the year.
        Nothing is fitted, nothing is scored, and no difference layers are
        written, because stage 4 has ground truth for the training year alone.
        Predicted minus observed across two different years is land-cover
        change, not model error, and writing it as a difference layer would
        invite exactly that misreading.

Outputs are keyed on the PHENOLOGY year, so an off-year map never overwrites the
training year's.

Outputs -> `stage5_phenology_model_prediction/run{N}/`:

    stage5_1_report_{SITE}_{YEAR}.json - EVERY RESULT LIVES HERE, so that stage
        5_4 only renders and never recomputes: per fold, pooled cross-validated,
        out of bag, held-out test tiles, per class MAE, RMSE, bias and R
        squared on each, error by mixedness stratum, site-wide raw sum, and
        feature importance. THE OUT-OF-BAG BLOCK IS A RANDOM HOLDOUT, not a
        spatial one, so it is optimistic; its gap against the cross-validated
        block measures how much a random split would flatter this model.
        FEATURE IMPORTANCE is stored three ways: impurity, which the joint model
        can only give for all outputs together; permutation on the train rows;
        and permutation on the test tiles, both per class per feature.
    fraction_predicted_{model}_{fw}_{SITE}_{YEAR}.tif - float32, 4 bands, the
        site-wide map, NaN where QA fails
    fraction_predicted_sum_{model}_{fw}_{SITE}_{YEAR}.tif - float32, the RAW sum
        of the four predicted fractions before any renormalisation
    class_predicted_{model}_{fw}_{SITE}_{YEAR}.tif - uint8, the dominant
        predicted class per Planet pixel, for looking at rather than for area
    fraction_difference_{model}_{fw}_{SITE}_{YEAR}.tif - float32, 4 bands,
        predicted minus observed over the labelled tiles, positive where the
        model predicts more of that class than stage 4 observed. TRAIN MODE
        ONLY.

and, one level up in `stage5_phenology_model_prediction/`:

    rfb_model_{SITE}_{YEAR}_{fw}_{model}_{target}_run{N}.pkl - ONE GZIPPED
        PICKLE PER MODEL, joint and independent separately, each holding the
        model fitted on all train tiles plus the feature names, framework,
        target and feature set. They sit beside the run directories rather than
        inside one because an off-year run writes to its own directory but must
        reach a model another run trained. Every stored field is checked on
        load: a forest will predict happily from the wrong columns in the wrong
        order and return plausible nonsense. Separate files so an off-year run
        can load the product alone without reading the diagnostic.

A RUN DIRECTORY WRITTEN BY A DIFFERENT FEATURE SET IS NEVER OVERWRITTEN. The
existing `run5` holds retired timing-only outputs and `run5_timing_evi` holds
the 13-feature ones, both produced before this consolidation. If the target
directory already carries a report naming another feature set, the script stops
and says so, because those runs are the controls cited in stage5_results.md.

ARGUMENTS

    config
        positional, required. Site config JSON, e.g. config/srer_2022.json.
    --run
        required. The STAGE 4 run supplying the fraction targets, read from
        `stage4_aggregation/run{N}/stage4_2_planet_blocks/`. It also identifies the saved model, since
        what a model was fitted on is what defines it.
    --output-run
        default: the same as --run. The STAGE 5 output label, written to
        `stage5_phenology_model_prediction/run{N}/`. Needed because stage 4
        targets live in run5 while the stage 5 `run5` and `run5_timing_evi`
        directories are frozen controls, so a new run reads one and writes
        another.
    --framework
        default C. Which RF-A framework's fractions are the target. C is the
        transferable product; D is the sanity check.
    --target
        default fraction_hard_count, the AREA fraction counted from the hard
        1 m classification, 0/9 through 9/9. `fraction_soft_mean` averages
        RF-A's probability vectors instead and is available but not the default.
    --no-predict
        flag. Fit and score only, skip the site-wide map.
    --normalisation
        default T1_G1, the end member normalisation. `none` fits the raw
        features and exists for the invariance check described below.

THIS SCRIPT TRAINS, AND ONLY TRAINS, AND ONLY ON SRER'S GROUND TRUTH YEAR. The
year comes from the config's `year`; there is no phenology-year argument. Stage
4 made a fractional cover map for that one year, so it is the only year anything
can be fitted or scored against. Applying the model to another year, or to any
other site, is stage 5_2's job.

NORMALISATION, AND WHY IT MUST NOT CHANGE THE SCORES HERE. Every feature is
expressed relative to the site's own end members: subtract m, the median of the
four class medians recorded by stage 4_7. m is one number per feature, the same
for every pixel, and a random forest splits one feature at a time, so a constant
shift moves every split by the same amount and cannot change a single tree.
Running with --normalisation none must therefore reproduce these scores. If it
does not, m is being applied per pixel somewhere and the transfer is unsound.
The shift earns its keep at OTHER sites, where each subtracts ITS OWN m and the
features arrive on a common, end-member-relative scale.

EXAMPLE COMMANDS. Activate the environment rather than calling the interpreter
by absolute path: a bare `/opt/miniconda3/envs/LCSC/bin/python` does not set
PROJ_DATA, PROJ then falls back to base's older proj.db, and the run dies at the
first raster write with `CRSError: The EPSG code is unknown` after the entire
fit has completed.

    conda activate LCSC

    # the normal run, phenology year taken from the config
    python run_stage5_1_fit_phenology_fractional_cover.py config/srer_2022.json --run 6 --framework C

    # scores only, no site-wide map
    python run_stage5_1_fit_phenology_fractional_cover.py config/srer_2022.json --run 6 --framework C --no-predict

Then, in order, using the same run label:

    python run_stage5_2_generate_phenology_regression_plots.py config/srer_2022.json --run 6
    python run_stage5_3_create_qgis_phenology_project.py config/srer_2022.json --run 6
"""

import argparse
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import rasterio
import sklearn
import xarray as xr
from constants import CLASS_COLORS, CLASS_LABELS, CLASS_NAMES, NODATA, SEVENTY
from helpers import endmember_stats_directory, planet_blocks_directory, resolve_config_path
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestRegressor

# LAYERS ARE IDENTIFIED BY product_lyr NUMBER, NEVER BY A HAND-TYPED NAME.
# Layers 2 to 8 are the seven timing metrics: OGI, 50PCGI, OGMx, Peak, OGD,
# 50PCGD, OGMn. Names, scales and fill values all come from PLSP_Layers.csv at
# run time. A hand-written name list is how "50PCGD" once appeared twice in this
# constant, which silently dropped the 50% green-up date and double-counted the
# green-down one - a typo that changes the feature set and raises no error.
RAW_TIMING_LAYER_NUMBERS = [2, 3, 4, 5, 6, 7, 8]
CYCLE_COUNT_LAYER_NUMBER = 1
QA_LAYER_NUMBER = 12
DERIVED_DURATION_FEATURES = [("DurGU", 4, 2), ("DurGD", 8, 6), ("LOS", 8, 2)]
ACCEPTED_QA_VALUES = (1, 2)

# THE GREENING GROUP. Layer 9 EVImax is the maximum EVI2 in the cycle, layer 10
# EVIamp its amplitude, layer 11 EVIarea the integral. THEIR SCALES ARE NOT 1
# AND NOT EQUAL TO EACH OTHER: EVImax and EVIamp are 0.0001, EVIarea is 0.01,
# all read from PLSP_Layers.csv by number. Both product tiers spell all three
# with the specification name, so unlike layers 3 and 7 they need no
# alternate-spelling fallback - verified against every SRER year 2017 to 2025.
GREENING_LAYER_NUMBERS = [9, 10, 11]
FEATURE_SET_NAME = "phenology"

# NORMALISATION IS WHAT MAKES ONE MODEL TRANSFERABLE. Every feature is expressed
# relative to the site's own end members: subtract m, the median of the four
# class medians, recorded by stage 4_7 from the hand-drawn polygons. T1 does the
# seven dates, G1 the three greening features, and the three durations are
# RECOMPUTED from the shifted dates so LOS = OGMn - OGI still holds exactly.
#
# AT SRER THIS CANNOT CHANGE THE FIT, and that is the point of checking it. m is
# one number per feature, subtracted from every pixel alike, and a random forest
# splits one feature at a time, so a constant shift moves every split by the
# same amount and changes no tree. The scores must reproduce the unnormalised
# run; if they do not, m is being applied per pixel somewhere. Run the same
# command with --normalisation none to produce that comparison.
#
# The shift only matters at OTHER sites, where each site subtracts ITS OWN m and
# the features arrive on a common, end-member-relative scale.
NORMALISATION_NAME = "T1_G1"

# WHERE THE ONE MODEL COMES FROM. RF-B is fitted at SRER 2022 and nowhere else,
# because stage 4 built a fractional cover map for that site-year alone. Stage
# 5_2 needs to name the same pickle from a different site's config, so the pair
# is a constant here rather than something each caller reconstructs.
TRAINING_SITE = "SRER"
TRAINING_YEAR = 2022

# THE TWO PRODUCT TIERS SPELL FOUR LAYERS DIFFERENTLY, and neither spelling can
# be assumed. PLSP_stage_nc uses the specification names 50PCGI and 50PCGD;
# PLSP_production_nc uses GI_50PC and GD_50PC. Layers are therefore resolved by
# trying the CSV name first and the alternate spelling second, so the same code
# reads either tier.
ALTERNATE_NETCDF_NAME = {"50PCGI": "GI_50PC", "50PCGD": "GD_50PC", "50PCGI_2": "GI_50PC_2", "50PCGD_2": "GD_50PC_2"}

# STAGE FILES DECLARE NO _FillValue AT ALL, while production files declare
# 32767. Fill therefore comes from PLSP_Layers.csv, never from the file
# attributes: an attrs-based reader would leave 32767 unmasked on 29.77% of
# SRER 2022 pixels and feed it in as a real day of year.
PREDICTION_CHUNK_ROWS = 500_000

# PERMUTATION IMPORTANCE IS CAPPED AND REPEATED, and both numbers are here
# rather than on the command line because they are properties of the analysis,
# not of a run. Each feature costs one full prediction per repeat per split, so
# 13 features x 3 repeats x 2 splits x 2 models is 156 predictions; the row cap
# is what keeps that to minutes instead of hours. Permutation importance is
# stable well below 100,000 rows, so the cap costs precision that cannot be
# seen while removing most of the runtime.
PERMUTATION_REPEATS = 3
PERMUTATION_MAX_ROWS = 100_000
MIXEDNESS_STRATA_BREAKDOWN = [("pure", 0.0, 0.001), ("slightly mixed", 0.001, 0.25), ("mixed", 0.25, 0.5), ("highly mixed", 0.5, 1.0)]


def lsp_netcdf_path(config, phenology_year):
    """Full path to the PlanetScope LSP netCDF for one year.

    The directory carries the `_NEON` suffix and the filename does not
    (instructions5.md section 5.1 naming trap), so both are derived here from
    the single `site_name` config key and cannot drift apart.

    Inputs: config - the site config dict; phenology_year - int
    Outputs: Path
    """
    grid_settings = config["stage1_3_planet_grid"]
    site_directory_name = config["site_name"]
    site_name_without_suffix = site_directory_name[: -len("_NEON")] if site_directory_name.endswith("_NEON") else site_directory_name
    # TWO TIERS, TWO FILENAME CONVENTIONS. Production files carry the full
    # site identity; stage files are named PLSP_{YEAR}.nc only. Both are tried
    # rather than configured per year, because which tier holds a given year
    # changes as the product is released.
    candidates = []
    for subdirectory in (grid_settings["lsp_subdir"], "PLSP_stage_nc", "PLSP_production_nc"):
        for filename in (f"{config['ameriflux_id']}_NEON_{site_name_without_suffix}_PLSP_{phenology_year}.nc", f"PLSP_{phenology_year}.nc"):
            candidate = resolve_config_path(grid_settings["planet_data_root"], subdirectory, site_directory_name, filename)
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("FAIL - no LSP netCDF found for " + str(phenology_year) + ". Tried:\n  " + "\n  ".join(str(candidate_path) for candidate_path in candidates))


def read_layer_specification(config, config_path):
    """The PLSP layer table, read from PLSP_Layers.csv rather than transcribed.

    The CSV is the product specification and is the authority for short_name,
    scale, offset, fill_value and valid range. Reading it means those values
    cannot drift out of step with a hand-copied table, and a product revision is
    picked up rather than silently ignored.

    Inputs: config; config_path - accepted for symmetry with the other readers,
            unused because the CSV path is recorded relative to this script
    Outputs: dict of {product_lyr: entry dict}
    """
    csv_path = (Path(__file__).resolve().parent / config["plsp_layers_csv"]).resolve()
    if not csv_path.exists():
        raise SystemExit(f"FAIL - PLSP layer specification not found: {csv_path}")
    specification = {}
    with open(csv_path, newline="") as handle:
        for csv_row in csv.DictReader(handle):
            short_name = csv_row["short_name"].strip()
            specification[int(csv_row["product_lyr"])] = {
                "short_name": short_name,
                "scale": float(csv_row["scale"]),
                "offset": float(csv_row["offset"]),
                "fill_value": int(csv_row["fill_value"]),
                "valid_min": int(csv_row["valid_min"]),
                "valid_max": int(csv_row["valid_max"]),
                "units": csv_row["units"].strip(),
            }
    return specification


def resolve_layer(dataset, specification, layer_number):
    """Fetch one layer by its product_lyr number, failing loudly if it is absent.

    Addressing layers by NUMBER rather than by name keeps the CSV-to-netCDF name
    discrepancy in one place, and makes a missing or renamed variable an error
    that names both spellings instead of a silently dropped feature.

    Inputs: dataset - open xarray Dataset; specification; layer_number - int
    Outputs: (DataArray, its specification entry)
    """
    entry = specification[layer_number]
    for candidate_name in (entry["short_name"], ALTERNATE_NETCDF_NAME.get(entry["short_name"])):
        if candidate_name and candidate_name in dataset.variables:
            return dataset[candidate_name], entry
    tried = [entry["short_name"], ALTERNATE_NETCDF_NAME.get(entry["short_name"])]
    raise SystemExit(f"FAIL - layer {layer_number} not found under any known spelling {[name for name in tried if name]}. Present variables: {sorted(dataset.data_vars)}")


def read_quality_mask(dataset, specification):
    """The Step 2 QA mask, with both conditions kept even though they overlap.

    BOTH CONDITIONS ARE APPLIED DELIBERATELY. At SRER 2021 `NumCycles == 1` and
    `QA in (1, 2)` select the identical 91.11% of pixels, so the second removes
    nothing the first did not. That co-incidence is a property of one site-year,
    not a guarantee, and it must be re-checked per site (section 2.4), so the
    code applies both and reports whether they agreed.

    Inputs: dataset - an open xarray Dataset, mask_and_scale=False;
            specification - the PLSP_Layers.csv table
    Outputs: (bool array [ny, nx], diagnostics dict)
    """
    cycle_count = resolve_layer(dataset, specification, CYCLE_COUNT_LAYER_NUMBER)[0].values
    quality_flag = resolve_layer(dataset, specification, QA_LAYER_NUMBER)[0].values
    has_single_cycle = cycle_count == 1
    has_accepted_quality = np.isin(quality_flag, ACCEPTED_QA_VALUES)
    diagnostics = {
        "num_cycles_equals_one": float(has_single_cycle.mean()),
        "qa_in_accepted_values": float(has_accepted_quality.mean()),
        "both_conditions": float((has_single_cycle & has_accepted_quality).mean()),
        "conditions_select_identical_pixels": bool(np.array_equal(has_single_cycle, has_accepted_quality)),
    }
    return has_single_cycle & has_accepted_quality, diagnostics


def read_phenology_feature_stack(dataset, specification):
    """The 13 phenology characteristics, masked and scaled, on the Planet grid.

    ORDER OF OPERATIONS IS FIXED AND IS THE WHOLE POINT: mask fill to NaN, cast
    to float, apply the per-layer scale, and only then compute the derived
    durations. Fill is 32767, which is Int16 max, so `OGMx - OGI` on the raw
    integers yields 32767 - 32767 = 0 - a plausible zero-day duration that
    passes every range check, on 8.89% of pixels at SRER 2021 (section 5.3
    trap 1). The greening layers share that fill value and are masked by the
    same rule, before scaling, for the same reason: 32767 * 0.0001 is 3.2767, a
    perfectly plausible EVI2 that no range check would catch.

    Scale, offset and fill come from PLSP_Layers.csv, not from the file's own
    attributes and not from a transcribed table. Scales differ per layer: the
    timing layers are 1, EVImax and EVIamp are 0.0001, EVIarea is 0.01. Reading
    the scale per layer by number is what makes carrying the greening group a
    one-constant change rather than new scaling code.

    FEATURE ORDER IS TIMING, THEN DURATIONS, THEN GREENING, and the greening
    group is appended last so that the first ten columns are bit-for-bit the
    stage 5_1 feature matrix. That keeps the two runs comparable by construction.

    Inputs: dataset - an open xarray Dataset, mask_and_scale=False;
            specification - the PLSP_Layers.csv table
    Outputs: (float32 array [n_features, ny, nx], list of feature names)
    """
    scaled_layers = {}
    for layer_number in RAW_TIMING_LAYER_NUMBERS + GREENING_LAYER_NUMBERS:
        band, entry = resolve_layer(dataset, specification, layer_number)
        raw_values = band.values
        fill_value = entry["fill_value"]
        scale_factor = entry["scale"]
        values = raw_values.astype("float32")
        if fill_value is not None:
            values[raw_values == fill_value] = np.nan
        scaled_layers[layer_number] = values * scale_factor + entry["offset"]

    feature_names = [specification[number]["short_name"] for number in RAW_TIMING_LAYER_NUMBERS]
    ordered_layers = [scaled_layers[number] for number in RAW_TIMING_LAYER_NUMBERS]
    for derived_name, minuend_number, subtrahend_number in DERIVED_DURATION_FEATURES:
        feature_names.append(derived_name)
        ordered_layers.append(scaled_layers[minuend_number] - scaled_layers[subtrahend_number])
    for layer_number in GREENING_LAYER_NUMBERS:
        feature_names.append(specification[layer_number]["short_name"])
        ordered_layers.append(scaled_layers[layer_number])

    feature_stack = np.stack(ordered_layers).astype("float32")
    return feature_stack, feature_names


def read_endmember_normalisation(config, stage4_run, feature_names):
    """The site's end member offsets from stage 4_7, checked against this feature set.

    m IS READ, NEVER RECOMPUTED HERE. Stage 4_7 owns the definition, from the
    hand-drawn polygons through the buffer and QA rules, and a second
    implementation of the same median is a second thing to drift.

    The statistics file's feature list must equal this run's, in order. A model
    normalised against a differently ordered m would subtract EVIarea's offset
    from OGI and produce a plausible-looking map of nothing.

    Inputs: config - the site config; stage4_run - the stage 4 run label;
            feature_names - this run's 13 names, in order
    Outputs: dict with offsets, the source path, its sha256, and the class
             medians, ready to travel with the model
    """
    stats_path = endmember_stats_directory(config["results_root"], stage4_run) / f"{config['site']}_{config['year']}_endmember_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"FAIL - no end member statistics at {stats_path}. Run run_stage4_7_compute_endmember_statistics.py --run {stage4_run} first; the site must PASS stage 4_6.")
    statistics = json.loads(stats_path.read_text())
    if statistics["feature_names"] != feature_names:
        raise SystemExit(f"FAIL - {stats_path.name} was written for {statistics['feature_names']} but this run uses {feature_names}")
    digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    return {
        "name": NORMALISATION_NAME,
        "features": feature_names,
        "offsets": statistics["m_median_of_class_medians"],
        "source": str(stats_path),
        "source_sha256": digest,
        "endmember_year": statistics["plsp_year"],
        "class_medians": {class_name: summary["median"] for class_name, summary in statistics["classes"].items()},
    }


def apply_endmember_normalisation(feature_stack, feature_names, specification, normalisation):
    """Subtract the site's end member offsets, then rebuild the durations.

    THE DURATIONS ARE RECOMPUTED, NOT SHIFTED BY THEIR OWN OFFSET. A duration is
    a difference of two dates, so once the dates move it must be taken again
    from the moved dates or the identity LOS = OGMn - OGI quietly stops holding
    and the feature set becomes internally inconsistent.

    Inputs: feature_stack [13, ny, nx]; feature_names; specification - the
            PLSP_Layers.csv table; normalisation - from
            read_endmember_normalisation, or None to leave the features alone
    Outputs: float32 array of the same shape
    """
    if normalisation is None:
        return feature_stack
    index_of = {name: position for position, name in enumerate(feature_names)}
    normalised = feature_stack.astype("float32", copy=True)
    for layer_number in RAW_TIMING_LAYER_NUMBERS + GREENING_LAYER_NUMBERS:
        position = index_of[specification[layer_number]["short_name"]]
        normalised[position] -= np.float32(normalisation["offsets"][position])
    for derived_name, minuend_number, subtrahend_number in DERIVED_DURATION_FEATURES:
        minuend = index_of[specification[minuend_number]["short_name"]]
        subtrahend = index_of[specification[subtrahend_number]["short_name"]]
        normalised[index_of[derived_name]] = normalised[minuend] - normalised[subtrahend]
    return normalised


def assert_grid_alignment(fraction_dataset, netcdf_dataset, grid):
    """Confirm the fraction raster and the netCDF are the same grid, cell for cell.

    Stage 4_1 wrote the fractions on the measured Planet grid so that no
    resampling is ever needed here. That is only safe if it is checked: a silent
    half-pixel or one-row disagreement would pair every block with the wrong
    phenology and produce a plausible-looking model of nothing.

    The netCDF stores cell CENTRES and the raster stores cell EDGES, so the
    half-pixel shift is applied before comparison.

    Inputs: fraction_dataset - open rasterio dataset; netcdf_dataset - open
            xarray Dataset; grid - the stage 1_3 grid dict
    Outputs: None, raises SystemExit on any mismatch
    """
    pixel_size = grid["planet_pixel_m"]
    netcdf_origin_x = float(netcdf_dataset.x[0]) - pixel_size / 2.0
    netcdf_origin_y = float(netcdf_dataset.y[0]) + pixel_size / 2.0
    problems = []
    if (fraction_dataset.width, fraction_dataset.height) != (netcdf_dataset.sizes["x"], netcdf_dataset.sizes["y"]):
        problems.append(f"dimensions {fraction_dataset.width} x {fraction_dataset.height} against netCDF {netcdf_dataset.sizes['x']} x {netcdf_dataset.sizes['y']}")
    if abs(fraction_dataset.transform.c - netcdf_origin_x) > 1e-10 or abs(fraction_dataset.transform.f - netcdf_origin_y) > 1e-10:
        problems.append(f"origin {fraction_dataset.transform.c}, {fraction_dataset.transform.f} against netCDF {netcdf_origin_x}, {netcdf_origin_y}")
    if abs(fraction_dataset.transform.a - pixel_size) > 1e-10:
        problems.append(f"pixel size {fraction_dataset.transform.a} against grid {pixel_size}")
    if problems:
        raise SystemExit("FAIL - fraction raster and LSP netCDF are not the same grid: " + "; ".join(problems))
    print("grid alignment verified: fraction raster index [row, col] is netCDF index [y, x], no transformation applied")


def assign_blocks_to_tiles(config, grid):
    """Which labelled NEON tile each Planet block belongs to, for spatial holdout.

    Stage 4_1 writes no tile raster, so membership is recomputed from the grid
    geometry. A block is attributed to the tile containing its CENTRE, which is
    unambiguous even though tile boundaries cut through blocks - no SRER tile is
    congruent with the Planet grid in both axes (section 5 Step 3).

    Inputs: config; grid - the stage 1_3 grid dict
    Outputs: (int16 array [ny, nx], -1 where no tile; list of tile ids in order)
    """
    pixel_size = grid["planet_pixel_m"]
    block_centre_x = grid["origin_x"] + (np.arange(grid["nx"]) + 0.5) * pixel_size
    block_centre_y = grid["origin_y"] - (np.arange(grid["ny"]) + 0.5) * pixel_size
    tile_index = np.full((grid["ny"], grid["nx"]), -1, dtype="int16")
    tile_ids = list(config["tiles"])
    tile_size_m = float(config["stage1_3_planet_grid"]["neon_tile_size_m"])
    for tile_number, tile_id in enumerate(tile_ids):
        tile_easting, tile_northing = (int(value) for value in tile_id.split("_"))
        inside_x = (block_centre_x >= tile_easting) & (block_centre_x < tile_easting + tile_size_m)
        inside_y = (block_centre_y >= tile_northing) & (block_centre_y < tile_northing + tile_size_m)
        tile_index[np.ix_(inside_y, inside_x)] = tile_number
    return tile_index, tile_ids


def compute_block_mixedness(fractions):
    """How mixed each block is: 1 minus its largest fraction.

    Zero for a perfectly pure block, 0.75 for an even four-way mix. Error is
    stratified by this because pure blocks are the easy case and a pooled score
    can hide complete failure on the mixtures, which is the regime that matters.

    Inputs: fractions - [n_blocks, n_classes]
    Outputs: [n_blocks] float array
    """
    return 1.0 - fractions.max(axis=1)


def fit_joint_multioutput_forest(features, fraction_targets, seed):
    """One RandomForestRegressor with a four-column target.

    Trees split on all four fractions together, so the model can exploit the
    fact that the classes are not independent.

    Inputs: features [n, 13]; fraction_targets [n, 4]; seed - int
    Outputs: fitted RandomForestRegressor
    """
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed, oob_score=True)
    model.fit(features, fraction_targets)
    return model


def fit_independent_per_class_forests(features, fraction_targets, seed):
    """Four separate RandomForestRegressors, one per class, same rows and features.

    The ablation against the joint model: if these score the same, the joint
    target is buying nothing.

    Inputs: features [n, 13]; fraction_targets [n, 4]; seed - int
    Outputs: list of four fitted RandomForestRegressors, in class-code order
    """
    per_class_models = []
    for class_column in range(fraction_targets.shape[1]):
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed, oob_score=True)
        model.fit(features, fraction_targets[:, class_column])
        per_class_models.append(model)
    return per_class_models


def predict_with_model(model_name, fitted_model, features):
    """Predict fractions with whichever model form was fitted.

    Inputs: model_name - "joint" or "independent"; fitted_model - the object or
            list returned by the corresponding fit function; features [n, 13]
    Outputs: [n, 4] predicted fractions, NOT renormalised
    """
    if model_name == "joint":  # NOTE same model for all classes
        return fitted_model.predict(features)
    else:  # NOTE separate model per class
        return np.column_stack([per_class.predict(features) for per_class in fitted_model])


# NOTE quality metrics
def score_fraction_predictions(true_fractions, predicted_fractions):
    """Per-class and pooled error, plus the simplex violation before renormalising.

    Bias is reported separately from MAE because a systematic under-prediction
    is what propagates downstream, exactly as at stage 3, while MAE alone cannot
    distinguish it from symmetric noise.

    Inputs: true_fractions, predicted_fractions - [n, 4]
    Outputs: dict of scores
    """
    error = predicted_fractions - true_fractions
    per_class_scores = {}
    for class_column, class_name in enumerate(CLASS_NAMES):
        column_error = error[:, class_column]
        column_truth = true_fractions[:, class_column]
        truth_variance = float(((column_truth - column_truth.mean()) ** 2).mean())
        per_class_scores[class_name] = {
            "mae": float(np.abs(column_error).mean()),
            "rmse": float(np.sqrt((column_error**2).mean())),
            "bias": float(column_error.mean()),
            "r2": float(1.0 - (column_error**2).mean() / truth_variance) if truth_variance > 0 else None,
        }
    raw_sum = predicted_fractions.sum(axis=1)
    return {
        "n": int(len(true_fractions)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "per_class": per_class_scores,
        "raw_sum_before_renormalisation": {
            "mean": float(raw_sum.mean()),
            "min": float(raw_sum.min()),
            "max": float(raw_sum.max()),
            "mean_absolute_deviation_from_one": float(np.abs(raw_sum - 1.0).mean()),
        },
    }


# NOTE mixedness of a 3mx3m block coming from compute_block_mixedness()
def score_by_mixedness_stratum(true_fractions, predicted_fractions, mixedness):
    """Error split by how mixed the block is.

    THIS IS THE HEADLINE, not the pooled score. Pure and near-pure blocks are
    the easy case and dominate the count, so a model can look strong while
    failing completely on mixtures.

    Inputs: true_fractions, predicted_fractions [n, 4]; mixedness [n]
    Outputs: dict of {stratum label: scores}
    """
    strata = {}
    for label, lower, upper in MIXEDNESS_STRATA_BREAKDOWN:
        selected = mixedness < upper if label == "pure" else (mixedness >= lower) & (mixedness < upper)
        if selected.any():
            strata[label] = score_fraction_predictions(true_fractions[selected], predicted_fractions[selected])
    return strata


def score_out_of_bag(model_name, fitted_model, true_fractions):
    """Score the out-of-bag predictions of the model fitted on all train rows.

    OOB IS A RANDOM HOLDOUT, NOT A SPATIAL ONE, and that is the entire reason to
    report it. Blocks are 3 m apart and strongly autocorrelated, so a block held
    out of one tree still has near-duplicate neighbours inside that tree's bag.
    OOB is therefore OPTIMISTIC, and THE GAP BETWEEN IT AND THE LEAVE-ONE-TILE-
    OUT SCORE IS THE MEASUREMENT: it is how much a random split would have
    flattered this model. Quote the cross-validated number as generalisation and
    never this one.

    Scored through the same function as every other split, so the numbers are
    directly comparable rather than sklearn's own oob_score_, which is R squared
    only and hides the per-class and raw-sum detail.

    A row that happened to be in-bag for every tree has no OOB prediction and
    arrives as NaN. Those rows are dropped and counted, not scored.

    Inputs: model_name - "joint" or "independent"; fitted_model; true_fractions
            [n_rows, 4], the same rows the model was fitted on
    Outputs: scores dict, with n_without_oob_prediction added
    """
    if model_name == "joint":
        out_of_bag_predictions = fitted_model.oob_prediction_
    else:
        out_of_bag_predictions = np.column_stack([per_class_model.oob_prediction_ for per_class_model in fitted_model])
    has_prediction = np.all(np.isfinite(out_of_bag_predictions), axis=1)
    scores = score_fraction_predictions(true_fractions[has_prediction], out_of_bag_predictions[has_prediction])
    scores["n_without_oob_prediction"] = int((~has_prediction).sum())
    return scores


def impurity_importance(model_name, fitted_model):
    """Mean decrease in impurity per feature, as the forest reports it.

    THE JOINT MODEL CANNOT GIVE THIS PER CLASS. Its trees split on all four
    outputs at once, so sklearn returns ONE vector for the whole model, not one
    per class. The independent model, being four separate forests, does give a
    vector per class. That asymmetry is why permutation importance exists below:
    it is the only per-class measure available for the product model.

    Impurity importance is also computed on the training data and is biased
    toward features with more distinct values, which here means the greening
    layers over the day-of-year ones. Read it as a cross-check, not as evidence.

    Inputs: model_name; fitted_model
    Outputs: {label: [one value per feature]}
    """
    if model_name == "joint":
        return {"all_outputs": [float(value) for value in fitted_model.feature_importances_]}
    return {class_name: [float(value) for value in per_class_model.feature_importances_] for class_name, per_class_model in zip(CLASS_NAMES, fitted_model)}


def score_permutation_importance(model_name, fitted_model, features, true_fractions, seed):
    """Per-class permutation importance: the MAE cost of shuffling each feature.

    THIS IS THE DEFENSIBLE MEASURE, and the only one that is per class for the
    joint model. Shuffle one feature column, predict again, and record how much
    each class's MAE rises. A feature the model genuinely uses for grass shows a
    large rise on grass and little elsewhere.

    Computed on whichever split is passed, so the train and test answers can be
    compared: a feature that matters in training but not out of tile was being
    used to memorise, not to generalise.

    Rows are capped at PERMUTATION_MAX_ROWS and drawn without replacement,
    because the estimate is stable long before the full 600,000 rows and the
    cost is one full forest prediction per feature per repeat.

    Inputs: model_name; fitted_model; features [n, 13]; true_fractions [n, 4];
            seed - int
    Outputs: dict with the per-class increase per feature, plus what it was
             measured on
    """
    generator = np.random.default_rng(seed)
    if len(features) > PERMUTATION_MAX_ROWS:
        selected = generator.choice(len(features), PERMUTATION_MAX_ROWS, replace=False)
        features, true_fractions = features[selected], true_fractions[selected]
    baseline_mae = np.abs(predict_with_model(model_name, fitted_model, features) - true_fractions).mean(axis=0)
    per_class_increase = {class_name: [] for class_name in CLASS_NAMES}
    for feature_index in range(features.shape[1]):
        repeats = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = features.copy()
            shuffled[:, feature_index] = generator.permutation(shuffled[:, feature_index])
            shuffled_mae = np.abs(predict_with_model(model_name, fitted_model, shuffled) - true_fractions).mean(axis=0)
            repeats.append(shuffled_mae - baseline_mae)
        mean_increase = np.mean(repeats, axis=0)
        for class_index, class_name in enumerate(CLASS_NAMES):
            per_class_increase[class_name].append(float(mean_increase[class_index]))
    return {
        "n_rows_used": int(len(features)),
        "n_repeats": PERMUTATION_REPEATS,
        "baseline_mae": {class_name: float(baseline_mae[class_index]) for class_index, class_name in enumerate(CLASS_NAMES)},
        "per_class_mae_increase": per_class_increase,
    }


def score_constant_mean_baseline(true_fractions):
    """The null model: predict the training mean fraction for every block.

    RF-B must beat this to have learned anything at all. Without it a figure
    like "MAE 0.12" carries no information.

    Inputs: true_fractions [n, 4]
    Outputs: dict of scores
    """
    constant_prediction = np.tile(true_fractions.mean(axis=0), (len(true_fractions), 1))
    return score_fraction_predictions(true_fractions, constant_prediction)


def predict_site_wide(model_name, fitted_model, feature_stack, usable_mask, grid):
    """Predict fractions for every QA-passing Planet pixel on the site.

    Chunked because the site is 11.1 million pixels: predicting in one call
    would hold the full feature matrix and every tree's output simultaneously.

    Inputs: model_name; fitted_model; feature_stack [10, ny, nx]; usable_mask
            [ny, nx]; grid
    Outputs: (fractions [4, ny, nx] with NaN where unusable, raw sum [ny, nx])
    """
    n_pixels = grid["ny"] * grid["nx"]
    flat_features = feature_stack.reshape(feature_stack.shape[0], -1)
    usable_positions = np.flatnonzero(usable_mask.ravel())
    predicted_flat = np.full((len(CLASS_NAMES), n_pixels), np.nan, dtype="float32")
    for start in range(0, len(usable_positions), PREDICTION_CHUNK_ROWS):
        chunk_positions = usable_positions[start : start + PREDICTION_CHUNK_ROWS]
        chunk_prediction = predict_with_model(model_name, fitted_model, flat_features[:, chunk_positions].T)
        predicted_flat[:, chunk_positions] = chunk_prediction.T.astype("float32")
        print(f"predicted {min(start + PREDICTION_CHUNK_ROWS, len(usable_positions)):,} of {len(usable_positions):,} pixels", end="\r")
    print(" " * SEVENTY, end="\r")
    predicted = predicted_flat.reshape(len(CLASS_NAMES), grid["ny"], grid["nx"])
    raw_sum = predicted.sum(axis=0)
    return predicted, raw_sum


def predicted_minus_observed(predicted_fractions, observed_fractions):
    """Predicted minus observed fraction, per class, where both exist.

    THIS IS THE ONLY LAYER THAT SHOWS WHERE THE MODEL IS WRONG. A summary MAE
    says how much error there is; this says where it is, and whether it is
    structured. A model that is uniformly noisy and a model that is
    systematically wrong over one landform can share an MAE and are not the same
    problem.

    Defined only over the labelled tiles, since observed fractions exist nowhere
    else - which is itself the point worth seeing on a map, because the rest of
    the site is extrapolation.

    Sign convention: POSITIVE means the model predicts MORE of that class than
    stage 4 observed.

    Inputs: predicted_fractions, observed_fractions - [n_classes, ny, nx]
    Outputs: float32 [n_classes, ny, nx], NaN where either side is missing
    """
    difference = predicted_fractions - observed_fractions
    both_present = np.isfinite(predicted_fractions) & np.isfinite(observed_fractions)
    difference[~both_present] = np.nan
    return difference.astype("float32")


def hard_class_from_fractions(predicted_fractions):
    """The dominant predicted class per Planet pixel, as a hard map.

    ARGMAX OF THE FRACTIONS, AND IT IS A DIFFERENT PRODUCT FROM THE FRACTIONS,
    not a better one. It answers "which class covers most of this 3 m pixel",
    which is legible on a map and comparable against a conventional land-cover
    product, while discarding the mixture that the fractions exist to carry. A
    pixel at 0.34 shrub / 0.33 grass / 0.33 bare becomes confidently shrub here.
    Use it to look at, never to compute area from - the fractions are the area
    product (instructions5.md section 5 Step 3).

    Inputs: predicted_fractions [n_classes, ny, nx], NaN where not predicted
    Outputs: uint8 [ny, nx] of class codes, NODATA where not predicted
    """
    predicted_somewhere = np.isfinite(predicted_fractions[0])
    dominant = np.full(predicted_fractions.shape[1:], NODATA, dtype="uint8")
    if predicted_somewhere.any():
        dominant[predicted_somewhere] = np.nanargmax(predicted_fractions[:, predicted_somewhere], axis=0).astype("uint8")
    return dominant


def write_class_raster(path, class_codes, grid):
    """Write a hard class map with the locked section 3 colour table embedded.

    The colour table travels inside the GeoTIFF so QGIS and GDAL both render the
    locked palette with no sidecar file (section 12 Q10).

    Inputs: path; class_codes - uint8 [ny, nx]; grid
    Outputs: None
    """
    transform = from_origin(grid["origin_x"], grid["origin_y"], grid["planet_pixel_m"], grid["planet_pixel_m"])
    profile = {
        "driver": "GTiff",
        "height": grid["ny"],
        "width": grid["nx"],
        "count": 1,
        "dtype": "uint8",
        "crs": grid["epsg"],
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(class_codes, 1)
        destination.write_colormap(1, {code: hex_to_rgba(CLASS_COLORS[code]) for code in CLASS_LABELS})
        destination.set_band_description(1, "dominant_predicted_class")


def hex_to_rgba(hex_colour):
    """Convert a #rrggbb string to an (r, g, b, 255) tuple for a GeoTIFF colormap."""
    digits = hex_colour.lstrip("#")
    return tuple(int(digits[offset : offset + 2], 16) for offset in (0, 2, 4)) + (255,)


def write_prediction_raster(path, data, grid, band_descriptions):
    """Write a prediction product on the Planet grid, aligned 1:1 with the netCDF."""
    data = data if data.ndim == 3 else data[np.newaxis, :, :]
    transform = from_origin(grid["origin_x"], grid["origin_y"], grid["planet_pixel_m"], grid["planet_pixel_m"])
    profile = {
        "driver": "GTiff",
        "height": grid["ny"],
        "width": grid["nx"],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": grid["epsg"],
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(data.astype("float32"))
        for band_number, description in enumerate(band_descriptions, start=1):
            destination.set_band_description(band_number, description)


def fitted_model_path(results_root, site, ground_truth_year, framework, model_name, target, run_label):
    """Where one fitted model is pickled.

    ONE FILE PER MODEL, not one file holding both. The joint model is the
    product and the independent one only a diagnostic, so an off-year run that
    wants the product should not have to read the diagnostic to reach it. They
    sit in the stage 5 root rather than inside a run directory because an
    off-year run writes to its own directory but must reach a model another run
    trained. The filename carries every field that changes what was fitted, so
    two frameworks, models or targets cannot silently share a file.

    Inputs: results_root - Path; site; ground_truth_year - int; framework;
            model_name - "joint" or "independent"; target; run_label
    Outputs: Path
    """
    directory = results_root / "stage5_phenology_model_prediction"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"rfb_model_{site}_{ground_truth_year}_{framework}_{model_name}_{target}_run{run_label}.pkl"


def save_fitted_model(path, model_name, fitted_model, report):
    """Pickle one fitted model, gzipped, with the metadata needed to check reuse.

    GZIPPED BECAUSE THESE ARE LARGE. Five forests of 200 trees grown on roughly
    600,000 rows came to 7.4 GB uncompressed in one file, inside a synced
    folder. Compression level 6 rather than 9: the last levels cost minutes on
    a file this size and buy little on forest internals.

    The model alone is not enough. A forest will predict from a feature matrix
    with the wrong columns in the wrong order and return plausible nonsense, so
    the feature names, framework, target, feature set and ground-truth year
    travel with it.

    Inputs: path - Path; model_name; fitted_model; report - dict
    Outputs: None, writes the pickle
    """
    payload = {
        "model": fitted_model,
        "model_name": model_name,
        "site": report["site"],
        "ground_truth_year": report["ground_truth_year"],
        "framework": report["framework"],
        "target": report["target"],
        "feature_set": report["feature_set"],
        "features": report["features"],
        "source_run": report["source_run"],
        # THE NORMALISATION TRAVELS WITH THE MODEL. Stage 5_2 subtracts the
        # TARGET site's own m, but it has to know which transform was used and
        # against which statistics, or it could feed raw features to a model
        # fitted on shifted ones and return a map that looks fine.
        "normalisation": report["normalisation"],
        "training_feature_range": report["training_feature_range"],
        "sklearn_version": sklearn.__version__,
    }
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_model_payload(path):
    """The whole pickled bundle: the model and everything saved beside it.

    Stage 5_2 needs the metadata as much as the forest - the feature order, the
    normalisation it was fitted with, and the training range it must report
    extrapolation against - so it reads the payload rather than the model alone.

    Inputs: path - Path to the gzipped pickle
    Outputs: dict
    """
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def load_fitted_model(path, model_name, report):
    """Load one pickled model, refusing anything that does not match this run.

    EVERY STORED FIELD IS CHECKED, not just the feature count. Predicting an off
    year with a model fitted on a different feature set, framework or target
    would produce a map that looks entirely normal and means nothing.

    Inputs: path - Path; model_name; report - dict for the run being predicted
    Outputs: the fitted model
    """
    if not path.exists():
        raise SystemExit(f"FAIL - no trained {model_name} model at {path}. Train it first by running this script for the ground truth year {report['ground_truth_year']}.")
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    for field in ("site", "ground_truth_year", "framework", "target", "feature_set", "features"):
        if payload.get(field) != report[field]:
            raise SystemExit(f"FAIL - {path.name} was fitted with {field} {payload.get(field)!r}, but this run needs {report[field]!r}. Retrain on the ground truth year.")
    print(f"loaded {model_name} model trained on {payload['ground_truth_year']} from {path.name}")
    return payload["model"]


def main():
    parser = argparse.ArgumentParser(description="Fit RF-B, unmixing PlanetScope phenology characteristics into fractional cover.")
    parser.add_argument("config", help="site config JSON")
    parser.add_argument("--run", required=True, help="stage 4 run label supplying the fraction targets, e.g. 5")
    parser.add_argument("--framework", default="C", help="RF-A framework whose fractions are the target (default C, the transferable product)")
    parser.add_argument("--target", default="fraction_hard_count", choices=["fraction_hard_count", "fraction_soft_mean"], help="which stage 4 fraction estimate to regress on (default fraction_hard_count, the area fraction from hard classification)")
    parser.add_argument("--output-run", default=None, help="stage 5 output label (default: the same as --run)")
    parser.add_argument("--no-predict", action="store_true", help="fit and score only, skip the site-wide map")
    parser.add_argument("--normalisation", default=NORMALISATION_NAME, choices=[NORMALISATION_NAME, "none"], help=f"end member normalisation (default {NORMALISATION_NAME}; 'none' fits the raw features, for the invariance check)")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    site, ground_truth_year = config["site"], config["year"]
    seed = config["BASE_SEED"] + ground_truth_year
    # THIS SCRIPT TRAINS, AND ONLY TRAINS, AND ONLY ON THE GROUND TRUTH YEAR.
    # Stage 4 made a fractional cover map for that one year, so it is the only
    # year a model can be fitted or scored against. Applying the model to any
    # other year, or to any other site, is stage 5_2's job: it loads this
    # script's pickle, reads THAT site's end members, and predicts.
    phenology_year = ground_truth_year

    results_root = resolve_config_path(config["results_root"])
    qa_directory = results_root / "stage1_data_and_features" / "qa"
    aggregation_directory = planet_blocks_directory(results_root, args.run)
    # THE INPUT RUN AND THE OUTPUT LABEL CAN DIFFER, and usually must. Stage 4
    # targets live in run5, while run5 and run5_timing_evi on the stage 5 side
    # are frozen controls, so a new stage 5 run reads run5 and writes elsewhere.
    # --output-run defaults to --run, so a fresh pair needs no extra flag.
    output_label = f"run{args.output_run or args.run}"
    output_directory = results_root / "stage5_phenology_model_prediction" / output_label
    output_directory.mkdir(parents=True, exist_ok=True)

    grid = json.loads((qa_directory / f"planet_grid_{site}_{ground_truth_year}.json").read_text())["grid"]
    layer_specification = read_layer_specification(config, args.config)

    print(f"RF-B - phenology characteristics to fractional cover - {site}")
    print("=" * SEVENTY)
    print(f"ground truth year {ground_truth_year} (stage 4 run {args.run}, framework {args.framework}, target {args.target})")
    if (args.output_run or args.run) != args.run:
        print(f"reading stage 4 run {args.run}, writing stage 5 {output_label}")
    print(f"feature set {FEATURE_SET_NAME}, 13 characteristics, writing to {output_label}")

    # NEVER OVERWRITE A RUN WRITTEN BY A DIFFERENT FEATURE SET. run5 holds the
    # retired timing-only outputs and run5_timing_evi the 13-feature ones, and
    # both are controls cited in stage5_results.md sections 9, 11 and 13.
    existing_report_path = output_directory / f"stage5_1_report_{site}_{ground_truth_year}.json"
    if existing_report_path.exists():
        existing_feature_set = json.loads(existing_report_path.read_text()).get("feature_set", "timing")
        if existing_feature_set != FEATURE_SET_NAME:
            raise SystemExit(f"FAIL - {output_directory} already holds a '{existing_feature_set}' run and this script writes '{FEATURE_SET_NAME}'. Pass a different --output-run label, or move that directory aside first.")
    print("")

    netcdf_path = lsp_netcdf_path(config, phenology_year)
    if not netcdf_path.exists():
        raise SystemExit(f"FAIL - LSP netCDF not found: {netcdf_path}")
    fraction_path = aggregation_directory / f"{args.target}_{args.framework}_{site}_{ground_truth_year}.tif"
    if not fraction_path.exists():
        raise SystemExit(f"FAIL - {fraction_path} not found. Run run_stage4_1_aggregate_base_map_to_planet_blocks.py --run {args.run} --frameworks {args.framework} first.")

    with xr.open_dataset(netcdf_path, mask_and_scale=False) as netcdf_dataset:
        with rasterio.open(fraction_path) as fraction_dataset:
            assert_grid_alignment(fraction_dataset, netcdf_dataset, grid)
            true_fraction_stack = fraction_dataset.read().astype("float32")
        quality_mask, quality_diagnostics = read_quality_mask(netcdf_dataset, layer_specification)
        feature_stack, feature_names = read_phenology_feature_stack(netcdf_dataset, layer_specification)

    # NORMALISE BEFORE ANYTHING ELSE TOUCHES THE FEATURES, so every score, every
    # importance and the pickled model all describe the same, end-member-relative
    # feature space. Normalising later, after the usable mask or the split, would
    # leave the saved model expecting inputs no one else produces.
    normalisation = read_endmember_normalisation(config, args.run, feature_names) if args.normalisation != "none" else None
    if normalisation:
        print(f"normalisation {normalisation['name']} from {Path(normalisation['source']).name}, end members {normalisation['endmember_year']}")
        print("m per feature: " + ", ".join(f"{name} {value:.3f}" for name, value in zip(feature_names, normalisation["offsets"])))
        feature_stack = apply_endmember_normalisation(feature_stack, feature_names, layer_specification, normalisation)
    else:
        print("normalisation none - raw features, for the invariance check against the normalised run")

    all_features_finite = np.all(np.isfinite(feature_stack), axis=0)
    usable_phenology = quality_mask & all_features_finite
    quality_diagnostics["all_features_finite"] = float(all_features_finite.mean())
    quality_diagnostics["usable_phenology"] = float(usable_phenology.mean())

    print(f"QA: NumCycles == 1 {quality_diagnostics['num_cycles_equals_one']:.2%}, QA in {ACCEPTED_QA_VALUES} {quality_diagnostics['qa_in_accepted_values']:.2%}, both {quality_diagnostics['both_conditions']:.2%}")
    print(f"the two conditions select identical pixels: {quality_diagnostics['conditions_select_identical_pixels']}")
    print(f"all {len(feature_names)} features finite {quality_diagnostics['all_features_finite']:.2%}, usable phenology {quality_diagnostics['usable_phenology']:.2%}")

    report = {
        "site": site,
        "ground_truth_year": ground_truth_year,
        "phenology_year": phenology_year,
        "mode": "train",
        "normalisation": normalisation if normalisation else {"name": "none"},
        "source_run": args.run,
        "output_label": output_label,
        "feature_set": FEATURE_SET_NAME,
        "framework": args.framework,
        "target": args.target,
        "features": feature_names,
        "qa": quality_diagnostics,
        "models": {},
    }

    has_ground_truth = np.all(np.isfinite(true_fraction_stack), axis=0)
    tile_index, tile_ids = assign_blocks_to_tiles(config, grid)
    trainable = has_ground_truth & usable_phenology & (tile_index >= 0)
    print(f"\nblocks with ground truth {has_ground_truth.sum():,}")
    print(f"blocks with usable phenology {usable_phenology.sum():,}")
    print(f"blocks with both, inside a labelled tile {trainable.sum():,}")
    if not trainable.any():
        raise SystemExit("FAIL - no block has both ground truth and usable phenology")

    row_positions = np.flatnonzero(trainable.ravel())
    feature_matrix = feature_stack.reshape(len(feature_names), -1)[:, row_positions].T
    fraction_matrix = true_fraction_stack.reshape(len(CLASS_NAMES), -1)[:, row_positions].T
    tile_of_row = tile_index.ravel()[row_positions]
    mixedness = compute_block_mixedness(fraction_matrix)

    train_tile_ids = [tile for tile, role in config["tiles"].items() if role == "train"]
    test_tile_ids = [tile for tile, role in config["tiles"].items() if role == "test"]
    is_train_row = np.isin(tile_of_row, [tile_ids.index(tile) for tile in train_tile_ids])
    print(f"training rows {is_train_row.sum():,} over {len(train_tile_ids)} train tiles, held-out test rows {(~is_train_row).sum():,} over {len(test_tile_ids)} test tiles")
    print(f"mixedness: pure {(mixedness < 0.001).mean():.1%}, median {np.median(mixedness):.3f}")

    baseline = score_constant_mean_baseline(fraction_matrix[is_train_row])
    print(f"\nBASELINE, predicting the constant training mean: MAE {baseline['mae']:.4f}")
    print("RF-B must beat this to have learned anything at all.")

    report["n_training_blocks"] = int(is_train_row.sum())
    report["n_test_blocks"] = int((~is_train_row).sum())
    report["train_tiles"] = train_tile_ids
    report["test_tiles"] = test_tile_ids
    report["pure_share"] = float((mixedness < 0.001).mean())
    # THE TRAINING RANGE TRAVELS WITH THE MODEL, because a random forest cannot
    # extrapolate: fed a value beyond anything it saw, it returns the edge of
    # what it saw, confidently and with no warning. Stage 5_2 reports the share
    # of each site's pixels that fall outside this range, which is the only
    # honest measure of how far a transfer is being stretched. p1 and p99 are
    # kept beside the extremes so one outlying training block cannot make the
    # range look wider than it usefully is.
    training_features = feature_matrix[is_train_row]
    report["training_feature_range"] = {
        "features": feature_names,
        "minimum": np.min(training_features, axis=0).tolist(),
        "maximum": np.max(training_features, axis=0).tolist(),
        "p1": np.percentile(training_features, 1, axis=0).tolist(),
        "p99": np.percentile(training_features, 99, axis=0).tolist(),
    }
    report["constant_mean_baseline"] = baseline

    for model_name in ("joint", "independent"):
        print(f"\n{'=' * SEVENTY}\nmodel: {model_name}")
        fit_function = fit_joint_multioutput_forest if model_name == "joint" else fit_independent_per_class_forests

        fold_scores = []
        pooled_truth, pooled_prediction, pooled_mixedness = [], [], []
        for held_out_tile in train_tile_ids:
            held_out_number = tile_ids.index(held_out_tile)
            fit_rows = is_train_row & (tile_of_row != held_out_number)
            held_rows = tile_of_row == held_out_number
            if not held_rows.any() or not fit_rows.any():
                continue
            fitted = fit_function(feature_matrix[fit_rows], fraction_matrix[fit_rows], seed)
            predicted = predict_with_model(model_name, fitted, feature_matrix[held_rows])
            truth = fraction_matrix[held_rows]
            scores = score_fraction_predictions(truth, predicted)
            fold_scores.append({"held_out": held_out_tile, **scores})
            print(f"held out {held_out_tile}: n {scores['n']:>7,} MAE {scores['mae']:.4f} shrub MAE {scores['per_class']['shrub']['mae']:.4f} raw sum {scores['raw_sum_before_renormalisation']['mean']:.4f}")
            pooled_truth.append(truth)
            pooled_prediction.append(predicted)
            pooled_mixedness.append(mixedness[held_rows])

        pooled_truth = np.vstack(pooled_truth)
        pooled_prediction = np.vstack(pooled_prediction)
        pooled_mixedness = np.concatenate(pooled_mixedness)
        cross_validated = score_fraction_predictions(pooled_truth, pooled_prediction)
        by_mixedness = score_by_mixedness_stratum(pooled_truth, pooled_prediction, pooled_mixedness)

        print(f"\n CROSS-VALIDATED over {len(fold_scores)} folds: n {cross_validated['n']:,} MAE {cross_validated['mae']:.4f} RMSE {cross_validated['rmse']:.4f}")
        print("per class MAE: " + " ".join(f"{name} {cross_validated['per_class'][name]['mae']:.4f}" for name in CLASS_NAMES))
        print("per class bias: " + " ".join(f"{name} {cross_validated['per_class'][name]['bias']:+.4f}" for name in CLASS_NAMES))
        print(f"raw sum before renormalisation: mean {cross_validated['raw_sum_before_renormalisation']['mean']:.4f}, mean absolute deviation from 1 {cross_validated['raw_sum_before_renormalisation']['mean_absolute_deviation_from_one']:.4f}")
        print("MAE by how mixed the block is:")
        for label, scores in by_mixedness.items():
            print(f"{label:<15} n {scores['n']:>8,} MAE {scores['mae']:.4f} shrub {scores['per_class']['shrub']['mae']:.4f}")

        print("\n fitting on all train tiles, evaluating on the held-out test tiles")
        fitted_on_train = fit_function(feature_matrix[is_train_row], fraction_matrix[is_train_row], seed)
        model_pickle_path = fitted_model_path(results_root, site, ground_truth_year, args.framework, model_name, args.target, args.run)
        save_fitted_model(model_pickle_path, model_name, fitted_on_train, report)
        print(f"saved {model_name} model to {model_pickle_path.name} ({model_pickle_path.stat().st_size / 1048576:.1f} MB gzipped)")
        out_of_bag = score_out_of_bag(model_name, fitted_on_train, fraction_matrix[is_train_row])
        print(f"OUT OF BAG on the train rows: n {out_of_bag['n']:,} MAE {out_of_bag['mae']:.4f} - a RANDOM holdout, so optimistic against the spatial folds above")
        if out_of_bag["n_without_oob_prediction"]:
            print(f"{out_of_bag['n_without_oob_prediction']:,} rows were in-bag for every tree and have no OOB prediction, dropped from this score")
        test_prediction = predict_with_model(model_name, fitted_on_train, feature_matrix[~is_train_row])
        test_truth = fraction_matrix[~is_train_row]
        test_scores = score_fraction_predictions(test_truth, test_prediction)
        test_by_mixedness = score_by_mixedness_stratum(test_truth, test_prediction, mixedness[~is_train_row])
        print(f"TEST TILES: n {test_scores['n']:,} MAE {test_scores['mae']:.4f} shrub MAE {test_scores['per_class']['shrub']['mae']:.4f} raw sum {test_scores['raw_sum_before_renormalisation']['mean']:.4f}")

        print(" permutation importance, train rows then test tiles")
        feature_importance = {
            "features": feature_names,
            "impurity": impurity_importance(model_name, fitted_on_train),
            "permutation_train": score_permutation_importance(model_name, fitted_on_train, feature_matrix[is_train_row], fraction_matrix[is_train_row], seed),
            "permutation_test": score_permutation_importance(model_name, fitted_on_train, feature_matrix[~is_train_row], fraction_matrix[~is_train_row], seed),
        }
        pooled_test_increase = np.mean([feature_importance["permutation_test"]["per_class_mae_increase"][class_name] for class_name in CLASS_NAMES], axis=0)
        ranked = sorted(zip(feature_names, pooled_test_increase), key=lambda pair: -pair[1])
        print(" top features on test, mean MAE increase over the four classes: " + ", ".join(f"{name} {value:+.4f}" for name, value in ranked[:5]))

        report["models"][model_name] = {
            "folds": fold_scores,
            "out_of_bag": out_of_bag,
            "feature_importance": feature_importance,
            "cross_validated": cross_validated,
            "cross_validated_by_mixedness": by_mixedness,
            "test_tiles": test_scores,
            "test_tiles_by_mixedness": test_by_mixedness,
        }

        if not args.no_predict:
            print("predicting site-wide")
            predicted_stack, raw_sum = predict_site_wide(model_name, fitted_on_train, feature_stack, usable_phenology, grid)
            stem = f"{model_name}_{args.framework}_{site}_{phenology_year}"
            write_prediction_raster(output_directory / f"fraction_predicted_{stem}.tif", predicted_stack, grid, CLASS_NAMES)
            write_prediction_raster(output_directory / f"fraction_predicted_sum_{stem}.tif", raw_sum, grid, ["raw_sum_before_renormalisation"])
            dominant_class = hard_class_from_fractions(predicted_stack)
            write_class_raster(output_directory / f"class_predicted_{stem}.tif", dominant_class, grid)
            difference_stack = predicted_minus_observed(predicted_stack, true_fraction_stack)
            write_prediction_raster(output_directory / f"fraction_difference_{stem}.tif", difference_stack, grid, CLASS_NAMES)
            finite_sum = raw_sum[np.isfinite(raw_sum)]
            report["models"][model_name]["site_wide"] = {
                "pixels_predicted": int(np.isfinite(raw_sum).sum()),
                "raw_sum_mean": float(finite_sum.mean()),
                "raw_sum_min": float(finite_sum.min()),
                "raw_sum_max": float(finite_sum.max()),
                "raw_sum_mean_absolute_deviation_from_one": float(np.abs(finite_sum - 1.0).mean()),
            }
            print(f"site-wide: {np.isfinite(raw_sum).sum():,} pixels, raw sum mean {finite_sum.mean():.4f}, mean absolute deviation from 1 {np.abs(finite_sum - 1.0).mean():.4f}")

    joint_mae = report["models"]["joint"]["cross_validated"]["mae"]
    independent_mae = report["models"]["independent"]["cross_validated"]["mae"]
    print(f"\n{'=' * SEVENTY}")
    print(f"joint MAE {joint_mae:.4f} against independent MAE {independent_mae:.4f}, difference {joint_mae - independent_mae:+.4f}")
    print("A difference near zero means the joint target buys nothing and the four classes are predicted independently anyway.")
    print(f"baseline MAE {baseline['mae']:.4f} - both models must beat this.")

    report_path = output_directory / f"stage5_1_report_{site}_{phenology_year}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
