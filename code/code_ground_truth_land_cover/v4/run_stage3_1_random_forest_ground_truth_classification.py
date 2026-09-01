"""Step 1d - per-pixel Random Forest classification (RF-A).

RF-A classifies at 1 m and outputs a hard class per pixel; the fractions used by
RF-B come later, from aggregating 16 of these to a PlanetScope block (Step 3).
RF-A is a CLASSIFIER, not a regressor - see instructions5.md section 4.3.

TRAINING IS PER PIXEL, NOT PER SEGMENT, and that is a deliberate departure from
the earlier section 5 Step 1d wording. SLIC segments are a uniform 9 px while
the median accepted shrub polygon is 5 px, so at a 70% segment-coverage rule the
whole train block yields 22 shrub segments against 738 bare - a 34:1 imbalance
that would make shrub effectively unpredictable. That is the v3 failure in
mirror image (instructions2.md section 4.5: a starved training set produced
93-97% tree cover). Per-pixel training sidesteps it: shrub gets 918 samples
instead of 22. The segment rasters are still produced and remain available for a
segment-level comparison later.

Labels are the UNION of hand-drawn polygons and accepted CHM shrub candidates,
exactly as counted by run_stage2_4_check_hand_labeling_progress.py. An accepted candidate
is an ordinary hand-validated label.

Validation is LEAVE-ONE-TILE-OUT over the train block. This is a DIAGNOSTIC, not
an accuracy assessment. Step 6 accuracy requires the Olofsson area-weighted
protocol on an independent probability sample (instructions5.md section 6);
cross-validation over training polygons is a different quantity and must never
be reported as map accuracy.

Frameworks A-E differ by input feature group only, so any accuracy difference is
attributable to the inputs (section 4.1). Note that D and E include CHM while
most shrub labels are CHM-derived, which makes their shrub score partly
circular - A-C carry the meaningful test of whether RGB, vegetation indices and
texture can recover CHM-defined shrub without CHM.

POLYGON SUBSAMPLING caps how many pixels any single polygon contributes. Without
it a 1386 m2 bare polygon donates 1386 near-identical, spatially autocorrelated
pixels while a 5 m2 shrub polygon donates 5. Measured on run 1: bare averaged
241 px/polygon against shrub at 7.4, so the forest saw roughly 34 independent
bare observations dressed up as 8200 samples. The nominal 10:1 class imbalance
was therefore real in weight but largely fake in information, and
class_weight="balanced" was compensating for redundancy rather than for genuine
scarcity. Capping per polygon attacks the cause instead of the symptom.

Sampling is random WITHIN each polygon, seeded, so it is reproducible and does
not preferentially take one part of a polygon.

RUNS are numbered and fully separated on disk. Every output lands under
stage3_classification/run{N}/, so a baseline stays intact and comparable after
the inputs change. Run 1 is the baseline with no subsampling and a partial shrub
candidate review; run 2 adds subsampling and the completed review. The features
and shadow directories are NOT run-scoped - they are stage 1 outputs shared by
every run.

SEGMENTS ARE NOT READ. An earlier design trained per segment and this script
bound a segments directory to read them from; the binding outlived the design
and sat unused. run_stage1_7_generate_segments.py has since been retired to
unused/ because nothing consumed its output at all - see unused/README.md.

Usage: python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 2
        python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 2 --frameworks A C
        python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 2 --max-pixels-per-polygon 100
        python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 1 --no-predict


        python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 3 --frameworks A B C --max-pixels-per-polygon 100
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from constants import CLASS_CODES, CLASS_COLORS, CLASS_LABELS, NODATA, SEVENTY
from helpers import resolve_config_path
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

FRAMEWORK_GROUPS = {
    "A": ["chromatic", "rgb_index", "brightness"],
    "B": ["chromatic", "rgb_index", "brightness", "nir_index"],
    "C": ["chromatic", "rgb_index", "brightness", "nir_index", "texture"],
    "D": ["chromatic", "rgb_index", "brightness", "nir_index", "texture", "chm"],
}


def validated_run_label(raw_label):
    """Validate a run label: an integer, or an integer with a suffix.

    Runs were originally numbered, but a bare number cannot say what a run IS -
    "run 99" was a smoke test for run 3 and nothing in the name recorded that.
    A suffixed label like 3_smoke carries the intent, sorts next to its parent
    run, and remains a valid directory name.

    Restricted to letters, digits, underscore and dash so the label can be used
    directly as a path component on any filesystem.

    Inputs: raw_label - the raw --run argument
    Outputs: the validated label string
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", raw_label):
        raise argparse.ArgumentTypeError(f"run label {raw_label!r} must contain only letters, digits, underscore or dash - it becomes a directory name")
    return raw_label


def hex_color_to_rgba(hex_color):
    """Convert a #rrggbb string to an (r, g, b, 255) tuple for a GeoTIFF colormap.

    Inputs: hex_color - hex color string such as "#c2b280"
    Outputs: 4-tuple of ints, (red, green, blue, alpha)
    """
    digits = hex_color.lstrip("#")
    return tuple(int(digits[offset : offset + 2], 16) for offset in (0, 2, 4)) + (255,)


def framework_band_columns(band_order, band_metadata, framework):
    """Column indices of the feature bands a framework is allowed to use.

    Frameworks differ by input group only, so the same feature stack is read for
    all of them and simply subset here.

    Inputs: band_order - list of band names in raster order; band_metadata -
             list of {name, group} dicts; framework - key into FRAMEWORK_GROUPS
    Outputs: (list of int column indices, list of band-name strings)
    """
    allowed_groups = set(FRAMEWORK_GROUPS[framework])
    group_of_band = {band["name"]: band["group"] for band in band_metadata}
    columns, band_names = [], []
    for column, band_name in enumerate(band_order):
        if group_of_band.get(band_name) in allowed_groups:
            columns.append(column)
            band_names.append(band_name)
    return columns, band_names


def read_label_polygons(label_dir, site, tile, year):
    """Training labels for a tile - hand-drawn polygons plus accepted candidates.

    The union is the same one run_stage2_4_check_hand_labeling_progress.py counts. An
    accepted CHM candidate was confirmed against RGB by the analyst, which is
    the same act as drawing one, so it is an ordinary label here.

    Inputs: label_dir - Path to stage2_labeling; site, tile, year
    Outputs: GeoDataFrame with class_code and geometry, or None
    """
    label_frames = []
    drawn_path = label_dir / f"training_polygons_{site}_{tile}_{year}.gpkg"
    if drawn_path.exists():
        try:
            drawn_polygons = gpd.read_file(drawn_path)
            if len(drawn_polygons):
                label_frames.append(drawn_polygons[["class_code", "geometry"]])
        except Exception:
            pass
    review_path = label_dir / f"shrub_review_{site}_{tile}_{year}.gpkg"
    if review_path.exists():
        for geometry_kind in ("polygon", "point"):
            try:
                candidates = gpd.read_file(review_path, layer=f"shrub_review_{site}_{tile}_{year}_{geometry_kind}")
            except Exception:
                continue
            if not len(candidates) or "reviewed" not in candidates.columns:
                continue
            is_reviewed = candidates["reviewed"].fillna(0).astype(int) == 1
            is_rejected = candidates["rejected"].fillna(0).astype(int) == 1 if "rejected" in candidates.columns else False
            accepted = candidates[is_reviewed & ~is_rejected]
            if len(accepted):
                label_frames.append(accepted[["class_code", "geometry"]])
    if not label_frames:
        return None
    all_labels = pd.concat(label_frames, ignore_index=True)
    all_labels = all_labels[all_labels["class_code"].notna()]
    return gpd.GeoDataFrame(all_labels, crs=label_frames[0].crs) if len(all_labels) else None


def rasterize_label_polygons(label_polygons, shape, transform):
    """Burn label polygons onto the 1 m analysis grid, with polygon identity.

    Classes are burned in a fixed order so that where polygons overlap, the
    later class wins deterministically rather than by draw order. Woody last:
    a shrub or tree polygon overlapping a bare one is the more specific claim.

    A parallel polygon-id raster is produced so pixels can be subsampled WITHIN
    a polygon. Identity is burned in the same order as the classes, so the id
    raster and the class raster always agree about who owns an overlap.

    Inputs: label_polygons - GeoDataFrame; shape - (rows, cols); transform - affine
    Outputs: (uint8 class array with NODATA where unlabelled,
              int32 polygon-id array with 0 where unlabelled)
    """
    label_raster = np.full(shape, NODATA, dtype="uint8")
    polygon_ids = np.zeros(shape, dtype="int32")
    next_polygon_id = 1
    for class_code in CLASS_CODES:
        polygons_of_class = label_polygons[label_polygons["class_code"].astype(int) == class_code]
        if not len(polygons_of_class):
            continue
        geometry_id_pairs = []
        for geometry in polygons_of_class.geometry:
            geometry_id_pairs.append((geometry, next_polygon_id))
            next_polygon_id += 1
        burned_ids = rasterize(
            geometry_id_pairs,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
        burned = burned_ids > 0
        label_raster[burned] = class_code
        polygon_ids[burned] = burned_ids[burned]
    return label_raster, polygon_ids


def subsample_pixels_within_each_polygon(polygon_ids, train_mask, max_pixels_per_polygon, random_generator):
    """Cap how many pixels each polygon contributes to training.

    Pixels inside one polygon are near-duplicates: same object, same lighting,
    adjacent ground. Beyond a few dozen they add weight without adding
    information, and they do it unevenly - a large bare polygon drowns out a
    small shrub one. Sampling is random within the polygon rather than taking a
    prefix, so no part of a polygon is systematically preferred.

    Inputs: polygon_ids - int32 polygon-id raster; train_mask - bool array of
             labelled and usable pixels; max_pixels_per_polygon - int cap, or
             None to disable; random_generator - a seeded numpy Generator
    Outputs: bool array, the retained subset of train_mask
    """
    if not max_pixels_per_polygon:
        return train_mask
    kept = np.zeros_like(train_mask)
    polygon_id_per_labelled_pixel = polygon_ids[train_mask]
    flat_positions = np.flatnonzero(train_mask)
    for polygon_id in np.unique(polygon_id_per_labelled_pixel):
        if polygon_id == 0:
            continue
        pixels_in_polygon = flat_positions[polygon_id_per_labelled_pixel == polygon_id]
        if len(pixels_in_polygon) > max_pixels_per_polygon:
            pixels_in_polygon = random_generator.choice(pixels_in_polygon, size=max_pixels_per_polygon, replace=False)
        kept.ravel()[pixels_in_polygon] = True
    return kept


class TileData(NamedTuple):
    """Everything one tile contributes, named so call sites read as English.

    A plain tuple here meant call sites indexed it positionally - `cache[t][0]`
    for the features and `cache[t][1]` for the labels, which said nothing about
    what is being fetched and silently returns the wrong array if the return
    order ever changes.
    """

    training_features: np.ndarray  # [n_labelled_pixels, n_framework_bands] float32
    training_labels: np.ndarray  # [n_labelled_pixels] uint8, class codes 0-3
    framework_stack: np.ndarray  # [n_framework_bands, rows, cols] float32, whole tile
    pixel_is_usable: np.ndarray  # [rows, cols] bool - all bands finite AND not shadow
    profile: dict  # rasterio profile, for writing outputs on the same grid
    label_raster: np.ndarray  # [rows, cols] uint8, NODATA where unlabelled


def load_tile_data(feature_dir, label_dir, shadow_dir, site, tile, year, band_columns, max_pixels_per_polygon, random_generator):
    """Feature matrix, label vector, and geometry for one tile.

    VALIDITY IS COMPUTED OVER THE FULL BAND STACK, not over the framework's
    subset. Frameworks A-E must differ by input features and nothing else
    (instructions5.md section 4.1), and a subset-derived mask would quietly
    break that: adding a band that carries more NaN would shrink the training
    set, so a framework could score differently because it trained on fewer
    pixels rather than because its features were better. Masking on the full
    stack makes every framework see exactly the same pixels by construction.

    ALL SHADOW PIXELS ARE DROPPED - both mask codes, deliberately. Reading the
    mask with .astype(bool) makes SHADOW_IS_TREE and SHADOW_IS_NODATA alike
    True, and that is the intended behaviour, not an accident of the cast:
    section 5 Step 1c (resolved 2026-08-18) never assigns shadow to a class.
    The goal is an accurate map over the majority of pixels, not a label for
    every pixel - a shadowed pixel was not observed well enough to classify,
    and inventing a class for it trades a known gap for an unknown error.

    An earlier version of the spec assigned canopy-adjacent shadow to tree.
    That would have handed tree up to 3% of every tile on a geometric argument
    rather than a spectral one, in a class already biased +4.5% by area.

    Inputs: feature_dir, label_dir, shadow_dir - Paths; site, tile, year;
             band_columns - band column indices for this framework;
             max_pixels_per_polygon - int cap or None; random_generator - seeded
             numpy Generator
    Outputs: TileData - see that class for what each field holds
    """
    with rasterio.open(feature_dir / f"features_{site}_{tile}_{year}.tif") as ds:
        all_band_stack = ds.read().astype("float32")
        profile, transform, shape = ds.profile, ds.transform, (ds.height, ds.width)

    pixel_is_usable = np.all(np.isfinite(all_band_stack), axis=0)
    framework_stack = all_band_stack[band_columns]

    shadow_path = shadow_dir / f"shadow_mask_ref_{site}_{tile}_{year}.tif"
    if shadow_path.exists():
        with rasterio.open(shadow_path) as ds:
            # Both codes at once, deliberately - see the docstring. Shadow is
            # never assigned to a class, so SHADOW_IS_TREE and SHADOW_IS_NODATA
            # are treated identically here.
            pixel_is_shadow = ds.read(1).astype(bool)
        pixel_is_usable &= ~pixel_is_shadow

    label_polygons = read_label_polygons(label_dir, site, tile, year)
    if label_polygons is not None:
        label_raster, polygon_ids = rasterize_label_polygons(label_polygons, shape, transform)
    else:
        label_raster = np.full(shape, NODATA, dtype="uint8")
        polygon_ids = np.zeros(shape, dtype="int32")

    train_mask = (label_raster != NODATA) & pixel_is_usable
    train_mask = subsample_pixels_within_each_polygon(polygon_ids, train_mask, max_pixels_per_polygon, random_generator)
    return TileData(
        training_features=framework_stack[:, train_mask].T,
        training_labels=label_raster[train_mask],
        framework_stack=framework_stack,
        pixel_is_usable=pixel_is_usable,
        profile=profile,
        label_raster=label_raster,
    )


def fit_random_forest(training_features, training_labels, seed):
    """Fit RF-A on per-pixel samples.

    class_weight is balanced because the classes are severely unequal in pixels
    even when the polygon counts look even: shrub polygons are small (median
    5 m2) and bare polygons large (median 82 m2), so shrub is ~6% of training
    pixels while bare is ~55%. Without weighting the forest simply under-calls
    shrub, which is the class the project most needs.

    Inputs: training_features - [n_pixels, n_bands]; training_labels -
             [n_pixels] class codes; seed - int
    Outputs: fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(training_features, training_labels)
    return model


def per_class_scores(true_labels, predicted_labels):
    """Per-class recall, precision and F1, plus macro-F1 and overall accuracy.

    Per class rather than pooled: overall accuracy at SRER is dominated by bare
    and grass and barely moves on shrub, the class that carries the scientific
    difficulty (instructions5.md section 6.4).

    Inputs: true_labels, predicted_labels - arrays of class codes
    Outputs: (dict of {class_code: {recall, precision, f1, support}},
              macro_f1 float, overall_accuracy float)
    """
    scores, f1_per_present_class = {}, []
    for class_code in CLASS_LABELS:
        is_truly_class = true_labels == class_code
        is_predicted_class = predicted_labels == class_code
        support = int(is_truly_class.sum())
        true_positives = int((is_truly_class & is_predicted_class).sum())
        recall = true_positives / support if support else float("nan")
        precision = true_positives / int(is_predicted_class.sum()) if is_predicted_class.sum() else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision > 0 and recall > 0) else 0.0
        scores[class_code] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_per_present_class.append(f1)
    overall_accuracy = float((true_labels == predicted_labels).mean()) if len(true_labels) else float("nan")
    return scores, float(np.mean(f1_per_present_class)) if f1_per_present_class else float("nan"), overall_accuracy


def print_per_class_scores(title, scores, macro_f1, overall_accuracy):
    """Print a per-class score block.

    Inputs: title - str; scores - dict from per_class_scores; macro_f1,
             overall_accuracy - floats
    Outputs: None
    """
    print(f"\n{title}")
    print(f"{'class':<8}{'support':>9}{'recall':>9}{'precision':>11}{'f1':>8}")
    for class_code, class_name in CLASS_LABELS.items():
        score = scores[class_code]
        if not score["support"]:
            print(f"{class_name:<8}{0:>9}{'-':>9}{'-':>11}{'-':>8}")
            continue
        print(f"{class_name:<8}{score['support']:>9}{score['recall']:>9.3f}{score['precision']:>11.3f}{score['f1']:>8.3f}")
    print(f"macro-F1 {macro_f1:.3f} overall {overall_accuracy:.3f} (overall is dominated by bare and grass - judge on per-class)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--run",
        type=validated_run_label,
        default="1",
        help="run label - a number, optionally suffixed (e.g. 3 or 3_smoke). Every output lands under run{LABEL}/ so baselines stay comparable",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="which frameworks to run",
    )
    parser.add_argument(
        "--max-pixels-per-polygon",
        type=int,
        default=None,
        help="cap pixels contributed by one polygon; overrides the config value; 0 disables",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="cross-validate only, skip full-tile prediction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a run directory that already holds a report - THIS DESTROYS A FROZEN BASELINE",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    seed = config["BASE_SEED"] + year
    results_root = resolve_config_path(config["results_root"])
    feature_dir = results_root / "stage1_data_and_features" / "features"
    shadow_dir = results_root / "stage1_data_and_features" / "shadow"
    label_dir = results_root / "stage2_labeling"

    # run directory: step 1d outputs are run-scoped, features and shadow are not
    run_dir = results_root / "stage3_classification" / f"run{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    classification_settings = config.get("stage3_1_classification", {})
    max_pixels_per_polygon = args.max_pixels_per_polygon if args.max_pixels_per_polygon is not None else classification_settings.get("max_pixels_per_polygon")
    max_pixels_per_polygon = max_pixels_per_polygon or None

    band_spec = json.loads((feature_dir / f"features_{site}_{year}_bands.json").read_text())
    band_order, band_metadata = band_spec["band_order"], band_spec["bands"]

    train_tiles = [tile for tile, role in config["tiles"].items() if role == "train"]
    test_tiles = [tile for tile, role in config["tiles"].items() if role == "test"]

    print(f"Step 1d - RF-A per-pixel classification - {site} {year}")
    print("=" * SEVENTY)
    print(f"run {args.run} seed {seed} train tiles {len(train_tiles)} test tiles {len(test_tiles)}")
    print(f"polygon subsampling: {('max ' + str(max_pixels_per_polygon) + ' px per polygon') if max_pixels_per_polygon else 'OFF - every polygon contributes all its pixels'}")
    print("DIAGNOSTIC ONLY - cross-validation over training polygons is not map accuracy.")
    print("Step 6 accuracy needs the Olofsson protocol on an independent sample (section 6).")

    # the report ACCUMULATES across runs. Running one framework must not erase
    # the others: a fresh dict here would mean `--frameworks C` silently deleted
    # A and B, and the comparison figure could only ever show whatever was in
    # the last command. Matches the append-a-named-section convention in
    # instructions1.md section 5.
    # named report_path, never `out`. An earlier version bound `out` to the
    # report path here and rebound it to the classification array in the
    # prediction loop below, so the final write became ndarray.write_text.
    report_path = run_dir / f"stage3_1_report_{site}_{year}.json"

    # A completed framework is a frozen baseline. Re-running it would silently
    # rewrite numbers a later run is compared against, and labels drift
    # continuously while labelling is in progress, so the rewrite would not even
    # reproduce the original. Refuse, and say which run to use instead.
    #
    # THE GUARD IS PER FRAMEWORK, NOT PER RUN. Running A today and B C D
    # tomorrow is the normal way a run gets built - frameworks are expensive and
    # are often added one at a time - and the report is explicitly designed to
    # accumulate them. An earlier version refused whenever the report FILE
    # existed, which blocked exactly that workflow and made the carry-forward
    # logic below unreachable: it could only ever run when there was nothing to
    # carry forward. Only an actual overwrite of an existing framework is
    # refused.
    if report_path.exists() and not args.force:
        existing_report = json.loads(report_path.read_text())
        already_written = set(existing_report.get("frameworks", {}))
        would_overwrite = sorted(already_written & set(args.frameworks))
        if would_overwrite:
            print(f"run {args.run} already holds framework(s): {', '.join(sorted(already_written))}")
            print(f"this command would overwrite: {', '.join(would_overwrite)}")
            print("refusing to overwrite a completed framework - use a different --run label for new work, or --force to overwrite deliberately")
            return
        # per-run provenance, carried in the config so it is recorded automatically
        # rather than depending on someone remembering. A run that changes several
        # things at once is only interpretable if that is written down at the time.
    run_note = classification_settings.get("run_notes", {}).get(str(args.run))
    if run_note:
        print(f"\nrun {args.run} note: {run_note}\n")
    else:
        print(f"\nNOTE: no run_notes entry for run {args.run} in config stage3_1_classification.run_notes - record what this run changes\n")

    report = {
        "run": args.run,
        "site": site,
        "year": year,
        "seed": seed,
        "training": "per-pixel",
        "max_pixels_per_polygon": max_pixels_per_polygon,
        "validation": "leave-one-tile-out over train",
        "note": "diagnostic, not a Step 6 accuracy assessment",
        "run_note": run_note,
        "tiles": dict(config["tiles"]),
        "frameworks": {},
    }
    if report_path.exists():
        try:
            previous_report = json.loads(report_path.read_text())
            report["frameworks"] = previous_report.get("frameworks", {})
            carried_forward = [name for name in sorted(report["frameworks"]) if name not in args.frameworks]
            if carried_forward:
                print(f"carrying forward earlier results for framework(s): {', '.join(carried_forward)}")
        except Exception as exc:
            print(f"warning: could not read the existing report, starting fresh: {exc}")

    for framework in args.frameworks:
        band_columns, band_names = framework_band_columns(band_order, band_metadata, framework)
        print(f"\n{'=' * 74}\nframework {framework} - {len(band_names)} features: {', '.join(band_names)}")

        tile_cache = {}
        for tile in train_tiles + test_tiles:
            # one generator per tile per framework, seeded identically, so every
            # framework subsamples the SAME pixels and the section 4.1
            # deconfounding survives subsampling
            tile_cache[tile] = load_tile_data(
                feature_dir,
                label_dir,
                shadow_dir,
                site,
                tile,
                year,
                band_columns,
                max_pixels_per_polygon,
                np.random.default_rng(seed),
            )

        train_pixels_per_class = Counter()
        for tile in train_tiles:
            train_pixels_per_class.update(tile_cache[tile].training_labels.tolist())
        print("train pixels per class: " + ", ".join(f"{name} {train_pixels_per_class.get(code, 0)}" for code, name in CLASS_LABELS.items()))
        if min(train_pixels_per_class.get(code, 0) for code in CLASS_LABELS) == 0:
            print("skipping: a class has no training pixels in the train block")
            continue

        # ---------------------------------------------------------- leave one tile out
        folds = []
        pooled_true_labels, pooled_predicted_labels = [], []
        for held_out_tile in train_tiles:
            fit_tiles = [tile for tile in train_tiles if tile != held_out_tile]
            fit_features = np.vstack([tile_cache[tile].training_features for tile in fit_tiles])
            fit_labels = np.concatenate([tile_cache[tile].training_labels for tile in fit_tiles])
            held_out_features = tile_cache[held_out_tile].training_features
            held_out_labels = tile_cache[held_out_tile].training_labels
            if not len(held_out_labels) or len(np.unique(fit_labels)) < 2:
                continue
            model = fit_random_forest(fit_features, fit_labels, seed)
            predicted_labels = model.predict(held_out_features)
            scores, macro, overall = per_class_scores(held_out_labels, predicted_labels)
            print_per_class_scores(f"held-out tile {held_out_tile} ({len(held_out_labels)} labelled px)", scores, macro, overall)
            folds.append(
                {
                    "held_out": held_out_tile,
                    "n": int(len(held_out_labels)),
                    "macro_f1": macro,
                    "overall": overall,
                    "per_class": {name: scores[code] for code, name in CLASS_LABELS.items()},
                }
            )
            pooled_true_labels.append(held_out_labels)
            pooled_predicted_labels.append(predicted_labels)

        if pooled_true_labels:
            pooled_true_labels = np.concatenate(pooled_true_labels)
            pooled_predicted_labels = np.concatenate(pooled_predicted_labels)
            scores, macro, overall = per_class_scores(pooled_true_labels, pooled_predicted_labels)
            print_per_class_scores(f"POOLED across {len(folds)} folds", scores, macro, overall)
            confusion = confusion_matrix(pooled_true_labels, pooled_predicted_labels, labels=list(CLASS_LABELS))
            print("\nconfusion (rows true, cols predicted)")
            print(f"{'':<8}" + "".join(f"{name:>8}" for name in CLASS_LABELS.values()))
            for true_row, class_code in enumerate(CLASS_LABELS):
                print(f"{CLASS_LABELS[class_code]:<8}" + "".join(f"{confusion[true_row, predicted_column]:>8}" for predicted_column in range(len(CLASS_LABELS))))
            report["frameworks"][framework] = {
                "features": band_names,
                "train_pixels_per_class": {name: int(train_pixels_per_class.get(code, 0)) for code, name in CLASS_LABELS.items()},
                "folds": folds,
                "pooled": {
                    "macro_f1": macro,
                    "overall": overall,
                    "per_class": {name: scores[code] for code, name in CLASS_LABELS.items()},
                },
                "confusion": confusion.tolist(),
            }

        if args.no_predict:
            continue

        # ------------------------------------------------------------- full prediction
        fit_features = np.vstack([tile_cache[tile].training_features for tile in train_tiles])
        fit_labels = np.concatenate([tile_cache[tile].training_labels for tile in train_tiles])
        model = fit_random_forest(fit_features, fit_labels, seed)
        feature_importance = sorted(zip(band_names, model.feature_importances_), key=lambda pair: -pair[1])
        print("\ntop features: " + ", ".join(f"{band} {weight:.3f}" for band, weight in feature_importance[:8]))
        report["frameworks"].setdefault(framework, {})["feature_importance"] = {band: float(weight) for band, weight in feature_importance}

        out_dir = run_dir / framework
        out_dir.mkdir(parents=True, exist_ok=True)
        for tile in train_tiles + test_tiles:
            tile_data = tile_cache[tile]
            framework_stack, profile = tile_data.framework_stack, tile_data.profile
            n_bands, rows, cols = framework_stack.shape
            pixel_rows = framework_stack.reshape(n_bands, -1).T
            predicted_classes = np.full(rows * cols, NODATA, dtype="uint8")
            usable = tile_data.pixel_is_usable.reshape(-1)
            if usable.any():
                predicted_classes[usable] = model.predict(pixel_rows[usable]).astype("uint8")
                class_probabilities = model.predict_proba(pixel_rows[usable]).astype("float32")
            predicted_classes = predicted_classes.reshape(rows, cols)

            profile.update(count=1, dtype="uint8", nodata=NODATA, compress="deflate")
            path = out_dir / f"classification_{framework}_{site}_{tile}_{year}.tif"
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(predicted_classes, 1)
                # color table travels inside the file, so QGIS and GDAL both render
                # the locked section 3 palette with no sidecar (section 12 Q10)
                ds.write_colormap(1, {code: hex_color_to_rgba(CLASS_COLORS[code]) for code in CLASS_LABELS})

                # NaN, not zero, outside the valid mask. predict_proba always sums
                # to 1 for a classified pixel, so an all-zero pixel could only ever
                # be masked - but written as 0 it reads as a legitimate probability
                # of zero and renders opaque. NaN matches the declared nodata and
                # renders transparent.
            probability_bands = np.full((len(CLASS_LABELS), rows * cols), np.nan, dtype="float32")
            for column, code in enumerate(model.classes_):
                probability_bands[int(code)][usable] = class_probabilities[:, column]
            probability_profile = dict(profile)
            probability_profile.update(count=len(CLASS_LABELS), dtype="float32", nodata=np.nan)
            with rasterio.open(
                out_dir / f"class_probability_{framework}_{site}_{tile}_{year}.tif",
                "w",
                **probability_profile,
            ) as ds:
                ds.write(probability_bands.reshape(len(CLASS_LABELS), rows, cols))
                for code, name in CLASS_LABELS.items():
                    ds.set_band_description(code + 1, f"p_{name}")

                    # section 3.1 prediction_quality: 1 = pure, 0 = even four-way mix.
                    # The hard label is argmax regardless of how weak the winner is - by
                    # design - so the strength of that call has to be carried separately
                    # or it is lost.
            prediction_quality = np.full(rows * cols, np.nan, dtype="float32")
            margin = np.full(rows * cols, np.nan, dtype="float32")
            if usable.any():
                clipped_probabilities = np.clip(class_probabilities, 1e-12, 1.0)
                entropy = -(clipped_probabilities * np.log(clipped_probabilities)).sum(axis=1)
                prediction_quality[usable] = 1.0 - entropy / np.log(len(CLASS_LABELS))
                sorted_probabilities = np.sort(class_probabilities, axis=1)
                margin[usable] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
            single_band_profile = dict(profile)
            single_band_profile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(
                out_dir / f"prediction_quality_{framework}_{site}_{tile}_{year}.tif",
                "w",
                **single_band_profile,
            ) as ds:
                ds.write(prediction_quality.reshape(rows, cols), 1)
            with rasterio.open(
                out_dir / f"margin_{framework}_{site}_{tile}_{year}.tif",
                "w",
                **single_band_profile,
            ) as ds:
                ds.write(margin.reshape(rows, cols), 1)

            share = {CLASS_LABELS[code]: float((predicted_classes == code).mean()) for code in CLASS_LABELS}
            masked = float((predicted_classes == NODATA).mean())
            median_quality = float(np.nanmedian(prediction_quality)) if usable.any() else float("nan")
            print(f"[{tile}] " + " ".join(f"{name} {fraction:.1%}" for name, fraction in share.items()) + f" masked {masked:.1%} median quality {median_quality:.2f}")

    report_path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
