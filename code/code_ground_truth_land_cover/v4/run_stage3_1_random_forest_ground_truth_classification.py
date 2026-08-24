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
candidate review; run 2 adds subsampling and the completed review. The features,
segments and shadow directories are NOT run-scoped - they are Step 1a-1c outputs
shared by every run.

Usage:  python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 2
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

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from constants import CLASS_COLORS, CLASS_LABELS, NODATA
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


def run_label(value):
    """Validate a run label: an integer, or an integer with a suffix.

    Runs were originally numbered, but a bare number cannot say what a run IS -
    "run 99" was a smoke test for run 3 and nothing in the name recorded that.
    A suffixed label like 3_smoke carries the intent, sorts next to its parent
    run, and remains a valid directory name.

    Restricted to letters, digits, underscore and dash so the label can be used
    directly as a path component on any filesystem.

    Inputs:  value - the raw --run argument
    Outputs: the validated label string
    """
    import argparse as _argparse

    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise _argparse.ArgumentTypeError(f"run label {value!r} must contain only letters, digits, underscore or dash - it becomes a directory name")
    return value


def hex_to_rgb(value):
    """Convert a #rrggbb string to an (r, g, b, 255) tuple for a GeoTIFF colormap.

    Inputs:  value - hex colour string
    Outputs: 4-tuple of ints
    """
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4)) + (255,)


def framework_band_indices(band_order, bands_meta, framework):
    """Column indices of the feature bands a framework is allowed to use.

    Frameworks differ by input group only, so the same feature stack is read for
    all of them and simply subset here.

    Inputs:  band_order - list of band names in raster order; bands_meta - list
             of {name, group} dicts; framework - key into FRAMEWORK_GROUPS
    Outputs: (list of int indices, list of band-name strings)
    """
    groups = set(FRAMEWORK_GROUPS[framework])
    group_of = {b["name"]: b["group"] for b in bands_meta}
    indices, names = [], []
    for i, name in enumerate(band_order):
        if group_of.get(name) in groups:
            indices.append(i)
            names.append(name)
    return indices, names


def read_labels(label_dir, site, tile, year):
    """Training labels for a tile - hand-drawn polygons plus accepted candidates.

    The union is the same one run_stage2_4_check_hand_labeling_progress.py counts. An
    accepted CHM candidate was confirmed against RGB by the analyst, which is
    the same act as drawing one, so it is an ordinary label here.

    Inputs:  label_dir - Path to stage2_labeling; site, tile, year
    Outputs: GeoDataFrame with class_code and geometry, or None
    """
    frames = []
    drawn = label_dir / f"training_polygons_{site}_{tile}_{year}.gpkg"
    if drawn.exists():
        try:
            gdf = gpd.read_file(drawn)
            if len(gdf):
                frames.append(gdf[["class_code", "geometry"]])
        except Exception:
            pass
    review = label_dir / f"shrub_review_{site}_{tile}_{year}.gpkg"
    if review.exists():
        for kind in ("polygon", "point"):
            try:
                gdf = gpd.read_file(review, layer=f"shrub_review_{site}_{tile}_{year}_{kind}")
            except Exception:
                continue
            if not len(gdf) or "reviewed" not in gdf.columns:
                continue
            reviewed = gdf["reviewed"].fillna(0).astype(int) == 1
            rejected = gdf["rejected"].fillna(0).astype(int) == 1 if "rejected" in gdf.columns else False
            keep = gdf[reviewed & ~rejected]
            if len(keep):
                frames.append(keep[["class_code", "geometry"]])
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out[out["class_code"].notna()]
    return gpd.GeoDataFrame(out, crs=frames[0].crs) if len(out) else None


def rasterize_labels(labels, shape, transform):
    """Burn label polygons onto the 1 m analysis grid, with polygon identity.

    Classes are burned in a fixed order so that where polygons overlap, the
    later class wins deterministically rather than by draw order. Woody last:
    a shrub or tree polygon overlapping a bare one is the more specific claim.

    A parallel polygon-id raster is produced so pixels can be subsampled WITHIN
    a polygon. Identity is burned in the same order as the classes, so the id
    raster and the class raster always agree about who owns an overlap.

    Inputs:  labels - GeoDataFrame; shape - (rows, cols); transform - affine
    Outputs: (uint8 class array with NODATA where unlabelled,
              int32 polygon-id array with 0 where unlabelled)
    """
    out = np.full(shape, NODATA, dtype="uint8")
    ids = np.zeros(shape, dtype="int32")
    next_id = 1
    for code in (0, 1, 2, 3):
        subset = labels[labels["class_code"].astype(int) == code]
        if not len(subset):
            continue
        shapes = []
        for geom in subset.geometry:
            shapes.append((geom, next_id))
            next_id += 1
        burned = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
        mask = burned > 0
        out[mask] = code
        ids[mask] = burned[mask]
    return out, ids


def subsample_by_polygon(ids, train_mask, max_per_polygon, rng):
    """Cap how many pixels each polygon contributes to training.

    Pixels inside one polygon are near-duplicates: same object, same lighting,
    adjacent ground. Beyond a few dozen they add weight without adding
    information, and they do it unevenly - a large bare polygon drowns out a
    small shrub one. Sampling is random within the polygon rather than taking a
    prefix, so no part of a polygon is systematically preferred.

    Inputs:  ids - int32 polygon-id raster; train_mask - bool array of labelled
             and valid pixels; max_per_polygon - int cap, or None to disable;
             rng - a seeded numpy Generator
    Outputs: bool array, the retained subset of train_mask
    """
    if not max_per_polygon:
        return train_mask
    kept = np.zeros_like(train_mask)
    flat_ids = ids[train_mask]
    positions = np.flatnonzero(train_mask)
    for polygon_id in np.unique(flat_ids):
        if polygon_id == 0:
            continue
        member = positions[flat_ids == polygon_id]
        if len(member) > max_per_polygon:
            member = rng.choice(member, size=max_per_polygon, replace=False)
        kept.ravel()[member] = True
    return kept


def load_tile(feature_dir, label_dir, shadow_dir, site, tile, year, indices, max_per_polygon, rng):
    """Feature matrix, label vector, and geometry for one tile.

    VALIDITY IS COMPUTED OVER THE FULL BAND STACK, not over the framework's
    subset. Frameworks A-E must differ by input features and nothing else
    (instructions5.md section 4.1), and a subset-derived mask would quietly
    break that: adding a band that carries more NaN would shrink the training
    set, so a framework could score differently because it trained on fewer
    pixels rather than because its features were better. Masking on the full
    stack makes every framework see exactly the same pixels by construction.

    Shadow pixels are dropped from training: section 5 Step 1c resolves shadow
    to tree or to nodata, so a shadowed pixel carries no reliable class.

    Inputs:  feature_dir, label_dir, shadow_dir - Paths; site, tile, year; indices -
             feature band column indices for the framework
    Outputs: (X float32 [n, k], y uint8 [n], stack float32 [k, rows, cols],
              valid bool [rows, cols], profile, label raster uint8)
    """
    with rasterio.open(feature_dir / f"features_{site}_{tile}_{year}.tif") as ds:
        full = ds.read().astype("float32")
        profile, transform, shape = ds.profile, ds.transform, (ds.height, ds.width)

    finite = np.all(np.isfinite(full), axis=0)
    stack = full[indices]

    shadow_path = shadow_dir / f"shadow_mask_ref_{site}_{tile}_{year}.tif"
    if shadow_path.exists():
        with rasterio.open(shadow_path) as ds:
            shadow = ds.read(1).astype(bool)
        finite &= ~shadow

    labels = read_labels(label_dir, site, tile, year)
    if labels is not None:
        label_raster, polygon_ids = rasterize_labels(labels, shape, transform)
    else:
        label_raster = np.full(shape, NODATA, dtype="uint8")
        polygon_ids = np.zeros(shape, dtype="int32")

    train_mask = (label_raster != NODATA) & finite
    train_mask = subsample_by_polygon(polygon_ids, train_mask, max_per_polygon, rng)
    X = stack[:, train_mask].T
    y = label_raster[train_mask]
    return X, y, stack, finite, profile, label_raster


def fit_random_forest_A_models(X, y, seed):
    """Fit RF-A on per-pixel samples.

    class_weight is balanced because the classes are severely unequal in pixels
    even when the polygon counts look even: shrub polygons are small (median
    5 m2) and bare polygons large (median 82 m2), so shrub is ~6% of training
    pixels while bare is ~55%. Without weighting the forest simply under-calls
    shrub, which is the class the project most needs.

    Inputs:  X - [n, k] features; y - [n] class codes; seed - int
    Outputs: fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def per_class_scores(y_true, y_pred):
    """Per-class recall, precision and F1, plus macro-F1 and overall accuracy.

    Per class rather than pooled: overall accuracy at SRER is dominated by bare
    and grass and barely moves on shrub, the class that carries the scientific
    difficulty (instructions5.md section 6.4).

    Inputs:  y_true, y_pred - arrays of class codes
    Outputs: (dict of {class_code: {recall, precision, f1, support}},
              macro_f1 float, overall float)
    """
    scores, f1s = {}, []
    for code in CLASS_LABELS:
        true_c = y_true == code
        pred_c = y_pred == code
        support = int(true_c.sum())
        tp = int((true_c & pred_c).sum())
        recall = tp / support if support else float("nan")
        precision = tp / int(pred_c.sum()) if pred_c.sum() else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision > 0 and recall > 0) else 0.0
        scores[code] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "support": support,
        }
        if support:
            f1s.append(f1)
    overall = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    return scores, float(np.mean(f1s)) if f1s else float("nan"), overall


def print_scores(title, scores, macro_f1, overall):
    """Print a per-class score block.

    Inputs:  title - str; scores - dict from per_class_scores; macro_f1,
             overall - floats
    Outputs: None
    """
    print(f"\n{title}")
    print(f"{'class':<8}{'support':>9}{'recall':>9}{'precision':>11}{'f1':>8}")
    for code, label in CLASS_LABELS.items():
        s = scores[code]
        if not s["support"]:
            print(f"{label:<8}{0:>9}{'-':>9}{'-':>11}{'-':>8}")
            continue
        print(f"{label:<8}{s['support']:>9}{s['recall']:>9.3f}{s['precision']:>11.3f}{s['f1']:>8.3f}")
    print(f"macro-F1 {macro_f1:.3f}   overall {overall:.3f}  (overall is dominated by bare and grass - judge on per-class)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--run",
        type=run_label,
        default="1",
        help="run label - a number, optionally suffixed (e.g. 3 or 3_smoke). Every output lands under run{LABEL}/ so baselines stay comparable",
    )
    ap.add_argument(
        "--frameworks",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="which frameworks to run",
    )
    ap.add_argument(
        "--max-pixels-per-polygon",
        type=int,
        default=None,
        help="cap pixels contributed by one polygon; overrides the config value; 0 disables",
    )
    ap.add_argument(
        "--no-predict",
        action="store_true",
        help="cross-validate only, skip full-tile prediction",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite a run directory that already holds a report - THIS DESTROYS A FROZEN BASELINE",
    )
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    site, year = config["site"], config["year"]
    seed = config["BASE_SEED"] + year
    results = resolve_config_path(config["results_root"])
    feature_dir = results / "stage1_data_and_features" / "features"
    segments_dir = results / "stage1_data_and_features" / "segments"
    shadow_dir = results / "stage1_data_and_features" / "shadow"
    label_dir = results / "stage2_labeling"

    # run directory: step 1d outputs are run-scoped, features/segments/shadow are not
    run_dir = results / "stage3_classification" / f"run{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = config.get("stage3_1_classification", {})
    max_per_polygon = args.max_pixels_per_polygon if args.max_pixels_per_polygon is not None else settings.get("max_pixels_per_polygon")
    max_per_polygon = max_per_polygon or None

    manifest = json.loads((feature_dir / f"features_{site}_{year}_bands.json").read_text())
    band_order, bands_meta = manifest["band_order"], manifest["bands"]

    train_tiles = [t for t, role in config["tiles"].items() if role == "train"]
    test_tiles = [t for t, role in config["tiles"].items() if role == "test"]

    print(f"Step 1d - RF-A per-pixel classification - {site} {year}")
    print("=" * 74)
    print(f"run {args.run}   seed {seed}   train tiles {len(train_tiles)}   test tiles {len(test_tiles)}")
    print(f"polygon subsampling: {('max ' + str(max_per_polygon) + ' px per polygon') if max_per_polygon else 'OFF - every polygon contributes all its pixels'}")
    print("DIAGNOSTIC ONLY - cross-validation over training polygons is not map accuracy.")
    print("Step 6 accuracy needs the Olofsson protocol on an independent sample (section 6).")

    # the report ACCUMULATES across runs. Running one framework must not erase
    # the others: a fresh dict here would mean `--frameworks C` silently deleted
    # A and B, and the comparison figure could only ever show whatever was in
    # the last command. Matches the append-a-named-section convention in
    # instructions1.md section 5.
    # named report_path, not out: the prediction loop below binds `out` to the
    # classification array, which would shadow this and turn the final write into
    # ndarray.write_text
    report_path = run_dir / f"stage3_1_report_{site}_{year}.json"

    # a completed run is a frozen baseline. Re-running it would silently rewrite
    # the numbers a later run is being compared against, and labels drift
    # continuously while labelling is in progress - so the rewrite would not even
    # reproduce the original. Refuse, and say which run to use instead.
    if report_path.exists() and not args.force:
        existing = json.loads(report_path.read_text())
        done = ", ".join(sorted(existing.get("frameworks", {})))
        print(f"run {args.run} already exists at {report_path}")
        print(f"  it holds framework(s): {done}")
        print("refusing to overwrite a completed run - use a different --run label for new work, or --force to overwrite deliberately")
        return
        # per-run provenance, carried in the config so it is recorded automatically
        # rather than depending on someone remembering. A run that changes several
        # things at once is only interpretable if that is written down at the time.
    run_note = settings.get("run_notes", {}).get(str(args.run))
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
        "max_pixels_per_polygon": max_per_polygon,
        "validation": "leave-one-tile-out over train",
        "note": "diagnostic, not a Step 6 accuracy assessment",
        "run_note": run_note,
        "tiles": dict(config["tiles"]),
        "frameworks": {},
    }
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text())
            report["frameworks"] = previous.get("frameworks", {})
            kept = [k for k in sorted(report["frameworks"]) if k not in args.frameworks]
            if kept:
                print(f"carrying forward earlier results for framework(s): {', '.join(kept)}")
        except Exception as exc:
            print(f"warning: could not read the existing report, starting fresh: {exc}")

    for framework in args.frameworks:
        indices, names = framework_band_indices(band_order, bands_meta, framework)
        print(f"\n{'=' * 74}\nframework {framework} - {len(names)} features: {', '.join(names)}")

        cache = {}
        for tile in train_tiles + test_tiles:
            # one generator per tile per framework, seeded identically, so every
            # framework subsamples the SAME pixels and the section 4.1
            # deconfounding survives subsampling
            cache[tile] = load_tile(
                feature_dir,
                label_dir,
                shadow_dir,
                site,
                tile,
                year,
                indices,
                max_per_polygon,
                np.random.default_rng(seed),
            )

        counts = Counter()
        for tile in train_tiles:
            counts.update(cache[tile][1].tolist())
        print("train pixels per class: " + ", ".join(f"{CLASS_LABELS[c]} {counts.get(c, 0)}" for c in CLASS_LABELS))
        if min(counts.get(c, 0) for c in CLASS_LABELS) == 0:
            print("skipping: a class has no training pixels in the train block")
            continue

            # ---------------------------------------------------------- leave one tile out
        folds = []
        y_all, p_all = [], []
        for held in train_tiles:
            fit_tiles = [t for t in train_tiles if t != held]
            X = np.vstack([cache[t][0] for t in fit_tiles])
            y = np.concatenate([cache[t][1] for t in fit_tiles])
            Xh, yh = cache[held][0], cache[held][1]
            if not len(yh) or len(np.unique(y)) < 2:
                continue
            model = fit_random_forest_A_models(X, y, seed)
            pred = model.predict(Xh)
            scores, macro, overall = per_class_scores(yh, pred)
            print_scores(f"held-out tile {held}  ({len(yh)} labelled px)", scores, macro, overall)
            folds.append(
                {
                    "held_out": held,
                    "n": int(len(yh)),
                    "macro_f1": macro,
                    "overall": overall,
                    "per_class": {CLASS_LABELS[c]: scores[c] for c in CLASS_LABELS},
                }
            )
            y_all.append(yh)
            p_all.append(pred)

        if y_all:
            y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
            scores, macro, overall = per_class_scores(y_all, p_all)
            print_scores(f"POOLED across {len(folds)} folds", scores, macro, overall)
            cm = confusion_matrix(y_all, p_all, labels=list(CLASS_LABELS))
            print("\nconfusion (rows true, cols predicted)")
            print(f"{'':<8}" + "".join(f"{CLASS_LABELS[c]:>8}" for c in CLASS_LABELS))
            for i, code in enumerate(CLASS_LABELS):
                print(f"{CLASS_LABELS[code]:<8}" + "".join(f"{cm[i, j]:>8}" for j in range(len(CLASS_LABELS))))
            report["frameworks"][framework] = {
                "features": names,
                "train_pixels_per_class": {CLASS_LABELS[c]: int(counts.get(c, 0)) for c in CLASS_LABELS},
                "folds": folds,
                "pooled": {
                    "macro_f1": macro,
                    "overall": overall,
                    "per_class": {CLASS_LABELS[c]: scores[c] for c in CLASS_LABELS},
                },
                "confusion": cm.tolist(),
            }

        if args.no_predict:
            continue

            # ------------------------------------------------------------- full prediction
        X = np.vstack([cache[t][0] for t in train_tiles])
        y = np.concatenate([cache[t][1] for t in train_tiles])
        model = fit_random_forest_A_models(X, y, seed)
        importance = sorted(zip(names, model.feature_importances_), key=lambda kv: -kv[1])
        print("\ntop features: " + ", ".join(f"{n} {v:.3f}" for n, v in importance[:8]))
        report["frameworks"].setdefault(framework, {})["feature_importance"] = {n: float(v) for n, v in importance}

        out_dir = run_dir / framework
        out_dir.mkdir(parents=True, exist_ok=True)
        for tile in train_tiles + test_tiles:
            _, _, stack, valid, profile, _ = cache[tile]
            k, rows, cols = stack.shape
            flat = stack.reshape(k, -1).T
            out = np.full(rows * cols, NODATA, dtype="uint8")
            good = valid.reshape(-1)
            if good.any():
                out[good] = model.predict(flat[good]).astype("uint8")
                proba = model.predict_proba(flat[good]).astype("float32")
            out = out.reshape(rows, cols)

            profile.update(count=1, dtype="uint8", nodata=NODATA, compress="deflate")
            path = out_dir / f"classification_{framework}_{site}_{tile}_{year}.tif"
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(out, 1)
                # colour table travels inside the file, so QGIS and GDAL both render
                # the locked section 3 palette with no sidecar (section 12 Q10)
                ds.write_colormap(1, {code: hex_to_rgb(CLASS_COLORS[code]) for code in CLASS_LABELS})

                # NaN, not zero, outside the valid mask. predict_proba always sums
                # to 1 for a classified pixel, so an all-zero pixel could only ever
                # be masked - but written as 0 it reads as a legitimate probability
                # of zero and renders opaque. NaN matches the declared nodata and
                # renders transparent.
            prob = np.full((len(CLASS_LABELS), rows * cols), np.nan, dtype="float32")
            for i, code in enumerate(model.classes_):
                prob[int(code)][good] = proba[:, i]
            pprofile = dict(profile)
            pprofile.update(count=len(CLASS_LABELS), dtype="float32", nodata=np.nan)
            with rasterio.open(
                out_dir / f"class_probability_{framework}_{site}_{tile}_{year}.tif",
                "w",
                **pprofile,
            ) as ds:
                ds.write(prob.reshape(len(CLASS_LABELS), rows, cols))
                for code, name in CLASS_LABELS.items():
                    ds.set_band_description(code + 1, f"p_{name}")

                    # section 3.1 prediction_quality: 1 = pure, 0 = even four-way mix.
                    # The hard label is argmax regardless of how weak the winner is - by
                    # design - so the strength of that call has to be carried separately
                    # or it is lost.
            quality = np.full(rows * cols, np.nan, dtype="float32")
            margin = np.full(rows * cols, np.nan, dtype="float32")
            if good.any():
                safe = np.clip(proba, 1e-12, 1.0)
                entropy = -(safe * np.log(safe)).sum(axis=1)
                quality[good] = 1.0 - entropy / np.log(len(CLASS_LABELS))
                ordered = np.sort(proba, axis=1)
                margin[good] = ordered[:, -1] - ordered[:, -2]
            qprofile = dict(profile)
            qprofile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(
                out_dir / f"prediction_quality_{framework}_{site}_{tile}_{year}.tif",
                "w",
                **qprofile,
            ) as ds:
                ds.write(quality.reshape(rows, cols), 1)
            with rasterio.open(
                out_dir / f"margin_{framework}_{site}_{tile}_{year}.tif",
                "w",
                **qprofile,
            ) as ds:
                ds.write(margin.reshape(rows, cols), 1)

            share = {CLASS_LABELS[c]: float((out == c).mean()) for c in CLASS_LABELS}
            masked = float((out == NODATA).mean())
            median_quality = float(np.nanmedian(quality)) if good.any() else float("nan")
            print(f"[{tile}] " + " ".join(f"{k2} {v:.1%}" for k2, v in share.items()) + f"   masked {masked:.1%}   median quality {median_quality:.2f}")

    report_path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
