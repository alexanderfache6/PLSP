"""Stage 5_2 - apply the SRER model to any site and any year, through its end members.

ONE MODEL, TRAINED ONCE, APPLIED EVERYWHERE. Stage 5_1 fits RF-B at SRER 2022
against RF-A's fractions. This script never fits anything: it loads that model
and predicts, which is the whole point of the design - no site but SRER has a
ground truth map to train against, and labelling one at every site is what the
end member approach exists to avoid.

WHAT MAKES THE TRANSFER POSSIBLE. Both sides are expressed relative to their own
end members. SRER's features had SRER's m subtracted before fitting; this site's
features have THIS SITE'S m subtracted before predicting, read from stage 4_7's
statistics for the site's own imagery year. A juniper site whose season runs
three weeks later than SRER's therefore arrives on the same scale as SRER,
because both are measured from their own four class centres rather than from the
calendar.

THERE ARE NO SCORES AWAY FROM SRER, and any number that looked like one would be
invented. Stage 4 has ground truth for SRER 2022 alone. What this script reports
instead are three honest measures of whether the prediction can be believed:

    extrapolation share - the share of pixels falling outside the range of
        features the model was trained on, per feature and overall. A RANDOM
        FOREST CANNOT EXTRAPOLATE: beyond the training range it returns the edge
        of what it saw, confidently and silently. A site mostly outside that
        range is producing flat, plausible, meaningless output.
    held-out polygons - stage 4_7 keeps 3 polygons per class out of the
        statistics. Their own class's predicted fraction should be near 1. This
        is the closest thing to an accuracy check a new site has.
    raw sum before renormalisation - from the independent model, whose four
        forests do not know about one another. A sum far from 1 means the four
        predictions disagree about how much ground there is.

THE MODEL IS REFUSED UNLESS IT MATCHES. Feature names and their ORDER, the
feature set, the framework, the target and the normalisation are all checked
against what this run computed. A forest will happily predict from a matrix with
the right shape and the wrong columns, and the map would look completely normal.

INPUTS

    an end member config, config/endmembers/{SITE}_{year}_endmembers.json
    the pickles written by stage 5_1
    stage 4_7 statistics for this site, so the site must PASS stage 4_6
    the PLSP netCDF for each predicted year, production tier or stage tier

OUTPUTS, per site and year, under
{results_root}/stage5_phenology_model_prediction/run{N}/{SITE}_{year}/

    fraction_predicted_{model}_{framework}_{SITE}_{year}.tif, four bands
    class_predicted_{model}_{framework}_{SITE}_{year}.tif, the dominant class
    fraction_predicted_sum_{model}_{framework}_{SITE}_{year}.tif, the raw sum
    extrapolation_{framework}_{SITE}_{year}.tif, features out of range per pixel
    stage5_2_report_{SITE}_{year}.json

ARGUMENTS

    config
        positional, required. The site's end member config JSON.
    --run
        required. The stage 4 run holding this site's end member statistics.
    --model-run
        required. The stage 5 run label in the model filename, e.g. 5.
    --years
        one or more years to predict. Default: the site's own imagery year,
        the year its end members were measured in.
    --framework
        default C, matching the model.
    --target
        default fraction_hard_count, matching the model.
    --models
        default joint. `independent` or `joint independent` as well.

EXAMPLE COMMANDS

    conda activate LCSC

    # the site's own year
    python run_stage5_2_predict_phenology_fractional_cover.py config/endmembers/WKG_2023_endmembers.json --run 5 --model-run 5

    # several years with the same end members
    python run_stage5_2_predict_phenology_fractional_cover.py config/endmembers/SRER_2022_endmembers.json --run 5 --model-run 5 --years 2022 2023
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import run_stage4_4_download_site_endmember_tiles as download_stage
import run_stage4_7_compute_endmember_statistics as stats_stage
import run_stage5_1_fit_phenology_fractional_cover as train_stage
import xarray as xr
from constants import CLASS_LABELS, CLASS_NAMES, SEVENTY
from helpers import endmember_imagery_year, resolve_config_path


def planet_grid_from_production(production_grid):
    """The grid dict stage 5_1's writers expect, built from the production PLSP grid.

    Stage 5_1 reads its grid from a stage 1_3 JSON that exists for SRER alone.
    Every other site has the same information in its production PLSP file, which
    stages 4_4 to 4_7 already read and check, so it is reshaped here rather than
    requiring stage 1 to have been run somewhere it never will be.

    Inputs: production_grid - from read_production_grid
    Outputs: dict with ny, nx, origin_x, origin_y, planet_pixel_m, epsg
    """
    return {
        "ny": len(production_grid["y_centres"]),
        "nx": len(production_grid["x_centres"]),
        "origin_x": production_grid["left"],
        "origin_y": production_grid["top"],
        "planet_pixel_m": production_grid["pixel_size"],
        "epsg": production_grid["epsg"],
    }


def check_model_matches(payload, model_name, feature_names, framework, target, normalisation_name, path):
    """Refuse a model that was not fitted on this feature space.

    ORDER IS CHECKED, NOT JUST MEMBERSHIP. The 13 columns are unlabelled once
    they reach the forest, so a list with the same names in another order gives
    a full, normal-looking map built from the wrong feature in every split.

    Inputs: payload - the unpickled dict; model_name; feature_names; framework;
            target; normalisation_name; path - for the message
    Outputs: None, raises SystemExit on any mismatch
    """
    expected = {"model_name": model_name, "features": feature_names, "framework": framework, "target": target, "feature_set": train_stage.FEATURE_SET_NAME}
    for field, wanted in expected.items():
        if payload.get(field) != wanted:
            raise SystemExit(f"FAIL - {path.name} carries {field} {payload.get(field)!r}, this run needs {wanted!r}")
    trained_with = (payload.get("normalisation") or {}).get("name", "none")
    if trained_with != normalisation_name:
        raise SystemExit(f"FAIL - {path.name} was fitted with normalisation {trained_with!r} but this run applies {normalisation_name!r}. A model fitted on shifted features cannot be fed raw ones, or the other way round.")
    if "training_feature_range" not in payload:
        raise SystemExit(f"FAIL - {path.name} predates the training range and cannot report extrapolation. Refit with stage 5_1.")


def extrapolation_per_pixel(feature_stack, usable_mask, training_range):
    """How many features fall outside the training range, per pixel and per feature.

    THE p1 TO p99 RANGE IS USED, not the absolute extremes. One outlying
    training block would otherwise widen the range enough to declare a genuinely
    novel site perfectly in range, which is the opposite of what this measures.

    Inputs: feature_stack [13, ny, nx]; usable_mask; training_range - from the
            model payload
    Outputs: (count per pixel as float32 with NaN where unusable, dict of
             per-feature shares, overall share of pixels with any feature out)
    """
    low = np.asarray(training_range["p1"], dtype="float32")[:, np.newaxis, np.newaxis]
    high = np.asarray(training_range["p99"], dtype="float32")[:, np.newaxis, np.newaxis]
    outside = (feature_stack < low) | (feature_stack > high)
    outside &= usable_mask[np.newaxis, :, :]
    usable_count = int(usable_mask.sum())
    per_feature = {name: float(outside[position][usable_mask].mean()) for position, name in enumerate(training_range["features"])}
    count_per_pixel = outside.sum(axis=0).astype("float32")
    count_per_pixel[~usable_mask] = np.nan
    any_outside = float((outside.any(axis=0) & usable_mask).sum() / usable_count) if usable_count else 0.0
    return count_per_pixel, per_feature, any_outside


def held_out_polygon_check(config, stage4_run, predicted_stack, production_grid):
    """The predicted fraction of its own class, at each polygon stage 4_7 held out.

    THE ONLY ACCURACY-LIKE CHECK A NEW SITE HAS. These polygons were kept out of
    the end member statistics precisely so that something independent remains:
    they are pure ground of a known class, so the model should give that class a
    fraction near 1 there. A low value means the transfer is not working at this
    site, whatever the map looks like.

    The cells are found by stage 4_7's own rule, so the check reads exactly the
    cells the statistics would have read.

    Inputs: config; stage4_run; predicted_stack [4, ny, nx]; production_grid
    Outputs: dict per class with n polygons, n cells and the mean own-class
             fraction, or None when the statistics name no held-out polygons
    """
    from helpers import endmember_polygon_path, endmember_stats_directory

    stats_path = endmember_stats_directory(config["results_root"], stage4_run) / f"{config['site']}_{endmember_imagery_year(config)}_endmember_stats.json"
    statistics = json.loads(stats_path.read_text())
    polygons = gpd.read_file(endmember_polygon_path(config, stage4_run), layer=stats_stage.progress_stage.POLYGON_LAYER_NAME, fid_as_index=True)
    if polygons.crs is not None and polygons.crs.to_string() != config["expected_crs"]:
        polygons = polygons.to_crs(config["expected_crs"])
    results = {}
    for class_code, class_name in CLASS_LABELS.items():
        held_out = statistics["classes"][class_name]["held_out_fids"]
        values = []
        for feature_id in held_out:
            if feature_id not in polygons.index:
                continue
            for row, column in stats_stage.whole_cells_in(polygons.geometry.loc[feature_id], production_grid):
                value = predicted_stack[class_code, row, column]
                if np.isfinite(value):
                    values.append(float(value))
        results[class_name] = {"n_polygons": len(held_out), "n_cells": len(values), "mean_own_class_fraction": float(np.mean(values)) if values else None}
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict fractional cover at any site and year with the SRER model, through end member normalisation.")
    parser.add_argument("config", help="the site's end member config JSON")
    parser.add_argument("--run", required=True, help="stage 4 run holding this site's end member statistics, e.g. 5")
    parser.add_argument("--model-run", required=True, help="stage 5 run label in the model filename, e.g. 5")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="years to predict (default: the site's own imagery year)")
    parser.add_argument("--framework", default="C", help="RF-A framework the model was trained against (default C)")
    parser.add_argument("--target", default="fraction_hard_count", help="target the model was trained on (default fraction_hard_count)")
    parser.add_argument("--models", nargs="+", default=["joint"], choices=["joint", "independent"], help="which fitted models to apply (default joint, the product)")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    site, site_name = config["site"], config["site_name"]
    endmember_year = endmember_imagery_year(config)
    years = args.years or [endmember_year]
    results_root = resolve_config_path(config["results_root"])

    print(f"Stage 5_2 - predicting fractional cover at {site} {site_name}")
    print("=" * SEVENTY)
    print(f"end members from {endmember_year}, predicting {', '.join(str(year) for year in years)}, framework {args.framework}, models {', '.join(args.models)}")

    production_grid = download_stage.read_production_grid(config)
    download_stage.check_crs_agrees(config, production_grid)
    grid = planet_grid_from_production(production_grid)
    specification = train_stage.read_layer_specification(config, args.config)
    feature_names = [specification[number]["short_name"] for number in train_stage.RAW_TIMING_LAYER_NUMBERS] + [name for name, _, _ in train_stage.DERIVED_DURATION_FEATURES] + [specification[number]["short_name"] for number in train_stage.GREENING_LAYER_NUMBERS]

    # THE SITE'S OWN m, NOT SRER'S. This is the entire transfer: SRER subtracted
    # its own end members before fitting, and every other site subtracts its own
    # before predicting, so both arrive on one end-member-relative scale.
    normalisation = train_stage.read_endmember_normalisation(config, args.run, feature_names)
    print(f"normalisation {normalisation['name']} from {Path(normalisation['source']).name}, end members {normalisation['endmember_year']}")

    models = {}
    for model_name in args.models:
        model_path = train_stage.fitted_model_path(results_root, "SRER", train_stage.TRAINING_YEAR, args.framework, model_name, args.target, args.model_run)
        if not model_path.exists():
            raise SystemExit(f"FAIL - no {model_name} model at {model_path}. Train it with run_stage5_1_fit_phenology_fractional_cover.py.")
        payload = train_stage.read_model_payload(model_path)
        check_model_matches(payload, model_name, feature_names, args.framework, args.target, normalisation["name"], model_path)
        models[model_name] = payload
        print(f"loaded {model_name} from {model_path.name}, trained at {payload['site']} {payload['ground_truth_year']} with scikit-learn {payload.get('sklearn_version', 'unknown')}")

    for year in years:
        print(f"\n{'=' * SEVENTY}\n{site} {year}")
        netcdf_path = stats_stage.lsp_netcdf_path(config, year)
        with xr.open_dataset(netcdf_path, mask_and_scale=False) as dataset:
            stats_stage.check_netcdf_on_production_grid(dataset, production_grid, netcdf_path)
            quality_mask, quality_diagnostics = train_stage.read_quality_mask(dataset, specification)
            feature_stack, netcdf_feature_names = train_stage.read_phenology_feature_stack(dataset, specification)
        if netcdf_feature_names != feature_names:
            raise SystemExit(f"FAIL - {Path(netcdf_path).name} gives {netcdf_feature_names}, the model needs {feature_names}")
        feature_stack = train_stage.apply_endmember_normalisation(feature_stack, feature_names, specification, normalisation)
        usable = quality_mask & np.all(np.isfinite(feature_stack), axis=0)
        print(f"{Path(netcdf_path).name}, usable phenology {usable.mean():.2%} of {grid['ny'] * grid['nx']:,} pixels")

        output_directory = results_root / "stage5_phenology_model_prediction" / f"run{args.model_run}" / f"{site}_{year}"
        output_directory.mkdir(parents=True, exist_ok=True)
        report = {
            "site": site,
            "site_name": site_name,
            "phenology_year": year,
            "endmember_year": endmember_year,
            "is_endmember_year": year == endmember_year,
            "years_from_endmembers": year - endmember_year,
            "plsp_file": str(netcdf_path),
            "stage4_run": args.run,
            "model_run": args.model_run,
            "framework": args.framework,
            "target": args.target,
            "features": feature_names,
            "normalisation": normalisation,
            "qa": quality_diagnostics,
            "usable_share": float(usable.mean()),
            "models": {},
        }

        first_payload = models[args.models[0]]
        extrapolation_count, per_feature, any_outside = extrapolation_per_pixel(feature_stack, usable, first_payload["training_feature_range"])
        train_stage.write_prediction_raster(output_directory / f"extrapolation_{args.framework}_{site}_{year}.tif", extrapolation_count, grid, ["features_outside_training_p1_p99"])
        report["extrapolation"] = {"any_feature_outside_share": any_outside, "per_feature_share": per_feature, "range_used": "p1 to p99 of the training rows"}
        worst = sorted(per_feature.items(), key=lambda pair: -pair[1])[:4]
        print(f"extrapolation: {any_outside:.1%} of usable pixels have at least one feature outside the training range")
        print("worst features: " + ", ".join(f"{name} {share:.1%}" for name, share in worst))

        for model_name in args.models:
            fitted_model = models[model_name]["model"]
            print(f"\nmodel: {model_name}")
            predicted_stack, raw_sum = train_stage.predict_site_wide(model_name, fitted_model, feature_stack, usable, grid)
            stem = f"{model_name}_{args.framework}_{site}_{year}"
            train_stage.write_prediction_raster(output_directory / f"fraction_predicted_{stem}.tif", predicted_stack, grid, CLASS_NAMES)
            train_stage.write_prediction_raster(output_directory / f"fraction_predicted_sum_{stem}.tif", raw_sum, grid, ["raw_sum_before_renormalisation"])
            train_stage.write_class_raster(output_directory / f"class_predicted_{stem}.tif", train_stage.hard_class_from_fractions(predicted_stack), grid)
            finite_sum = raw_sum[np.isfinite(raw_sum)]
            held_out = held_out_polygon_check(config, args.run, predicted_stack, production_grid)
            report["models"][model_name] = {
                "pixels_predicted": int(np.isfinite(raw_sum).sum()),
                "raw_sum_mean": float(finite_sum.mean()),
                "raw_sum_min": float(finite_sum.min()),
                "raw_sum_max": float(finite_sum.max()),
                "raw_sum_mean_absolute_deviation_from_one": float(np.abs(finite_sum - 1.0).mean()),
                "mean_fraction_per_class": {class_name: float(np.nanmean(predicted_stack[class_code])) for class_code, class_name in enumerate(CLASS_NAMES)},
                "held_out_polygons": held_out,
            }
            print(f"{np.isfinite(raw_sum).sum():,} pixels predicted, raw sum mean {finite_sum.mean():.4f}, range {finite_sum.min():.4f} to {finite_sum.max():.4f}")
            print("mean fraction: " + ", ".join(f"{class_name} {np.nanmean(predicted_stack[class_code]):.3f}" for class_code, class_name in enumerate(CLASS_NAMES)))
            print("held-out polygons, own class fraction: " + ", ".join(f"{class_name} {values['mean_own_class_fraction']:.3f} over {values['n_cells']} cells" if values["mean_own_class_fraction"] is not None else f"{class_name} none" for class_name, values in held_out.items()))

        report_path = output_directory / f"stage5_2_report_{site}_{year}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {report_path}")

    print("\n" + "=" * SEVENTY)
    print(f"{site}: {len(years)} year(s) predicted. There is no ground truth away from SRER 2022, so no scores were produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
