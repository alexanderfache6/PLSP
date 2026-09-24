"""Stage 5_3 - load the RF-B predictions into QGIS for visual review.

RUNS HEADLESS FROM A SHELL ONLY, inside the QGIS conda environment. It is not
written to run inside the QGIS Python console: it parses command-line arguments
and always bootstraps its own QgsApplication. Running it needs that environment
and two variables:

    export PYTHONPATH=$CONDA_PREFIX/share/qgis/python
    export QT_QPA_PLATFORM=offscreen
    export PROJ_DATA=$CONDA_PREFIX/share/proj PROJ_LIB=$PROJ_DATA

ARGUMENTS, the same two as stage 5_2 so one label is typed once and passed to
each script in turn:

    config
        positional, required. Site config JSON, e.g.
        config/srer_2022.json.
    --run
        required. Run LABEL, the directory name under
        `stage5_phenology_model_prediction/` with the leading `run`
        stripped. Stage 5_1 now fits one feature set, so a label is normally
        just the stage 4 run number; the historical `5` and `5_timing_evi`
        directories remain readable.

THE RUN LABEL NAMES THE STAGE 5 DIRECTORY ONLY. Which stage 4 run supplied the
fraction targets is read from that run's report, under `source_run`, because a
feature-set variant such as the retired 5_timing_evi wrote to its own stage 5
directory while still training against stage 4 run 5. Deriving both from one
label would look for a `stage4_aggregation/run5_timing_evi/` that does not
exist, so this indirection is kept for those historical runs. The
pairing is therefore self-describing and cannot be got wrong by typing.

EXAMPLE COMMANDS

    # the current run
    python run_stage5_3_create_qgis_phenology_project.py config/srer_2022.json --run 6

    # a historical run, retired feature sets included
    python run_stage5_3_create_qgis_phenology_project.py config/srer_2022.json --run 5_timing_evi

WHY THIS EXISTS. Stage 5_1 scores the model against held-out tiles and stage 5_2
plots those scores, but neither shows WHERE the model is wrong. A fractional
cover map can carry a respectable MAE while being systematically wrong over one
landform, and only looking at it against the imagery reveals that.

PLSP INPUT LAYERS ARE NOT LOADED, and cannot be. The phenology metrics live in a
netCDF, and GDAL in both the pipeline and the QGIS environment is built without
the netCDF and HDF5 drivers - `gdal.GetDriverByName("netCDF")` returns None, so
QGIS cannot open the product at all. Viewing the inputs alongside the
predictions requires either exporting the layers to GeoTIFF first or installing
the driver with `conda install -c conda-forge libgdal-netcdf`. Neither is done
here.

WHAT TO LOOK FOR, in order:

    1. Hard class against the 10 cm RGB. The dominant predicted class should
       follow real structure - washes, canopy patches, bare interfluves. If it
       looks like noise at Planet scale, the model has not learned geography.
    2. Predicted shrub fraction against OBSERVED shrub fraction from stage 4,
       over the ten labelled tiles. This is the only ground where the two can be
       compared directly, and it is where the reported MAE comes from.
    3. Predicted shrub fraction OUTSIDE the labelled tiles. There is no ground
       truth there, so the question is whether the pattern remains plausible or
       degenerates - the labelled tiles are 10 km2 of a roughly 100 km2
       footprint and were chosen to span the CHM shrub quintiles, not at random,
       so this is extrapolation.
    4. The RAW SUM layer. Neither model enforces the simplex, so this is the map
       of where the four fractions fail to reconstruct full cover. Structure in
       it - a region that consistently sums to 0.8 - is a model defect with a
       location, which no summary statistic gives.
    5. Joint against independent, on the same class. If the two maps are
       indistinguishable, the joint four-column target is buying nothing.

EVERY LAYER LEGEND IS COLLAPSED and everything except one reference layer starts
unchecked. Two models times six layers plus the observed stack is a long panel.

LAYER TREE, top to bottom. QgsLayerTreeGroup appends, so predictions sit above
the observations they are compared against and RGB is the basemap:

    RF-B joint - hard class, four soft fraction layers, four predicted-minus-
    observed difference layers, then the raw sum
    RF-B independent - the same
    observed, stage 4 - the fraction target the model was trained against, plus
    pure end members and the valid pixel count
    RGB - 10 cm imagery, all tiles
"""

import argparse
import json
import os

from constants import CLASS_COLORS, CLASS_LABELS, CLASS_NAMES
from helpers import expand_path, planet_blocks_directory
from qgis.core import (
    QgsApplication,
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsMultiBandColorRenderer,
    QgsPalettedRasterRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
)
from qgis.PyQt.QtGui import QColor

FRAMEWORK = "C"
MODEL_NAMES = ["joint", "independent"]
DIVERGING_LOW = "#8c5109"
DIVERGING_MID = "#f5f5f5"
DIVERGING_HIGH = "#1f5566"
MAGNITUDE_COLORS = ["#f4f6f7", "#cfdde2", "#8fb3bf", "#4d8496", "#1f5566"]


def style_fraction_band(layer, band_number, class_colour):
    """White-to-class-colour ramp over 0 to 1, for one class's percent cover.

    Percent cover is a magnitude, so a single hue, and ramping toward the
    class's own locked colour keeps identity consistent with the classification
    rasters. White at zero rather than a second hue avoids implying a polarity
    that does not exist.

    Inputs: layer - QgsRasterLayer; band_number - 1-based; class_colour - hex
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(0.0, QColor("#ffffff"), "0"),
            QgsColorRampShader.ColorRampItem(0.5, QColor(class_colour).lighter(140), "0.5"),
            QgsColorRampShader.ColorRampItem(1.0, QColor(class_colour), "1"),
        ]
    )
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band_number, shader))


def style_raw_sum(layer):
    """Diverging ramp centred on exactly 1, the value the fractions should sum to.

    THIS IS THE ONE LAYER THAT EARNS A DIVERGING RAMP. Everything else here is a
    magnitude, but the raw sum has a meaningful midpoint - 1.0 is correct, below
    is under-predicted cover and above is over-predicted - so a two-hue ramp with
    a neutral centre is the correct encoding rather than a decorative one. The
    range is fixed at 0.5 to 1.5 so the midpoint stays at 1 and two models can be
    compared without the stretch shifting underneath them.

    Inputs: layer - QgsRasterLayer
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(0.5, QColor(DIVERGING_LOW), "0.5 under"),
            QgsColorRampShader.ColorRampItem(1.0, QColor(DIVERGING_MID), "1.0 exact"),
            QgsColorRampShader.ColorRampItem(1.5, QColor(DIVERGING_HIGH), "1.5 over"),
        ]
    )
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def style_difference(layer, band_number):
    """Diverging ramp centred on zero, for predicted minus observed.

    A SIGNED ERROR HAS A TRUE MIDPOINT, so it earns a diverging ramp: brown
    where the model predicts less of the class than stage 4 observed, neutral at
    zero, blue where it predicts more. Fixed at -0.5 to +0.5 so zero stays at
    the centre and classes and models stay comparable, rather than each layer
    stretching to its own range and hiding which errors are large.

    Inputs: layer - QgsRasterLayer; band_number - 1-based
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(-0.5, QColor(DIVERGING_LOW), "-0.5 under-predicted"),
            QgsColorRampShader.ColorRampItem(0.0, QColor(DIVERGING_MID), "0 exact"),
            QgsColorRampShader.ColorRampItem(0.5, QColor(DIVERGING_HIGH), "+0.5 over-predicted"),
        ]
    )
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band_number, shader))


def style_magnitude(layer, ramp_minimum, ramp_maximum):
    """Single-hue light-to-dark ramp, deliberately outside the class palette.

    Keeping magnitude layers off the class hues means a magnitude cell can never
    be misread as a class mark.

    Inputs: layer - QgsRasterLayer; ramp_minimum, ramp_maximum - floats bounding
            the ramp, in the layer's own units
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    steps = len(MAGNITUDE_COLORS) - 1
    ramp.setColorRampItemList([QgsColorRampShader.ColorRampItem(ramp_minimum + (ramp_maximum - ramp_minimum) * step / steps, QColor(MAGNITUDE_COLORS[step]), f"{ramp_minimum + (ramp_maximum - ramp_minimum) * step / steps:.2f}") for step in range(len(MAGNITUDE_COLORS))])
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def style_class_codes(layer, label_prefix):
    """Locked section 3 palette with class names in the legend.

    A colour table inside the GeoTIFF gives colours but no labels, and an
    unlabelled legend defeats the point given the palette fails a normal-vision
    separation check on grass against bare.

    Inputs: layer - QgsRasterLayer of uint8 class codes; label_prefix - str
    Outputs: None, styled in place
    """
    classes = [QgsPalettedRasterRenderer.Class(code, QColor(CLASS_COLORS[code]), f"{label_prefix}{CLASS_LABELS[code]}") for code in CLASS_LABELS]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))


def add_layer(project, group, layer, checked):
    """Register a layer, append it to a group, and collapse its legend.

    Inputs: project; group - QgsLayerTreeGroup; layer - QgsRasterLayer;
            checked - bool visibility
    Outputs: True when the layer was valid and added
    """
    if not layer.isValid():
        print(f"WARNING: invalid layer skipped: {layer.name()}")
        return False
    project.addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setItemVisibilityChecked(checked)
    node.setExpanded(False)
    return True


def build_model_group(project, root, model_name, phenology_directory, site, year, is_first_model):
    """One collapsed group per model: hard class, soft fractions, then the raw sum.

    HARD CLASS LEADS because it is the layer that can be judged against the RGB
    at a glance. The soft fractions below it are the actual product; the hard
    class is a view of them (stage 5_1 writes it by argmax and it must never be
    used to compute area).

    Inputs: project; root; model_name; phenology_directory; site; year;
            is_first_model - bool, only the leading model loads anything checked
    Outputs: int, number of layers added
    """
    group = root.addGroup(f"RF-B {model_name}")
    stem = f"{model_name}_{FRAMEWORK}_{site}_{year}"
    added = 0

    class_path = os.path.join(phenology_directory, f"class_predicted_{stem}.tif")
    if os.path.exists(class_path):
        layer = QgsRasterLayer(class_path, f"dominant class {model_name}")
        if layer.isValid():
            style_class_codes(layer, "predicted ")
            added += add_layer(project, group, layer, is_first_model)

    fraction_path = os.path.join(phenology_directory, f"fraction_predicted_{stem}.tif")
    if os.path.exists(fraction_path):
        for class_code, class_name in enumerate(CLASS_NAMES):
            layer = QgsRasterLayer(fraction_path, f"{class_name}")
            if layer.isValid():
                style_fraction_band(layer, class_code + 1, CLASS_COLORS[class_code])
                added += add_layer(project, group, layer, False)

    difference_path = os.path.join(phenology_directory, f"fraction_difference_{stem}.tif")
    if os.path.exists(difference_path):
        for class_code, class_name in enumerate(CLASS_NAMES):
            layer = QgsRasterLayer(difference_path, f"{class_name} predicted minus observed {model_name}")
            if layer.isValid():
                style_difference(layer, class_code + 1)
                added += add_layer(project, group, layer, False)

    sum_path = os.path.join(phenology_directory, f"fraction_predicted_sum_{stem}.tif")
    if os.path.exists(sum_path):
        layer = QgsRasterLayer(sum_path, f"raw sum before renormalisation {model_name}")
        if layer.isValid():
            style_raw_sum(layer)
            added += add_layer(project, group, layer, False)

    group.setExpanded(False)
    return added


def build_observed_group(project, root, aggregation_directory, site, year, target_name):
    """The stage 4 fractions the model was trained against, for direct comparison.

    Loaded with the same ramps as the predictions so the two can be toggled
    against each other without the colour scale shifting underneath.

    Inputs: project; root; aggregation_directory; site; year; target_name - the
            stage 4 product used as the regression target
    Outputs: int, number of layers added
    """
    group = root.addGroup("observed, stage 4")
    added = 0

    observed_path = os.path.join(aggregation_directory, f"{target_name}_{FRAMEWORK}_{site}_{year}.tif")
    if os.path.exists(observed_path):
        for class_code, class_name in enumerate(CLASS_NAMES):
            layer = QgsRasterLayer(observed_path, f"{class_name}")
            if layer.isValid():
                style_fraction_band(layer, class_code + 1, CLASS_COLORS[class_code])
                added += add_layer(project, group, layer, False)

    pure_path = os.path.join(aggregation_directory, f"pure_endmember_{FRAMEWORK}_{site}_{year}.tif")
    if os.path.exists(pure_path):
        layer = QgsRasterLayer(pure_path, "pure end members, 8+ of 9")
        if layer.isValid():
            style_class_codes(layer, "pure ")
            added += add_layer(project, group, layer, False)

    count_path = os.path.join(aggregation_directory, f"valid_pixel_count_{FRAMEWORK}_{site}_{year}.tif")
    if os.path.exists(count_path):
        layer = QgsRasterLayer(count_path, "valid pixel count, 0 to 9")
        if layer.isValid():
            style_magnitude(layer, 0, 9)
            added += add_layer(project, group, layer, False)

    group.setExpanded(False)
    return added


def build_rgb_group(project, root, config, data_directory):
    """The 10 cm RGB basemap, one layer per tile.

    Check 1 is the whole reason this is loaded: the hard class map is judged
    against real structure in the imagery, not against other model output.
    """
    group = root.addGroup("RGB 10 cm basemap")
    rgb_spec = config["products"]["rgb"]
    added = 0
    for tile in config["tiles"]:
        path = expand_path(data_directory, rgb_spec["folder"], rgb_spec["pattern"].format(tile=tile))
        if not os.path.exists(path):
            continue
        layer = QgsRasterLayer(path, f"RGB {tile}")
        if not layer.isValid():
            continue
        layer.setRenderer(QgsMultiBandColorRenderer(layer.dataProvider(), 1, 2, 3))
        added += add_layer(project, group, layer, True)
    group.setExpanded(False)
    return added


def parse_arguments():
    """Config path and run label from the command line, matching stage 5_2.

    The three stage 5 scripts are now called the same way, so a run label is
    typed once and passed to each in turn rather than edited into a constant at
    the top of this file and forgotten.

    Outputs: argparse.Namespace with .config and .run
    """
    parser = argparse.ArgumentParser(description="Build the QGIS review project for one RF-B run.")
    parser.add_argument("config", help="site config JSON")
    parser.add_argument("--run", required=True, help="run label naming the stage 5 directory, e.g. 6")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    site, year = config["site"], config["year"]
    results_root = os.path.expanduser(config["results_root"])
    phenology_directory = os.path.join(results_root, "stage5_phenology_model_prediction", f"run{args.run}")
    data_directory = expand_path(config["data_root"], config["site_name"])

    report_path = os.path.join(phenology_directory, f"stage5_1_report_{site}_{year}.json")
    if not os.path.exists(report_path):
        raise SystemExit(f"MISSING {report_path} - run run_stage5_1_fit_phenology_fractional_cover.py --run {args.run} first")
    report = json.load(open(report_path))

    # THE STAGE 4 RUN IS READ FROM THE REPORT, NOT FROM THE RUN LABEL. A feature
    # set variant such as the retired 5_timing_evi wrote to its own directory while
    # still drawing its targets from stage 4 run 5, so deriving the aggregation
    # directory from the label would send this script to a stage4_aggregation
    # directory that does not exist. Every stage 5_1 report records the run that
    # supplied its targets, which makes the pairing self-describing.
    source_run = report.get("source_run", args.run)
    aggregation_directory = str(planet_blocks_directory(results_root, source_run))
    if not os.path.isdir(aggregation_directory):
        raise SystemExit(f"MISSING {aggregation_directory} - the report names source_run {source_run}, which has no stage 4 planet_blocks directory")

    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(config["expected_crs"]))
    # ABSOLUTE layer paths. QGIS defaults to paths relative to the .qgz, which
    # break silently the moment the project file changes directory depth.
    project.writeEntry("Paths", "/Absolute", True)
    root = project.layerTreeRoot()

    total = 0
    for model_index, model_name in enumerate(MODEL_NAMES):
        if model_name in report.get("models", {}):
            total += build_model_group(project, root, model_name, phenology_directory, site, year, model_index == 0)
    total += build_observed_group(project, root, aggregation_directory, site, year, report["target"])
    total += build_rgb_group(project, root, config, data_directory)

    output_path = os.path.join(phenology_directory, f"phenology_review_{site}_{year}.qgz")
    project.write(output_path)

    print(f"loaded {total} layers from run {args.run}, models {', '.join(sorted(report.get('models', {})))}")
    print(f"saved project: {output_path}")
    print("")
    print(f"phenology year {report['phenology_year']}, ground truth year {report['ground_truth_year']}, target {report['target']}, framework RF-A_{report['framework']}")
    print(f"feature set {report.get('feature_set', 'timing')}, targets from stage 4 run {source_run}")
    if not report.get("is_training_year", True):
        print("")
        print(f"OFF YEAR - these maps are predicted for {report['phenology_year']} from a model fitted")
        print(f"against {report.get('training_year', report['ground_truth_year'])} ground truth. Stage 4 has no fractional cover map for this year,")
        print("so the observed layers and the difference layers below belong to the training year.")
    print("")
    print("PLSP input layers are NOT loaded: GDAL here is built without the netCDF driver, so QGIS cannot open the product. Either export the layers to GeoTIFF or install the driver with conda install -c conda-forge libgdal-netcdf.")
    print("")
    print(
        "Check, in order: (1) hard class against the RGB, it should follow washes and canopy rather than look like noise; (2) predicted against observed shrub fraction over the ten labelled tiles, which is where the reported MAE comes from; (3) predicted shrub OUTSIDE those tiles, which is extrapolation with no ground truth; (4) the raw sum layer, where structure means a model defect with a location; (5) joint against independent on the same class, where indistinguishable maps mean the joint target buys nothing."
    )


def run():
    """Bootstrap a headless QGIS, then build the project.

    Without QgsApplication.initQgis() the GDAL and OGR providers are never
    registered, every QgsRasterLayer comes back invalid, and the project saves
    empty with no error. The prefix comes from sys.prefix, which is the conda
    environment running this interpreter, so nothing is hard-coded to a machine.
    """
    import sys

    QgsApplication.setPrefixPath(sys.prefix, True)
    application = QgsApplication([], False)
    application.initQgis()
    try:
        main()
    finally:
        application.exitQgis()


run()
