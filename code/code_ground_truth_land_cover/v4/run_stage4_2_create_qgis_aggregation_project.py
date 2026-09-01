"""Load the stage 4_1 fraction products into QGIS for visual review.

Run inside QGIS: Plugins -> Python Console -> Show Editor -> Open Script -> Run.
QGIS must be its own conda environment (LCSC_QGIS, instructions5.md section 7A) -
it pulls a large Qt stack and must not contaminate the pipeline environment.

WHY THIS EXISTS. Stage 4_1 verifies its own arithmetic: fractions sum to 1.0,
the transform matches the Planet grid, valid counts span 0 to N^2. None of that
shows whether a block's fractions describe the ground under it. A block is nine
1 m pixels, so it can be checked directly by eye against the classification it
came from - which is the one check no assertion can make.

WHAT TO LOOK FOR, in order:

    1. Pick a block and compare its fraction against the 1 m classification
       underneath. Nine pixels, so the hard-count fraction can only be a
       multiple of 1/9 - 0.111, 0.222, 0.333 and so on. A value between those
       steps means the block was built from the wrong denominator.
    2. Toggle valid pixel count. Blocks below 9 should sit on tile edges, on
       shadow, and along the unflown strip of 517000_3531000 - not scattered
       through open ground. Interior speckle would mean shadow is being detected
       where there is none.
    3. Toggle block prediction quality against the shrub fraction. Low quality
       should concentrate where shrub and grass meet, which is where the model
       is genuinely undecided. Low quality in the middle of pure bare would be a
       warning.
    4. THE DECISIVE ONE, and the reason both frameworks are loaded. Compare
       shrub fraction under RF-A_C against RF-A_D over the same ground. They
       disagree on 19.5% of blocks by three or more pixels of nine, and mean
       shrub cover differs by 20% (results/stage4_results.md section 7). The
       pixel-level scores do not predict a gap that size, because it lives on
       ground the training polygons never covered. Look at where the two part
       company and judge which is reading the RGB correctly.
    5. Toggle PLANET EVIamp. Fractions are the ground truth this product exists
       to align with the Planet data, so high shrub and tree cover should track
       the EVIamp pattern. It will not match cleanly - that mismatch is the
       whole reason RF-B is being fitted - but a total absence of correspondence
       would mean something upstream is wrong.

FRACTION LAYERS ARE ONE PER CLASS, not one four-band layer. A four-band raster
renders as false-colour RGB, which mixes three classes into one image and drops
the fourth. Percent cover of a single class is a MAGNITUDE, so each class gets
its own single-band layer ramped from white to that class's locked section 3
colour - the same identity system the classification uses.

Only the hard-count fraction is loaded by default. Soft mean and confidence
weighted are loaded unchecked: they answer narrower questions (sub-pixel mixture
and ambiguity weighting) and three sets of four class layers on top of each
other would make the panel unusable.

The project is read-only and regenerated from scratch each run. Every layer is a
view onto what stage 4_1 wrote, nothing here is editable, and there is no
unsaved-edit hazard unlike the labeling project.

PURE END MEMBERS LEAD EACH FRAMEWORK GROUP. They are what Step 4 selects and
what anchors RF-B at Step 5, and a block is pure when at least 8 of its 9 pixels
share one class. That is a COUNT, not a percentage, for the same reason the
retention rule is: at N = 3 the achievable hard-count fractions are multiples of
1/9, so the spec's ">= 90%" admitted only 9-of-9 blocks because 8/9 = 0.889 is
below 0.90. Correcting it roughly doubles the pool - shrub goes from 90,111 to
172,028 blocks under RF-A_C.

EVERY LAYER LEGEND IS COLLAPSED and every group starts closed. With 47 layers,
expanded legends push the tree past the panel and the groups stop being
navigable.

LAYER TREE, top of the panel to bottom. QgsLayerTreeGroup appends, so the block
products sit above the 1 m pixels they were built from and RGB is the basemap:

    RF-A_C - pure end members, then hard count per class, then soft mean and
    confidence weighted unchecked
    RF-A_D - the same, all unchecked, for the section 7 comparison
    block diagnostics - valid pixel count and block prediction quality
    classification 1 m - the pixels each block was aggregated from
    PLANET EVIamp - the LSP data the fractions will be matched against
    RGB - 10 cm imagery, all tiles
"""

import json
import os

from constants import CLASS_COLORS, CLASS_LABELS, CLASS_NAMES
from helpers import expand_path
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

CONFIG = os.path.expanduser("~/Documents/GitHub/PLSP/code/code_ground_truth_land_cover/v4/config/srer_2022.json")
RUN = "5"

FRACTION_KINDS = [
    ("fraction_hard_count", "hard count", True),
    ("fraction_soft_mean", "soft mean", False),
    ("fraction_confidence_weighted", "confidence weighted", False),
]
QUALITY_COLORS = ["#f4f6f7", "#cfdde2", "#8fb3bf", "#4d8496", "#1f5566"]


def style_fraction(layer, band, colour):
    """White-to-class-colour ramp over 0 to 1, for one class's percent cover.

    Percent cover is a magnitude, so a single hue. Ramping toward the class's
    own locked colour keeps identity consistent with the classification raster,
    and white at zero rather than a second hue avoids implying a polarity that
    does not exist.

    Inputs: layer - QgsRasterLayer; band - 1-based band number; colour - hex
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(0.0, QColor("#ffffff"), "0"),
            QgsColorRampShader.ColorRampItem(0.5, QColor(colour).lighter(140), "0.5"),
            QgsColorRampShader.ColorRampItem(1.0, QColor(colour), "1"),
        ]
    )
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band, shader))


def style_valid_pixel_count(layer, per_block, min_valid):
    """Paletted 0 to N^2, with the retention cut visible as a colour break.

    Paletted rather than a continuous ramp because the count is discrete and
    only three cuts exist at N = 3. Blocks that fail the rule are red and blocks
    that pass are green, so the cost of the threshold is legible at a glance
    rather than requiring the legend to be read.

    Inputs: layer - QgsRasterLayer; per_block - N^2; min_valid - retention cut
    Outputs: None, styled in place
    """
    classes = []
    for count in range(per_block + 1):
        if count >= min_valid:
            colour = QColor("#1f5566") if count == per_block else QColor("#4d8496")
            label = f"{count} of {per_block} - kept"
        else:
            colour = QColor("#c0563a").lighter(160 - 10 * count)
            label = f"{count} of {per_block} - dropped"
        classes.append(QgsPalettedRasterRenderer.Class(count, colour, label))
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))


def style_magnitude(layer, low, high):
    """Single-hue light-to-dark ramp, deliberately outside the class palette.

    Used for block prediction quality and EVIamp. Keeping magnitude layers off
    the class hues means a magnitude cell can never be misread as a class mark.

    Inputs: layer - QgsRasterLayer; low, high - floats bounding the ramp
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    steps = len(QUALITY_COLORS) - 1
    ramp.setColorRampItemList([QgsColorRampShader.ColorRampItem(low + (high - low) * i / steps, QColor(QUALITY_COLORS[i]), f"{low + (high - low) * i / steps:.2f}") for i in range(len(QUALITY_COLORS))])
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def style_classification(layer):
    """Locked section 3 palette with class NAMES in the legend.

    The GeoTIFF carries a colour table, but a colour table gives colors without
    labels, and an unlabelled legend defeats the point given the palette fails a
    normal-vision separation check on grass versus bare.

    Inputs: layer - QgsRasterLayer of uint8 class codes
    Outputs: None, styled in place
    """
    classes = [QgsPalettedRasterRenderer.Class(code, QColor(CLASS_COLORS[code]), f"{code} {CLASS_LABELS[code]}") for code in CLASS_LABELS]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))


def add_layer(project, group, layer, checked):
    """Register a layer and append it to a group.

    Inputs: project - QgsProject; group - QgsLayerTreeGroup; layer -
            QgsRasterLayer; checked - bool visibility
    Outputs: True when the layer was valid and added
    """
    if not layer.isValid():
        print(f"WARNING: invalid layer skipped: {layer.name()}")
        return False
    project.addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setItemVisibilityChecked(checked)
    # every layer legend collapsed. With 47 layers, expanded legends push the
    # tree past the panel and the groups stop being navigable.
    node.setExpanded(False)
    return True


def style_pure_endmember(layer):
    """Locked section 3 palette over the dominant class code, NODATA elsewhere.

    A pure end-member raster is categorical, not a magnitude: the value is which
    class owns the block, so it wears the same colors as the classification.

    Inputs: layer - QgsRasterLayer of uint8 class codes with 255 nodata
    Outputs: None, styled in place
    """
    classes = [QgsPalettedRasterRenderer.Class(code, QColor(CLASS_COLORS[code]), f"pure {CLASS_LABELS[code]}") for code in CLASS_LABELS]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))


def build_framework_group(project, root, framework, aggregation_dir, site, year, is_first, min_pure, per_block):
    """One collapsed group per framework: its pure end members, then its fractions.

    PURE END MEMBERS LEAD, because they are what Step 4 selects and what anchors
    RF-B at Step 5. A block is pure when at least `min_pure` of its `per_block`
    pixels share one class - 8 or 9 of 9 - which is a COUNT, not a percentage.
    At N = 3 the achievable fractions are multiples of 1/9, so the spec's
    ">= 90%" rule admitted only 9-of-9 blocks and discarded every 8-of-9 one,
    roughly halving the pool. See stage4_1 for the measurement.

    The fraction layers below them are the evidence: a block flagged pure should
    show its class at 0.889 or 1.000 and everything else near zero.

    Inputs: project; root - layer tree root; framework - letter; aggregation_dir
            - path to the run's stage4 outputs; site, year; is_first - bool, the
            leading framework is the one loaded checked; min_pure, per_block
    Outputs: int, number of layers added
    """
    group = root.addGroup(f"RF-A_{framework}")
    added = 0
    pure_path = os.path.join(aggregation_dir, f"pure_endmember_{framework}_{site}_{year}.tif")
    if os.path.exists(pure_path):
        layer = QgsRasterLayer(pure_path, f"pure end members RF-A_{framework} ({min_pure}+ of {per_block})")
        if layer.isValid():
            style_pure_endmember(layer)
            added += add_layer(project, group, layer, is_first)
    for kind, label, kind_default in FRACTION_KINDS:
        path = os.path.join(aggregation_dir, f"{kind}_{framework}_{site}_{year}.tif")
        if not os.path.exists(path):
            continue
        for code, name in enumerate(CLASS_NAMES):
            layer = QgsRasterLayer(path, f"{name} {label} RF-A_{framework}")
            if not layer.isValid():
                continue
            style_fraction(layer, code + 1, CLASS_COLORS[code])
            added += add_layer(project, group, layer, is_first and kind_default)
    group.setExpanded(False)
    return added


def main():
    config = json.load(open(CONFIG))
    site, year = config["site"], config["year"]
    results_root = os.path.expanduser(config["results_root"])
    aggregation_dir = os.path.join(results_root, "stage4_aggregation", f"run{RUN}")
    classification_dir = os.path.join(results_root, "stage3_classification", f"run{RUN}")
    qa_dir = os.path.join(results_root, "stage1_data_and_features", "qa")
    data_dir = expand_path(config["data_root"], config["site_name"])

    report_path = os.path.join(aggregation_dir, f"stage4_1_report_{site}_{year}.json")
    if not os.path.exists(report_path):
        raise SystemExit(f"MISSING {report_path} - run run_stage4_1_aggregate_to_planet_blocks.py --run {RUN} first")
    report = json.load(open(report_path))
    frameworks = sorted(report.get("frameworks", {}))
    if not frameworks:
        raise SystemExit("the stage 4_1 report holds no frameworks")

    per_block = int(report["grid"]["N"]) ** 2
    min_valid = int(report["min_valid_pixels_per_block"])
    min_pure = int(report["frameworks"][frameworks[0]].get("min_pure_pixels_per_block", 8))

    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(config["expected_crs"]))
    # ABSOLUTE layer paths. QGIS defaults to paths relative to the .qgz, which
    # break silently the moment the project file changes directory depth - a
    # results-directory rename is enough, and the failure is invisible until the
    # project is opened.
    project.writeEntry("Paths", "/Absolute", True)
    root = project.layerTreeRoot()

    total = 0
    for index, framework in enumerate(frameworks):
        total += build_framework_group(project, root, framework, aggregation_dir, site, year, index == 0, min_pure, per_block)

    diagnostics = root.addGroup("block diagnostics")
    lead = frameworks[0]
    count_path = os.path.join(aggregation_dir, f"valid_pixel_count_{lead}_{site}_{year}.tif")
    if os.path.exists(count_path):
        layer = QgsRasterLayer(count_path, f"valid pixel count RF-A_{lead}")
        style_valid_pixel_count(layer, per_block, min_valid)
        total += add_layer(project, diagnostics, layer, False)
    quality_path = os.path.join(aggregation_dir, f"block_prediction_quality_{lead}_{site}_{year}.tif")
    if os.path.exists(quality_path):
        layer = QgsRasterLayer(quality_path, f"block prediction quality RF-A_{lead}")
        style_magnitude(layer, 0.0, 1.0)
        total += add_layer(project, diagnostics, layer, False)
    diagnostics.setExpanded(False)

    pixels = root.addGroup(f"classification 1 m RF-A_{lead}")
    for tile in config["tiles"]:
        path = os.path.join(classification_dir, lead, f"classification_{lead}_{site}_{tile}_{year}.tif")
        if not os.path.exists(path):
            continue
        layer = QgsRasterLayer(path, f"class {tile}")
        style_classification(layer)
        total += add_layer(project, pixels, layer, False)
    pixels.setExpanded(False)

    planet = root.addGroup("PLANET")
    variable = config["stage1_3_planet_grid"].get("verification_variable", "EVIamp")
    lsp_year = config["stage1_3_planet_grid"]["grid_source_year"]
    lsp_path = os.path.join(qa_dir, f"planet_{variable}_{site}_{lsp_year}.tif")
    if os.path.exists(lsp_path):
        layer = QgsRasterLayer(lsp_path, f"PLANET {variable} {lsp_year}")
        style_magnitude(layer, 0.0, 0.55)
        total += add_layer(project, planet, layer, False)
    planet.setExpanded(False)

    rgb_group = root.addGroup("RGB 10 cm basemap")
    rgb_spec = config["products"]["rgb"]
    for tile in config["tiles"]:
        path = expand_path(data_dir, rgb_spec["folder"], rgb_spec["pattern"].format(tile=tile))
        if not os.path.exists(path):
            continue
        layer = QgsRasterLayer(path, f"RGB {tile}")
        if not layer.isValid():
            continue
        layer.setRenderer(QgsMultiBandColorRenderer(layer.dataProvider(), 1, 2, 3))
        total += add_layer(project, rgb_group, layer, True)
    rgb_group.setExpanded(False)

    out = os.path.join(aggregation_dir, f"aggregation_review_{site}_{year}.qgz")
    project.write(out)

    print(f"loaded {total} layers from run {RUN}, frameworks {', '.join(frameworks)}")
    print(f"saved project: {out}")
    print("")
    print(f"block = {per_block} one-metre pixels, retained at >= {min_valid} valid")
    print(f"hard-count fractions can only be multiples of 1/{per_block} = {1 / per_block:.3f}")
    print("")
    print(
        "Check, in order: (1) a block's fraction against the 1 m classification underneath, at multiples of 1/9; (2) valid pixel count below 9 sits on tile edges, shadow and the unflown strip, not scattered through open ground; (3) low block prediction quality concentrates where shrub meets grass; (4) THE DECISIVE ONE - shrub fraction RF-A_C against RF-A_D, which disagree on 19.5% of blocks by three or more pixels of nine; (5) high shrub and tree cover should broadly track the EVIamp pattern."
    )


def run():
    """Run inside QGIS, or bootstrap a headless QGIS when launched from a shell.

    The other QGIS scripts in this project are pasted into the QGIS Python
    console, where QgsApplication already exists. This one is also useful from
    the command line, and there QgsApplication must be created and initQgis()
    called before anything else: without it the GDAL and OGR providers are never
    registered, every QgsRasterLayer comes back invalid, and the project saves
    with nothing in it and no error.

    The prefix comes from sys.prefix, which is the conda environment running
    this interpreter, so nothing is hard-coded to one machine.

    Headless use needs two environment variables set by the caller:

        PYTHONPATH=$CONDA_PREFIX/share/qgis/python
        QT_QPA_PLATFORM=offscreen
    """
    import sys

    if QgsApplication.instance() is not None:
        main()
        return
    QgsApplication.setPrefixPath(sys.prefix, True)
    application = QgsApplication([], False)
    application.initQgis()
    try:
        main()
    finally:
        application.exitQgis()


run()
