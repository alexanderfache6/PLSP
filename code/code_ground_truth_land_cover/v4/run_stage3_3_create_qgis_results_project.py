#!/usr/bin/env python3
"""Load Step 1d classification results into QGIS for visual review.

Run inside QGIS: Plugins -> Python Console -> Show Editor -> Open Script -> Run,
or paste the file in. It builds the layer tree, applies the LOCKED section 3
class colors, and saves a .qgz next to the results so the project can simply be
reopened afterwards.

Written as a QGIS script rather than a hand-authored .qgs because the project XML
schema is version-specific and silently degrades when it does not match the
running QGIS; the API does not.

    CONFIG below are the only things to edit.

RUNS are numbered and separated on disk. Everything is read from
stage3_classification/run{N}/, matching run_stage3_1_random_forest_ground_truth_classification.py, so a baseline
run stays loadable after the inputs change.

The run is chosen two ways, because this script has two callers:

    --run 2 from a shell, matching run_stage3_1_random_forest_ground_truth_classification.py and
                  run_stage3_2_generate_ground_truth_classification_plots.py

The flag wins when present. The saved .qgz lands inside the run directory, so
loading run 1 and run 2 produces two separate projects rather than overwriting
one.

GROUPING IS BY METHOD, NOT BY TILE, which is the opposite of the labeling
project and deliberate. Labeling is done one tile at a time, so tile groups suit
it. Reviewing results means comparing frameworks over the same ground, so the
tiles of one framework must toggle together as a unit - otherwise switching from
A to B means ticking five boxes and unticking five more.

Layer tree, top of the panel to bottom. QgsLayerTreeGroup appends, so groups land
in exactly this order and RGB sits underneath everything as the basemap:

    framework A classification + per-class probability, all tiles
    framework B ...
    ... one group per framework found on disk
    RGB 10 cm imagery, all tiles

Within a framework group, classification layers sit above probability layers, and
probability layers load UNCHECKED. Four probability bands per tile times five
tiles is twenty layers; drawing them all would obscure the classification and
make the project slow to open. Tick the one being interrogated.

FRAMEWORKS ARE DISCOVERED FROM DISK, never listed in this file. Every immediate
subdirectory of stage3_classification that holds a classification raster
becomes a group, named after the directory. Run framework C tomorrow and it
appears on the next load with no edit here - which is the point, since this
project is read-only and gets rebuilt rather than maintained.

Nothing in the project is editable: every layer is a raster and the .qgz is
regenerated from scratch each run. There is no unsaved-edit hazard, unlike the
labeling project.
"""

import argparse
import glob
import json
import os
import sys

from constants import CLASS_COLORS, CLASS_LABELS, FRAMEWORK_ORDER
from helpers import expand_path
from qgis.core import (
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

# --------------------------------------------------------------------- CONFIG
CONFIG = os.path.expanduser("~/Documents/GitHub/PLSP/code/code_ground_truth_land_cover/v4/config/srer_2022.json")
# -----------------------------------------------------------------------------


def resolve_run():
    """Run number from --run

    argparse is not used directly on sys.argv because the QGIS Python Console
    populates argv with the QGIS application's own arguments, which would make
    a strict parser fail. Only an explicit --run is honoured; anything else is
    ignored and the constant applies.

    Inputs: none, reads sys.argv
    Outputs: int run number
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default=None)
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known.run


def discover_frameworks(cls_root, site, year):
    """Framework directories present on disk, in preferred display order.

    A directory qualifies when it holds at least one classification raster named
    for it. Discovery rather than a hard-coded list means a framework added
    later appears automatically, and a framework not yet run is simply absent
    instead of logging a warning per tile.

    Inputs: cls_root - path to stage3_classification; site, year
    Outputs: list of (framework name, directory path) tuples
    """
    found = []
    for entry in sorted(os.listdir(cls_root)):
        directory = os.path.join(cls_root, entry)
        if not os.path.isdir(directory):
            continue
        if not glob.glob(os.path.join(directory, f"classification_{entry}_{site}_*_{year}.tif")):
            continue
        found.append((entry, directory))
    rank = {name: i for i, name in enumerate(FRAMEWORK_ORDER)}
    return sorted(found, key=lambda item: (rank.get(item[0], len(rank)), item[0]))


def style_classification(layer):
    """Paletted renderer using the locked section 3 class colors.

    The GeoTIFF already carries an embedded colour table written by
    run_stage3_1_random_forest_ground_truth_classification.py, so QGIS would render this correctly unaided. The
    renderer is set explicitly anyway to guarantee the class NAMES appear in the
    legend - a colour table gives colors but no labels, and an unlabelled
    legend defeats the point given the palette fails a normal-vision separation
    check on grass versus bare.

    Inputs: layer - a QgsRasterLayer of uint8 class codes
    Outputs: None, styled in place
    """
    classes = [QgsPalettedRasterRenderer.Class(code, QColor(CLASS_COLORS[code]), f"{code} {CLASS_LABELS[code]}") for code in CLASS_LABELS]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))


def style_probability(layer, band, colour):
    """Single-hue ramp from transparent-white to the class colour, 0 to 1.

    Each probability band is ramped toward its OWN class colour, so a stack of
    probability layers reads as the same identity system as the classification.
    White at zero rather than a second hue: this is a magnitude, and a two-hue
    ramp would imply a polarity that does not exist.

    Inputs: layer - QgsRasterLayer; band - 1-based band number; colour - hex
    Outputs: None, styled in place
    """
    shader = QgsRasterShader()
    function = QgsColorRampShader()
    function.setColorRampType(QgsColorRampShader.Interpolated)
    function.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(0.0, QColor("#ffffff"), "0.0"),
            QgsColorRampShader.ColorRampItem(0.5, QColor(colour).lighter(140), "0.5"),
            QgsColorRampShader.ColorRampItem(1.0, QColor(colour), "1.0"),
        ]
    )
    shader.setRasterShaderFunction(function)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), band, shader))


def add_layer(project, group, layer, checked, expanded):
    """Register a layer and append it to a group, setting its legend state.

    Inputs: project - QgsProject; group - QgsLayerTreeGroup; layer -
             QgsRasterLayer; checked - bool visibility; expanded - bool legend
    Outputs: True when the layer was valid and added
    """
    if not layer.isValid():
        print(f"WARNING: invalid layer skipped: {layer.name()}")
        return False
    project.addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setExpanded(expanded)
    node.setItemVisibilityChecked(checked)
    return True


def build_framework_group(project, root, framework, cls_dir, site, year, tiles):
    """One collapsed group holding every tile's classification and probabilities.

    Classification layers first so they sit above the probability layers they
    summarize. Probabilities load unchecked - see the module docstring.

    Inputs: project, root - QGIS objects; framework - key string; cls_dir -
             directory holding the framework's rasters; site, year; tiles - dict
             of {tile: role}
    Outputs: int, number of layers added
    """
    group = root.addGroup(f"RF-A_{framework}")
    added = 0

    for tile, role in tiles.items():
        path = os.path.join(cls_dir, f"classification_{framework}_{site}_{tile}_{year}.tif")
        if not os.path.exists(path):
            continue
        layer = QgsRasterLayer(path, f"class RF-A_{framework} {tile} ({role})")
        if layer.isValid():
            style_classification(layer)
        added += add_layer(project, group, layer, False, False)

    for tile, role in tiles.items():
        path = os.path.join(cls_dir, f"class_probability_{framework}_{site}_{tile}_{year}.tif")
        if not os.path.exists(path):
            continue
        for code, name in CLASS_LABELS.items():
            layer = QgsRasterLayer(path, f"p({name}) RF-A_{framework} {tile}")
            if layer.isValid():
                # band order is class-code order, written by run_stage3_1_random_forest_ground_truth_classification.py
                style_probability(layer, code + 1, CLASS_COLORS[code])
            added += add_layer(project, group, layer, False, False)

    group.setExpanded(False)
    return added


def build_rgb_group(project, root, config, data, tiles):
    """One group holding the 10 cm RGB basemap for every tile.

    Added last so it lands at the bottom of the panel and sits underneath the
    results rather than covering them.

    Inputs: project, root - QGIS objects; config - parsed config; data - site data
             directory; tiles - dict of {tile: role}
    Outputs: int, number of layers added
    """
    product = config["products"]["rgb"]
    group = root.addGroup("RGB 10cm")
    added = 0
    for tile, role in tiles.items():
        path = os.path.join(data, product["folder"], product["pattern"].format(tile=tile))
        layer = QgsRasterLayer(path, f"RGB {tile} ({role})")
        if layer.isValid():
            layer.setRenderer(QgsMultiBandColorRenderer(layer.dataProvider(), 1, 2, 3))
        added += add_layer(project, group, layer, True, False)
    group.setExpanded(False)
    return added


def main():
    config = json.load(open(CONFIG))
    site, year = config["site"], config["year"]
    results = os.path.expanduser(config["results_root"])
    data = expand_path(config["data_root"], config["site_name"])
    run = resolve_run()
    cls_root = os.path.join(results, "stage3_classification", f"run{run}")
    if not os.path.isdir(cls_root):
        print(f"no such run directory: {cls_root}")
        return
    tiles = config["tiles"]

    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(config["expected_crs"]))
    # ABSOLUTE layer paths, deliberately. QGIS defaults to paths relative to the
    # .qgz, which silently break the moment the project file changes directory
    # depth - a rename of a results directory is enough, and the failure is
    # invisible until the project is opened. Absolute paths cost portability
    # between machines; these projects are regenerated per machine anyway.
    project.writeEntry("Paths", "/Absolute", True)
    root = project.layerTreeRoot()

    found = []
    for framework, cls_dir in discover_frameworks(cls_root, site, year):
        added = build_framework_group(project, root, framework, cls_dir, site, year, tiles)
        found.append((framework, added))

    rgb_added = build_rgb_group(project, root, config, data, tiles)

    out = os.path.join(cls_root, f"results_{site}_{year}.qgz")
    project.write(out)

    if not found:
        print("no classification rasters found - run run_stage3_1_random_forest_ground_truth_classification.py first")
    print(f"run {run}")
    for framework, added in found:
        print(f"RF-A_{framework}: {added} layers")
    print(f"RGB 10cm: {rgb_added} layers")
    print(f"total {len(project.mapLayers())} layers in {len(found) + 1} groups")
    print(f"saved project: {out}")
    print("\nGrouped by METHOD, not by tile, so a whole framework toggles as one unit when comparing.")
    print("Probability layers load unchecked - four bands times five tiles would obscure the classification. Tick the one being interrogated.")
    print("Classification rasters also carry an embedded colour table, so they render correctly in any GDAL reader without this project.")
    print("Frameworks are discovered from disk - run another one and re-run this script; nothing here needs editing.")


main()
