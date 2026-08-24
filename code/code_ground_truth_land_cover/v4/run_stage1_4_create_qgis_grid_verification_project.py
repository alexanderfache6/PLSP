#!/usr/bin/env python3
"""Load the Step 0c grid layers into QGIS for the mandatory visual alignment check.

Run inside QGIS: Plugins -> Python Console -> Show Editor -> Open Script -> Run.
QGIS must be its own conda environment (LCSC_QGIS, instructions5.md section 7A) -
it pulls a large Qt stack and must not contaminate the pipeline environment.

WHY THIS EXISTS AS A SEPARATE STEP. instructions5.md section 5 Step 3 requires
both grids exported and confirmed by eye against the NEON tiles BEFORE anything
is computed. run_stage1_3_define_planet_grid.py proves the arithmetic is
self-consistent - uniform spacing, integer N, whole-metre edges. It cannot prove
the grid sits where the imagery sits. A CRS mislabelled in the LSP product, a
half-pixel centre/edge convention error, or a coregistration offset between the
LSP product and the AOP flight would all pass every numeric check and still put
every N x N block on the wrong ground. Only looking at it catches that.

WHAT TO LOOK FOR, in order, at the corner window of any tile:

    1. The 3 m planet cells and the 1 m analysis cells must nest exactly - three
       analysis cells across each planet cell, edges coincident, no sliver
       anywhere along the shared boundary. A visible sliver means the origin or
       the centre/edge half-pixel shift is wrong.
    2. The tile boundary will CUT THROUGH planet cells on most tiles. That is
       expected and measured, not a defect: at SRER no tile is congruent in both
       axes (offsets of 0, 1 or 2 m). It is the reason Step 3 must mosaic before
       blocking rather than aggregating each tile in isolation.
    3. The cell edges must line up with real edges in the 10 cm RGB underneath -
       a road margin, a canopy edge, a fence line. If the grid is systematically
       shifted against the imagery, that is a coregistration problem between the
       LSP product and the AOP flight, and it invalidates Step 3 entirely.
    4. `tile_footprints_cropped` must cover the part of each tile that carries
       usable ground truth. Where it falls short of `tile_footprints`, that
       ground truth has no Planet pixel and is dropped from Steps 3-6.
    5. THE DECISIVE CHECK - tick the exported LSP layer on. Its pixel blocks must
       coincide exactly with the 3 m outlines, and toggling it against the RGB
       must show dark EVIamp over woody canopy and bright EVIamp over bare
       ground. Checks 1-4 compare grid outlines against imagery and against each
       other; only this one compares the grid against the Planet data it is
       built to address. A whole- or half-pixel shift here invalidates every
       block in Step 3 and passes checks 1-4 undetected.

The project is read-only and regenerated from scratch each run - every layer is
a view onto the GeoPackage written by Step 0c, nothing here is editable, and
there is no unsaved-edit hazard unlike the labeling project.

LAYER TREE, top of the panel to bottom. QgsLayerTreeGroup appends, so the fine
grids sit above the coarse ones and RGB is the basemap underneath everything:

    analysis cells 1 m          hairline, hollow
    planet cells 3 m            bold, hollow - the grid being verified
    verification windows        dashed, marks where the cell grids are drawn
    tile footprints cropped     what actually enters Steps 3-6
    tile footprints             the full 1 km NEON tiles
    planet footprint            the LSP extent - the SRER focus area
    PLANET EVIamp               the real LSP data on the measured grid, OFF by
                                default - toggle it against the RGB
    RGB                         10 cm imagery, all tiles

Every grid layer is hollow. A filled cell would hide the imagery, and the
imagery is the thing being checked against.
"""

import json
import os

from qgis.core import (
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsMultiBandColorRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor

# --------------------------------------------------------------------- CONFIG
CONFIG = os.path.expanduser(
    "~/Documents/GitHub/PLSP/code/code_ground_truth_land_cover/v4/config/srer_2022.json"
)
# -----------------------------------------------------------------------------

ANALYSIS_COLOR = "#ffffff"
PLANET_COLOR = "#e6007e"
WINDOW_COLOR = "#00e5ff"
TILE_COLOR = "#ffd400"
CROPPED_COLOR = "#00c853"
FOOTPRINT_COLOR = "#ff6d00"


def expand(root, *parts):
    return os.path.join(os.path.expanduser(str(root)), *parts)


def hollow(color, width, style="solid"):
    """A no-fill outline symbol, so the imagery underneath stays readable."""
    return QgsFillSymbol.createSimple(
        {
            "color": "transparent",
            "style": "no",
            "outline_color": color,
            "outline_width": width,
            "outline_style": style,
        }
    )


def add_vector(
    project, root, gpkg, layer_name, title, color, width, style="solid", checked=True
):
    """Register one GeoPackage layer with a hollow outline renderer."""
    layer = QgsVectorLayer(f"{gpkg}|layername={layer_name}", title, "ogr")
    if not layer.isValid():
        print(
            f"MISSING layer {layer_name} in {gpkg} - run run_stage1_3_define_planet_grid.py first"
        )
        return None
    layer.setRenderer(QgsSingleSymbolRenderer(hollow(color, width, style)))
    project.addMapLayer(layer, False)
    node = root.addLayer(layer)
    node.setItemVisibilityChecked(checked)
    return layer


def add_lsp_layer(project, root, path, title):
    """The exported LSP layer, stretched 2-98% over a yellow-to-green ramp.

    THIS IS THE LAYER THAT CLOSES THE LOOP. The cell outlines show where the grid
    claims Planet pixels are; this shows where the Planet data actually is. Its
    blockiness IS the Planet pixel - if the 3 m outlines do not sit exactly on
    the blocks of this raster, the grid is wrong, and no amount of comparing
    outlines against each other would have revealed it.

    Loaded UNCHECKED. It is opaque and would hide the 10 cm imagery; the intended
    use is to toggle it on and off against the RGB, which is how a shift shows up.
    """
    layer = QgsRasterLayer(path, title)
    if not layer.isValid():
        print(
            f"MISSING {path} - re-run run_stage1_3_define_planet_grid.py to export it"
        )
        return None
    stats = layer.dataProvider().bandStatistics(1)
    low, high = stats.minimumValue, stats.maximumValue
    ramp = QgsColorRampShader(low, high)
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(low, QColor("#f7f7c8"), f"{low:.3f}"),
            QgsColorRampShader.ColorRampItem(
                (low + high) / 2.0, QColor("#7cb342"), f"{(low + high) / 2.0:.3f}"
            ),
            QgsColorRampShader.ColorRampItem(high, QColor("#1b5e20"), f"{high:.3f}"),
        ]
    )
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))
    project.addMapLayer(layer, False)
    node = root.addLayer(layer)
    node.setItemVisibilityChecked(False)
    return layer


def add_rgb_group(project, root, cfg, data_dir):
    """The 10 cm RGB basemap, one layer per tile, in a collapsed group.

    Check 3 above is the whole reason this is loaded: grid edges are compared
    against real edges in the imagery, not against other grid lines.
    """
    group = root.addGroup("RGB 10 cm basemap")
    pattern = cfg["products"]["rgb"]
    loaded = 0
    for tile in cfg["tiles"]:
        path = expand(data_dir, pattern["folder"], pattern["pattern"].format(tile=tile))
        if not os.path.exists(path):
            continue
        layer = QgsRasterLayer(path, f"RGB {tile}")
        if not layer.isValid():
            continue
        layer.setRenderer(QgsMultiBandColorRenderer(layer.dataProvider(), 1, 2, 3))
        project.addMapLayer(layer, False)
        group.addLayer(layer)
        loaded += 1
    group.setExpanded(False)
    return loaded


def first_corner_window(cfg):
    """South-west corner window of the first configured tile, for the zoom hint."""
    tile = next(iter(cfg["tiles"]))
    easting, northing = (int(v) for v in tile.split("_"))
    half = float(cfg["stage1_3_planet_grid"]["verification_window_m"]) / 2.0
    return tile, easting - half, northing - half, easting + half, northing + half


def main():
    cfg = json.load(open(CONFIG))
    site, year = cfg["site"], cfg["year"]
    qa_dir = expand(cfg["results_root"], "stage1_data_and_features", "qa")
    gpkg = os.path.join(qa_dir, f"planet_grid_{site}_{year}.gpkg")
    if not os.path.exists(gpkg):
        raise SystemExit(
            f"MISSING {gpkg} - run run_stage1_3_define_planet_grid.py first"
        )

    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(cfg["expected_crs"]))
    # ABSOLUTE layer paths, deliberately. QGIS defaults to paths relative to the
    # .qgz, which silently break the moment the project file changes directory
    # depth - a rename of a results directory is enough, and the failure is
    # invisible until the project is opened. Absolute paths cost portability
    # between machines; these projects are regenerated per machine anyway.
    project.writeEntry("Paths", "/Absolute", True)
    root = project.layerTreeRoot()

    add_vector(
        project,
        root,
        gpkg,
        "analysis_cells_verification",
        "analysis cells 1 m",
        ANALYSIS_COLOR,
        "0.08",
    )
    add_vector(
        project,
        root,
        gpkg,
        "planet_cells_verification",
        "planet cells 3 m",
        PLANET_COLOR,
        "0.4",
    )
    add_vector(
        project,
        root,
        gpkg,
        "verification_windows",
        "verification windows",
        WINDOW_COLOR,
        "0.6",
        style="dash",
    )
    add_vector(
        project,
        root,
        gpkg,
        "tile_footprints_cropped",
        "tile footprints cropped to LSP",
        CROPPED_COLOR,
        "0.8",
    )
    add_vector(
        project,
        root,
        gpkg,
        "tile_footprints",
        "tile footprints full 1 km",
        TILE_COLOR,
        "0.5",
    )
    add_vector(
        project,
        root,
        gpkg,
        "planet_footprint",
        "planet footprint LSP extent",
        FOOTPRINT_COLOR,
        "1.0",
    )

    variable = cfg["stage1_3_planet_grid"].get("verification_variable", "EVIamp")
    lsp_year = cfg["stage1_3_planet_grid"]["grid_source_year"]
    lsp_path = os.path.join(qa_dir, f"planet_{variable}_{site}_{lsp_year}.tif")
    lsp_layer = add_lsp_layer(project, root, lsp_path, f"PLSP {variable} {lsp_year}")

    data_dir = expand(cfg["data_root"], cfg["site_name"])
    rgb_count = add_rgb_group(project, root, cfg, data_dir)

    out = os.path.join(qa_dir, f"grid_verification_{site}_{year}.qgz")
    project.write(out)

    tile, x_min, y_min, x_max, y_max = first_corner_window(cfg)
    grid = cfg["stage1_3_planet_grid"]["measured"]
    print(f"loaded {len(project.mapLayers())} layers ({rgb_count} RGB tiles)")
    print(f"saved project: {out}")
    print("")
    print(
        f"planet pixel {grid['planet_pixel_m']} m, N = {grid['N']}, origin {grid['origin_x']}, {grid['origin_y']}"
    )
    print("")
    print(f"ZOOM HERE FIRST - south-west corner window of {tile}:")
    print(
        f"{x_min:.0f},{y_min:.0f} : {x_max:.0f},{y_max:.0f} (paste into the coordinate box, or use Zoom to Layer on verification windows)"
    )
    print("")
    print(
        "Confirm, in order: (1) three 1 m cells span each 3 m cell with no sliver; (2) the tile boundary cutting through planet cells is expected - no SRER tile is congruent in both axes; (3) cell edges line up with real edges in the RGB, not systematically offset from them; (4) tile footprints cropped covers the ground truth that will enter Steps 3-6."
    )
    if lsp_layer:
        print("")
        print(
            f"(5) THE DECISIVE ONE - tick '{variable}' on and confirm its pixel blocks coincide EXACTLY with the 3 m outlines, then toggle it against the RGB and confirm dark {variable} sits over woody canopy and bright {variable} over bare ground. The outlines only say where the grid claims Planet pixels are; this layer is where the Planet data actually is. A half-pixel or whole-pixel shift between them invalidates every block in Step 3 and is invisible in checks 1-4."
        )
    print("")
    print("Only after this passes may Step 3 aggregation run.")


main()
