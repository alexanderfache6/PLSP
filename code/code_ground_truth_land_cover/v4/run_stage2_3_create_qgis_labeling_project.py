#!/usr/bin/env python3
"""Load every Step 4.2 labeling layer into QGIS, styled and ready to edit.

Run inside QGIS: Plugins -> Python Console -> Show Editor -> Open Script -> Run,
or paste the file in. It builds the layer tree, applies the LOCKED section 3
class colours, turns class_code into a dropdown, and saves a .qgz next to the
labeling outputs so the project can simply be reopened afterwards.

Written as a QGIS script rather than a hand-authored .qgs because the project
XML schema is version-specific and silently degrades when it does not match the
running QGIS; the API does not.

    CONFIG below is the only thing to edit.

Layer tree, one group per tile, top of the panel to bottom. QgsLayerTreeGroup
appends, so layers are added in exactly this order and RGB lands underneath
everything rather than covering it:

    training_polygons   draw labelled polygons here (editable)
    labeling_zones      candidate sites - fill class_code here (editable)
    cluster_map         k=16 zones, provenance only - never a label
    shrub review        CHM-derived shrub candidates to accept or reject
    CHM                 canopy height, for the shrub/tree call
    SAVI                greenness, for the bare/grass call
    RGB                 10 cm imagery - the basemap you actually label from
"""

import json
import os

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsDefaultValue,
    QgsEditorWidgetSetup,
    QgsFillSymbol,
    QgsMarkerSymbol,
    QgsMultiBandColorRenderer,
    QgsPalettedRasterRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor

# --------------------------------------------------------------------- CONFIG
CONFIG = os.path.expanduser(
    "~/Documents/GitHub/PLSP/code/code_ground_truth_land_cover/v4/config/srer_2022.json"
)
# -----------------------------------------------------------------------------

# LOCKED, section 3 - do not renumber or recolour
CLASS_COLORS = {0: "#c2b280", 1: "#7cb342", 2: "#8d6e63", 3: "#1b5e20"}
CLASS_LABELS = {0: "bare", 1: "grass", 2: "shrub", 3: "tree"}
UNLABELLED_COLOR = "#e6007e"  # deliberately outside the earth/green family

CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
]


def expand(root, *parts):
    return os.path.join(os.path.expanduser(str(root)), *parts)


def marker(color, size, outline, width):
    return QgsMarkerSymbol.createSimple(
        {
            "name": "circle",
            "size": size,
            "color": color,
            "outline_color": outline,
            "outline_width": width,
        }
    )


def fill(color, outline, width, style):
    return QgsFillSymbol.createSimple(
        {
            "color": color,
            "outline_color": outline,
            "outline_width": width,
            "style": style,
        }
    )


def style_class_field(layer, geom):
    """Categorized renderer on class_code + a value-map dropdown for editing."""
    categories = []
    for code, colour in CLASS_COLORS.items():
        symbol = (
            marker(colour, "3", "black", "0.3")
            if geom == "point"
            else fill(colour, "black", "0.4", "solid")
        )
        categories.append(
            QgsRendererCategory(code, symbol, f"{code} {CLASS_LABELS[code]}")
        )
    blank = (
        marker(UNLABELLED_COLOR, "2.4", "white", "0.4")
        if geom == "point"
        else fill("transparent", UNLABELLED_COLOR, "0.5", "no")
    )
    categories.append(QgsRendererCategory(None, blank, "unlabelled"))
    layer.setRenderer(QgsCategorizedSymbolRenderer("class_code", categories))

    idx = layer.fields().indexOf("class_code")
    if idx >= 0:
        value_map = {"map": [{CLASS_LABELS[c]: str(c)} for c in sorted(CLASS_LABELS)]}
        layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", value_map))


def style_review_state(layer, geom):
    """Categorized renderer on reviewed, so the worklist is readable at a glance.

    Pending candidates render magenta, matching the unlabelled convention used
    by the zone and polygon layers, and recolour once reviewed. Without this a
    zero-click accept is indistinguishable from a candidate never looked at,
    which is unworkable at 150 per tile.

    Inputs:  layer - a QgsVectorLayer of shrub candidates; geom - "polygon" or
             "point"
    Outputs: None, the layer is styled in place
    """
    shrub = CLASS_COLORS[2]
    states = [(0, UNLABELLED_COLOR, "pending"), (1, shrub, "reviewed")]
    categories = []
    for value, colour, label in states:
        if geom == "point":
            symbol = marker(colour, "2.6", "white", "0.4")
        else:
            symbol = (
                fill("transparent", colour, "0.6", "no")
                if value == 0
                else fill(colour, "black", "0.4", "solid")
            )
        categories.append(QgsRendererCategory(value, symbol, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer("reviewed", categories))

    for field, options in (
        ("reviewed", {"0": "0 pending", "1": "1 reviewed"}),
        ("rejected", {"0": "0 keep", "1": "1 reject"}),
    ):
        idx = layer.fields().indexOf(field)
        if idx >= 0:
            layer.setEditorWidgetSetup(
                idx,
                QgsEditorWidgetSetup(
                    "ValueMap",
                    {"map": [{text: value} for value, text in options.items()]},
                ),
            )


def pseudocolor(layer, vmin, vmax, ramp):
    """Interpolated single-band ramp between vmin and vmax."""
    shader = QgsRasterShader()
    fcn = QgsColorRampShader()
    fcn.setColorRampType(QgsColorRampShader.Interpolated)
    items = []
    for i, colour in enumerate(ramp):
        value = vmin + (vmax - vmin) * i / (len(ramp) - 1)
        items.append(
            QgsColorRampShader.ColorRampItem(value, QColor(colour), f"{value:.2f}")
        )
    fcn.setColorRampItemList(items)
    shader.setRasterShaderFunction(fcn)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def set_derived_defaults(layer, tile, cluster_layer):
    """Auto-fill tile and cluster_id at draw time - both are recoverable from geometry.

    Making the analyst type them is wasted effort and a source of typos. The
    raster layer is referenced by id rather than name because the display name
    contains spaces. point_on_surface is used instead of centroid: a centroid can
    fall outside a concave polygon and would sample the wrong cluster.

    Both fields are set read-only in the form and surfaced in a map tip, so the
    derived values can be checked while drawing but cannot be typed over.

    run_stage2_4_check_hand_labeling_progress.py --fill remains the backstop, and recomputes
    both for any polygon drawn before this was in place.
    """
    fields = layer.fields()
    idx_tile = fields.indexOf("tile")
    if idx_tile >= 0:
        layer.setDefaultValueDefinition(idx_tile, QgsDefaultValue(f"'{tile}'", False))
    idx_cluster = fields.indexOf("cluster_id")
    if idx_cluster >= 0 and cluster_layer is not None and cluster_layer.isValid():
        expr = f"raster_value('{cluster_layer.id()}', 1, point_on_surface($geometry))"
        layer.setDefaultValueDefinition(idx_cluster, QgsDefaultValue(expr, True))

    # both are derived, so show them but do not let them be typed over
    form = layer.editFormConfig()
    for idx in (idx_tile, idx_cluster):
        if idx >= 0:
            form.setReadOnly(idx, True)
    layer.setEditFormConfig(form)

    # hover to check the derived values without opening the form
    layer.setMapTipTemplate(
        "<b>class</b> [% coalesce(class_code, 'UNLABELLED') %]<br><b>cluster</b> [% coalesce(cluster_id, '-') %]<br><b>tile</b> [% coalesce(tile, '-') %]<br><b>area</b> [% round($area, 1) %] m2"
    )
    layer.setDisplayExpression("coalesce(cluster_id, '-')")


def build_tile_group(project, root, cfg, tile, role, data, lab_dir, site, year):
    """Add one tile's six layers, in panel order, and return the editable ones."""
    prod = cfg["products"]
    group = root.addGroup(f"{role.upper()}  {tile}")

    poly_path = os.path.join(lab_dir, f"training_polygons_{site}_{tile}_{year}.gpkg")
    poly = QgsVectorLayer(
        f"{poly_path}|layername=training_polygons_{site}_{tile}_{year}",
        f"POLYGONS  {tile}  <- draw here",
        "ogr",
    )
    if poly.isValid():
        style_class_field(poly, "polygon")

    zone_path = os.path.join(lab_dir, f"labeling_zones_{site}_{tile}_{year}.gpkg")
    zones = QgsVectorLayer(
        f"{zone_path}|layername=labeling_zones_{site}_{tile}_{year}",
        f"zones  {tile}  ({role})",
        "ogr",
    )
    if zones.isValid():
        style_class_field(zones, "point")

    # section 4.2 stage 1b: CHM-derived shrub candidates, stratified review subset.
    # the full shrub_candidates_* set is provenance and is deliberately not loaded -
    # it runs to ~14k features per tile and would make the project unusable.
    review_path = os.path.join(lab_dir, f"shrub_review_{site}_{tile}_{year}.gpkg")
    shrub_poly = QgsVectorLayer(
        f"{review_path}|layername=shrub_review_{site}_{tile}_{year}_polygon",
        f"shrub review  {tile}  <- accept/reject",
        "ogr",
    )
    if shrub_poly.isValid():
        style_review_state(shrub_poly, "polygon")

    clu = QgsRasterLayer(
        os.path.join(lab_dir, f"cluster_map_{site}_{tile}_{year}.tif"),
        f"clusters k16  {tile}",
    )
    if clu.isValid():
        classes = [
            QgsPalettedRasterRenderer.Class(
                i + 1, QColor(CLUSTER_COLORS[i]), f"cluster {i + 1}"
            )
            for i in range(cfg["stage2_1_labeling_zones"]["k"])
        ]
        clu.setRenderer(QgsPalettedRasterRenderer(clu.dataProvider(), 1, classes))
        clu.setOpacity(0.75)

    chm = QgsRasterLayer(
        os.path.join(
            data, prod["chm"]["folder"], prod["chm"]["pattern"].format(tile=tile)
        ),
        f"CHM  {tile}",
    )
    if chm.isValid():
        # 0 to H_TREE_MIN to 5 m: the range the shrub/tree cut lives in
        pseudocolor(chm, 0.0, 5.0, ["#ffffff", "#fdbb84", "#7f0000"])
        chm.setOpacity(0.75)

    savi = QgsRasterLayer(
        os.path.join(
            data,
            prod["vi"]["folder"],
            prod["vi"]["pattern"].format(tile=tile, index="SAVI"),
        ),
        f"SAVI  {tile}",
    )
    if savi.isValid():
        pseudocolor(savi, 0.0, 0.6, ["#8c510a", "#f6e8c3", "#01665e"])
        savi.setOpacity(0.75)

    if clu.isValid():
        project.addMapLayer(
            clu, False
        )  # register early so raster_value() can resolve it by id
    if poly.isValid():
        set_derived_defaults(poly, tile, clu)

    rgb = QgsRasterLayer(
        os.path.join(
            data, prod["rgb"]["folder"], prod["rgb"]["pattern"].format(tile=tile)
        ),
        f"RGB 10cm  {tile}",
    )
    if rgb.isValid():
        rgb.setRenderer(QgsMultiBandColorRenderer(rgb.dataProvider(), 1, 2, 3))

    # top of panel -> bottom; RGB last so it sits under everything as a basemap.
    # expanded controls the legend node: the two editable layers show their
    # class_code categories so labelling progress is readable at a glance,
    # while the reference rasters stay collapsed to keep the panel short.
    # (layer, hidden, expanded)
    ordered = [
        (poly, False, True),
        (zones, False, True),
        (shrub_poly, False, True),
        (clu, True, False),
        (chm, True, False),
        (savi, True, False),
        (rgb, False, False),
    ]
    for layer, hidden, expanded in ordered:
        if not layer.isValid():
            print(f"WARNING: invalid layer skipped in {tile}: {layer.name()}")
            continue
        if layer.id() not in project.mapLayers():
            project.addMapLayer(layer, False)
        node = group.addLayer(layer)
        node.setExpanded(expanded)
        if hidden:
            node.setItemVisibilityChecked(False)

    # groups collapsed by default - five tiles x six layers is a long panel,
    # and labelling is done one tile at a time
    group.setExpanded(False)
    return [layer for layer in (poly, zones, shrub_poly) if layer.isValid()]


def main():
    cfg = json.load(open(CONFIG))
    site, year = cfg["site"], cfg["year"]
    results = os.path.expanduser(cfg["results_root"])
    data = expand(cfg["data_root"], cfg["site_name"])
    lab_dir = os.path.join(results, "stage2_labeling")

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

    editable = []
    for tile, role in cfg["tiles"].items():
        editable += build_tile_group(
            project, root, cfg, tile, role, data, lab_dir, site, year
        )

    out = os.path.join(lab_dir, f"labeling_{site}_{year}.qgz")
    project.write(out)
    print(
        f"loaded {len(project.mapLayers())} layers into {len(cfg['tiles'])} tile groups"
    )
    print(f"editable layers: {len(editable)}")
    print(f"saved project: {out}")
    print(
        "\nclass_code is a dropdown (bare/grass/shrub/tree). Toggle editing per layer, fill or draw, then save the layer. Cluster maps are off by default - they are provenance only and must never be mapped to a class."
    )
    print(
        "\nshrub review layers hold the stratified CHM candidate subset (section 4.2 stage 1b). Pending candidates are magenta and recolour to shrub brown once reviewed = 1. Accept: reviewed = 1. Reject: reviewed = 1 and rejected = 1, then clear class_code. The full shrub_candidates_* set is not loaded - it is provenance only and far too large to render."
    )


main()
