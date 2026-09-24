"""Stage 4_5 - build the QGIS project in which a site's end members are drawn.

ONE SCRIPT FOR EVERY SITE, NEON AND AMERIFLUX ALIKE. It reads what stage 4_4
fetched and lays out, for one site, everything needed to draw bare, grass, shrub
and tree polygons on imagery, next to the PLSP cells those polygons will be
read from.

POLYGONS ARE NEVER DELETED. The polygon GeoPackage is created only when it does
not exist; if it exists it is opened as it is and never rewritten, truncated or
replaced. The same holds for every derived vector layer here: a layer is added
to its GeoPackage only if absent, and no layer is ever removed. The project can
therefore be rebuilt at any time, for example after 4_4 fetches a new site,
without touching a single drawn polygon.

LAYERS, top to bottom. Anything not yet available is added as an empty
placeholder named with the reason, so every site's project has the same shape.

    end member polygons - the layer to draw in, field class_code 0 bare,
        1 grass, 2 shrub, 3 tree, chosen from a dropdown, plus an optional
        tile field, as in stage 2
    phenocams - phenocam 1 and 2 where recorded, as reference positions; empty
        at SJER, which has neither
    PLSP grid, imagery year - one 3 m cell per feature, ONLY over the
        downloaded imagery, and drawn ONLY when zoomed in closer than 1:2,000
    PLSP QA, imagery year - 1 and 2 accepted, 3 and 4 rejected
    PLSP EVImax, imagery year - in EVI2 units
    CHM, NEON only - every tile 4_4 measured, as low below 0.7 m (bare or
        grass), medium 0.7 m to 2.0 m (shrub), high 2.0 m and above (tree)
    imagery - the three NEON RGB tiles, or the NAIP window

NO RF-A PRODUCT IS LOADED, at SRER or anywhere. The polygons are meant to be an
independent check on RF-A, and a visible model output invites tracing it.

THE PLSP LAYERS ARE FROM THE SITE'S IMAGERY YEAR, the year its NEON or NAIP
imagery was flown closest to 2022: 2022 at SRER, 2023 at SJER. Polygons, PLSP
layers and every file here carry that one year, so a site is labelled against
the phenology of the year its imagery shows. Only RF-B's training is fixed to
SRER 2022.

WHY THE PLSP LAYERS ARE WRITTEN AS GEOTIFFS, NOT READ LIVE FROM THE NETCDF.
Years from 2022 on are in the stage tier, and GDAL misreads it twice with its
default settings. Measured on SRER 2022:

    1. ROWS COME BACK UPSIDE DOWN. The stage file carries no grid_mapping link,
       so GDAL assumes a bottom-up file and flips it. GDAL's top row of EVImax
       reads 2846, 2939, 3033, 3082, which is the netCDF's BOTTOM row; the real
       top row is 1711, 1737, 1719, 1730. GDAL_NETCDF_BOTTOMUP=NO fixes it.
    2. FILL IS REWRITTEN. GDAL enforces the file's valid range and turns all
       183,285 fill pixels of 32767 into -32767, while reporting nodata as
       -32767. HONOUR_VALID_RANGE=NO keeps 32767.

With both options set, GDAL's EVImax matches a direct netCDF read exactly:
checksum 35,376,899,221 and 183,285 fill pixels in both, and QA is identical.
A VRT pointing at the netCDF would be drawn by QGIS with GDAL's defaults, so it
would be upside down, which is why the layers are written out once here.

THE GRID IS THE PRODUCTION GRID. The stage file's x and y coordinate arrays are
compared, value for value, with the site's production file before anything is
written, and the script stops on any difference (results/stage4_results.md
section 9). They are read through GDAL's multidimensional API, which reads the
stage file correctly; the classic API exposes neither its coordinates nor its
GeoTransform attribute. Fill values come from PLSP_Layers.csv, not the file.

EVERY EXPORT CHECKS ITS OWN ORIENTATION. The multidimensional API returns rows in
file order with no flipping, so the first row of each exported layer is compared
with the first row read that way, and the script stops if they differ.

INPUTS

    site config - site_name, data_root, results_root, expected_crs, products,
        phenocam_csv, plsp_layers_csv, year, optional stage4_4_endmember_tiles
    stage 4_4 report - tile_selection_{site_name}.json and, for NEON,
        tile_stats_{site_name}.csv, in the site's endmembers folder
    PLSP netCDF for the imagery year - stage tier first, then production

OUTPUTS, all in {results_root}/stage4_aggregation/run{N}/stage4_6_labeling_progress/{site_name}/

    endmember_polygons_{site_name}_{year}.gpkg - HAND LABELS, created if absent
    endmember_labeling_{site_name}_{year}.qgz - the project, rebuilt each run
    plsp_evimax_{site_name}_{year}.tif, plsp_qa_{site_name}_{year}.tif
    plsp_grid_{site_name}_{year}.gpkg - one layer per imagery tile or window
    reference_points_{site_name}.gpkg - layer phenocams
    chm_mosaic_{site_name}_{year}.vrt - NEON only

ARGUMENTS

    config
        positional, required. The site config JSON.
    --run
        required. The stage 4 run label the endmembers folder sits under,
        e.g. 5.

EXAMPLE COMMANDS

    conda activate LCSC_QGIS
    export PYTHONPATH=$CONDA_PREFIX/share/qgis/python
    export QT_QPA_PLATFORM=offscreen
    export PROJ_DATA=$CONDA_PREFIX/share/proj PROJ_LIB=$PROJ_DATA

    python run_stage4_5_create_qgis_endmember_labeling_project.py config/endmembers/SRER_2022_endmembers.json --run 5

RUNS HEADLESS FROM A SHELL ONLY, bootstrapping its own QgsApplication.
"""

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from constants import CLASS_COLORS, CLASS_LABELS, SEVENTY
from helpers import endmember_directory, endmember_imagery_year, endmember_polygon_path, read_phenocam_positions, read_selected_site_row, resolve_config_path, resolve_script_relative
from osgeo import gdal, ogr, osr
from qgis.core import (
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsEditorWidgetSetup,
    QgsFillSymbol,
    QgsMarkerSymbol,
    QgsMultiBandColorRenderer,
    QgsPalettedRasterRenderer,
    QgsPalLayerSettings,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor

# fail loudly: without this GDAL and OGR return None on error and carry on
gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

DEFAULT_PLANET_DATA_ROOT = "~/Dropbox/planet/data/planet"
DEFAULT_NAIP_ROOT = "~/Dropbox/planet/data/NAIP"
EVIMAX_LAYER_NUMBER = 9
QA_LAYER_NUMBER = 12
ACCEPTED_QA_VALUES = (1, 2)
GRID_VISIBLE_BELOW_SCALE = 2000.0
FLOAT_NODATA = -9999.0
POLYGON_LAYER_NAME = "endmember_polygons"
PHENOCAM_LAYER_NAME = "phenocams"
MAGNITUDE_COLORS = ["#f4f6f7", "#cfdde2", "#8fb3bf", "#4d8496", "#1f5566"]
QA_COLORS = {1: "#2e7d32", 2: "#9ccc65", 3: "#e0a96d", 4: "#b23b3b"}
QA_OPACITY = 0.75  # 25% transparency, so the imagery stays readable underneath


def read_layer_fill_values(config):
    """Fill value and scale for the layers written here, from PLSP_Layers.csv.

    The CSV is the authority. The stage files declare no fill at all, so the
    value cannot be taken from them.

    Inputs: config - needs plsp_layers_csv
    Outputs: {layer number: {short_name, scale, fill_value}}
    """
    csv_path = resolve_script_relative(config["plsp_layers_csv"])
    wanted = {}
    with open(csv_path, newline="") as handle:
        for csv_row in csv.DictReader(handle):
            layer_number = int(csv_row["product_lyr"])
            if layer_number in (EVIMAX_LAYER_NUMBER, QA_LAYER_NUMBER):
                wanted[layer_number] = {"short_name": csv_row["short_name"].strip(), "scale": float(csv_row["scale"]), "fill_value": int(csv_row["fill_value"])}
    return wanted


def planet_data_root(config):
    """The PLSP product root, from the config or the default.

    Inputs: config
    Outputs: str
    """
    return config.get("planet_data_root") or config.get("stage1_3_planet_grid", {}).get("planet_data_root") or DEFAULT_PLANET_DATA_ROOT


def read_production_reference(config):
    """Transform, size and CRS of the site's production PLSP grid, through GDAL.

    GDAL reads production files correctly: they link their variables to a CF
    crs variable. The UTM zone comes from that CRS's central meridian and must
    equal the config's expected_crs.

    Inputs: config
    Outputs: dict with file, geotransform, width, height, epsg, left, right,
             bottom, top, pixel_size
    """
    directory = resolve_config_path(planet_data_root(config), "PLSP_production_nc", config["site_name"])
    candidates = sorted(glob.glob(str(directory / "*PLSP_*.nc")))
    if not candidates:
        raise SystemExit(f"FAIL - no production PLSP file in {directory}")
    dataset = gdal.Open(f'NETCDF:"{candidates[0]}":EVImax')
    transform = dataset.GetGeoTransform()
    spatial_reference = osr.SpatialReference(wkt=dataset.GetProjection())
    central_meridian = spatial_reference.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN)
    epsg = f"EPSG:326{int((central_meridian + 183) / 6):02d}"
    if epsg != config["expected_crs"]:
        raise SystemExit(f"FAIL - config expected_crs is {config['expected_crs']} but the production grid is in {epsg}")
    width, height = dataset.RasterXSize, dataset.RasterYSize
    dataset = None
    x_centres, y_centres = read_coordinate_arrays(candidates[0])
    return {
        "file": candidates[0],
        "x_centres": x_centres,
        "y_centres": y_centres,
        "geotransform": transform,
        "width": width,
        "height": height,
        "epsg": epsg,
        "pixel_size": transform[1],
        "left": transform[0],
        "top": transform[3],
        "right": transform[0] + transform[1] * width,
        "bottom": transform[3] + transform[5] * height,
    }


def read_coordinate_arrays(path):
    """x and y cell-centre arrays of a PLSP file, in file order.

    Through GDAL's multidimensional API, which reads the stage tier correctly
    where the classic API does not.

    Inputs: path - netCDF path
    Outputs: (x float64 array, y float64 array)
    """
    dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    group = dataset.GetRootGroup()
    x_centres = np.asarray(group.OpenMDArray("x").ReadAsArray(), dtype="float64")
    y_centres = np.asarray(group.OpenMDArray("y").ReadAsArray(), dtype="float64")
    group = None
    dataset = None
    return x_centres, y_centres


def read_first_row_in_file_order(path, variable_name, width):
    """The first row of a variable exactly as stored, for the orientation check.

    Inputs: path; variable_name; width - number of columns
    Outputs: 1-D array of raw values
    """
    dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    group = dataset.GetRootGroup()
    first_row = np.asarray(group.OpenMDArray(variable_name).ReadAsArray(array_start_idx=[0, 0], count=[1, width]))[0]
    group = None
    dataset = None
    return first_row


def find_plsp_file(config, year):
    """The PLSP netCDF for a year, stage tier first, then production.

    Inputs: config; year - int
    Outputs: (path str, tier str) or (None, None)
    """
    root = planet_data_root(config)
    stage_path = resolve_config_path(root, "PLSP_stage_nc", config["site_name"], f"PLSP_{year}.nc")
    if stage_path.exists():
        return str(stage_path), "stage"
    production = sorted(glob.glob(str(resolve_config_path(root, "PLSP_production_nc", config["site_name"]) / f"*PLSP_{year}.nc")))
    if production:
        return production[0], "production"
    return None, None


def open_plsp_variable(path, variable_name):
    """Open one PLSP variable with the two settings that make GDAL read it right.

    GDAL_NETCDF_BOTTOMUP=NO stops the stage tier being flipped, and
    HONOUR_VALID_RANGE=NO stops fill being rewritten. See the module notes for
    the measurements behind both.

    Inputs: path; variable_name
    Outputs: gdal.Dataset
    """
    gdal.SetConfigOption("GDAL_NETCDF_BOTTOMUP", "NO")
    try:
        return gdal.OpenEx(f'NETCDF:"{path}":{variable_name}', open_options=["HONOUR_VALID_RANGE=NO"])
    finally:
        gdal.SetConfigOption("GDAL_NETCDF_BOTTOMUP", None)


def check_grid_matches_production(path, tier, reference):
    """Stop unless this file's x and y arrays are the production grid's.

    Compared value for value, not by a recorded transform, because the stage
    tier's transform is not linked to its variables.

    Inputs: path; tier; reference - from read_production_reference
    Outputs: None, raises on any difference
    """
    x_centres, y_centres = read_coordinate_arrays(path)
    if x_centres.shape != reference["x_centres"].shape or y_centres.shape != reference["y_centres"].shape:
        raise SystemExit(f"FAIL - {Path(path).name} ({tier}) grid is {x_centres.size} x {y_centres.size}, production is {reference['x_centres'].size} x {reference['y_centres'].size}")
    if not (np.allclose(x_centres, reference["x_centres"]) and np.allclose(y_centres, reference["y_centres"])):
        raise SystemExit(f"FAIL - {Path(path).name} ({tier}) x/y differ from the production grid")


def write_plsp_geotiff(path, tier, layer, reference, output_path, as_scaled_float, plsp_year):
    """Write one PLSP layer as a GeoTIFF on the production grid, once.

    Reused if it already exists: a year's PLSP values do not change.

    Inputs: path; tier; layer - {short_name, scale, fill_value}; reference;
            output_path; as_scaled_float - True for EVImax, False for QA;
            plsp_year - int
    Outputs: output_path
    """
    if output_path.exists():
        return output_path
    netcdf_name = layer["short_name"]
    check_grid_matches_production(path, tier, reference)
    dataset = open_plsp_variable(path, netcdf_name)
    raw = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    if not np.array_equal(raw[0], read_first_row_in_file_order(path, netcdf_name, reference["width"])):
        raise SystemExit(f"FAIL - {netcdf_name} from {Path(path).name} came back reordered; its first row does not match the file. Nothing written.")
    is_fill = raw == layer["fill_value"]
    if as_scaled_float:
        values = raw.astype("float32") * layer["scale"]
        values[is_fill] = FLOAT_NODATA
        data_type, nodata = gdal.GDT_Float32, FLOAT_NODATA
    else:
        values = raw.astype("int16")
        data_type, nodata = gdal.GDT_Int16, layer["fill_value"]
    target = gdal.GetDriverByName("GTiff").Create(str(output_path), reference["width"], reference["height"], 1, data_type, options=["COMPRESS=DEFLATE", "TILED=YES"])
    target.SetGeoTransform(reference["geotransform"])
    spatial_reference = osr.SpatialReference()
    spatial_reference.SetFromUserInput(reference["epsg"])
    target.SetProjection(spatial_reference.ExportToWkt())
    band = target.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(values)
    band.SetDescription(f"{netcdf_name} {plsp_year}, {tier} tier")
    target = None
    print(f"wrote {output_path.name}: {int(is_fill.sum()):,} fill pixels set to nodata")
    return output_path


def open_or_create_geopackage(path):
    """Open a GeoPackage for update, creating the file only if it is absent.

    Never deletes the file or any layer in it.

    Inputs: path - Path
    Outputs: ogr.DataSource
    """
    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        return driver.Open(str(path), 1)
    return driver.CreateDataSource(str(path))


def spatial_reference_for(epsg):
    """An OGR spatial reference with x/y axis order, for an EPSG string.

    Inputs: epsg - e.g. "EPSG:32612"
    Outputs: osr.SpatialReference
    """
    spatial_reference = osr.SpatialReference()
    spatial_reference.SetFromUserInput(epsg)
    spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_reference


def ensure_polygon_file(path, epsg):
    """Create the end member polygon GeoPackage, ONLY if it does not exist.

    AN EXISTING FILE IS NEVER OPENED FOR WRITING HERE. It holds hand labels.

    Inputs: path; epsg
    Outputs: bool, True if the file was created on this run
    """
    if path.exists():
        return False
    datasource = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = datasource.CreateLayer(POLYGON_LAYER_NAME, spatial_reference_for(epsg), ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("class_code", ogr.OFTInteger64))
    layer.CreateField(ogr.FieldDefn("tile", ogr.OFTString))
    datasource = None
    return True


def ensure_phenocam_layer(path, cameras, epsg):
    """Add the phenocam point layer, only if absent, even when there are none.

    Inputs: path; cameras - from read_phenocam_positions; epsg
    Outputs: bool, True if the layer was created on this run
    """
    datasource = open_or_create_geopackage(path)
    if datasource.GetLayerByName(PHENOCAM_LAYER_NAME) is not None:
        datasource = None
        return False
    target_reference = spatial_reference_for(epsg)
    layer = datasource.CreateLayer(PHENOCAM_LAYER_NAME, target_reference, ogr.wkbPoint)
    for field_name, field_type in (("number", ogr.OFTInteger), ("name", ogr.OFTString), ("latitude", ogr.OFTReal), ("longitude", ogr.OFTReal)):
        layer.CreateField(ogr.FieldDefn(field_name, field_type))
    to_site = osr.CoordinateTransformation(spatial_reference_for("EPSG:4326"), target_reference)
    for camera in cameras:
        easting, northing, _ = to_site.TransformPoint(camera["longitude"], camera["latitude"])
        feature = ogr.Feature(layer.GetLayerDefn())
        for field_name in ("number", "name", "latitude", "longitude"):
            feature.SetField(field_name, camera[field_name])
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(easting, northing)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
    datasource = None
    return True


def ensure_grid_layer(path, layer_name, bounds, reference):
    """Add a PLSP cell layer covering bounds, only if that layer is absent.

    Every cell whose square touches bounds is written, clipped to the site
    grid, with its column and row in the PLSP array so a cell can be named.

    Inputs: path; layer_name; bounds - left, right, bottom, top; reference
    Outputs: number of cells written, 0 if the layer already existed
    """
    datasource = open_or_create_geopackage(path)
    if datasource.GetLayerByName(layer_name) is not None:
        datasource = None
        return 0
    pixel = reference["pixel_size"]
    first_column = max(0, math.floor((bounds["left"] - reference["left"]) / pixel))
    last_column = min(reference["width"] - 1, math.ceil((bounds["right"] - reference["left"]) / pixel) - 1)
    first_row = max(0, math.floor((reference["top"] - bounds["top"]) / pixel))
    last_row = min(reference["height"] - 1, math.ceil((reference["top"] - bounds["bottom"]) / pixel) - 1)
    layer = datasource.CreateLayer(layer_name, spatial_reference_for(reference["epsg"]), ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("plsp_col", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("plsp_row", ogr.OFTInteger))
    definition = layer.GetLayerDefn()
    written = 0
    layer.StartTransaction()
    for row in range(first_row, last_row + 1):
        cell_top = reference["top"] - row * pixel
        for column in range(first_column, last_column + 1):
            cell_left = reference["left"] + column * pixel
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in ((cell_left, cell_top), (cell_left + pixel, cell_top), (cell_left + pixel, cell_top - pixel), (cell_left, cell_top - pixel), (cell_left, cell_top)):
                ring.AddPoint_2D(x, y)
            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)
            feature = ogr.Feature(definition)
            feature.SetField("plsp_col", column)
            feature.SetField("plsp_row", row)
            feature.SetGeometry(polygon)
            layer.CreateFeature(feature)
            written += 1
    layer.CommitTransaction()
    datasource = None
    return written


def product_path(config, product_key, tile_id):
    """Where a NEON product file for one tile sits, from the config layout.

    Mirrors stage 1_1's expected_paths, which cannot be imported here because
    its module needs packages the QGIS environment lacks.

    Inputs: config; product_key - "rgb" or "chm"; tile_id
    Outputs: Path
    """
    product = config["products"][product_key]
    return resolve_config_path(config["data_root"], config["site_name"], product["folder"], product["pattern"].format(tile=tile_id))


def read_stage4_4_outputs(config, directory):
    """The 4_4 selection report and, for NEON, the list of measured tiles.

    Inputs: config; directory - the site's endmembers folder
    Outputs: (report dict or None, list of measured tile ids)
    """
    report_path = directory / f"tile_selection_{config['site_name']}.json"
    if not report_path.exists():
        return None, []
    report = json.loads(report_path.read_text())
    measured = []
    table_path = directory / f"tile_stats_{config['site_name']}.csv"
    if table_path.exists():
        with open(table_path, newline="") as handle:
            measured = [csv_row["tile_id"] for csv_row in csv.DictReader(handle)]
    return report, measured


def placeholder_layer(name, reason):
    """An empty layer standing in for data that is not there yet.

    Inputs: name; reason
    Outputs: QgsVectorLayer with no geometry and no features
    """
    return QgsVectorLayer("None", f"{name} - NOT AVAILABLE: {reason}", "memory")


def style_chm_bands(layer):
    """Low, medium and high height classes at the section 3 thresholds.

    Discrete ramps colour a value by the first break at or above it, so the
    breaks sit just below 0.7 m and 2.0 m to keep a pixel of exactly 0.7 m in
    shrub and one of exactly 2.0 m in tree, as the height rule defines them.

    Inputs: layer - CHM QgsRasterLayer
    Outputs: None
    """
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Discrete)
    ramp.setColorRampItemList(
        [
            QgsColorRampShader.ColorRampItem(float(np.nextafter(0.7, 0)), QColor(CLASS_COLORS[0]), "low, below 0.7 m - bare or grass"),
            QgsColorRampShader.ColorRampItem(float(np.nextafter(2.0, 0)), QColor(CLASS_COLORS[2]), "medium, 0.7 to 2.0 m - shrub"),
            QgsColorRampShader.ColorRampItem(float("inf"), QColor(CLASS_COLORS[3]), "high, 2.0 m and above - tree"),
        ]
    )
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def style_evimax(layer, ramp_minimum, ramp_maximum):
    """Single-hue magnitude ramp in EVI2 units.

    Inputs: layer; ramp_minimum, ramp_maximum - floats
    Outputs: None
    """
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    steps = len(MAGNITUDE_COLORS) - 1
    ramp.setColorRampItemList([QgsColorRampShader.ColorRampItem(ramp_minimum + (ramp_maximum - ramp_minimum) * step / steps, QColor(MAGNITUDE_COLORS[step]), f"{ramp_minimum + (ramp_maximum - ramp_minimum) * step / steps:.2f}") for step in range(len(MAGNITUDE_COLORS))])
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def style_qa(layer):
    """QA as categories, accepted against rejected, at 25% transparency.

    The imagery underneath has to stay readable while drawing, since a polygon
    is placed on what the ground looks like, not on where QA happens to be
    good. 25% transparency is QGIS's own wording for 75% opacity.

    Inputs: layer
    Outputs: None
    """
    classes = [QgsPalettedRasterRenderer.Class(value, QColor(colour), f"{value} {'accepted' if value in ACCEPTED_QA_VALUES else 'rejected'}") for value, colour in QA_COLORS.items()]
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
    layer.renderer().setOpacity(QA_OPACITY)


def style_polygons(layer):
    """One colour per class, half transparent so the imagery shows through, plus a class_code dropdown.

    The dropdown is a value map, as in style_class_field in stage 2_3: the form
    lists the class names and stores the class code, so a polygon can only be
    given 0, 1, 2 or 3. It is stored in the project, not the GeoPackage, so
    rebuilding the project restores it without touching a polygon.

    Inputs: layer
    Outputs: None
    """
    categories = []
    for class_code, class_name in CLASS_LABELS.items():
        symbol = QgsFillSymbol.createSimple({"color": QColor(CLASS_COLORS[class_code]).name(), "outline_color": "#000000", "outline_width": "0.3"})
        symbol.setOpacity(0.5)
        categories.append(QgsRendererCategory(class_code, symbol, f"{class_code} {class_name}"))
    layer.setRenderer(QgsCategorizedSymbolRenderer("class_code", categories))

    field_index = layer.fields().indexOf("class_code")
    if field_index >= 0:
        value_map = {"map": [{CLASS_LABELS[class_code]: str(class_code)} for class_code in sorted(CLASS_LABELS)]}
        layer.setEditorWidgetSetup(field_index, QgsEditorWidgetSetup("ValueMap", value_map))


def style_grid(layer):
    """Outline-only cells, drawn only when zoomed in.

    Inputs: layer
    Outputs: None
    """
    layer.renderer().setSymbol(QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "#ffd400", "outline_width": "0.15"}))
    layer.setScaleBasedVisibility(True)
    layer.setMinimumScale(GRID_VISIBLE_BELOW_SCALE)
    layer.setMaximumScale(0.0)


def style_phenocams(layer):
    """A visible marker with the camera name beside it.

    Inputs: layer
    Outputs: None
    """
    layer.renderer().setSymbol(QgsMarkerSymbol.createSimple({"name": "star", "color": "#ff1744", "outline_color": "#ffffff", "size": "4"}))
    labels = QgsPalLayerSettings()
    labels.fieldName = "name"
    layer.setLabeling(QgsVectorLayerSimpleLabeling(labels))
    layer.setLabelsEnabled(True)


def add_layer(project, group, layer, visible):
    """Add a layer to a group with its legend collapsed.

    Inputs: project; group; layer; visible
    Outputs: 1 if added, 0 if the layer was invalid
    """
    if not layer.isValid():
        print(f"INVALID layer skipped: {layer.name()}")
        return 0
    project.addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setItemVisibilityChecked(visible)
    node.setExpanded(False)
    return 1


def raster_percentiles(path, low, high):
    """Two percentiles of a raster's valid values, for a display stretch.

    Inputs: path; low, high - percentiles 0 to 100
    Outputs: (float, float)
    """
    dataset = gdal.Open(str(path))
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    dataset = None
    valid = values[values != nodata] if nodata is not None else values.ravel()
    return float(np.percentile(valid, low)), float(np.percentile(valid, high))


def main():
    parser = argparse.ArgumentParser(description="Build the QGIS project for drawing one site's end member polygons.")
    parser.add_argument("config", help="site config JSON")
    parser.add_argument("--run", required=True, help="stage 4 run label the endmembers folder sits under, e.g. 5")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    site_name = config["site_name"]
    imagery_year = endmember_imagery_year(config)
    plsp_year = imagery_year
    directory = endmember_directory(config, args.run)
    directory.mkdir(parents=True, exist_ok=True)
    site_row = read_selected_site_row(resolve_script_relative(config["phenocam_csv"]), site_name)
    is_neon = site_row.get("is_neon", "").strip() in ("1", "1.0", "True", "true")
    report, measured_tiles = read_stage4_4_outputs(config, directory)
    if report is not None and int(report["imagery_year"]) != imagery_year:
        raise SystemExit(f"FAIL - the 4_4 report used imagery year {report['imagery_year']} but the config now says {imagery_year}")
    reference = read_production_reference(config)
    layer_specs = read_layer_fill_values(config)

    print(f"Stage 4_5 - end member labelling project - {site_name} ({site_row.get('site_id', '')}, {site_row.get('neon_domain', '')})")
    print("=" * SEVENTY)
    print(f"site type {'NEON' if is_neon else 'AmeriFlux'}, imagery {imagery_year}, PLSP layers {plsp_year}, grid {reference['epsg']}")
    print(f"4_4 report {'found' if report else 'NOT FOUND, imagery and grid will be placeholders'}")

    polygon_path = endmember_polygon_path(config, args.run)
    created = ensure_polygon_file(polygon_path, reference["epsg"])
    print(f"polygons {polygon_path.name}: {'created empty' if created else 'exists, left untouched'}")

    cameras = read_phenocam_positions(site_row)
    reference_points_path = directory / f"reference_points_{site_name}.gpkg"
    created = ensure_phenocam_layer(reference_points_path, cameras, reference["epsg"])
    print(f"phenocams: {len(cameras)} recorded, layer {'created' if created else 'exists, left untouched'}")

    plsp_path, plsp_tier = find_plsp_file(config, plsp_year)
    evimax_path = directory / f"plsp_evimax_{site_name}_{plsp_year}.tif"
    qa_path = directory / f"plsp_qa_{site_name}_{plsp_year}.tif"
    if plsp_path:
        print(f"PLSP {plsp_year}: {Path(plsp_path).name}, {plsp_tier} tier")
        write_plsp_geotiff(plsp_path, plsp_tier, layer_specs[EVIMAX_LAYER_NUMBER], reference, evimax_path, True, plsp_year)
        write_plsp_geotiff(plsp_path, plsp_tier, layer_specs[QA_LAYER_NUMBER], reference, qa_path, False, plsp_year)
    else:
        print(f"PLSP {plsp_year}: not on disk yet")

    grid_path = directory / f"plsp_grid_{site_name}_{plsp_year}.gpkg"
    imagery_entries = []
    grid_layer_names = []
    if report and is_neon:
        for selected in report["selected"]:
            rgb_path = product_path(config, "rgb", selected["tile_id"])
            label = f"RGB {imagery_year} {selected['tile_id']}, {selected['role']} - {selected['end_member_classes']}"
            imagery_entries.append((label, rgb_path))
            if rgb_path.exists():
                easting, northing = (int(value) for value in selected["tile_id"].split("_"))
                layer_name = f"grid_{selected['tile_id']}"
                written = ensure_grid_layer(grid_path, layer_name, {"left": easting, "right": easting + 1000, "bottom": northing, "top": northing + 1000}, reference)
                print(f"grid {layer_name}: {'created, ' + format(written, ',') + ' cells' if written else 'exists, left untouched'}")
                grid_layer_names.append(layer_name)
    elif report and not is_neon:
        naip_path = Path(report["imagery_path"])
        imagery_entries.append((f"NAIP {imagery_year} window", naip_path))
        if naip_path.exists():
            written = ensure_grid_layer(grid_path, "grid_naip_window", report["window"], reference)
            print(f"grid grid_naip_window: {'created, ' + format(written, ',') + ' cells' if written else 'exists, left untouched'}")
            grid_layer_names.append("grid_naip_window")

    chm_vrt_path = directory / f"chm_mosaic_{site_name}_{imagery_year}.vrt"
    chm_files = [str(product_path(config, "chm", tile_id)) for tile_id in measured_tiles if product_path(config, "chm", tile_id).exists()] if is_neon else []
    if chm_files:
        gdal.BuildVRT(str(chm_vrt_path), chm_files)
        print(f"CHM mosaic: {len(chm_files)} tiles")

    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem(reference["epsg"]))
    project.writeEntry("Paths", "/Absolute", True)
    root = project.layerTreeRoot()
    added = 0

    group = root.addGroup("end member polygons - DRAW HERE")
    polygon_layer = QgsVectorLayer(f"{polygon_path}|layername={POLYGON_LAYER_NAME}", f"end member polygons {imagery_year}", "ogr")
    style_polygons(polygon_layer)
    added += add_layer(project, group, polygon_layer, True)
    group.setExpanded(True)

    group = root.addGroup("reference")
    phenocam_layer = QgsVectorLayer(f"{reference_points_path}|layername={PHENOCAM_LAYER_NAME}", "phenocams" if cameras else "phenocams - none recorded for this site", "ogr")
    style_phenocams(phenocam_layer)
    added += add_layer(project, group, phenocam_layer, True)
    group.setExpanded(False)

    group = root.addGroup(f"PLSP {plsp_year}")
    if grid_layer_names:
        for layer_name in grid_layer_names:
            grid_layer = QgsVectorLayer(f"{grid_path}|layername={layer_name}", f"PLSP grid {layer_name.replace('grid_', '')}, zoom in closer than 1:{GRID_VISIBLE_BELOW_SCALE:,.0f}", "ogr")
            style_grid(grid_layer)
            added += add_layer(project, group, grid_layer, True)
    else:
        added += add_layer(project, group, placeholder_layer("PLSP grid", "no imagery downloaded"), False)
    if plsp_path:
        qa_layer = QgsRasterLayer(str(qa_path), f"PLSP QA {plsp_year}")
        style_qa(qa_layer)
        added += add_layer(project, group, qa_layer, False)
        evimax_layer = QgsRasterLayer(str(evimax_path), f"PLSP EVImax {plsp_year}")
        style_evimax(evimax_layer, *raster_percentiles(evimax_path, 2, 98))
        added += add_layer(project, group, evimax_layer, False)
    else:
        added += add_layer(project, group, placeholder_layer(f"PLSP QA {plsp_year}", "PLSP file not generated"), False)
        added += add_layer(project, group, placeholder_layer(f"PLSP EVImax {plsp_year}", "PLSP file not generated"), False)
    group.setExpanded(False)

    if is_neon:
        group = root.addGroup(f"CHM {imagery_year}")
        if chm_files:
            chm_layer = QgsRasterLayer(str(chm_vrt_path), f"CHM {imagery_year}, {len(chm_files)} tiles")
            style_chm_bands(chm_layer)
            added += add_layer(project, group, chm_layer, False)
        else:
            added += add_layer(project, group, placeholder_layer("CHM", "not downloaded"), False)
        group.setExpanded(False)

    group = root.addGroup(f"imagery {imagery_year}")
    if not imagery_entries:
        added += add_layer(project, group, placeholder_layer("imagery", "stage 4_4 has not run" if report is None else "nothing selected"), False)
    for label, image_path in imagery_entries:
        if image_path.exists():
            image_layer = QgsRasterLayer(str(image_path), label)
            # BANDS 1, 2 AND 3 ARE RED, GREEN AND BLUE for both sources. NEON
            # RGB has only those three; a NAIP window has a fourth, near
            # infrared, which stage 4_4 keeps because it separates green grass
            # from bare soil. It is not rendered here: polygons are drawn on
            # what the ground looks like, in natural colour.
            image_layer.setRenderer(QgsMultiBandColorRenderer(image_layer.dataProvider(), 1, 2, 3))
            added += add_layer(project, group, image_layer, True)
        else:
            added += add_layer(project, group, placeholder_layer(label, "not downloaded"), False)
    group.setExpanded(False)

    project_path = directory / f"endmember_labeling_{site_name}_{imagery_year}.qgz"
    project.write(str(project_path))
    print("\n" + "=" * SEVENTY)
    print(f"{added} layers, saved {project_path}")
    print(f"draw polygons in {polygon_path.name}, class_code 0 bare, 1 grass, 2 shrub, 3 tree, at least 15 per class")


def run():
    """Bootstrap a headless QGIS, build the project, and exit cleanly on failure.

    Without QgsApplication.initQgis() the GDAL and OGR providers are never
    registered, every layer comes back invalid, and the project saves empty
    with no error.

    A FAILURE IS PRINTED BEFORE QGIS SHUTS DOWN. Shutting QGIS down while an
    error still holds GDAL objects segfaults, which exits with code 139 and
    swallows the message. So the message is printed and flushed first, and the
    error and its references are released before exitQgis runs.
    """
    import gc

    QgsApplication.setPrefixPath(sys.prefix, True)
    application = QgsApplication([], False)
    application.initQgis()
    exit_code = 0
    try:
        main()
    except SystemExit as stop:
        if isinstance(stop.code, str):
            print(stop.code, file=sys.stderr, flush=True)
            exit_code = 1
        else:
            exit_code = stop.code or 0
    except Exception as error:
        print(f"FAIL - {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    gc.collect()
    application.exitQgis()
    sys.exit(exit_code)


run()
