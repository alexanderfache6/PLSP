import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


def resolve_config_path(root, *parts):
    """Expand a config path and join sub-paths onto it."""
    return Path(str(root)).expanduser().joinpath(*parts)


def expand_path(root, *parts):
    return os.path.join(os.path.expanduser(str(root)), *parts)


def read_rgb_at_scale(path, scale_m):
    """Read NEON RGB decimated to scale_m, with unflown ground masked to NaN.

    SHARED BY STAGE 1_5 AND STAGE 1_6 ON PURPOSE. Both need RGB at
    TEXTURE_SCALE, and both were reading it with their own copy of this
    function - so when the unflown-ground defect was found in one, the other
    still had it. One implementation, one place to fix.

    NEON RGB DECLARES NO NODATA, AND THAT IS A DEFECT, NOT A CURIOSITY. Ground
    the camera did not fly is written as all-zero, and with no declared nodata
    GDAL returns those zeros as ordinary data. Nothing downstream can tell them
    from genuinely black ground: zero is finite, so the usable-pixel test in
    stage 3 accepts it and the classifier assigns a land-cover class to ground
    that was never photographed.

    Measured at SRER, 517000_3531000: 1.137% of the tile is all-zero RGB while
    CHM nodata is only 0.083% - the lidar flew it, the camera did not. Those
    pixels became k-means cluster 4 in its entirety and bled into clusters 5, 6,
    9 and 12, which made the stage 2 coverage gate demand hand labels over
    unphotographed ground. In stage 1_6 the same zeros entered the POOLED luma
    histogram, dragging the site-wide 20th percentile down and re-cutting shadow
    on every tile. No other tile at SRER is affected.

    A PRODUCT'S FLIGHT COVERAGE MUST BE TESTED ON THAT PRODUCT. Checking CHM
    says nothing about the camera - the instruments have different footprints,
    and the tile-eligibility check in instructions5.md section 2A missed this
    case by testing CHM alone.

    The zero test runs at NATIVE resolution, before decimation: averaging first
    blends black and lit pixels into a dark-but-nonzero fringe that no threshold
    recovers cleanly. The majority rule then matches the shadow aggregation in
    stage 1_6, so the boundary lands in one place rather than eroding or
    dilating the hole.

    Inputs:  path - RGB GeoTIFF; scale_m - target pixel size in metres
    Outputs: (arr float32 [3, h, w] with NaN where unflown, transform, crs, bounds)
    """
    with rasterio.open(path) as ds:
        width = int(round((ds.bounds.right - ds.bounds.left) / scale_m))
        height = int(round((ds.bounds.top - ds.bounds.bottom) / scale_m))
        unflown_native = (ds.read() == 0).all(axis=0).astype("float32")
        arr = ds.read(out_shape=(ds.count, height, width), resampling=Resampling.average, out_dtype="float32",)
        transform = from_bounds(*ds.bounds, width, height)
        unflown = np.empty((height, width), dtype="float32")
        reproject(source=unflown_native, destination=unflown, src_transform=ds.transform, src_crs=ds.crs, dst_transform=transform, dst_crs=ds.crs, resampling=Resampling.average,)
        arr[:, unflown > 0.5] = np.nan
        return arr, transform, ds.crs, ds.bounds
