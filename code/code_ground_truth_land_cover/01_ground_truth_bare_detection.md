# 01 Ground Truth — Bare Detection

## Purpose

Phase 2 of the ground truth land cover pipeline. Identifies bare-soil pixels at 1m resolution across NEON SRER 2022 tiles by thresholding the pre-computed SAVI vegetation index (NEON product DP3.30026.002). Uses a site-wide threshold determined from a pooled multi-tile histogram plus visual overlay verification against 10cm RGB (DP3.30010.001). Produces per-tile bare mask and confidence rasters that feed forward into shadow reassignment (Phase 3), tree detection (Phase 4), and final land cover assembly (Phase 6).

Transferable design: SAVI is derivable from NAIP RGB+NIR at non-NEON AmeriFlux sites, so the same thresholding logic applies downstream once the SAVI raster is provided.

## Workflow

1. Pool R, G, B, NDVI, SAVI values across the configured tile list at 1m resolution.
2. Compute candidate SAVI thresholds (Otsu, 15th percentile, 25th percentile) from pooled SAVI.
3. Plot pooled histograms (RGB overlaid, NDVI, SAVI with candidate thresholds).
4. For each tile, render a 1x3 overlay of the 10cm RGB with the bare mask at each candidate threshold. Used for visual selection.
5. Commit the final threshold (`SAVI_BARE_THRESHOLD`) to the notebook.
6. For each tile, apply the threshold and write a bare mask GeoTIFF and a bare confidence GeoTIFF preserving CRS and transform from the source SAVI raster.

## Tunable parameters

| Parameter | Default | Description |
|---|---|---|
| `OUT_SHAPE_1M` | `(1000, 1000)` | Target grid shape for 1m rasters (VI and CHM native). |
| `RGB_SHAPE_10CM` | `(10000, 10000)` | Native RGB shape at 10cm. Used when reading RGB for overlays. |
| `UPSAMPLE_FACTOR` | `10` | Nearest-neighbor factor for upsampling 1m bare masks to 10cm to overlay on RGB. |
| `OVERLAY_ALPHA` | `0.45` | Transparency of the bare mask overlay on RGB (0 = invisible, 1 = opaque). |
| `OVERLAY_COLOR` | `(1.0, 0.0, 0.0)` | RGB color of the bare mask overlay. Red by default. |
| Otsu candidate | computed | Otsu's method on pooled SAVI. Automated but sensitive to distribution skew. |
| p15 candidate | computed | 15th percentile of pooled SAVI. Aggressive bare threshold. |
| p25 candidate | computed | 25th percentile of pooled SAVI. Conservative bare threshold. |
| `SAVI_BARE_THRESHOLD` | `0.2` | Final SAVI cutoff. Set manually after inspecting histograms and overlays. Pixels with SAVI < threshold are labeled bare. |
| `ground.TILE_IDS` | from helpers | Tile list used for pooling and per-tile processing. Site-wide threshold assumed representative when all target tiles are pooled together. |

## Outputs

Written to `ground.OUTPUT_DIR = SAVE_DIR / SITE_ID`.

### Diagnostic outputs (one per notebook run)

- **`pooled_histograms_{YEAR}.png`**
  3-row figure: pooled RGB overlaid histograms (DN 0-255), pooled NDVI histogram, pooled SAVI histogram with Otsu/p15/p25 candidate thresholds shown as vertical lines. Used to inspect distribution shape and pick a reasonable threshold interactively.

- **`bare_overlay_{tile_id}_{YEAR}.png`** (one per tile, generated during tuning only)
  1x3 panel: 10cm RGB with the bare mask overlaid in translucent red, one panel per candidate threshold (Otsu, p15, p25). Titles include threshold value and bare fraction for that tile. Used to visually judge which threshold best separates bare from vegetated on the ground.

### Product outputs (per tile, final)

- **`bare_mask_{tile_id}_{YEAR}.tif`** — uint8, 1m
  Binary bare mask. 1 = bare (SAVI < `SAVI_BARE_THRESHOLD`), 0 = not bare, 255 = nodata. CRS, transform, and grid inherited from the source SAVI raster. Consumed by Phase 3 (shadow-bare overlap flagging), Phase 5 (excludes bare pixels from shrub/grass classification), and Phase 6 (assembles into the final categorical land cover raster with priority tree > shrub > grass > bare).

- **`bare_confidence_{tile_id}_{YEAR}.tif`** — float32, 1m
  Per-pixel bare confidence, `clip((threshold - SAVI) / threshold, 0, 1)`. Larger values = SAVI further below threshold = more confident bare. 0 for non-bare valid pixels. NaN for nodata. Overwritten to 0 at shadow-bare overlap pixels by Phase 3. Feeds Phase 6 confidence stratification and any downstream weighted analyses.

## Assessment for parameter tuning

- **Pooled SAVI histogram bimodality.** In arid open canopy, expect a weakly bimodal or single-mode-with-heavy-shoulder distribution. If bimodality is absent, Otsu will misplace the threshold and manual selection based on overlays is required.
- **Bare fraction sanity check.** SRER is estimated 40-60% bare/sparse ground in most locations. A threshold producing <10% or >75% pooled bare fraction is suspect.
- **Overlay agreement across tiles.** If the same threshold produces visually accurate bare masks on all 3 tiles, it's safe to fix site-wide. If one tile requires a noticeably different threshold, this points to either radiometric drift within the flight, ecological gradient, or terrain effects — flag for TPI stratification in a later revision.
- **Bare confidence distribution.** After running the mask write step, inspect the bare_confidence histogram. A distribution concentrated at high confidence (>0.6) means bare pixels are well below the threshold. A distribution concentrated near 0 means bare pixels are barely below threshold and the classification is fragile — indicates the threshold sits inside a busy part of the SAVI histogram.

## Dependencies

- `00_ground_truth_helpers` — provides `TILE_IDS`, `YEAR`, `SITE_ID`, `SITE_NAME`, `OUTPUT_DIR`, `SAVE_DIR`, `SITE_DATA_PATH`, `build_paths(tile_id)`.
- Python: `numpy`, `rasterio`, `matplotlib`, `scikit-image` (for `threshold_otsu`).