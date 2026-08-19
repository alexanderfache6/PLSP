# 02 Ground Truth — Shadow Detection

## Purpose

Phase 3 of the ground truth land cover pipeline. Detects and classifies shadow pixels in the 10cm NEON RGB imagery (DP3.30010.001) to (a) remove shadow bias from the Phase 2 bare mask and (b) provide shadow-tree candidates as an input to the Phase 4 tree union.

The core assumption from the arid-open-canopy setting at SRER: any shadow adjacent to a CHM return above 2m is tree-cast. Non-adjacent shadow is flagged as low-confidence and retained for downstream weighting rather than removed.

Transferable design: the brightness + blue-shift rule uses RGB only and applies unchanged at NAIP-only AmeriFlux sites. CHM adjacency requires a height layer (3DEP LiDAR at AmeriFlux sites where available; skip the reassignment step where not).

## Workflow

1. Pool 10cm simple-mean brightness across the configured tile list; compute the site-wide 20th-percentile brightness threshold.
2. Per tile:
   a. Detect 10cm shadow candidates: `(R+G+B)/3 < threshold` AND `B > R`.
   b. Aggregate to 1m with a >70% majority rule over each 10x10 block.
   c. Reassign shadow pixels within 5m of any CHM > 2m pixel to a tree-shadow mask.
   d. Retain non-adjacent shadow as a low-confidence mask.
   e. Overwrite `bare_confidence` to 0 at pixels that are simultaneously bare (Phase 2) and shadow.

## Tunable parameters

| Parameter | Default | Description |
|---|---|---|
| `BRIGHTNESS_PERCENTILE` | `20` | Percentile of pooled 10cm brightness (DN 0-255) used as the shadow threshold. Lower = fewer shadow pixels detected; higher = more, including dark bare soil false positives. |
| `AGG_FRACTION` | `0.70` | Minimum fraction of 10cm subpixels that must be shadow for the 1m pixel to be marked shadow. Higher = stricter, sharper shadow boundaries; lower = catches shadow edges but adds mixed pixels. |
| `UPSAMPLE_FACTOR` | `10` | 10cm subpixels per 1m pixel per axis. Fixed by data resolution; do not change unless input resolutions change. |
| `CHM_TREE_HEIGHT` | `2.0` (m) | Minimum CHM height for tree footprint used in shadow-to-tree adjacency. Matches the operational tree threshold used elsewhere in the workflow. Lower to catch tall shrub shadows as tree-shadow; raise to require unambiguous canopy trees. |
| `DILATION_METERS` | `5` | Dilation radius (m == 1m pixels) around CHM > 2m for the shadow-to-tree adjacency zone. Larger = more shadow pixels get reassigned to tree; smaller = only pixels immediately adjacent. 5m fits typical D14 mesquite/paloverde shadow lengths at flight sun angles. |

## Outputs

Written to `ground.OUTPUT_DIR = SAVE_DIR / SITE_ID`.

Per tile, all 1m, uint8 GeoTIFFs unless noted (1 = True, 0 = False, 255 = nodata):

- **`shadow_mask_1m_{tile_id}_{YEAR}.tif`**
  Raw 1m shadow mask. All pixels where the brightness + blue-shift rule fires at 10cm and the 70% aggregation majority holds at 1m. This is the full shadow inventory before any interpretation. Used for QA and to derive the two downstream masks.

- **`shadow_tree_mask_{tile_id}_{YEAR}.tif`**
  Shadow pixels that fall within the 5m dilated CHM > 2m zone. Interpreted as tree-cast shadow on the ground under or near a canopy. Feeds directly into Phase 4 as one of the three components of the final tree union (CHM > 2m ∪ shadow-tree ∪ RGB detector > 0.7).

- **`shadow_lowconf_mask_{tile_id}_{YEAR}.tif`**
  Shadow pixels outside the dilated tree zone. Cannot be confidently reassigned to any class. Retained as a flag for downstream shrub/grass classification: pixels here should carry reduced weight during training and reduced confidence in Phase 6 assembly.

- **`bare_confidence_{tile_id}_{YEAR}.tif`** (overwritten in place)
  The Phase 2 bare confidence layer is modified: any pixel that was flagged bare AND is shadow gets confidence set to 0. The bare mask uint8 raster itself is not modified — the confidence layer alone captures the downgrade. Downstream Phase 6 uses the confidence layer, not the raw mask, for assembly and reporting.

## Assessment for parameter tuning

- **10cm shadow fraction (printed).** Rough expectation for D14 mid-day summer flights: 5-15% shadow at 10cm. Much higher (>25%) indicates the brightness threshold is too lax and dark soil is being flagged; much lower (<2%) indicates the threshold is too strict and real shadow edges are being missed.
- **1m shadow fraction (printed).** Should be lower than the 10cm fraction due to the 70% majority rule. If the drop is dramatic (e.g., 12% at 10cm → 1% at 1m), the shadows are fragmented sub-1m; consider lowering `AGG_FRACTION` to 0.5. If nearly identical, shadows are large solid blocks and the current rule is fine.
- **shadow -> tree vs. shadow -> low-conf split.** Rough expectation: majority of shadow pixels should be tree-adjacent in a purely-vegetated arid site, since most shadow is cast by mesquite/paloverde. If low-confidence dominates, either shadow is over-detected (revisit brightness threshold) or the CHM adjacency zone is too small (raise `DILATION_METERS`).
- **n_updated in bare_confidence.** Should be a modest fraction of the previously bare pixels (a few percent). If it's very large (>20%), the Phase 2 SAVI threshold may be flagging shaded vegetation as bare, and the SAVI cutoff should be revisited alongside shadow detection.
- **Visual overlay QA (recommended).** Overlay `shadow_tree_mask` and `shadow_lowconf_mask` on the 10cm RGB in QGIS. Tree-shadow should sit under and adjacent to visible canopies. Low-confidence shadow scattered over bare ground with no nearby canopy suggests brightness threshold false positives.

## Dependencies

- `00_ground_truth_helpers` — provides `TILE_IDS`, `YEAR`, `SITE_ID`, `OUTPUT_DIR`, `build_paths(tile_id)` returning `(rgb, ndvi, savi, chm)`.
- Requires Phase 2 outputs (`bare_confidence_{tile_id}_{YEAR}.tif`) to be present; the overwrite step is skipped with a warning if missing.
- Python: `numpy`, `rasterio`, `scipy.ndimage` (for `binary_dilation`).