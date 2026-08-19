# 03 Ground Truth — Tree Detection

## Purpose

Phase 4 of the ground truth land cover pipeline. Combines LiDAR CHM crown delineation (as high-confidence training samples) with an SLIC + random-forest RGB tree detector to produce a per-tile tree mask, confidence layer, crown polygons, and a transferable RF model.

Two products:
- **Site-specific** tree mask, tree confidence, and crown geometries at 1m for SRER 2022. Uses CHM plus RGB spectral/texture.
- **Transferable** RF model exportable to non-NEON AmeriFlux sites (WKG, WJS) that have only NAIP RGB+NIR. Feature set restricted to what NAIP can reproduce.

Independent of Phase 3 shadow reassignment in current form — shadow-tree union can be added later at the raster stage.

## Workflow

1. **4.1 LiDAR crown detection.** Variable-window local maxima on CHM (min height 1.5m for training), marker-controlled watershed for crown polygons, filter for high-confidence crowns. Write peak points and crown outlines as GeoPackages.
2. **4.2 SLIC segmentation.** Partition 1m RGB into ~100,000 compact superpixels (~10 pixels each).
3. **4.3 Label segments.** Assign 'tree' (>50% inside a filter_passed crown), 'nontree' (>70% of pixels with CHM < 0.5m — no NDVI constraint), or 'ambiguous'.
4. **4.4 Extract features.** Per-segment spectral means/stds (R, G, B, NDVI, SAVI), texture (3x3 local variance of green), shape (area, perimeter, eccentricity, solidity, compactness), context (5x5 local NDVI). NaN values from empty segments filled to 0.
5. **4.5 Train RF.** Leave-one-tile-out cross-validation, then final model trained on all pooled tree/nontree segments. Save pickle for transfer.
6. **4.6 Predict.** Apply RF to all segments; write per-segment tree probability raster.
7. **4.7 Union.** Tree mask = (CHM > 2m) UNION (RGB prob > 0.7). Note the operational 2m is distinct from the 1.5m detection threshold used only during crown extraction.
8. **4.8 Confidence.** High = CHM and RGB agree; Medium = CHM-only or RGB prob > 0.85; Low = RGB-only prob 0.7-0.85.

## Tunable parameters

### Phase 4.1 — crown detection

| Parameter | Default | Description |
|---|---|---|
| `CHM_TREE_DETECT_HEIGHT` | `1.5` (m) | Min height for tree top detection and watershed mask floor. Lowered from 2m to capture short mesquite as training samples. |
| `LOCAL_MAX_BASE_RADIUS` | `0.5` (m) | Base of the variable-window formula: `radius = base + coef * height`. Prevents window collapse for shortest peaks. |
| `LOCAL_MAX_HEIGHT_COEF` | `0.05` | Slope of window radius vs height. Higher = larger windows for tall trees (fewer peaks per crown); lower = tighter windows (better small-tree resolution but more multi-peak-per-crown risk). |
| `CROWN_MIN_AREA` | `1.0` (m²) | Min filter_passed crown area. |
| `CROWN_MAX_HEIGHT_STD` | `3.0` (m) | Max within-crown height std. Rejects merged multi-tree crowns. |
| `CROWN_MIN_MAX_HEIGHT` | `2.0` (m) | Min peak height for filter_passed. Matches the operational tree/shrub boundary. |

### Phase 4.2 — SLIC

| Parameter | Default | Description |
|---|---|---|
| `SLIC_N_SEGMENTS` | `100_000` | Target superpixel count per 1000x1000 tile. Mean segment ≈ 10 pixels ≈ 3m diameter. |
| `SLIC_COMPACTNESS` | `10` | Balance between color-boundary and spatial regularity. Higher = more square. |

### Phase 4.3 — segment labeling

| Parameter | Default | Description |
|---|---|---|
| `LABEL_TREE_MIN_CROWN_FRACTION` | `0.5` | Min fraction of segment pixels inside a filter_passed crown to label 'tree'. |
| `LABEL_NONTREE_CHM_MAX` | `0.5` (m) | Max CHM for a pixel to count toward non-tree. CHM-only criterion; no NDVI constraint. |
| `LABEL_NONTREE_MIN_FRACTION` | `0.7` | Min fraction of low-CHM pixels within a segment to label 'nontree'. |

### Phase 4.7 — union and confidence

| Parameter | Default | Description |
|---|---|---|
| `CHM_TREE_OPERATIONAL_HEIGHT` | `2.0` (m) | Operational tree/shrub boundary for the final union. Distinct from `CHM_TREE_DETECT_HEIGHT`. |
| `TREE_PROB_THRESHOLD` | `0.7` | Min RF tree probability for a segment to enter the tree union. |
| `RGB_HIGH_CONF_PROB` | `0.85` | RF probability threshold above which an RGB-only tree is Medium confidence rather than Low. |

### Model

| Parameter | Default | Description |
|---|---|---|
| `RF_FEATURE_COLUMNS` | 17 features | Transferable feature set: R/G/B/NDVI/SAVI means and stds, green local variance mean, NDVI context mean, shape features. |
| `GRID_SHAPE_1M` | `(1000, 1000)` | Grid size for all 1m rasters. Matches AOP 1km tile at 1m. |

## Outputs

Written to `ground.OUTPUT_DIR = SAVE_DIR / SITE_ID`.

### Per tile

- **`tree_crown_peaks_{tile_id}_{YEAR}.gpkg`** — point layer
  One point per detected tree top. Attributes: `tree_id`, `height` (m), `filter_passed` (bool). Direct output of the variable-window local-maxima step. Used both internally as seeds for watershed and externally for tree density / spatial pattern analyses.

- **`tree_crown_outlines_{tile_id}_{YEAR}.gpkg`** — polygon layer
  One polygon per watershed-delineated crown. Attributes: `tree_id`, `area_m2`, `max_height`, `mean_height`, `height_std`, `edge_clipped`, `filter_passed`. The `filter_passed = True` subset is the high-confidence set used to label RGB segments as 'tree'.

- **`tree_mask_{tile_id}_{YEAR}.tif`** — uint8, 1m
  Binary tree mask. 1 = tree, 0 = not tree, 255 = nodata. Union of (CHM > `CHM_TREE_OPERATIONAL_HEIGHT`) and (RF tree probability > `TREE_PROB_THRESHOLD`). Feeds Phase 6 as the tree component of the final categorical map (priority tree > shrub > grass > bare).

- **`tree_confidence_{tile_id}_{YEAR}.tif`** — uint8, 1m
  Tree confidence stratification. 0 = not tree, 1 = low (RGB-only prob 0.7-0.85), 2 = medium (CHM-only OR RGB-only prob > 0.85), 3 = high (both CHM and RGB agree). Used to weight downstream shrub/grass training and to report cover uncertainty.

- **`tree_segment_prob_{tile_id}_{YEAR}.tif`** — float32, 1m
  Per-pixel RF tree probability (propagated from the segment probability). -1 = nodata. Used for probability histogram QA and for tuning `TREE_PROB_THRESHOLD`.

### Site-wide

- **`rgb_tree_detector_{YEAR}.pkl`** — pickle
  Trained random forest classifier and feature column list. Portable to AmeriFlux sites; applied to NAIP-derived features to produce a tree probability map without any LiDAR input. Along with a feature specification (to be documented separately), this is the transferable deliverable.

## Assessment for parameter tuning

Iterate in this order on the 3-tile test set before scaling to the full site.

### 1. LiDAR crown detection (Phase 4.1 outputs)

Overlay `tree_crown_peaks_*.gpkg` on 10cm RGB in QGIS. Every visible crown should have one peak; every peak should sit on a visible crown.

- Multiple peaks per crown → increase `LOCAL_MAX_HEIGHT_COEF` (0.05 → 0.08).
- One peak covering multiple visible crowns → decrease `LOCAL_MAX_HEIGHT_COEF` or `LOCAL_MAX_BASE_RADIUS`.
- Peaks over obvious shrubs → keep `CHM_TREE_DETECT_HEIGHT` at 1.5m for detection and rely on `CROWN_MIN_MAX_HEIGHT = 2.0` to remove them from filter_passed.

Overlay `tree_crown_outlines_*.gpkg` (filter_passed=True) on RGB. Polygons should tightly enclose visible crowns.

- Distribution of `area_m2` should center at 3-30 m² for mesquite/paloverde. Heavy skew < 2 m² = over-segmentation; heavy skew > 50 m² = merged multi-tree slipping through filter (tighten `CROWN_MAX_HEIGHT_STD` to 2.0).

### 2. Segment labels

Printed counts of tree/nontree/ambiguous per tile. Rough targets:

- tree segments: 5-20% of total (matches expected woody cover).
- nontree segments: 20-60% of total. If this is < 5% (as seen in the initial run), the non-tree criterion is too strict — raise `LABEL_NONTREE_CHM_MAX` from 0.5 to 1.0 or lower `LABEL_NONTREE_MIN_FRACTION` from 0.7 to 0.5. Classifier trained on a starved non-tree set will over-predict tree.
- ambiguous: 30-70% is normal.

### 3. LOO CV metrics

Reported per fold. Watch for:

- Perfect (100%) accuracy across all folds = suspicious. Either training set is too narrow (test set has same characteristics) or classes are trivially separable in the small labeled subset. Real generalization is tested on the ambiguous majority, which LOO CV does not see.
- Cross-fold variability in tree-class recall (e.g., 0.6 in one fold, 0.9 in another) = per-tile radiometric or ecological variation; expand training tiles before final transfer.

### 4. Feature importances

Printed after final training. Expected top features: `ndvi_mean`, `savi_mean`, `ndvi_context_mean`, `green_local_var_mean`. If raw R/G/B means dominate, the model is fitting per-tile radiometric bias and won't transfer well to NAIP — apply per-tile spectral normalization or drop raw means in favor of stds and differences.

### 5. Segment probability raster

Load `tree_segment_prob_*.tif` in QGIS. Histogram should be bimodal — large peak near 0, smaller peak near 1, few pixels in 0.3-0.7. Unimodal near 0.5 = poor separation, revisit features/labels. Spatial pattern of 0.5-0.7 probability shows which segments are most sensitive to `TREE_PROB_THRESHOLD` choice.

### 6. Final tree mask

Overlay `tree_mask_*.tif` on 10cm RGB. Compute site-wide tree cover fraction; expect 5-20% for SRER. Values above 30% almost certainly indicate the non-tree training issue from step 2. Values below 3% indicate the RF is missing real trees — lower `TREE_PROB_THRESHOLD` or add training samples from short-tree height ranges.

### 7. Tree confidence distribution

Distribution across confidence levels (1/2/3) of tree pixels. Rough target: majority high (both sources), significant medium (single strong source), small low (RGB-only weak). If > 30% of tree pixels are low confidence, the RF is doing most of the work on weak signals and CHM isn't reinforcing; revisit CHM operational height or RF training set.

## Dependencies

- `00_ground_truth_helpers` — provides `TILE_IDS`, `YEAR`, `SITE_ID`, `OUTPUT_DIR`, `build_paths(tile_id)` returning `(rgb, ndvi, savi, chm)`.
- Independent of Phase 2 and Phase 3; can run standalone.
- Python: `numpy`, `pandas`, `rasterio`, `geopandas`, `shapely`, `scipy.ndimage`, `scikit-image`, `scikit-learn`.