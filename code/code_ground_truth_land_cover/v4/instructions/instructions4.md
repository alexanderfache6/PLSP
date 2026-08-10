# Ground Truth Classification Framework — NEON SRER

## 1. Objective

Develop and compare multiple ground-truth land-cover classification frameworks (4 classes: **tree, shrub, grass, bare**) at NEON SRER, aggregate to PlanetScope pixel scale, derive pure end members for random forest training on PlanetScope phenology metrics, and assess transferability to AmeriFlux site WKG. Compare against RAP fractional cover as a later validation step.

## 2. Site / Data

**Site**: Santa_Rita_Experimental_Range_NEON
**Site ID**: SRER
**Domain**: D14
**Acquisition date (vintage)**: 2022-08 — consistent across all products
**CRS**: UTM

### Base data path
```
.../Dropbox/planet/data/NEON/Santa_Rita_Experimental_Range_NEON/
```
Local path now; end of path will follow cluster (SCC) path convention when migrated.

### Data products

| Layer | Product folder | Product code | File pattern |
|---|---|---|---|
| RGB (10 cm) | `NEON_images-camera-ortho-mosaic` | `DP3.30010.001` | `{YYYY}_{SITEID}_{N}_{EASTING}_{NORTHING}_image.tif` |
| Veg indices (1 m) | `NEON_indices-veg-spectrometer-bidir-mosaic` | `DP3.30026.002` | subfolder `NEON_{DOMAIN}_{SITEID}_DP3_{EASTING}_{NORTHING}_bidirectional_VegIndices/`, files `NEON_{DOMAIN}_{SITEID}_DP3_{EASTING}_{NORTHING}_bidirectional_{INDEX}.tif` |
| CHM / LiDAR (1 m) | `NEON_struct-ecosystem` | `DP3.30015.001` | `NEON_{DOMAIN}_{SITEID}_DP3_{EASTING}_{NORTHING}_CHM.tif` |

**Indices available**: NDVI, SAVI, EVI. Priority for drylands: **SAVI > NDVI > EVI**.
Assumption to verify: SAVI/EVI files follow the same naming convention as the NDVI example.

### Tiles (5 total, 1 km × 1 km, key = `{EASTING}_{NORTHING}`)

| Tile | Role |
|---|---|
| `511000_3527000` | Train |
| `511000_3528000` | Train |
| `511000_3529000` | Train |
| `515000_3530000` | Test (held out) |
| `515000_3531000` | Test (held out) |

Train tiles form a contiguous block at easting 511000; test tiles are a spatially separate contiguous pair at easting 515000 — provides a within-site spatial generalization check ahead of the WKG transfer test.

**Phenocam**: located within the train block (511000 tiles). Use as an independent phenology ground-truth reference for the random forest phenology-metrics step.

## 3. Compute environment

Python, conda environment, rasterio/GDAL stack. Additional coding conventions/helper functions to be supplied via Claude Code terminal session (reference material, separate from this instruction file).

## 4. Output directory structure

```
.../Dropbox/planet/data/NEON/Santa_Rita_Experimental_Range_NEON/results/
    01_pixel_classification/       # per-framework 1 m hard classification (A-E)
    02_aggregation_mask/           # 4x4 m moving-window % cover per class
    03_pure_endmembers/            # pure end-member windows + validation
    04_rf_phenology_classification/# random forest soft classification (PlanetScope metrics) — pending PlanetScope data
    05_accuracy_assessment/        # sampled points, manual labels, per-framework accuracy
    06_rap_comparison/             # RAP 10 m vs. ground-truth vs. Planet mask — pending RAP data
    07_transferability_wkg/        # WKG transfer test
```
Each step gets its own numbered subdirectory; outputs from each framework (A–E) kept separate within `01_pixel_classification/`.

## 5. Processing pipeline

### Step 1 — Per-pixel hard classification (1 m)
Build 5 parallel frameworks, each outputting a single-class label per 1 m pixel (tree/shrub/grass/bare):

- **A. RGB only** — object detection, sub-tracks:
  - Traditional: segmentation (incl. OBIA — multiresolution/superpixel segmentation) + thresholding (incl. RGB color indices: ExG, ExGR, VARI, GLI) + trained classifier (RF/SVM)
  - U-Net-style semantic segmentation
  - DeepForest (NEON-trained crown detection) / Detectree2 (comparison)
  - SAM (Segment Anything Model) — zero-shot segment proposals + color/texture-based classification
- **B. RGB + vegetation indices** (SAVI priority)
- **C. RGB + vegetation indices + texture**
  - Texture candidates: GLCM (contrast, homogeneity, entropy, correlation), Local Binary Patterns (LBP), moving-window std dev/variance, Gabor filters, wavelet-based texture
- **D. RGB + vegetation indices + texture + LiDAR (CHM)**
- **E. Full combined model** — all layers above, same dual-track (traditional + DL) approach as A

Baseline classification logic to encode:
- CHM → isolates tree
- SAVI → isolates bare
- Texture → separates shrub vs. grass
- LiDAR (CHM) → sanity check for bare/grass vs. shrub/tree separation

### Step 2 — Aggregation to PlanetScope scale
4×4 m moving window over each framework's 1 m hard classification → % cover per class per window (mask layer).

### Step 3 — Pure end-member identification
Flag windows with ≈100% single-class cover as candidate pure end members. Export for manual validation.

### Step 4 — Random forest classification (PlanetScope phenology metrics)
**Pending**: PlanetScope phenology time-series data to be provided later.
- Training data: validated pure end members (Step 3)
- Output: soft classification (fractional cover, 4 classes) per 4×4 m window
- Compare RF soft output vs. aggregated ground-truth window classification (Step 2)
- Cross-check against phenocam signal (train block)

### Step 5 — Accuracy assessment
Sampling applied at two levels, same method for both:
1. 1 m pixel level, per class
2. 4×4 m pure end-member windows, per class

X samples per class per level — **left as a variable, to be set later**. Manual labeling performed by user. Same stratified sampling applied across all 5 frameworks (A–E) for comparison; select best-performing framework.

### Step 6 — RAP comparison
**Pending**: RAP data path to be provided later.
Compare RAP (10 m, fractional cover) vs. ground-truth mask (1 m aggregated to 10 m) vs. PlanetScope mask (4×4 m).

### Step 7 — Transferability testing
- Phase 1: framework development/selection on SRER train tiles, validated on SRER test tiles
- Phase 2: transfer best framework to WKG (AmeriFlux site)
  - Constraint: NAIP imagery only, 0.6 m, RGB+NIR (vegetation indices still computable), LiDAR offset 1–2 yrs — framework must tolerate this
  - Test whether pure end members transfer within the same ecoregion or require per-site derivation
  - Working assumption: sites outside a shared ecoregion require independent end-member sets

## 6. Open items (deferred, not blocking initial framework build)

- PlanetScope phenology data path/metrics — pending
- RAP data path — pending
- Accuracy assessment sample size (X per class) — TBD
- SAVI/EVI file naming convention — assumed to mirror NDVI pattern, needs verification against actual files
- UTM zone — assumed standard NEON zone for SRER, needs verification against actual file CRS
- WKG data paths — needed at Step 7