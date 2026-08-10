# NEON Ground Truth Land Cover Pipeline — Session Summary

## Project Context

Land surface phenology and plant functional type detection using PlanetScope satellite imagery, validated against high-resolution ground truth generated from NEON Airborne Observation Platform (AOP) data. Target site: **SRER (Santa Rita Experimental Range)**, NEON Domain D14 (Desert Southwest), 2022 flight year. Long-term goal includes transferring the tree detection method to AmeriFlux sites (WKG, WJS) that lack NEON's LiDAR/hyperspectral inputs and have only NAIP RGB+NIR imagery.

Target classes: **bare / grass / shrub / tree**, at 1m resolution, aggregated to 3m for PlanetScope compatibility.

## Data Products Used

| Product | Resolution | Role |
|---|---|---|
| DP3.30010.001 (RGB camera ortho-mosaic) | 10cm | Spatial detail, texture, shadow detection, DeepForest input |
| DP3.30026.002 (vegetation indices: NDVI, SAVI, EVI, ARVI, PRI) | 1m | Bare/vegetated split, shrub/grass classification features |
| DP3.30015.001 (Canopy Height Model) | 1m | Tree detection, crown delineation |

Restricted to these three products specifically because they mimic what's available at non-NEON AmeriFlux sites (NAIP RGB+NIR → SAVI/NDVI, USGS 3DEP LiDAR → CHM), keeping the pipeline transferable.

## Pipeline Structure

Code organized as numbered notebooks importing shared config/paths from `00_ground_truth_helpers.py` (`TILE_IDS`, `YEAR`, `SITE_ID`, `OUTPUT_DIR`, `build_paths(tile_id)`). Tile-by-tile processing for parallelization; 3-tile test set before scaling to full site.

### Phase 2 — Bare Detection (`01_ground_truth_bare_detection.ipynb`)

- Pool SAVI/NDVI/RGB across tiles at 1m, compute candidate thresholds (Otsu, p15, p25).
- Visual RGB overlay verification at 1m display resolution (downsampled from 10cm to avoid OOM).
- Final threshold: **SAVI < 0.2** → bare.
- Outputs: `bare_mask_{tile}_{YEAR}.tif` (uint8), `bare_confidence_{tile}_{YEAR}.tif` (float32, linear distance-below-threshold scaled to [0,1]).

### Phase 3 — Shadow Detection (`02_ground_truth_shadow_detection.ipynb`)

- Brightness threshold (Rec. 709 luma: `0.2126R + 0.7152G + 0.0722B`) pooled across tiles at 20th percentile, combined with blue-shift rule (`B > R`) for shadow detection at 10cm.
- Aggregate to 1m via >70% majority rule.
- Shadow within 5m of CHM > 2m → reassigned to tree-shadow mask (input to Phase 4 union).
- Remaining shadow → low-confidence flag.
- Shadow-bare overlap → `bare_confidence` overwritten to 0 in place.
- Outputs: `shadow_mask_1m`, `shadow_tree_mask`, `shadow_lowconf_mask` (all uint8), updated `bare_confidence`.

### Phase 4 — Tree Detection, Approach 1: CHM + RF (`03_ground_truth_tree_detection.ipynb`)

- **4.1** LiDAR crown detection: variable-window local maxima on CHM (`radius = base_radius + height_coef × height`), watershed segmentation, filtered by area/edge/height-std. Crown geometry finalized as **circles centered at the tree-top peak**, radius from the same height formula (not fit to watershed boundary). Attributes recomputed from pixels inside the circular footprint.
- **4.2–4.4** SLIC segmentation (1m, ~100k segments/tile) + per-segment features (spectral, texture, shape, context) restricted to RGB/NDVI/SAVI for transferability.
- **4.5** Random forest trained on segments labeled via LiDAR crown overlap; leave-one-tile-out cross-validation.
- **4.6–4.8** Predict, union with CHM > 2m, confidence stratification (high/medium/low).
- **Known issue identified but not yet fixed in this thread**: initial run produced 93–97% tree cover due to a starved non-tree training set (`LABEL_NONTREE_CHM_MAX`/`NDVI_MAX`/`MIN_FRACTION` too strict). Fix specified: loosen to CHM-only criterion (`LABEL_NONTREE_CHM_MAX = 0.5`, `LABEL_NONTREE_MIN_FRACTION = 0.7`, drop NDVI constraint), and split `CHM_TREE_DETECT_HEIGHT` (1.5m, training) from `CHM_TREE_OPERATIONAL_HEIGHT` (2.0m, final union) as separate parameters — these were being conflated in the original code.
- Outputs: `tree_crown_peaks_*.gpkg`, `tree_crown_outlines_*.gpkg`, `tree_mask_*.tif`, `tree_confidence_*.tif`, `tree_segment_prob_*.tif`, `rgb_tree_detector_{YEAR}.pkl` (transferable model).

### Phase 4 — Tree Detection, Approach 2: DeepForest (in progress, not yet in notebook form)

Pivoted to test **DeepForest** (pretrained `weecology/deepforest-tree`, no fine-tuning) as a simpler, inherently transferable alternative — RGB-only, trained substantially on NEON imagery already.

- `predict_tile()` on native 10cm RGB, `patch_size=800`, DeepForest defaults for overlap/IoU.
- Keep all detections (no score filtering), rectangular bounding boxes retained as-is.
- Outputs planned: `tree_crown_outlines_deepforest_{tile}_{YEAR}.gpkg` (boxes + score + label), `tree_score_deepforest_{tile}_{YEAR}.tif` (rasterized score, 0 outside boxes, max-score-on-overlap via ascending-sort + `MergeAlg.replace`), `tree_mask_deepforest_{tile}_{YEAR}.tif` (binary).
- This is Option A: full replacement of the CHM/SLIC/RF approach for this test, not a hybrid.

## Current Blocker: GPU Access on BU SCC

Extended troubleshooting sequence, not yet fully resolved:

1. Initial `qrsh -l gpus=1` landed on a **Tesla K40m** (compute capability 3.5) — incompatible with any current PyTorch build, producing a misleading "no driver found" error.
2. Corrected request: `qrsh -l gpus=1 -l gpu_c=6.0 -pe omp 4` → landed on **Tesla P100-PCIE-12GB** (compute capability 6.0, driver 580.159.04, CUDA 13.0) — compatible in principle.
3. `torch.cuda.is_available()` still returned `False` despite correct hardware. Diagnosed through: environment/session mismatch (ruled out — same hostname), Jupyter kernel not inheriting shell environment (kernel restarted, issue persisted), CPU-only PyTorch build (ruled out — `2.13.0+cu130` confirmed CUDA-enabled).
4. Root `torch.cuda.init()` error: same "no driver found" message even with correct GPU and matching PyTorch build.
5. `libcuda.so.1` confirmed present and registered via `ldconfig` at `/lib64`.
6. `module load cuda/13.2` loaded successfully; `nvcc` and toolkit paths confirmed. **Open hypothesis**: the loaded module's `LD_LIBRARY_PATH` may be shadowing the real driver library with a compile-time stub `libcuda.so` from the toolkit's own `lib64` — check pending (`find /share/pkg.8/cuda/13.2/install -name "libcuda.so*"`).
7. **Status at end of session: still unresolved.** Last message reports drivers "still don't work" but no diagnostic output provided for the stub-library check or a fresh-kernel `torch.cuda.init()` retest.

## Secondary Blocker: GDAL Tiling Requirement

`predict_tile(..., dataloader_strategy='window')` requires internally tiled GeoTIFFs for memory-efficient windowed reads; NEON RGB products are likely strip-organized, causing a `ValueError`. Two fixes proposed, neither yet executed:

- **Preferred long-term**: `gdal_translate -of GTiff -co TILED=YES <input> <output>` to preprocess RGB tiles once, cache tiled copies.
- **Faster unblock for testing**: switch `dataloader_strategy` from `'window'` to `'batch'` (loads full image, avoids tiling requirement; acceptable at 10000×10000×3 scale for a 3-tile test).

## Standing Conventions Established This Session

- All functions require explicit pydoc (purpose/inputs/outputs), no optional/default arguments — all parameters passed positionally at call sites.
- No line-wrapping at 80 characters; long lines kept as single lines.
- No leading whitespace in `print()` statements.
- Naming convention for alternate-method outputs: insert method name before tile ID (e.g., `tree_mask_deepforest_{tile}_{YEAR}.tif`).
- Each phase notebook paired with a `.md` file documenting purpose, tunable parameters (with defaults and tuning guidance), and full output descriptions.
- Confirm understanding and provide a step list before writing any code; wait for explicit confirmation.

## Immediate Next Steps

1. Resolve GPU driver visibility (check for stub `libcuda.so` shadowing in the CUDA 13.2 module path; retest `torch.cuda.init()` from a kernel launched fresh in the module-loaded shell).
2. Resolve GDAL tiling error (either preprocess with `gdal_translate` or switch to `dataloader_strategy='batch'` for the test run).
3. Run DeepForest 3-tile test once unblocked; compare tree cover fraction and spatial pattern against the CHM/RF approach's outputs (once that approach's non-tree-labeling fix is also applied).
4. Apply the identified fix to the CHM/RF pipeline's Phase 4.3 labeling parameters (currently specified but not yet re-run).
5. Decide between CHM/RF vs. DeepForest as the primary method based on 3-tile comparison, or formalize a validation protocol between the two.