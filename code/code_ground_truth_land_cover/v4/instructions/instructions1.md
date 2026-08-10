# PlanetScope LSP Timing-Only Phenology Clustering for Rangeland PFT Fractional Cover
## Project Summary

---

## 1. Objective

Test whether **timing-only phenology metrics** derived from high-frequency PlanetScope imagery (3.7 m, ~1–2 day revisit) can be used to:
1. Identify pure/near-pure endmember signatures for plant functional types (PFTs), and
2. Ultimately produce a **fractional cover map** (bare / grass / shrub / tree) for semi-arid rangelands, using **timing metrics alone** (no amplitude/greenness magnitude features).

Ground truth for validation: NAIP (0.6 m RGB+NIR) imagery combined with lidar-derived height stratification, producing PFT reference maps.

**Secondary goal:** test whether identified endmembers are **transferable across sites/biomes** (Great Plains vs. Southwest desert rangelands).

---

## 2. Study Sites

Nine AmeriFlux sites originally considered, spanning Great Plains and Southwest US semi-arid rangelands:

| Site | Region | Ecosystem |
|---|---|---|
| US-ARM | Southern Great Plains (OK) | Winter wheat / mixed crop-pasture |
| US-xKZ (Konza) | Central Great Plains (KS) | Tallgrass prairie |
| US-Wkg (Kendall) | Chihuahuan Desert (AZ) | C4 desert grassland — **pilot site** |
| US-Whs (Lucky Hills) | Chihuahuan Desert (AZ) | Creosote shrubland |
| US-SRG / US-SRM | Sonoran/Madrean (AZ) | Grassland / mesquite savanna |
| US-Ses | Chihuahuan Desert (NM) | Creosote shrubland |
| US-Mpj | Colorado Plateau (NM) | Piñon-juniper woodland |
| US-Wjs | Colorado Plateau (NM) | Juniper savanna — **pilot site (planned next)** |

**Pilot sites chosen:** US-Wkg (grass/bare endmember test) and US-Wjs (juniper/tree endmember test, not yet run).

---

## 3. Key Design Decisions (Locked In)

- **Class taxonomy:** collapsed from 4 originally-desired classes down to **bare, grass, shrub, tree** — feasible given available lidar coverage was sparse at most sites.
- **Features: timing-only**, no amplitude/magnitude. 7 raw Moon et al. PlanetScope LSP metrics + 3 derived durations = **10 total features**:
  - Raw: OGI, 50PCGI, OGMx, Peak, OGD, 50PCGD, OGMn
  - Derived: DurGU (OGMx−OGI), DurGD (OGMn−OGD), LOS (OGMn−OGI)
- **1-cycle constraint:** only unimodal (single growing season) pixels used; verified in advance at all sites.
- **QA:** trust the Moon et al. QA layer as sole validity filter (NumCycles = 1 strict, QA ∈ {1,2} = high+medium quality). No downstream NaN-dropping needed as a separate step.
- **Standardization:** independent z-score StandardScaler fit per site per year.
- **Climate anchoring:** planned for cross-site work — site-relative and biome-relative anomalies (not yet implemented, reserved for cross-site phase).
- **Sampling:** stratified grid (30 px cells, 1 sample/cell), year-varying random seed (`base_seed + year`).

---

## 4. Literature Review Findings (Early Phase)

Reviewed temporal unmixing and phenology-based PFT separation literature. Key takeaways:
- Timing-only separation is well-supported for **C3 vs C4 grass** (offset peak timing) and **evergreen vs deciduous** woody/tree classes (SOS/EOS presence/absence).
- Weakest link: **grass vs sparse shrub** and **bare vs senesced grass**, both of which have degenerate or overlapping timing signatures.
- Prior work (Fisher & Mustard genetic-algorithm mixture inversion; Okujeni/Kowalski regression-based unmixing with synthetic training data) provided the methodological basis for the unmixing approach eventually planned for fractional cover.

---

## 5. Pipeline Architecture

All code collocated in a single Python module (`plsp_clustering.py`) per project convention — no cross-module imports; functions passed by reference (e.g., `update_report_fn`) rather than imported across files. Each site/run has its own JSON config and results directory (`{output_path}/{site_id}_{method}_{run_number}/`), with a single JSON `report.json` file that every step appends a named section to, giving full run provenance.

### Phase 1 — Data Preparation
| Step | Purpose |
|---|---|
| 1.1 | Load 7 raw PlanetScope LSP timing-metric GeoTIFFs per year into stacked arrays (fill value → NaN), with shape/CRS/transform validation within and across years. |
| 1.2 | Build per-year boolean QA mask (NumCycles + QA layers, AND logic), log retention stats to report. Serves as sole validity filter for the pipeline. |
| 1.3 | Stratified grid sampling of QA-passing pixels (30px cells, seed = base + year), writing per-year CSV of raw metric + QA values at sampled locations. |
| 1.4 | Compute 3 derived duration features (DurGU, DurGD, LOS) via explicit operand/operation specs in config (no formula parsing — safety-restricted to subtraction only), write combined feature CSV. |
| 1.5 | Fit independent StandardScaler per year on the 10 features; overwrite with z-scored values; save scaler (joblib) and scaled feature CSV; defensive NaN assertion before fitting (trusts QA layer). |

### Phase 2 — Clustering (K-means then GMM)
| Step | Purpose |
|---|---|
| 2.1 | PCA + correlation diagnostics (diagnostic only — not used for clustering input). Confirms expected redundancy among peak-adjacent timing metrics. |
| 2.2 | K-means diagnostic sweep, k=3–20, per year. Records inertia, silhouette, cluster sizes. Returns both raw sweep results and summary DataFrame. |
| 2.3 | K-means validity diagnostic plots: elbow, silhouette, and cluster-size stacked bar chart (largest→smallest per k). Manual review only — no automatic k selection. |
| 2.4 | GMM diagnostic sweep, k=3–20, both "full" and "diag" covariance types. Records BIC, AIC, log-likelihood, silhouette, convergence, cluster sizes. |
| 2.5 | GMM validity diagnostic plots: BIC/AIC, silhouette (both covariance types), and per-covariance-type cluster-size bar charts. |
| 2.6 | **Manual** k (and covariance type) selection — analyst reviews Steps 2.3/2.5 outputs, supplies flat lists of chosen k per year, validated against swept ranges, written into config as `selected_clusters`. |
| 2.7 | Refit and persist only the analyst-chosen K-means and GMM models (joblib, filename includes k), plus combined per-pixel hard labels + GMM soft posteriors CSV. Only step in Phase 2 that saves models. |
| 2.8 | Full-raster prediction: apply persisted models to every QA-passing pixel (not just samples) across the whole 10 km × 10 km footprint. Output: GeoTIFF (native UTM CRS from QA layer) + categorical PNG (tab20 colormap, black nodata) per method. |

### Phase 3 — Endmember Identification and Validation
| Step | Purpose |
|---|---|
| 3.1 | Load persisted K-means/GMM models fresh from disk (fresh-start assumption); extract centroids (K-means) and means + normalized (k, F, F) covariances (GMM, handling both "full" and "diag" shapes). |
| 3.2 | Pairwise centroid distance matrices: **Euclidean** for K-means; **Mahalanobis** for GMM using pooled (averaged) covariance per cluster pair, with ridge regularization for numerical stability. |
| 3.3 | Annotated heatmap visualizations of both distance matrices. |
| 3.4 | Per-cluster interior-pixel sampling (5×5 neighborhood majority filter to avoid boundary/speckle pixels) — 20 samples/cluster, written as **editable GeoPackage** (not CSV) with empty nullable-integer `naip_class` column (0=bare, 1=grass, 2=shrub, 3=tree) for manual QGIS annotation against NAIP. |
| 3.5 | Purity assessment: after manual QGIS annotation, compute per-cluster class distribution, majority-class purity fraction, and Shannon entropy. Stacked bar chart (per-bar segments sorted by count, cluster order preserved on x-axis) with purity-fraction line on secondary y-axis. |
| 3.6 | Cluster-to-class mapping using purity thresholds: **≥0.80 = high confidence, ≥0.50 = moderate confidence** (both assigned to majority class), **<0.50 = "mixed"** (own category, rendered orange). Reclassified full-raster GeoTIFF + PNG produced (5 categories: bare/grass/shrub/tree/mixed + nodata). |

---

## 6. Key Findings (Wkg, 2023, k=10)

**Visual diagnostics before formal analysis:**
- Geometric rectangular block pattern (upper-left of raster) confirmed to be edge effects of bad QA pixels (not a genuine land-cover artifact) — deprioritized as endmember candidate.
- K-means and GMM cluster IDs **do not correspond spatially** across methods (same cluster number, different real-world meaning) — confirmed cluster identity is not portable between algorithms.
- GMM produced a distinctive thin dendritic pattern suspected to trace drainage/riparian vegetation.
- A sharp lateral raster boundary attributed to a genuine elevation ridge / bare-spot transition (confirmed against NAIP, not an artifact).

**Purity results (K-means, k=10, all 20/20 points annotated per cluster):**
- Only **2 of 10 clusters** met the "clean" bar: Cluster 2 (100% bare) and Cluster 6 (80% grass).
- Several clusters landed at or near 50% purity — essentially coin-flip majority calls (clusters 0, 1, 4, 8, 9).
- **Bare fragmented across 4 clusters** (0, 2, 4, 7) with widely varying purity (50–100%) — suggests either sub-classes within "bare" (soil type, litter amount) or over-segmentation at k=10.
- **Shrub was the most confused class** (clusters 1, 5, 8; purity 40–70%) — consistent with known heterogeneity of shrub phenology (evergreen vs. deciduous species) at these sites.
- **Tree (cluster 3, 75% purity)** was the second-cleanest signal after bare.
- **~25% of pixels landed in "mixed" (purity <0.50)** after reclassification — left unresolved for now; the remaining 75% (assigned to bare/grass/shrub/tree) was judged to describe the landscape well for this single site.

**Decision:** did not pursue centroid-distance-based inference to resolve mixed pixels at this stage — current 4-class assignment (excluding mixed) considered sufficient for Wkg alone.

---

## 7. Current Status

- Phase 1–3 pipeline fully built and run **once**, for **one site (Wkg), one year (2023)**.
- Second pilot site (Wjs, juniper savanna) **not yet run** through the pipeline.
- Cross-site/cross-biome comparison **not yet started** — this requires climate-anchored (site-relative anomaly) feature representations, which were designed conceptually earlier but not yet implemented in code.

---

## 8. Immediate Next Step

**Run the full Phase 1 → Phase 3 pipeline on US-Wjs** (juniper savanna), ideally matching Wkg's year, to obtain its own purity-confirmed non-mixed clusters. Then:
1. Compute cross-site centroid distances in **climate-anchored** feature space (not raw DOY, since Wkg and Wjs have offset phenological calendars).
2. Compare whether Wkg's confirmed grass/bare/tree/shrub clusters sit close to Wjs's equivalent confirmed clusters in anchored space.
3. This comparison is the core test of the project's transferability hypothesis — whether timing-only endmembers generalize across biomes.

Extending to additional years per site (interannual stability testing) is a planned refinement **after** the cross-site question is answered, not before.

---

## 9. Deferred / Not Yet Addressed

- Resolution of "mixed" cluster pixels (~25% of raster) — considered a follow-on step (centroid-distance-based inference to nearest confident cluster), deliberately not pursued yet.
- Climate-anchored (site-relative / biome-relative anomaly) feature engineering — designed conceptually, not yet coded.
- Multi-year (Approach B: concatenated multi-year feature vectors) clustering — designed conceptually (Approach A per-year first, then B), not yet implemented.
- Fractional cover / unmixing model (the ultimate project goal) — not started; current work is entirely in the clustering/endmember-identification diagnostic phase.
- Field-plot validation data (Jornada LTER, ARS Walnut Gulch, Santa Rita SRER transects, NEON plot data) — sources identified and documented, not yet integrated.