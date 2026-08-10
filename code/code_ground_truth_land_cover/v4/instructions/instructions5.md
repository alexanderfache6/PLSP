# Ground Truth Land Cover — v4 Execution Specification

**Status**: authoritative. This document supersedes `instructions4.md` and governs all v4 work.
**Created**: 2026-08-09

## 0. Document hierarchy

| Document | Status | Use |
|---|---|---|
| `instructions5.md` (this file) | **Authoritative** | Spec and execution plan going forward |
| `instructions4.md` | Superseded | Prior spec; retained for provenance |
| `instructions1.md` | Archived context | Upstream PlanetScope LSP clustering project (Wkg/Wjs). Explains where the phenology metrics and the unsupervised-clustering label option come from |
| `instructions2.md` | Archived context | v3 session summary (bare/shadow/tree notebooks). Shadow method and standing conventions carried forward from here |
| `instructions3.md` | Archived context | Raw dictated brain-dump. **Contains heavy speech-to-text corruption** ("bear"=bare, "rainforest"/"rental force"=random forest, "lighter"=LiDAR, "nap"=NAIP, "Cypress sites"=across sites, "vegetation industries"=vegetation indices). Do not read literally |

---

## 1. Objective

Build and compare multiple ground-truth land-cover classification frameworks (**tree / shrub / grass / bare**, plus **shadow** as a handled non-terminal class) at NEON SRER; aggregate 1 m hard classifications to PlanetScope pixel scale for fractional cover; derive validated pure end members; train a PlanetScope phenology model for fractional cover; and assess transferability across a network of three paired NEON/AmeriFlux domains (§2), beginning with SRER → WKG. Compare against RAP fractional cover as a later validation step.

---

## 2. Study region and site network

### 2.1 Site network — three paired domains

The study region is **three NEON domains**, each contributing one **NEON site** (full AOP: 10 cm RGB, 1 m VI, 1 m CHM) paired with one nearby **AmeriFlux site** (NAIP RGB+NIR only, plus offset 3DEP LiDAR).

Each pair is a self-contained transferability test: build ground truth at the data-rich NEON site, transfer to the data-poor AmeriFlux partner within the same domain and vegetation class.

| Domain | Domain name | Veg. class | NEON site | AmeriFlux partner | Ecosystem |
|---|---|---|---|---|---|
| **D14** | Desert Southwest | **GRA** — grasslands | **xSR** (SRER) | **WKG** | Semi-arid desert grassland / mesquite savanna |
| **D13** | Southern Rockies & Colorado Plateau | **SAV** — savannahs | **xMB** (MOAB) | **WJS** | Piñon-juniper / juniper savanna |
| **D06** | Prairie Peninsula | **GRA** — grasslands | **xKZ** (KONZ) | **KFB** — Konza Prairie LTER (4B) | Tallgrass prairie |

NEON sites carry AmeriFlux registrations under their `x`-prefixed IDs (`US-xSR`, `US-xMB`, `US-xKZ`), so flux data is available on both sides of every pair.

**D06 is co-located, and that makes it the control.** `US-KFB` is Konza Prairie LTER watershed 4B — the same Konza Prairie as NEON `xKZ`, differing in burn treatment rather than location. The D14 and D13 pairs are separated by tens of kilometres and differ ecologically as well as by sensor; D06 does not. That makes D06 the **cleanest available Axis-1 test**: transfer failure there is attributable to sensor and resolution alone, because the ecology is effectively held constant. Run D06 as the sensor-transfer control before interpreting D14 or D13 results. The burn-treatment difference is itself a variable to record — fire history drives litter and standing-dead cover, which bears directly on the bare vs. senesced-grass separation (`instructions1.md` §4).

**References**:
- NEON field site map and info — https://www.neonscience.org/field-site-map-and-info
- AmeriFlux site search / mapping tool — https://ameriflux.lbl.gov/sites/site-search/?mapping-tool

**Current scope**: SRER only. The other five sites are the roadmap; the pipeline must be built so adding one is a config file plus data, with zero code edits (R8).

### 2.2 What this network is designed to test

The three pairs give two independent axes of transfer, which must be reported separately:

**Axis 1 — within-domain transfer (NEON → AmeriFlux).** Same domain, same vegetation class, different sensor. Isolates the **sensor and resolution** problem (AOP vs. NAIP) from the ecological one. This is §8 Step 8 Route 1 vs. Route 2.

**Axis 2 — cross-domain transfer.** D14 and D06 are **both GRA grasslands but ecologically very different** — hot semi-arid C4 desert grassland versus mesic tallgrass prairie. That contrast is the strongest available test of the working assumption that sites outside a shared ecoregion require independent end-member sets: same nominal vegetation class, very different phenology. D13 (SAV) supplies the woody-dominated contrast and is where tree and shrub signal is strongest.

### 2.3 Phenology calendars differ across domains — raw DOY is not comparable

This is the central cross-site obstacle, already identified in `instructions1.md` §§3 and 8:

| Domain | Growth driver | Approximate green-up |
|---|---|---|
| D14 (SRER, WKG) | North American monsoon | Jul–Sep |
| D13 (MOAB, WJS) | Cool-season moisture + monsoon | Bimodal — spring and late summer |
| D06 (KONZ, KFB) | Spring warming + summer rainfall | Apr–Sep |

Consequences that must be handled before any cross-domain comparison:

1. **Raw DOY features cannot be compared across domains.** A grass end member at SRER and a grass end member at KONZ will differ in `OGI` by months for reasons that are climatic, not compositional. `instructions1.md` §8 specifies **climate-anchored** (site-relative and biome-relative anomaly) features for exactly this; that work is designed but not yet implemented and is a **prerequisite for Axis 2**, not for Axis 1.
2. **The `NumCycles == 1` strict filter is a live risk at D13 — see §2.4.** Bimodal cool-season-plus-monsoon systems legitimately produce two cycles, and the strict filter would discard those pixels wholesale.
3. **Acquisition-date checks are per site, not global** (R7). The monsoon-peak timing that makes SRER 2022-08 favorable has no equivalent meaning at D06.

### 2.4 Initial step — NumCycles distribution per site

**This runs first, before any other analysis, at every site in the network.** It is cheap (one layer, one histogram) and it decides whether the project's core 1-cycle constraint is viable at that site.

`instructions1.md` §3 locks in a **`NumCycles == 1` strict** filter, stating unimodality was verified in advance at all sites — but that was the earlier nine-site list, and **D13 (MOAB, WJS) is expected to be genuinely bimodal**: cool-season moisture drives a spring green-up and the North American monsoon drives a second late-summer green-up in the same calendar year. Where that happens, `NumCycles == 2` is the *correct* description of the vegetation, and the strict filter discards those pixels as though they were bad data.

**Procedure** — per site, per year:

1. Read layer 1 (`NumCycles`, valid 0–6, fill 32767) across the full site footprint.
2. Mask fill; tabulate the **count and percentage of pixels at each value 0–6**.
3. Cross-tabulate `NumCycles` against `QA` (layer 12) — a 2-cycle pixel at QA 1 is a real second cycle; a 2-cycle pixel at QA 3–4 may be noise-driven.
4. Map `NumCycles` spatially. Bimodality that is spatially structured (drainages, aspect, vegetation type) is ecological; salt-and-pepper bimodality is likely fitting noise.
5. Report the **retained fraction under `NumCycles == 1` strict**, and the additional fraction that would be recovered by admitting `NumCycles == 2`.

**Decision rule**: if the 1-cycle retained fraction is high, proceed unchanged. If a site loses a large share of pixels — the expectation at D13 — the 1-cycle constraint must be **revisited explicitly**, not silently absorbed as data loss. Options in that case, to be decided with the results in hand:

- Restrict to cycle 1 of a 2-cycle pixel and accept partial description.
- Treat 1-cycle and 2-cycle pixels as separate populations with separate models.
- Extend the feature set to the cycle-2 layers (13–23) for bimodal pixels, which changes feature dimensionality across sites and would break direct model transfer (R1).

Note the interaction with §5.3 trap 5: the current spec asserts cycle-2 layers are fill for all retained pixels. That assertion holds **only** under `NumCycles == 1` strict, and any relaxation of the filter invalidates it.

Outputs → `00_qa/numcycles_distribution_{SITE}_{YEAR}.json` plus a per-site histogram and map. **Run this for all six sites as early as PlanetScope LSP data allows** — it is a prerequisite for scoping, not a per-site step to be deferred until that site's turn.

### 2.5 Site verification required before use

| Item | Status |
|---|---|
| `US-KFB` — Konza Prairie LTER (4B) | Confirmed. Co-located with NEON `xKZ`; record the burn treatment |
| NEON AOP flight years available per site (xSR, xMB, xKZ) | Needed — governs which years §6's stability check can cover |
| NAIP acquisition years and dates per AmeriFlux site | Needed (R7) |
| 3DEP LiDAR year per AmeriFlux site and offset from imagery | Needed (§11 check #27) |
| UTM zone per site | Verify from file CRS; differs across domains |
| Ecoregion definition | **Resolved — NEON domain** (R1, §12 Q4). All pairs are within-domain, so all three are model-transfer tests |
| Pair separation distance and any EPA L3 boundary crossed | Record per pair for interpreting Route 1 outcomes (R1) |
| PlanetScope LSP coverage per site and year | Needed |

---

## 2A. SRER — current site

**Site**: Santa_Rita_Experimental_Range_NEON (`SRER`), NEON Domain D14, AmeriFlux `US-xSR`
**Acquisition vintage**: 2022-08 (monsoon peak — grasses green; see §9 temporal note)
**CRS**: UTM (verify against actual file CRS; do not assume the zone)

### Data products

| Layer | Product code | Resolution | Product folder / pattern |
|---|---|---|---|
| RGB ortho-mosaic | `DP3.30010.001` | 10 cm | `NEON_images-camera-ortho-mosaic` / `{YYYY}_{SITEID}_{N}_{TILE}_image.tif` |
| Vegetation indices | `DP3.30026.002` | 1 m | `NEON_indices-veg-spectrometer-bidir-mosaic` / `NEON_{DOMAIN}_{SITEID}_DP3_{TILE}_bidirectional_VegIndices/` |
| CHM | `DP3.30015.001` | 1 m | `NEON_struct-ecosystem` / `NEON_{DOMAIN}_{SITEID}_DP3_{TILE}_CHM.tif` |

Products deliberately restricted to these three: they mimic what is available at non-NEON AmeriFlux sites (NAIP RGB+NIR → SAVI/NDVI; USGS 3DEP → CHM), preserving transferability.

**Index priority for drylands**: SAVI > NDVI > EVI.
**Verify before coding**: SAVI/EVI filenames are *assumed* to mirror the NDVI pattern — confirm against the actual directory listing.

### Tiles (1 km × 1 km, key = `{EASTING}_{NORTHING}`)

| Tile | Role |
|---|---|
| `511000_3527000` | Train |
| `511000_3528000` | Train |
| `511000_3529000` | Train |
| `515000_3530000` | Test (held out) |
| `515000_3531000` | Test (held out) |

Train tiles are a contiguous block at easting 511000; test tiles are a spatially separate contiguous pair at easting 515000, giving a within-site spatial generalization check ahead of the WKG transfer.

**Action required**: the existing `00_ground_truth_helpers.py` carries a different 3-tile set. Confirm all five tiles above are downloaded for all three products before Step 0 completes; download any missing tiles.

**Phenocam**: located within the 511000 train block. Use as an independent phenology reference in Step 5.

### Paths

- **Local (now)**: `.../Dropbox/planet/data/NEON/Santa_Rita_Experimental_Range_NEON/`
- **Cluster (SCC)**: `/projectnb/modislc/users/fache/data/NEON/{SITE_NAME}/`, results under `/projectnb/modislc/users/fache/results/`

Local paths are used for development. **All deep-learning tracks run on the SCC** (see §8 prerequisites). Path root must be a single configurable constant in the helpers module so the switch is one edit, not a search-and-replace.

### Output directory structure

```
results/
    00_qa/                          # per-tile QA, CHM noise floor, grid alignment reports
    01_pixel_classification/        # per-framework 1 m hard classification (A-E)
    02_aggregation_mask/            # N x N m window % cover per class
    03_pure_endmembers/             # pure end-member windows + validation
    04_rf_phenology/                # PlanetScope fractional-cover model
    05_accuracy_assessment/         # shared sample set, manual labels, per-framework accuracy
    06_rap_comparison/              # RAP 10 m vs. ground truth vs. Planet
    07_transferability_wkg/         # WKG transfer test
```

---

## 3. Class definitions

### Class codes — LOCKED (cross-project convention)

**These integer codes are locked and must not be renumbered.** They match `instructions1.md` §3.4's `naip_class` convention, so outputs are directly interoperable between v4, the LSP clustering project, and any site added later — no lookup table, no per-framework or per-site variants. Any code, GeoPackage attribute, colormap, or confusion matrix uses these values.

| Code | Class | Terminal? | Reference rule (SRER, CHM available) | Description |
|---|---|---|---|---|
| **0** | **bare** | Yes | SAVI < `SAVI_BARE_MAX` and CHM < `H_GRASS_MAX` | Soil, rock, litter, standing dead with no green signal |
| **1** | **grass** | Yes | Herbaceous. CHM < `H_GRASS_MAX` and SAVI >= `SAVI_BARE_MAX` | Perennial and annual graminoids; herbaceous forbs |
| **2** | **shrub** | Yes | Woody. `H_GRASS_MAX` <= CHM < `H_TREE_MIN` | Low woody vegetation; includes cacti and succulents |
| **3** | **tree** | Yes | Woody. CHM >= `H_TREE_MIN` | Tall woody vegetation |
| **4** | **shadow** | **No** | Detected per §5 Step 1c | Intermediate only. Resolved to tree (3) if within `SHADOW_TREE_RADIUS` of CHM >= `H_TREE_MIN`, else masked to nodata. **Never appears in a final product** |
| **255** | **nodata** | n/a | — | Fill / masked / outside footprint. uint8 rasters |

**`mixed` is not a class code.** It is a **continuous confidence layer** (§3.1), so a pixel always carries both its plurality class *and* how strongly it resembles that class — rather than losing the class label to an ambiguity code, or collapsing a graded quantity into a boolean.

> **Threshold values are provisional and will be refined** — the *codes* above are locked, the *rules* are not. Thresholds are a starting point for the first run, expected to iterate against visual inspection and Step 6 accuracy. Any code written against them must read config constants, never inline literals, so refinement is a config edit.

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| `H_TREE_MIN` | **2.0 m** | Locked per decision |
| `H_GRASS_MAX` | **0.3 m** (provisional) | Must be validated against the measured CHM noise floor — see below |
| `SAVI_BARE_MAX` | **0.2** | Carried forward from v3 bare detection |
| `SHADOW_TREE_RADIUS` | **5 m** | Carried forward from v3 |

### Species context (SRER)

Velvet mesquite (*Prosopis velutina*) is the dominant woody species and straddles the shrub/tree boundary — many individuals sit at 1.5–3 m. Creosote (*Larrea*), burroweed (*Isocoma*), cholla and prickly pear are shrub. Lehmann lovegrass and native gramas are grass.

### Three required safeguards

1. **The 2.0 m cut splits mesquite.** Persist per-crown CHM statistics (min/mean/max/p90, pixel count) as attributes on the crown vector outputs, so the shrub/tree threshold can be re-cut analytically without re-running the pipeline.
2. **0.3 m may sit inside CHM noise.** Before locking `H_GRASS_MAX`, measure the empirical CHM distribution over visually-confirmed bare areas and set the threshold above the observed noise floor. Record the measurement in `00_qa/`.
3. **1 m pixels are mixed — handled as continuous confidence, not a boolean flag.** See §3.1.

### 3.1 Mixture as a confidence layer

1 m pixels in semi-arid rangeland are usually mixed. Rather than discarding that information into a boolean flag, **mixture is carried through the whole pipeline as a continuous confidence measure**: low mixture means high purity and strong class resemblance; a near-even mixture means low confidence in the predicted cover. It propagates into aggregation, end-member selection, model training, and accuracy reporting.

#### Two outputs per 1 m pixel, always produced together

| Output | Content | Purpose |
|---|---|---|
| **Hard label** | Plurality class, codes 0–3 (§3) | The discrete product; accuracy assessment; Step 3 counting |
| **Soft vector** | Per-class membership `p = [p_bare, p_grass, p_shrub, p_tree]`, sums to 1 | Confidence; soft aggregation; training weights |

Hard label = **argmax of `p`**. Ties break in order **tree > shrub > grass > bare**.

#### Where the soft vector comes from

- **Learned frameworks (A–E, RF)**: the classifier's per-class probability output directly.
- **Rule-based reference path (§3 threshold rules)**: distance-to-threshold scaled to [0, 1], the convention already established in v3 (`instructions2.md` Phase 2, `bare_confidence` as linear distance below the SAVI threshold), then normalized across classes.
- **DL tracks**: softmax output (DL-2), or detection score (DL-1).

#### Confidence measure

Primary, for continuity with `instructions1.md` §3.5, which already uses Shannon entropy for cluster purity:

```
confidence = 1 − H(p) / log(K),    H(p) = −Σ p_c · log p_c,    K = 4
```

Range [0, 1]: **1.0 = pure** (all membership in one class), **0.0 = perfectly even** four-way mixture. This is the layer meant by "mixed" throughout this document.

Also record, as a secondary diagnostic, the **margin** `p_(1) − p_(2)` between the top two classes. Entropy and margin disagree in an informative way: a pixel split 50/50 between grass and shrub has moderate entropy but near-zero margin, and it is the margin that identifies the specific confusions worth investigating (grass/shrub, bare/senesced grass).

#### Outputs

`confidence_{framework}_{tile}_{YEAR}.tif` (float32, [0,1]),
`margin_{framework}_{tile}_{YEAR}.tif` (float32),
`class_prob_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands, band order = class codes 0–3).

Retaining the full probability stack is what makes soft aggregation (Step 3) possible. Do not harden and discard.

#### This layer is `prediction_quality`

Throughout this document and in config, the §3.1 confidence layer is named **`prediction_quality`** — it measures how confident the *classification* is. It is distinct from **`data_quality`** (PlanetScope `numObs`, `QA`, `NumCycles`), which measures whether the *observations* were good enough to fit a phenology curve. The two are never blended into one score; see the table in §5.3.

#### Downstream use — four places

1. **Step 3 aggregation** — soft aggregation, and confidence weighting of the hard count.
2. **Step 4 end members** — a block that is 95% grass built from low-confidence pixels is a weaker end member than one that is 92% grass at high confidence. Rank by purity **and** confidence.
3. **Step 5 RF-B training** — block confidence becomes the regression `sample_weight`. Blocks with noisy targets contribute less.
4. **Step 6 accuracy** — report accuracy stratified by confidence, not only pooled.

### Transferability constraint

The CHM rules above define the **reference** labels only. CHM is not reliably available at WKG (NAIP-era LiDAR is offset 1–2 years). Frameworks A–C must therefore reproduce these labels **from RGB + vegetation indices + texture alone**. This is the central design tension and is handled explicitly in §4.

---

## 4. Framework design

### 4.1 Deconfounding

Frameworks A–E vary **input layers only**. The algorithm is held fixed across all five so that any accuracy difference is attributable to the inputs, which is the actual research question.

**Fixed algorithm for A–E**: segmentation (SLIC at 1 m) + per-segment feature extraction + Random Forest classifier.

| Framework | Inputs | Transferable to WKG? |
|---|---|---|
| A | RGB only | Yes |
| B | RGB + vegetation indices (SAVI priority) | Yes |
| C | RGB + VI + texture | Yes |
| D | RGB + VI + texture + CHM | Degraded (offset LiDAR) |
| E | All layers + full feature set | Degraded |

**Priority order for feature emphasis**: vegetation indices and texture first; **LiDAR/CHM is a sanity check between classes, not a primary driver**. All available data is used, but the transferable frameworks (A–C) are the ones carried to WKG.

**Deep-learning tracks are a separate, smaller comparison** — not folded into A–E:

| Track | Method | Notes |
|---|---|---|
| DL-1 | DeepForest (`weecology/deepforest-tree`, pretrained, no fine-tuning) | RGB 10 cm, tree crowns only. **Role: independent tree-crown validation backup**, not a competing 4-class framework — trained on NEON RGB, so it is an outside opinion on crowns (§12 Q9). NAIP use requires the degrade-and-compare test in Q9a |
| DL-2 | U-Net semantic segmentation, 4-class | Requires the hand-labeled training set from §4.2 |
| DL-3 | SAM zero-shot segment proposals + color/texture classification | Optional, run only if DL-1/DL-2 leave gaps |

DL tracks output tree (DL-1) or full 4-class (DL-2/DL-3) maps evaluated on the same shared sample set as A–E. **DL-1 is scored on the tree class only** — it is a cross-check on tree detection, not a candidate for framework selection (§12 Q9).

**Reference vs. candidate framing**: Framework D/E (CHM-bearing) is expected to be the most accurate at SRER, but is the *least* transferable. It therefore serves as the **reference/labeler** against which the transferable candidates (A–C, DL) are measured — the "best framework" selection in Step 6 selects among transferable candidates, judged by agreement with both D/E and the manual sample set.

### 4.2 Label source — hand-labeled, with clustering used to target the labeling

**Hand-labeled polygons are the ground truth.** Unsupervised clustering is **not** a competing label source; it runs first, as a targeting aid that decides *where* the human should label. The two stages are sequential, not parallel.

> **The cluster ID is never a label.** Clustering says *where to look*; the analyst says *what it is*. No cluster-to-class assignment is performed, and no cluster ID propagates into the training labels. This keeps the ground truth fully human-determined and avoids importing the clustering's errors into the labels it was meant to help collect.

#### Stage 1 — K-means to define labeling zones

Purpose: unguided hand labeling drifts toward the visually obvious. A labeler naturally picks clean, unambiguous patches, which starves the classifier of exactly the intermediate and difficult cases that decide Step 6 accuracy. Clustering the feature space first exposes the full range of what is actually present, so labeling can be allocated across it rather than across whatever caught the eye.

- **Simple K-means**, per decision — no GMM, no covariance-type sweep, no soft assignment at this stage. This is a targeting tool, and its output is a set of zones to inspect.
- **Input**: the 1 m feature stack, z-scored with a per-site `StandardScaler` (R4; matches `instructions1.md` §3). Use the **full stack including CHM** — this stage runs only at NEON sites and only to place labels, so the transferability constraint on A–C does not apply here.
- **Deliberately over-segment**: choose `k` well above the 4 classes — **k ≈ 15–20**. Over-segmentation is the *goal*. `instructions1.md` §6 found bare fragmenting across 4 clusters at k=10 (varying soil, litter), and that fragmentation is information: it marks distinct appearances of one class that the labeling must cover.
- **`k` is chosen once and does not need optimizing.** No elbow or silhouette sweep — a suboptimal `k` yields slightly redundant zones, which costs a little labeling effort and nothing else.
- **Seed** `BASE_SEED + year` (§12 Q8).

#### Stage 2 — Zone-guided hand labeling

- **Per-cluster interior-pixel sampling** to place candidate labeling sites: 5×5 majority filter to avoid boundary and speckle pixels, matching the protocol proven in `instructions1.md` §3.4. Emit ~20 candidate sites per cluster.
- Written as an **editable GeoPackage** for QGIS, with an empty nullable-integer `class_code` column (§3 codes 0–3), plus `cluster_id` retained as an attribute **for provenance only**.
- The analyst draws polygons at or around these sites on 10 cm RGB, with CHM and SAVI available as reference overlays.
- **Coverage requirement**: every cluster must receive labeled polygons, and the existing per-class minimum still holds — **>= 50 polygons per class per tile role** (train block / test block). Clusters constrain *where* labels are placed; they do not relax *how many* per class are needed.

#### Outputs

- `labeling_zones_{tile}_{YEAR}.gpkg` — candidate sites with `cluster_id`, `class_code` (empty, to be filled)
- `cluster_map_{tile}_{YEAR}.tif` — the k-means map, retained for provenance and the diagnostic below
- `training_polygons_{tile}_{YEAR}.gpkg` — the delivered hand labels, attribute `class_code` (0–3)
- `training_labels_{tile}_{YEAR}.tif` — rasterized to 1 m (uint8, 255 = unlabeled)

#### A free diagnostic worth keeping

After labeling, cross-tabulate `cluster_id` against the assigned `class_code`. A cluster receiving a single class is a clean, separable region of feature space. A cluster receiving several classes is genuinely ambiguous **in the features the classifier will use** — which both predicts where frameworks A–E will struggle and independently corroborates the §3.1 confidence layer. Record the cross-tabulation; it costs nothing beyond labels already collected.

### 4.3 The two random forests — RF-A and RF-B

**These are two different models with different inputs, different targets, and different failure modes. Do not conflate them.** Both understandings below are confirmed and settled.

| | **RF-A** — ground truth | **RF-B** — phenology |
|---|---|---|
| Pipeline step | Step 1d | Step 5 |
| Input features | 1 m spectral / texture / VI / CHM stack (per framework A–E) | 10 PlanetScope LSP timing metrics |
| Spatial unit | 1 m pixel (via SLIC segment) | One Planet pixel (~4 m, native) |
| Model form | **Classifier** — hard class per pixel | **Multi-output regressor** — 4 fractions per pixel |
| Fractional cover obtained by | Aggregating N×N = 16 hard 1 m labels to a Planet block (Step 3) | Predicted directly by the model |
| Training data | §4.2 labels (hand polygons or clustered+annotated) | **All valid Planet blocks — mixed and pure alike** |
| Role of pure end members | n/a | **Anchors and validation only** |

**RF-A — confirmed.** Classify at 1 m, then aggregate 16 × 1 m pixels into one Planet block for fractional cover. Fractions are counts of hard labels, so they are exact by construction. There is no calibration question here.

**RF-B — confirmed.** Planet's native pixel *is* the ~4 m cell, so there is no sub-pixel level to classify and aggregate; the model maps one 4 m feature vector directly to four fractions. Training uses **all pixel types, not only pure ones** — Step 3 supplies true fractions for every valid block across all five tiles, so the regressor sees the full mixing range. **Pure end members serve as anchors and validation**, and as the transfer basis at Step 8 — they are not the training set.

**The trap this avoids**: training RF-B as a classifier on pure end members and reading per-class vote proportions as fractional cover. Such a model has only ever seen fractions of 0 and 1; its vote share on a mixed pixel measures distance to a decision boundary, not mixing proportion. A genuinely 50/50 grass/shrub pixel routinely returns 0.85/0.15.

### 4.4 Transferability rules (binding)

Every step must run unchanged at a non-NEON site given only NAIP RGB+NIR (and optionally offset LiDAR). The rules below are constraints on *how code is written*, not aspirations — a step that violates one is not transferable and must be reworked before it is accepted.

#### R1 — Three distinct kinds of transfer, each scoped to a different distance

| Kind | What moves | **Claimed scope** | Tested at |
|---|---|---|---|
| **Pipeline transfer** | The *method* only. Re-run every step at the new site, retrain everything | **Between ecoregions** — the universal claim | Step 8 Route 2, Axis 2 |
| **Model transfer** | A serialized trained model (`.pkl`) applied to new-site features | **Within an ecoregion only** | Step 8 Route 1, Axis 1 |
| **End-member transfer** | Site-derived pure end members applied to new-site phenology | **Within an ecoregion only** | Step 8 Route 1, Axis 1 |

**This scoping is a design decision, not an empirical finding** (§12 Q4). The pipeline is the portable artifact and must run anywhere; a fitted model is expected to hold only among sites sharing an ecoregion. Across ecoregions, retrain — do not attempt to carry weights or end members, and do not report a cross-ecoregion failure of model transfer as a failure of the method.

Report each kind separately. "The framework transferred" is meaningless without saying which of the three is claimed and over what distance.

**What this obliges**, for the within-ecoregion pairs where model transfer *is* attempted:
- **R2 and R4 become hard requirements, not good practice.** A transferred model reads features by position and name; if `TEXTURE_SCALE` differs between the two sites, or a per-site scaler is missing, the model is silently reading different variables. Feature vectors must match in dimension, order, and definition.
- **Class priors will still differ** (R6). Random forests are sensitive to prior shift, so a transferred model may need recalibration even within an ecoregion. Report per-class recall before and after any adjustment.

> **"Same ecoregion" means "same NEON domain" — RESOLVED (§12 Q4).** The NEON domain is the operative unit throughout this project. Every NEON/AmeriFlux pair in §2.1 sits within one domain by construction, so **all three pairs are model-transfer tests**, and cross-domain work (Axis 2) is pipeline-transfer only.
>
> | Pair | Domain | Transfer kind attempted |
> |---|---|---|
> | **D14** — SRER / WKG | D14 Desert Southwest | Model + end-member (Route 1), then pipeline (Route 2) |
> | **D13** — MOAB / WJS | D13 S. Rockies & Colorado Plateau | Model + end-member (Route 1), then pipeline (Route 2) |
> | **D06** — KONZ / KFB | D06 Prairie Peninsula | Model + end-member (Route 1), then pipeline (Route 2) |
>
> One caveat to record rather than resolve: a NEON domain is a broad climate/vegetation envelope, coarser than an EPA Level III ecoregion, so a within-domain pair may still span a finer ecological boundary — D14's SRER/WKG separation crosses a Sonoran/Madrean/Chihuahuan semi-desert grassland transition. **Record the separation distance and any EPA Level III boundary crossed for each pair** in the run report, so that a Route 1 failure can be interpreted against how far the transfer actually reached. This does not change the design: domains define scope, the record explains outcomes. **D06 remains the cleanest test** — co-located, so a Route 1 failure there is attributable to sensor and resolution alone (§2.1).

#### R2 — Common analysis scale, fixed before feature extraction

NEON RGB is 10 cm; NAIP is 60 cm. **Texture is scale-dependent**: GLCM and LBP computed at 10 cm then aggregated to 1 m are *not the same feature* as the same statistics computed at 60 cm. A model trained on one and applied to the other is reading a different variable under the same column name.

**Rule**: define a single `TEXTURE_SCALE` constant, set to the coarsest resolution across all target sites (**0.6 m**, set by NAIP). At SRER, degrade 10 cm RGB to 0.6 m *before* computing texture. Never compute texture at native NEON resolution for any feature that feeds A–C.

Native 10 cm is retained only for (a) visual inspection and manual labeling, and (b) DeepForest (DL-1), which is inherently resolution-specific and is not claimed to be scale-transferable.

#### R3 — No absolute spectral thresholds in transferable frameworks

NEON vegetation indices come from an atmospherically corrected, BRDF-corrected, narrowband hyperspectral product. NAIP is uncorrected 8-bit broadband DN, mosaicked across dates and sun angles. **SAVI = 0.2 does not mean the same thing in both.** A hard-coded threshold will silently mis-segment at the transfer site.

**Rule**: absolute thresholds (`SAVI_BARE_MAX` etc.) are permitted **only** in the reference/labeling path at SRER (§3, framework D/E). Transferable frameworks A–C must use either:
- percentile-based thresholds computed per site from the site's own distribution, or
- learned decision boundaries from the classifier, with no thresholding step at all.

Every absolute threshold in the codebase must be tagged in config as `reference_only: true` or `transferable: true`. Report both tags at Step 0.

#### R4 — Per-site feature standardization

Any model applied across sites (R1 model transfer) requires features in comparable units. Fit an independent `StandardScaler` per site and persist it, matching the convention already used in `instructions1.md` §3. A model transferred without its scaler is invalid.

#### R5 — Common 1 m analysis grid

0.6 m does not divide evenly into 1 m, and neither 0.6 m nor 0.1 m divides evenly into the Planet pixel. **Rule**: all sites resample to a common 1 m analysis grid as the single canonical intermediate, and the N×N aggregation to Planet (Step 3) always starts from that 1 m grid. Resampling method is fixed and identical across sites: nearest-neighbour for categorical, bilinear for continuous. Record the method in the run report.

#### R6 — Class prior shift is expected

SRER is mesquite savanna; WKG (Kendall) is C4 desert grassland with far less woody cover. A model trained where trees are common will encounter a very different class balance. **Rule**: report the per-class prior at both sites, and evaluate transfer with per-class metrics (macro-F1, per-class recall), never overall accuracy — OA at WKG will be dominated by grass and bare and will look good regardless of whether woody classes work at all.

#### R7 — Acquisition phenology must be checked, not assumed

SRER ground truth is 2022-08, monsoon peak, grasses green. NAIP in Arizona is flown across a range of summer dates and may land **pre-monsoon**, when grass is senesced and grass/bare separation collapses (`instructions1.md` §4 already identifies bare vs. senesced grass as a known degenerate pair). **Rule**: the NAIP acquisition date per tile is a required Step 0 check at any transfer site, and a pre-monsoon acquisition is a documented blocker, not something to work around silently.

#### R8 — Site-specific values live in config, never in code

Planet pixel size, grid origin, UTM zone, tile lists, acquisition dates, class priors, and all thresholds are per-site config entries. Adding a site must be a new config file plus data, with zero code edits. This is the practical test of every rule above.

---

## 5. Processing pipeline

### Step 0 — Data audit and grid definition

**Run the full checklist in §11 before any other step.** No downstream step begins until §11 passes, or until a specific failure is explicitly waived in writing and recorded.

Step 0 finalizes four values that every later step reads from config: the Planet aggregation factor `N`, the Planet grid origin, `TEXTURE_SCALE`, and `H_GRASS_MAX`.

Outputs → `00_qa/`, as both a machine-readable `data_audit_{SITE}_{YEAR}.json` and a human-readable summary.

### Step 1 — Per-pixel classification at 1 m

**1a. Feature construction.** Resample RGB to the 1 m analysis grid (R5) and separately to `TEXTURE_SCALE` = 0.6 m for texture (R2). Compute RGB color indices (ExG, ExGR, VARI, GLI). Load SAVI/NDVI at 1 m. Compute texture at **0.6 m** and aggregate to 1 m: GLCM (contrast, homogeneity, entropy, correlation), LBP, moving-window std dev. Load CHM at 1 m (frameworks D/E only). Native 10 cm RGB is retained only for manual labeling and DL-1.

**1b. Segmentation.** SLIC at 1 m, ~100k segments/tile. Per-segment spectral, texture, shape, and context features.

**1c. Shadow detection** (reinstated from v3, `instructions2.md` Phase 3):
- Rec. 709 luma (`0.2126R + 0.7152G + 0.0722B`) thresholded at the **pooled 20th percentile** — percentile-based per R3, so it transfers — combined with a blue-shift rule (`B > R`). Computed at `TEXTURE_SCALE` (0.6 m) per R2, not at native 10 cm.
- Aggregate to 1 m via > 70% majority.
- **Shadow is its own class.** Resolution rule: shadow within `SHADOW_TREE_RADIUS` (5 m) of CHM >= `H_TREE_MIN` is **assigned to tree**; all remaining shadow is **ignored** (masked to nodata, excluded from training, from Step 3 aggregation denominators, and from accuracy assessment).
- At transferable frameworks (A–C) without CHM, the tree-proximity test uses the framework's own predicted tree mask.

**1d. Classification.** RF over segments, trained on the §4.2 hand labels, leave-one-tile-out CV within the train block, evaluated on the held-out test tiles.

**1e. Soft outputs.** Retain the classifier's per-class probability vector for every pixel and derive the confidence and margin layers per §3.1. **Do not harden and discard the probabilities** — Step 3 soft aggregation and Step 5 sample weighting both depend on them.

Outputs per framework → `01_pixel_classification/{framework}/`:
`classification_{framework}_{tile}_{YEAR}.tif` (uint8, 0–3, 255 nodata),
`class_prob_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands in class-code order),
`confidence_{framework}_{tile}_{YEAR}.tif` (float32, [0,1]),
`margin_{framework}_{tile}_{YEAR}.tif` (float32),
`shadow_mask_{framework}_{tile}_{YEAR}.tif` (uint8).

### Step 2 — PlanetScope QA masking

Apply the PlanetScope QA layers to remove faulty Planet pixels: **`NumCycles == 1` strict (layer 1) AND `QA ∈ {1, 2}` (layer 12)**. Full layer specification and handling traps are in §5.2–5.3 — in particular, **fill (32767) must be masked to NaN and cast to float before any arithmetic**, or derived durations silently evaluate to 0 on fill pixels.

QA-failing Planet cells are excluded from Steps 3–5. Log retention statistics, and log the `numObs` (layer 24) distribution for the surviving pixels.

### Step 3 — Aggregation to PlanetScope scale

**N × N** blocks (N from Step 0, expected 4) **aligned to the PlanetScope pixel grid**, over each framework's 1 m hard classification → percent cover per class per block.

> **"Moving window" in the source documents means grid-aligned, one block per Planet pixel** — the window steps by N, landing exactly on Planet pixel boundaries. It does **not** mean a stride-1 sliding window. Blocks are therefore non-overlapping and stand in one-to-one correspondence with Planet pixels, which is what Step 5 requires: a stride-1 window would produce overlapping, non-independent samples that cannot be matched to Planet pixels.

Shadow-masked and nodata 1 m pixels are excluded from the denominator; blocks with < 75% valid 1 m pixels are dropped.

#### Three fraction estimates per block, all produced

Mixture confidence (§3.1) feeds directly into this step. Compute all three and carry them forward; they answer different questions and their disagreement is itself diagnostic.

Let `i` index the valid 1 m pixels in a block, `c` a class, `p_ic` the soft membership, and `w_i` the pixel confidence.

**(a) Hard count — the area fraction.**
```
f_c = count(label_i == c) / n_valid
```
Unweighted count of plurality labels. This is a true **area fraction** and is the reference estimate: it is what "percent cover" conventionally means, and it is what Step 6's manual labeling validates against.

**(b) Soft mean — the expected fraction.**
```
f_c = Σ_i p_ic / n_valid
```
Averages the membership vectors instead of hardening first. A 1 m pixel genuinely half grass and half shrub contributes 0.5 to each rather than 1.0 to whichever narrowly won. **This is the better-matched target for RF-B**, because sub-pixel mixture is exactly what the phenology model is being asked to predict, and hardening throws that signal away before the model ever sees it.

**(c) Confidence-weighted count.**
```
f_c = Σ_i w_i · [label_i == c] / Σ_i w_i
```
Down-weights ambiguous pixels in the hard count.

> **Caveat on (c), stated so it is not misread**: weighting changes what the number *means*. (a) is the fraction of block area whose plurality class is `c`; (c) is a confidence-weighted quantity that is **no longer strictly an area fraction** and will not agree with a manual area estimate. Use (c) as a diagnostic and sensitivity check, not as the headline product. Where (a) and (b) diverge sharply, the block is mixture-dominated and worth inspecting.

**Block confidence** = mean of `w_i` over valid pixels in the block. Carried forward as a band and used at Step 4 (end-member ranking), Step 5 (regression `sample_weight`), and Step 6 (stratified accuracy).

Outputs → `02_aggregation_mask/`, per framework:
`frac_hard_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands, class-code order),
`frac_soft_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands),
`frac_wtd_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands),
`block_confidence_{framework}_{tile}_{YEAR}.tif` (float32),
`valid_count_{framework}_{tile}_{YEAR}.tif` (uint8).

### Step 4 — Pure end-member identification

Flag blocks with **>= 90%** single-class cover (hard fraction, estimate (a)) as candidate pure end members.

**Purity and confidence are two different things, and both gate selection.** A block that is 95% grass assembled from low-confidence 1 m pixels is a weaker end member than one that is 92% grass at high confidence — the first is a block of ambiguous pixels that happened to fall the same way, the second is genuinely pure grass. Rank candidates by **purity × block confidence**, and record both separately so the trade-off stays visible rather than being buried in a single score.

- Require both `hard_fraction >= 0.90` **and** `block_confidence` above a threshold set from its observed distribution (§11 check #22).
- Expect strong class imbalance: pure bare and pure grass will be abundant; pure tree and pure shrub rare. Record per-class counts; if any class yields fewer than 30 blocks, report it rather than silently lowering either threshold.
- End-member blocks cluster spatially. Use **spatial** (block/tile) holdout, never random holdout, in Step 5.
- Export for manual validation as GeoPackage with class, hard fraction, soft fraction, block confidence, and tile attributes.

Outputs → `03_pure_endmembers/`.

### Step 5 — PlanetScope phenology fractional cover

#### 5.1 Data location, naming, and available years

**Format**: NetCDF (`.nc`), one file per site per year — **not GeoTIFF**. See §5.3 trap 6 for the handling consequences.

**Path template**:
```
{DATA_ROOT}/planet/PLSP_production_nc/{SITE_DIR}/{SITE_ID}_NEON_{SITE_NAME}_PLSP_{YEAR}.nc
```

**SRER example**:
```
.../Dropbox/planet/data/planet/PLSP_production_nc/Santa_Rita_Experimental_Range_NEON/
    US-xSR_NEON_Santa_Rita_Experimental_Range_PLSP_2017.nc
```

| Token | SRER value | Notes |
|---|---|---|
| `DATA_ROOT` | local `.../Dropbox/planet/data`; cluster `/projectnb/modislc/users/fache/data` | Shared with the NEON tree (`{DATA_ROOT}/NEON/...`) — **one root constant serves both** |
| `SITE_DIR` | `Santa_Rita_Experimental_Range_NEON` | **With** the `_NEON` suffix |
| `SITE_ID` | `US-xSR` | AmeriFlux ID, hyphenated |
| `SITE_NAME` | `Santa_Rita_Experimental_Range` | **Without** the `_NEON` suffix |
| `YEAR` | 2017–2021 | See availability below |

> **Naming trap**: the directory carries `_NEON` and the filename does not. `Santa_Rita_Experimental_Range_NEON` vs. `Santa_Rita_Experimental_Range` are two distinct strings in the same path. Derive one from the other in the helpers module — never store both independently, or they will drift.

The path splits cleanly into `{DATA_ROOT}/planet/...` for PLSP and `{DATA_ROOT}/NEON/...` for AOP, so migrating local → cluster is one constant, matching the existing `00_ground_truth_helpers.py` convention (R8).

**Available years**:

| Years | Status |
|---|---|
| **2017–2021** | **Available now.** Development and feature-characterization years (§5.1a item 3). Not used to train RF-B |
| **2022** | **The target year** — PlanetScope prediction and validation, matching the 2022-08 ground truth. **Pending generation; Step 5 is gated on it** |
| **2023–2025** | To be generated per site. Reserved for the §6 multi-year stability work |

#### 5.1a Year alignment — 2022 is the target year (resolves Q1)

**Decision: PLSP 2022 is the year used for PlanetScope prediction and validation**, matching the 2022-08 SRER ground truth exactly. The 2022–2025 products will be generated per site; 2022 is the one Step 5 targets.

> **Explicit dependency**: PLSP 2022 **does not exist yet**. Only 2017–2021 are available today. Every instruction in Step 5 that names 2022 is conditional on that product being generated first. This is a scheduling dependency, not an unresolved methodological choice — the year is decided, the data is pending. Nothing in Step 5's design depends on the interim years, and the interim gap has its own specific use (item 3 below).

Consequences:

1. **Step 5 is gated on the 2022 product.** RF-B training and validation both use PLSP 2022 against 2022 ground truth — same year, no temporal offset, no stationarity assumption required. This is the clean case, and it removes the largest methodological caveat any interim-year substitute would have carried. **Do not begin Step 5 against 2017–2021 as a stand-in**; the earlier options of "use 2021 with a 1-year offset" or "train across five years as replicates" are **withdrawn**, not pending.
2. **Sequencing.** Steps 0–4 (ground truth, aggregation, end members) run to completion on the 2022 NEON flight without needing PLSP at all. Only Step 5 waits. Build and validate the ground-truth side first; the dependency is one-directional.
3. **Use 2017–2021 for feature characterization in the meantime.** Five years of LSP metrics over a landscape whose 2022 cover is known measures **interannual variability of the timing metrics for effectively unchanged cover**. That separates two things the project otherwise cannot distinguish: metrics varying because *cover* changed versus because *climate* varied. The per-pixel spread across 2017–2021 for stable-cover pixels estimates climatic noise in the features, and therefore sets a **floor on achievable RF-B accuracy** before a single 2022 model is fitted. This is characterization, not training — it produces an error bound, not a model. Run it while waiting; it is free information and it calibrates expectations for Step 5.
4. **§6's stability check remains required, with a narrower job.** With prediction and validation both on 2022, it no longer guards a temporal offset. It now answers whether end members and the RF-B model hold across years — the multi-year and cross-site ambitions in §2 and §6 still depend on it. SRER has documented mesquite encroachment, so interannual woody change is real, not hypothetical.

**Also run §2.4 (NumCycles distribution) across all available years**, not just the target year. Monsoon strength varies year to year, so the bimodality question may have a different answer in a wet year than a dry one, and a single-year check could mislead. The 2017–2021 series is directly useful here, and this check does not need to wait for 2022.

#### 5.2 Product specification — PlanetScope LSP metric layers

24 layers, 1-based `product_lyr`. All layers are **Int16** with **fill value 32767**.

| Lyr | Short name | Long name | Units | Scale | Valid min | Valid max | Used |
|---|---|---|---|---|---|---|---|
| 1 | `NumCycles` | Number of phenological cycles | cycles | 1 | 0 | 6 | **QA filter** |
| 2 | `OGI` | Onset Greenness Increase (15% greenness increase) | DOY | 1 | -181 | 548 | **Feature** |
| 3 | `50PCGI` | 50 Percent Greenness Increase (50% increase) | DOY | 1 | -181 | 548 | **Feature** |
| 4 | `OGMx` | Onset Greenness Maximum (90% increase) | DOY | 1 | -181 | 548 | **Feature** |
| 5 | `Peak` | Date of Cycle Peak | DOY | 1 | 1 | 366 | **Feature** |
| 6 | `OGD` | Onset Greenness Decrease (10% decrease) | DOY | 1 | -181 | 548 | **Feature** |
| 7 | `50PCGD` | 50 Percent Greenness Decrease (50% decrease) | DOY | 1 | -181 | 548 | **Feature** |
| 8 | `OGMn` | Onset Greenness Minimum (85% decrease) | DOY | 1 | -181 | 548 | **Feature** |
| 9 | `EVImax` | Maximum EVI2 during cycle | – | 0.0001 | 0 | 10000 | Excluded (amplitude) |
| 10 | `EVIamp` | EVI2 amplitude during cycle | – | 0.0001 | 0 | 10000 | Excluded (amplitude) |
| 11 | `EVIarea` | Integrated EVI2 during cycle | – | 0.01 | 0 | 32766 | Excluded (amplitude) |
| 12 | `QA` | Overall QA for cycle 1 | – | 1 | 1 | 4 | **QA filter** |
| 13–23 | `*_2` | Cycle 2 equivalents of layers 2–12 | | | | | Unused (1-cycle constraint) |
| 24 | `numObs` | Days with clear observations in calendar year | count | 1 | 0 | 366 | **Confidence** |

**Features (10, timing-only)** — per `instructions1.md` §3:
- Raw (7): layers 2–8 — `OGI`, `50PCGI`, `OGMx`, `Peak`, `OGD`, `50PCGD`, `OGMn`
- Derived (3): `DurGU = OGMx − OGI`, `DurGD = OGMn − OGD`, `LOS = OGMn − OGI`

**QA filter** (`instructions5.md` Step 2, per `instructions1.md` §3): `NumCycles == 1` strict **AND** `QA ∈ {1, 2}` (high + medium). QA values 3–4 are low/poor and are dropped. This is the sole validity filter; no separate downstream NaN-dropping step.

#### 5.3 Product handling — five traps

These follow directly from the layer specification above and must be handled in this order.

1. **Int16 overflow on derived features — the dangerous one.** Fill is 32767, which is `Int16` max. Computing `DurGU = OGMx − OGI` on raw arrays yields `32767 − 32767 = 0` for fill pixels: a plausible-looking zero-day duration that passes every range check. **Mask fill to NaN and cast to float *before* any arithmetic.** Never compute derived features on the raw integer arrays.

2. **Timing metrics are extended DOY, not circular.** Valid range is **−181 to 548**, so a cycle may legitimately start in the previous calendar year (negative DOY) or end in the next (> 366). Two consequences: (a) **no circular statistics are needed** — treat these as ordinary continuous variables; (b) **a `value < 0` filter would silently discard valid early-onset pixels.** Filter on the fill value only, then range-check against the per-layer bounds in the table.

3. **`Peak` has a different valid range from every other timing layer** (1–366, not −181–548). It cannot cross a year boundary while its neighbours can. Range-check per layer, never with one shared bound, and be aware that a cycle spanning the year boundary will pair an out-of-calendar `OGI` with an in-calendar `Peak`.

4. **Scale factors are not uniform.** Timing layers are scale 1; the EVI layers are 0.0001 or 0.01. Since the design is timing-only, no scaling applies to any feature actually used — but **do not blanket-apply a single scale factor** across the stack, and re-check this if the amplitude layers are ever reinstated (see 5.3).

5. **Cycle-2 layers (13–23) are unused by construction.** The `NumCycles == 1` filter guarantees any surviving pixel has cycle-2 layers at fill. Do not read them; if they are read, assert they are fill as a QA check.

6. **NetCDF, not GeoTIFF — the read path is different.** Use `xarray`/`netCDF4`, not `rasterio`. Three specific consequences:
   - **Georeferencing is not guaranteed.** NetCDF carries CRS and grid via CF conventions (a `crs`/`spatial_ref` variable plus 1-D `x`/`y` coordinate arrays), not a GeoTIFF geotransform. Extract CRS and affine explicitly and verify against the NEON tiles; do not assume either is present.
   - **Layers may be named variables rather than a band axis.** This is an improvement — variable names should match the `short_name` column in §5.2, removing the band-order risk of check #30. **Verify the names match** rather than assuming positional order; if the file instead stores a single array with a layer dimension, positional order applies and check #30 stands in full.
   - **`xarray` may auto-apply `_FillValue` and `scale_factor`** from CF attributes on read (`mask_and_scale`, default `True`). This is convenient but must be **verified, not assumed** — if it fires, trap 1 is already handled; if it does not, fill arrives as raw 32767 and trap 1 is live. Check which, and record it. Applying scaling twice is equally damaging.

**Additional confidence filter — `numObs` (layer 24).** Not currently in the `instructions1.md` pipeline. A pixel with few clear observations produces poorly constrained curve fits and therefore unreliable timing metrics, even at QA 1–2. **Record the `numObs` distribution at Step 0 and evaluate a minimum-observation threshold**; carry `numObs` through as a per-pixel confidence attribute regardless of whether it becomes a hard filter.

> **Two confidences — data quality vs. prediction quality. Keep them separate.**
>
> | | **Data quality** | **Prediction quality** |
> |---|---|---|
> | Layers | `numObs` (24), `QA` (12), `NumCycles` (1) | Block confidence (§3.1) |
> | Side of the model | **X** — the phenology features | **y** — the ground-truth fractions |
> | Asks | *Was the observation good enough to fit a curve?* | *Was the classification confident enough to trust the fraction?* |
> | Fails when | Cloud, sparse revisit, poor curve fit | Genuine sub-pixel mixture, spectrally ambiguous cover |
> | Config name | `data_quality` | `prediction_quality` |
>
> These are different quantities that fail for unrelated reasons: a perfectly observed pixel can be hopelessly mixed, and a pure grass block can sit under a cloud-starved time series. **Never blend them into one score** — a single number cannot say which side is weak, which is exactly the diagnosis needed when RF-B underperforms. Keep the two names distinct in config, in the report, and in every output attribute table.
>
> If both enter the RF-B `sample_weight`, combine them as an explicit product of the two named terms and persist each factor separately alongside the product, so the contribution of each stays recoverable after the fact.

#### 5.4 Amplitude layers — available but excluded by design

`EVImax`, `EVIamp`, and `EVIarea` (layers 9–11) are present in the product and deliberately **not** used: the project's core hypothesis is that **timing alone** suffices (`instructions1.md` §1). They are noted here because they are the obvious ablation. If timing-only performance is inadequate at Step 6, adding them is the first fallback — but that is a **change to the research question**, not a tuning step, and must be recorded as such.

#### 5.5 Model form — multi-output regression, not classification

The ground-truth RF (Step 1) classifies at 1 m and aggregates 16 pixels to get fractions; that is a valid count. The phenology model is different: **Planet's native pixel *is* the ~4 m cell**, so there is no sub-pixel level to classify and aggregate. The model must map one 4 m feature vector directly to four fractions.

Do **not** train a classifier on pure end members and read per-class vote proportions as fractional cover. A classifier trained only on pure pixels has seen fractions of 0 and 1 only; its vote share on a mixed pixel measures distance to a decision boundary, not mixing proportion. A genuinely 50/50 grass/shrub pixel routinely returns 0.85/0.15.

Instead:
- **X** = 10 timing metrics per Planet pixel. **y** = the Step 3 fractions — use the **soft mean, estimate (b)**, as the primary target, since sub-pixel mixture is precisely what the model is being asked to predict and the hard count discards it. Fit against the hard count (a) as well and report both; a large gap indicates the model is limited by label hardening rather than by the features.
- **`sample_weight` = `prediction_quality`** (block confidence, §3.1 and Step 3). Blocks whose targets were assembled from ambiguous 1 m pixels carry noisier `y` and should influence the fit less. This is the main reason the confidence layer is propagated this far. If `data_quality` is also folded in, keep it a separate named factor per §5.3 — never a single blended score.
- Train a **multi-output RF regressor on all valid blocks**, mixed and pure alike — Step 3 supplies true fractions across all five tiles, so no synthetic mixtures are needed at SRER.
- Pure end members (Step 4) serve as **anchors, validation, and the WKG transfer basis** — not as the sole training set.
- Constrain or post-normalize predictions to sum to 1 and lie in [0, 1].
- Cross-check the seasonal signal against the phenocam in the train block.

Outputs → `04_rf_phenology/`.

### Step 6 — Accuracy assessment

**Governing protocol**: Olofsson, Foody, Herold, Stehman, Woodcock & Wulder (2014), *Good practices for estimating area and assessing accuracy of land change*, Remote Sensing of Environment 148:42–57 — https://www.sciencedirect.com/science/article/pii/S0034425714000704

This is not a loose reference. **Accuracy is estimated area-weighted, and reported per land-cover class with confidence intervals**, following the estimators below. Raw sample counts are never reported as accuracy.

#### 6.1 Why area weighting is mandatory here

A stratified sample over-samples rare classes by design. At SRER, tree and shrub are rare and bare and grass are abundant (§11 check #20), so raw counts from a stratified sample **do not** reflect the map's actual composition. Computing accuracy from those counts silently assumes equal class areas and produces a number that describes the sample rather than the landscape. Every estimate below therefore reweights sample counts by the map-area proportion of the stratum they came from.

#### 6.2 Sampling design

**Strata = map class × confidence bin** (§3.1). Crossing the two satisfies two requirements at once: Olofsson's requirement that strata be defined on the map being assessed, and the §3.1 requirement that low-confidence pixels be populated well enough to validate the confidence layer. Record `W_h`, the area proportion of every stratum `h`, from the full map — these are the weights that make the estimators area-weighted.

**Sample size**, replacing the earlier "X per class, TBD":

```
n = ( Σ_h W_h · S_h / S(Ô) )²        S_h = sqrt( U_h · (1 − U_h) )
```

where `U_h` is a conjectured user's accuracy for stratum `h` and `S(Ô)` the target standard error of overall accuracy. Use `S(Ô) = 0.01`, and conjecture `U_h` ≈ 0.6–0.7 for shrub and for low-confidence bins, ≈ 0.8–0.9 for bare and grass at high confidence — deliberately pessimistic, since underestimating `U` only oversamples.

**Allocation**: proportional to `W_h`, then raised to a **floor of 50–100 samples in any rare stratum** (Olofsson §5.1.2). Without that floor, tree and shrub confidence intervals will be too wide to separate frameworks — which is the entire purpose of this step.

**Single shared sample set**, drawn **once** and evaluated by every framework, so comparisons are direct.

> **A tension to handle explicitly.** Olofsson stratifies on the map being assessed, but frameworks A–E produce *different* maps, so a per-framework stratification would give each framework a different sample and destroy comparability. Resolution: stratify **once** on the reference framework (D/E, §4.1), draw a probability sample, and **record every inclusion probability**. The estimators below remain unbiased for every framework, because the sample is a valid probability sample of the whole area regardless of which map defines the strata. The cost is precision, not bias: for a framework whose rare-class map disagrees sharply with the stratification, rare-class intervals will be wider. Where that happens, augment per-framework and **recompute the weights** — never merge an augmented sample in at equal weight.

**Two assessment levels**, same protocol:
1. 1 m pixel level, per class.
2. N × N m pure end-member blocks, per class.

**Response design**: manual labeling by the user on 10 cm RGB, against the §3 class definitions. Record the labeling protocol and any ambiguous-case rules — the reference labels are the measurement instrument, and an undocumented protocol is not reproducible.

#### 6.3 Estimators

Build the error matrix in **estimated area proportions**, not counts:

```
p̂_ij = W_i · n_ij / n_i·
```

`n_ij` = samples of map class `i` whose reference label is `j`; `n_i·` = total sampled in stratum `i`.

| Quantity | Estimator |
|---|---|
| Overall accuracy | `Ô = Σ_i p̂_ii` |
| User's accuracy (class `i`) | `Û_i = p̂_ii / p̂_i·` — commission |
| Producer's accuracy (class `j`) | `P̂_j = p̂_jj / p̂_·j` — omission |
| **Error-adjusted area** (class `j`) | `Â_j = A_total · p̂_·j` |

**Report 95% confidence intervals (±1.96·SE) on every one of these**, using the stratified standard-error formulas in Olofsson §§4.3–4.4. An accuracy figure without an interval is not reportable here.

**The error-adjusted area `Â_j` is a deliverable in its own right**, not a diagnostic. It will differ from the naive pixel count of class `j`, and the difference *is* the map's bias for that class. For a product whose purpose is ground truth, that number matters as much as the accuracy figures.

#### 6.4 Framework selection

Select the best-performing **transferable** framework (A–C, DL) per §4.1 on **per-class user's and producer's accuracy with intervals**, not on overall accuracy.

Overall accuracy at SRER is dominated by bare and grass — the abundant, easy classes — and will barely move on shrub, historically the weakest (`instructions1.md` §6: 40–70% purity). A framework can gain overall accuracy while getting worse at the classes the project exists to separate.

- **Primary criterion**: producer's and user's accuracy for **shrub** and for **grass**, the pair that carries the scientific difficulty.
- **Secondary**: the remaining per-class figures, then overall accuracy.
- **Report intervals and honour them.** With realistic sample sizes, several frameworks will likely be statistically indistinguishable. Say so when it happens and select on transferability or simplicity instead — do not rank frameworks on differences smaller than their confidence intervals.

#### 6.5 Confidence-layer validation

Because strata already cross class with confidence bin, per-bin accuracy falls out of §6.3 directly.

- **Report accuracy per confidence bin**, not only pooled. Pooled accuracy conflates two very different failures: being wrong on genuinely ambiguous pixels is reasonable behaviour, being wrong on high-confidence pixels is a broken model.
- **Accuracy must rise monotonically with confidence.** If it does not, the §3.1 confidence layer is not measuring what it claims, and every downstream use — end-member ranking (Step 4), RF-B `sample_weight` (Step 5) — is unsound. Plot accuracy against confidence bin and check the trend before relying on it. This is the check that makes the confidence design falsifiable.

#### 6.6 Scope limit

This protocol governs **categorical** accuracy: the 1 m classification and the block-level end-member classification. It does **not** apply to RF-B's continuous fractional-cover predictions, which need regression metrics — see §12 Q3 (per-class RMSE, MAE, bias, 1:1 scatter). Do not force fractional output into an error matrix.

Outputs → `05_accuracy_assessment/`: the sample set with inclusion probabilities and stratum weights, reference labels, the area-proportion error matrix per framework, all estimators with 95% intervals, error-adjusted areas, and the accuracy-vs-confidence plot.

### Step 7 — RAP comparison

**Pending**: RAP data path. Compare RAP (10 m fractional cover) vs. ground truth (1 m aggregated to the RAP grid) vs. the Planet-scale mask (Step 3). Aggregation to 10 m must align to the RAP grid, same rule as Step 0.

### Step 8 — Transferability to WKG

- **Phase 1**: framework development on SRER train tiles, validated on SRER test tiles.
- **Phase 2**: transfer to WKG. Two routes, tested in order. SRER and WKG share NEON domain D14, so **Route 1 is in scope** (R1).

  **Route 1 — transfer the model and end members directly.** Apply the SRER-trained model and SRER-derived pure end members to WKG features and phenology, without rebuilding ground truth. This is the **within-ecoregion model-transfer claim** (R1) and the ideal outcome. Requires the persisted per-site scaler (R4) and identical feature definitions (R2); check per-class recall for prior shift (R6) before concluding anything.

  **Route 2 — build WKG its own ground truth layer** by running the transferable pipeline (frameworks A–C) on **NAIP imagery**. NAIP is 0.6 m RGB+NIR, so vegetation indices and texture are both computable and Steps 1–4 run unchanged; this yields native WKG fractional-cover blocks and therefore native WKG training data for RF-B.

  Route 2 is the **expected path across an ecoregion boundary** and the fallback if Route 1 fails within one; it is the reason frameworks A–C are constrained to RGB + VI + texture in §4.1. **It also removes the need for synthetic mixtures at WKG** (see §9 risk 1): with its own ground truth layer, WKG has true mixed-pixel fractions and RF-B can be trained there exactly as at SRER.

  Constraints at WKG: NAIP 0.6 m RGB+NIR; LiDAR offset 1–2 years, so CHM is a cross-check at best and frameworks D/E are not carried over.

  Working assumption: sites outside a shared ecoregion require independent end-member sets. Comparing Route 1 against Route 2 is the direct test of that assumption.

- **Phase 3 — extend across the network (§2).** SRER → WKG is the D14 pair only, and tests **Axis 1** (within-domain, sensor transfer). Repeating the same two routes at D13 (MOAB → WJS) and D06 (KONZ → KFB) completes Axis 1. **D06 is the cleanest model-transfer test in the network** — co-located, certainly within one ecoregion, so a Route 1 failure there is attributable to sensor and resolution alone (§2.1).

  Comparing *between* domains is **Axis 2**, and by R1 that is a **pipeline-transfer test only**: retrain per ecoregion, and do not attempt to carry models or end members across. Axis 2 additionally requires the climate-anchored features described in §2.3, which are not yet implemented. Do not attempt it on raw DOY features — the phenological calendars are offset by months for climatic rather than compositional reasons, and a cross-domain end-member comparison on raw DOY would measure climate, not composition.

---

## 6. Multi-year operation and stability

Ground truth may be produced for **several years** where NEON flight data is available.

**Required check**: a **year-to-year stability assessment** of both the 1 m classifications and the derived pure end members. Shrub encroachment, grass mortality, and interannual monsoon variability all move these. Specifically:
- Per-class area change between years over the same tiles.
- Persistence rate of pure end-member blocks across years (fraction still pure in year *t+1*).
- Whether the Step 5 regressor trained on year *t* holds on year *t+1*.

Until this is run, the pipeline carries an explicit assumption: **2022-08 ground truth is applied to LSP year(s) to be named**, and that assumption must be recorded in the run report.

---

## 7. Standing conventions

Carried forward from `instructions2.md`:

- All functions require explicit pydoc (purpose / inputs / outputs).
- **No optional or default arguments** — all parameters passed positionally at call sites.
- No line-wrapping at 80 characters; long lines kept as single lines.
- No leading whitespace in `print()` statements.
- Alternate-method outputs insert the method name before the tile ID: `tree_mask_deepforest_{tile}_{YEAR}.tif`.
- Each phase notebook is paired with a `.md` documenting purpose, tunable parameters (with defaults and tuning guidance), and full output descriptions.
- **Confirm understanding and provide a step list before writing any code; wait for explicit confirmation.**
- Shared config and paths live in a single helpers module (`TILE_IDS`, `YEAR`, `SITE_ID`, `OUTPUT_DIR`, `build_paths(tile_id)`). Add `build_plsp_path(site, year)` per §5.1, deriving the no-suffix `SITE_NAME` from `SITE_DIR` rather than storing both.
- Tile-by-tile processing for parallelization.
- **Per-site JSON config**; every site-varying value lives there, never inline in code (R8).
- **Run provenance**: each run gets its own results directory and a single `report.json` to which every step appends a named section — the convention proven in `instructions1.md` §5. Record resampling methods, thresholds actually used, seeds, and every §11 observed value.
- **`data_quality` and `prediction_quality` are distinct, reserved names** (§3.1, §5.3). Never merge them into a generic `confidence` or `quality` field in any raster, attribute table, config key, or report section.
- **Seeds**: `base_seed + year`, matching `instructions1.md` §3, applied to SLIC, RF, and all sampling.

---

## 8. Prerequisites and known blockers

Carried from `instructions2.md` §§6–7 — **DL tracks are gated on both**:

1. **GPU on SCC.** `torch.cuda.is_available()` returned `False` on a compatible P100 despite a CUDA-enabled build. Open hypothesis: the `cuda/13.2` module's `LD_LIBRARY_PATH` shadows the real driver with a compile-time stub `libcuda.so` from the toolkit's own `lib64`. Check pending. Request line that lands correct hardware: `qrsh -l gpus=1 -l gpu_c=6.0 -pe omp 4`.
2. **GDAL internal tiling.** `predict_tile(dataloader_strategy='window')` requires internally tiled GeoTIFFs; NEON RGB is likely strip-organized. Preferred fix: `gdal_translate -of GTiff -co TILED=YES` once, cache tiled copies. Faster unblock for testing: `dataloader_strategy='batch'`.

---

## 9. Known risks

1. **Timing metrics do not mix linearly.** A 50/50 grass/shrub pixel's SOS is not the mean of the two SOS values — the upstream curve-fitting is nonlinear. Linear mixing is a much weaker assumption for timing metrics than for reflectance, which is what the Okujeni/Kowalski precedent assumed. **Largely mitigated** by Step 8 Route 2: building WKG its own ground truth layer from NAIP gives true mixed-pixel fractions there, so synthetic linear mixtures are never required. The risk only returns if Route 2 proves infeasible at some future site — in which case validate synthetic mixing against the SRER regressor (where true mixed labels exist) before relying on it.
2. **Step 5 is gated on delivery of the PLSP 2022 product, which does not yet exist.** Prediction and validation both target 2022 to match the ground-truth flight year (§5.1a), so Step 5 cannot start until that product is generated. Mitigated by sequencing: Steps 0–4 are independent of PLSP and proceed in parallel. The residual risk is schedule, not method — but it is a real dependency and should be tracked as one.
3. **Shrub is the historically worst class.** `instructions1.md` §6 found shrub the most confused class (40–70% purity), attributed to evergreen/deciduous heterogeneity. Expect it to be the accuracy floor.
4. **Grass vs. bare is date-dependent.** The 2022-08 monsoon-peak acquisition is favorable; a dry-season acquisition would collapse this separation.
5. **Manual labeling has its own error rate**, particularly shrub vs. grass at 10 cm. Consider double-labeling a subset to estimate labeler consistency.

---

## 10. Open items

| Item | Status |
|---|---|
| PlanetScope phenology data path | **Resolved — §5.1**. NetCDF; 2017–2021 available now |
| Which PLSP year for Step 5 | **Resolved — 2022** (§5.1a), matching the ground-truth flight year |
| PLSP 2022 product generation | **Pending — gates Step 5 only.** Steps 0–4 proceed independently. Track as a schedule dependency |
| `numObs` minimum-observation threshold | TBD — evaluate at Step 0 check #35 |
| RAP data path | Pending |
| Accuracy assessment sample size | **Resolved — §6.2**. Computed from the Olofsson formula with `S(Ô) = 0.01`; floor of 50–100 in rare strata. Conjectured `U_h` values still to be fixed |
| SAVI/EVI filename convention | Assumed to mirror NDVI; verify against actual files |
| UTM zone | Assumed standard NEON zone for SRER; verify against file CRS |
| WKG data paths | Needed at Step 8 |
| `NumCycles == 1` viability at D13 (and all sites) | **Run §2.4 first** — decides whether the 1-cycle constraint holds |
| NEON AOP flight years per site; NAIP and 3DEP years per AmeriFlux site | Needed — see §2.4 |
| Climate-anchored features for cross-domain (Axis 2) comparison | Designed in `instructions1.md` §8, not implemented |
| Presence of all 5 tiles for all 3 products | Verify at Step 0 |
| `H_GRASS_MAX` final value | Provisional 0.3 m; set from measured CHM noise floor at Step 0 |
| PlanetScope pixel size and grid origin | Read from product file at Step 0 |

---

## 11. Data checks (Step 0 checklist)

Every check writes a pass/fail plus the observed value into `00_qa/data_audit_{SITE}_{YEAR}.json`. Checks marked **BLOCKER** stop the pipeline; others are recorded and reviewed.

### 11.1 Inventory

| # | Check | Fail condition |
|---|---|---|
| 1 | All 5 tiles present for all 3 products (15 files/dirs) | **BLOCKER** if any missing |
| 2 | SAVI/EVI filenames match the NDVI pattern | **BLOCKER** — assumption from `instructions4.md`, never verified |
| 3 | RGB tile dimensions = 10000 × 10000 at 10 cm (1 km) | Report; partial tiles are usable but must be flagged |
| 4 | VI and CHM tile dimensions = 1000 × 1000 at 1 m | Report |
| 5 | No duplicate or overlapping tile footprints | Report |

### 11.2 Georeferencing and alignment

| # | Check | Fail condition |
|---|---|---|
| 6 | CRS identical across all products and all tiles; UTM zone recorded explicitly | **BLOCKER** on mismatch |
| 7 | VI and CHM 1 m grids share an identical origin (no half-pixel offset) | **BLOCKER** — a silent half-pixel shift corrupts every class rule in §3 |
| 8 | RGB 10 cm grid nests exactly within the 1 m grid | **BLOCKER** |
| 9 | Coregistration test: pick a hard edge (road, building, large crown), cross-correlate RGB against CHM and against SAVI, report offset in metres | Report; > 1 m offset is a **BLOCKER** |
| 10 | Planet LSP raster CRS matches, and its footprint covers all 5 tiles | **BLOCKER** |
| 11 | Planet pixel size read from the product file — **never hard-coded** as 3, 3.7, or 4 m. Record it; derive `N = round(planet_pixel_size / 1.0)` | **BLOCKER** if `N` is non-integer within tolerance |
| 12 | Planet grid origin recorded; confirm 1 m blocks tile it exactly with no fractional remainder | **BLOCKER** |

### 11.3 Radiometry and value sanity

| # | Check | Fail condition |
|---|---|---|
| 13 | **VI scale factor.** NEON VI rasters may be stored as scaled integers. Read dtype and any scale/offset metadata; confirm SAVI/NDVI land in [-1, 1] after scaling | **BLOCKER** — an unapplied scale factor silently breaks `SAVI_BARE_MAX` = 0.2 with no error |
| 14 | RGB band count (3 vs 4) and dtype (uint8 vs uint16); identify any alpha band | **BLOCKER** if unexpected — an alpha band read as blue corrupts the shadow rule |
| 15 | Nodata / fill value recorded per product and confirmed actually applied (not left as a raw sentinel such as -9999 or 65535 entering arithmetic) | **BLOCKER** |
| 16 | CHM range: no negatives, and max plausible for SRER (values > ~15 m are suspect) | Report |
| 17 | CHM noise floor: distribution over visually-confirmed bare areas; set `H_GRASS_MAX` above it (§3 safeguard 2) | Report — sets a config value |
| 18 | Per-tile percentage of nodata in the VI mosaic — bidirectional mosaics carry flightline gaps | Report; > 5% flags the tile |
| 19 | Visual scan for cloud, cloud shadow, and mosaic seams in RGB | Report |

### 11.4 Class and sampling sanity

| # | Check | Fail condition |
|---|---|---|
| 19a | All class-valued rasters use the locked §3 codes (0–3 terminal, 4 shadow intermediate only, 255 nodata); assert no value outside that set, and no 4 in any final product | **BLOCKER** |
| 20 | Per-class pixel counts under the §3 reference rules, per tile | Report — establishes the class prior for R6 |
| 21 | Both train and test blocks contain all four classes | **BLOCKER** if a class is absent from either |
| 22 | Expected pure end-member counts per class at the 90% threshold (§5 Step 4) | Report; < 30 for any class is flagged, not silently worked around |
| 22a | Distribution of the 1 m confidence layer and of block confidence (§3.1), per class and per tile — sets the Step 4 confidence threshold | Report — sets a config value |
| 22b | Agreement between hard (a) and soft (b) fraction estimates per block (§5 Step 3); map the disagreement | Report — large systematic divergence means the site is mixture-dominated at 1 m and Step 5 targets should be revisited |
| 23 | Phenocam location falls inside the 511000 train block; record its footprint | Report |

### 11.5 Cross-site checks (run at every transfer site)

| # | Check | Fail condition |
|---|---|---|
| 24 | NAIP acquisition date per tile — pre- vs. post-monsoon (R7) | **BLOCKER** if pre-monsoon |
| 25 | NAIP band order and NIR band position confirmed (not assumed) | **BLOCKER** |
| 26 | NAIP native resolution confirmed as 0.6 m; if a site is coarser, `TEXTURE_SCALE` must be raised globally and SRER features recomputed (R2) | **BLOCKER** |
| 27 | LiDAR acquisition year vs. imagery year; record the offset | Report — governs whether CHM is usable as a cross-check |
| 28 | Site VI distributions compared against SRER (histogram overlap), to quantify the R3 threshold-transfer problem | Report |
| 29 | Class prior at the transfer site vs. SRER (R6) | Report |

### 11.6 PlanetScope LSP product checks

Run against the layer specification in §5.2.

| # | Check | Fail condition |
|---|---|---|
| 30 | 24 layers present, and **variable names match the `short_name` column** in §5.2 (or, if stored positionally, band order matches `product_lyr`) | **BLOCKER** — silently reading `OGD` as `OGMx` produces plausible but wrong durations |
| 30a | NetCDF CRS and affine extracted from CF metadata and verified against the NEON tiles (§5.3 trap 6) | **BLOCKER** |
| 30b | Whether `xarray` auto-applied `_FillValue`/`scale_factor` on read — determined and recorded, not assumed | **BLOCKER** — governs whether trap 1 is already handled or still live |
| 30c | PLSP file present for every site × year 2017–2021; filename tokens parse per §5.1 | Report |
| 31 | Dtype is Int16 and fill is 32767 as documented | **BLOCKER** |
| 32 | Fill masked to NaN and cast to float **before** derived-feature arithmetic; assert no derived duration equals exactly 0 from a fill pair (§5.3 trap 1) | **BLOCKER** |
| 33 | Per-layer range check against §5.2 bounds — `Peak` at 1–366, other timing layers at −181–548. Confirm no filter drops negative DOY (§5.3 traps 2–3) | **BLOCKER** |
| 34 | After `NumCycles == 1`, assert cycle-2 layers (13–23) are all fill | Report — a failure means the QA filter is not doing what is assumed |
| 35 | `numObs` (layer 24) distribution over QA-passing pixels; evaluate a minimum-observation threshold | Report — sets a config value |

### 11.7 Tile-edge handling

Texture windows, SLIC segments, and the shadow-proximity test all run off tile edges. **Decide once and apply everywhere**: read a buffer from neighbouring tiles where available, and where not (site boundary) mask an edge margin equal to the largest window radius used. Record the margin. Do not leave edge behaviour implicit — it is a common source of seam artifacts in tiled mosaics.

---

## 12. Open questions to resolve

Distinct from §10 (missing inputs); these are decisions not yet made.

### Q1 — Which PlanetScope year(s)? — **RESOLVED: 2022**
**PLSP 2022 is used for PlanetScope prediction and validation**, matching the 2022-08 SRER ground truth exactly — same year, no temporal offset, no stationarity assumption. 2022–2025 will be generated per site; 2023–2025 are reserved for the §6 multi-year stability work. Full detail in §5.1a.

**The year is decided; the data is pending.** PLSP 2022 does not exist yet — only 2017–2021 are available today — so Step 5 carries a scheduling dependency on that product being generated. The earlier alternatives (use 2021 with a 1-year offset, or train across five years as replicates) are **withdrawn**, not still open.

Consequence for sequencing: **Steps 0–4 do not depend on PLSP at all** and should be built and validated first. Use the available 2017–2021 series in the meantime for the feature-variability characterization in §5.1a item 3, which bounds achievable RF-B accuracy before any 2022 model is fitted.

### Q2 — What metric selects the "best" framework at Step 6? — **RESOLVED**
**Area-weighted accuracy, per land-cover class, following Olofsson et al. (2014) good practices.** Full protocol in **§6.1–6.6**, including the estimators, the sample-size formula, and the stratification design.

Selection is on **per-class user's and producer's accuracy with 95% confidence intervals**, with shrub and grass as the primary criterion — not on overall accuracy, which is dominated by the abundant easy classes and would reward a framework that improves on bare and grass while getting worse at the classes the project exists to separate. Where frameworks are statistically indistinguishable, say so and select on transferability or simplicity rather than ranking on differences smaller than the intervals.

This supersedes the earlier macro-F1 recommendation. Macro-F1 weights classes equally, which is a reasonable heuristic but is not area-weighted and yields no confidence intervals or error-adjusted area estimates.

### Q3 — How is RF-B evaluated?
Fractional-cover regression needs different metrics from classification. Recommend per-class RMSE and MAE on held-out blocks, plus systematic bias (mean signed error) per class, plus a 1:1 scatter of predicted vs. true fraction. Overall R² alone will hide class-specific failure.

### Q4 — Does the model transfer, or only the pipeline? — **RESOLVED: both, at different distances**

- **Model transfer** (and end-member transfer) is claimed **within an ecoregion**, and is the ideal outcome for a paired site.
- **Pipeline transfer** is claimed **between ecoregions**. Across an ecoregion boundary, retrain — carrying weights or end members is out of scope by design, not a failure to be diagnosed.

Full statement in **R1** (§4.4), including what this obliges: per-site scaler persistence (R4) and identical feature definitions (R2) become hard requirements wherever model transfer is attempted, and class-prior shift (R6) may still require recalibration even within an ecoregion.

**"Ecoregion" is defined as the NEON domain.** All three §2.1 pairs sit within one domain, so **all three attempt model transfer**; only cross-domain comparison (Axis 2) is pipeline-only.

**Consequence for effort**: model serialization and per-site scaler persistence are load-bearing for the within-domain claim at every pair, so build them properly rather than treating them as optional. See the caveat in R1 about recording pair separation distance — it does not change scope, but it is what lets a Route 1 failure be interpreted.

### Q5 — Are class codes locked across projects? — **RESOLVED: yes, locked**
Confirmed as a **locked cross-project convention**, matching `instructions1.md` §3.4's `naip_class`. See §3 for the authoritative table. Outputs are interoperable across v4, the LSP clustering project, and any future site without a lookup table. **Do not renumber**, and do not introduce per-framework or per-site variants.

### Q6 — Is `mixed` ever a reportable output? — **RESOLVED: yes, as a confidence layer**
Mixture is **not** a flag and **not** a class. It is a **continuous confidence layer** — low mixture means high purity and strong class resemblance; near-even mixtures mean low confidence in the predicted cover. Full specification in **§3.1**, with downstream use in Step 3 (soft and weighted aggregation), Step 4 (end-member ranking), Step 5 (RF-B `sample_weight`), and Step 6 (stratified accuracy).

This also anticipates the `instructions1.md` §6 finding that ~25% of pixels landed in "mixed" in the clustering work: a graded confidence layer degrades gracefully where a boolean flag would simply discard a quarter of the raster.

**Remaining sub-question**: whether the confidence layer is *published* as a standalone product alongside the classification, or kept as an internal quality band. Recommend publishing it — a fractional-cover map without an uncertainty layer is hard to use, and it is already computed.

### Q7 — Label sources: compared before or after framework selection? — **RESOLVED: neither, the question dissolves**

**Hand-labeled classes are the ground truth.** Unsupervised clustering is not a competing label source, so there is no comparison to schedule and no label-source × framework matrix. Clustering runs *first*, as a simple k-means whose only job is to identify hand-labeling zones; the analyst then labels within them. Full design in **§4.2**.

This is a better arrangement than the comparison it replaces: it removes a whole axis from the run matrix, keeps the ground truth fully human-determined, and still extracts the real value clustering offered — coverage of the feature space, so labeling is not biased toward the visually obvious. The cluster ID never becomes a label.

### Q8 — Random seed convention — **RESOLVED**
**`seed = BASE_SEED + year`, with `BASE_SEED = 6`.** Applied to SLIC, RF (both RF-A and RF-B), stratified sampling, and every other stochastic step.

- **Varies by year** — each year is independently reproducible rather than sharing a single draw.
- **Constant across sites** — the same year uses the same seed at every site in the network (§2). A difference between SRER and WKG results is therefore never attributable to the seed, which is what keeps the Axis-1 transfer comparison clean. This does not mean identical *selections* across sites: footprints and valid-pixel counts differ, so the same seed yields different samples. It removes the seed as a confounder, not the sampling variability.
- **Never hard-code the derived value.** Store `BASE_SEED = 6` in config and compute `BASE_SEED + year` at the point of use, so the convention stays visible and any year added later follows it automatically.
- Record the resolved seed for every stochastic step in `report.json` (§7).

Matches the `base_seed + year` convention already used in `instructions1.md` §3.

### Q9 — Is DeepForest still in scope? — **RESOLVED: yes, as a tree-crown validation backup**

DeepForest (`weecology/deepforest-tree`) stays as **DL-1**, repositioned: it is **not** a competing framework for the 4-class map, but an **independent cross-check on the tree class**. Its value is precisely that it was trained on NEON RGB — it is an outside opinion on crowns, derived from the same imagery type but a completely different method from the CHM/SLIC/RF path.

**Role**: run at NEON sites on native 10 cm RGB; compare crown count, crown area, and spatial pattern against the Step 1 tree class and the CHM-derived crowns. Agreement raises confidence in the tree class; systematic disagreement localizes where the tree detection is wrong. It contributes to Step 6 as a **tree-class-only** comparison, not as an A–E framework.

Still gated on the GPU blocker (§8), and still the one method exempted from the R2 scale rule.

#### Q9a — Can DeepForest run on NAIP? — the honest answer is "yes, degraded, and it must be measured first"

This matters because the tree-crown backup is most useful exactly where CHM is weakest — the AmeriFlux sites.

**The resolution problem.** DeepForest was trained on NEON RGB at ~10 cm. NAIP is 0.6 m — **6× coarser**. The model is sensitive to ground sample distance, because a RetinaNet learns crowns at a characteristic pixel size. Concretely, at SRER/WKG a mesquite crown of 2–5 m is:
- **20–50 px** across at 10 cm — comfortably within the trained regime;
- **3–8 px** across at 0.6 m — far below it.

**Standard mitigation**: upsample NAIP to ~10 cm GSD before prediction, so crowns present at the pixel scale the model expects. This adds no information — the model sees interpolated blobs — but it is the difference between "wrong scale" and "right scale, less detail," and it is what the DeepForest documentation recommends when GSD differs from training.

**Expected outcome, stated in advance**: large isolated crowns plausibly detected; small, young, or clustered woody plants largely missed. That is a problem at WKG specifically, where the woody component is sparse and small — the hard case, not the easy one. Additional NAIP mismatches: 8-bit uncorrected radiometry mosaicked across dates and sun angles (versus NEON's corrected AOP), and a 4-band RGB+NIR stack where DeepForest expects 3-band RGB — drop NIR for DeepForest, use it for the vegetation indices.

**Do not adopt it on NAIP without this experiment first — degrade-and-compare:**

1. Take NEON SRER 10 cm RGB and run DeepForest natively → **reference crowns**.
2. Degrade the same imagery to 0.6 m to simulate NAIP; upsample back to 10 cm; run DeepForest again → **simulated-NAIP crowns**.
3. Compare against the reference: detection rate as a function of crown diameter, false-positive rate, and crown-area bias.

This yields a quantified answer — *DeepForest at NAIP resolution recovers X% of crowns above Y m diameter* — over a scene where the true answer is already known from CHM. It is cheap, needs no new data, and converts an assumption into a measured degradation curve that also tells you the **minimum detectable crown size** at NAIP resolution. If step 3 shows recovery is adequate for the crown sizes that matter at WKG, use it there; if not, DL-1 stays a NEON-only cross-check and the AmeriFlux tree class rests on frameworks A–C.

**Optional extension if the degradation proves too severe**: fine-tune DeepForest on 0.6 m simulated-NAIP chips using the native-10 cm detections as labels. This is well-posed — NEON supplies both the imagery and, via CHM, an independent crown check — and produces a NAIP-resolution crown detector without any manual crown digitizing. Treat as a follow-on, not part of the initial run.
