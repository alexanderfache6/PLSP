# Ground Truth Land Cover — v4 Execution Specification

**Status**: authoritative. This document supersedes `instructions4.md` and governs all v4 work.
**Created**: 2026-08-09

## 0. Document hierarchy

| Document                       | Status            | Use                                                                                                                                                                                                                                                          |
| ------------------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `instructions5.md` (this file) | **Authoritative** | Spec and execution plan going forward                                                                                                                                                                                                                        |
| `instructions4.md`             | Superseded        | Prior spec; retained for provenance                                                                                                                                                                                                                          |
| `instructions1.md`             | Archived context  | Upstream PlanetScope LSP clustering project (Wkg/Wjs). Explains where the phenology metrics and the unsupervised-clustering label option come from                                                                                                           |
| `instructions2.md`             | Archived context  | v3 session summary (bare/shadow/tree notebooks). Shadow method and standing conventions carried forward from here                                                                                                                                            |
| `instructions3.md`             | Archived context  | Raw dictated brain-dump. **Contains heavy speech-to-text corruption** ("bear"=bare, "rainforest"/"rental force"=random forest, "lighter"=LiDAR, "nap"=NAIP, "Cypress sites"=across sites, "vegetation industries"=vegetation indices). Do not read literally |
|                                |                   |                                                                                                                                                                                                                                                              |

### 0.1 Two numbering systems, and which is which

**This document numbers PIPELINE STEPS. Filenames number EXECUTION ORDER. They are different systems and they are not expected to agree.**

| | System | Where it lives | Example |
|---|---|---|---|
| **Steps** | logical pipeline stages, §5 | this document only | "Step 3 — Aggregation to PlanetScope scale" |
| **Stages** | the order scripts are actually run | filenames, config keys, results directories | `run_stage3_1_random_forest_ground_truth_classification.py` |

They diverge because the pipeline is not executed in the order it is described. Most importantly, **hand labeling (§4.2) must run after feature generation (Step 1a) and before classification (Step 1d)** — `run_stage2_1_find_kmeans_cluster_labeling_zones.py` fits its clusters on the Step 1a feature stack, and Step 1d trains on the labels that come out of it. Numbering files by spec step therefore produced a sort order that was actively misleading: `step1d` sorted before `step4_2a`, the reverse of how they run.

**The stages:**

| Stage | What it does | Spec steps covered | Results directory |
|---|---|---|---|
| **1** | data setup and feature generation | Step 0, Step 1a–1c | `stage1_data_and_features/` (with `qa/`, `features/`, `segments/`, `shadow/`) |
| **2** | hand labeling | §4.2 | `stage2_labeling/` |
| **3** | ground-truth classification | Step 1d–1e | `stage3_classification/run{N}/` |
| **4** | aggregation to Planet scale | Step 2, Step 3 | `stage4_aggregation/` |
| **5** | pure end members | Step 4 | `stage5_pure_endmembers/` |
| **6** | phenology regression | Step 5 | `stage6_rf_phenology/` |
| **7** | accuracy assessment | Step 6 | `stage7_accuracy_assessment/` |
| **8** | RAP comparison | Step 7 | `stage8_rap_comparison/` |
| **9** | transferability | Step 8 | `stage9_transferability_wkg/` |

**Config keys follow the stage of the script that reads them** — `stage1_3_planet_grid`, `stage2_1_labeling_zones`, `stage3_1_classification`, `stage4_1_aggregation` — so a key is traceable to exactly one script.

> **NAMING COLLISION, stated so it is not tripped over.** §4.2 uses "Stage 1 / Stage 1b / Stage 2" for the three *labeling* phases (k-means zones, CHM shrub candidates, hand labeling). Those are internal to §4.2 and are **unrelated** to the execution stages above — all three of them live inside execution stage 2. Read "Stage" inside §4.2 as a labeling phase; everywhere else it means execution order.

---

## 1. Objective

Build and compare multiple ground-truth land-cover classification frameworks (**tree / shrub / grass / bare**, plus **shadow** as a handled non-terminal class) at NEON SRER; aggregate 1 m hard classifications to PlanetScope pixel scale for fractional cover; derive validated pure end members; train a PlanetScope phenology model for fractional cover; and assess transferability across a network of paired NEON/AmeriFlux domains (§2), beginning with SRER → WKG. Compare against RAP fractional cover as a later validation step.

---

## 2. Study region and site network

### 2.1 Site network — paired domains

The study region is a set of **NEON domains**, each contributing one **NEON site** (full AOP: 10 cm RGB, 1 m VI, 1 m CHM) paired with one nearby **AmeriFlux site** (NAIP RGB+NIR only, plus offset 3DEP LiDAR).

Each pair is a self-contained transferability test: build ground truth at the data-rich NEON site, transfer to the data-poor AmeriFlux partner within the same domain and vegetation class.

**Selection method.** All 104 sites in the PLSP product were assigned to a NEON domain by point-in-polygon against the official NEON domain boundaries (`NEONDomains_2024`), restricted to the lower 48, then filtered to rangeland IGBP classes (OSH, CSH, GRA, WSA, SAV) with croplands excluded. Generated by `code/selected_sites_info/02_generate_site_domain_pairs/`.

**Hard constraint**: both members of a pair must exist in the PLSP product *Land Surface Phenology, Eddy Covariance Tower Sites, North America, 2017–2021*. This eliminates most candidates, including two whole domains (§2.1a).

#### The three dryland / rangeland pairs

| Pair | Domain | Domain name | NEON site | IGBP | AmeriFlux partner | IGBP | Sep. | Ecosystem |
|---|---|---|---|---|---|---|---|---|
| **1** | **D14** | Desert Southwest | **US-xSR** (SRER) | **OSH** | **US-Wkg** (Kendall Grasslands) | **GRA** | 87 km | Semi-arid desert grassland / mesquite savanna |
| **2** | **D15** | Great Basin | **US-xNQ** (ONAQ, Onaqui-Ault) | **OSH** | **US-Rws** (Reynolds Creek Wyoming big sagebrush) | **OSH** | 485 km | Wyoming big sagebrush steppe |
| **3** | **D13** | Southern Rockies & Colorado Plateau | **US-xMB** (MOAB) | **OSH** | **US-Wjs** (Willard Juniper Savannah) | **SAV** | 529 km | Piñon-juniper / juniper savanna |

#### Site attributes — as recorded in the AmeriFlux site table

| Site | Role | Domain | IGBP | Köppen | MAP | AmeriFlux record | Phenocam |
|---|---|---|---|---|---|---|---|
| `US-xSR` | NEON | D14 | OSH | Bsk | 346 mm | 2017–2024 | `NEON.D14.SRER.DP1.00042` / `.00033` |
| `US-Wkg` | AmeriFlux | D14 | GRA | Bsk | 407 mm | 2004–2025 | `kendall` |
| `US-xNQ` | NEON | D15 | OSH | Dfb ⚠ | 288 mm | 2017–2024 | `NEON.D15.ONAQ.DP1.00042` / `.00033` |
| `US-Rws` | AmeriFlux | D15 | OSH | Bsk | 290 mm | 2014–2023 | `arsgreatbasinltar098` |
| `US-xMB` | NEON | D13 | OSH | Bsk | 319 mm | 2017–2024 | `NEON.D13.MOAB.DP1.00042` / `.00033` |
| `US-Wjs` | AmeriFlux | D13 | SAV | Bsk | 361 mm | 2007–2025 | `junipersavannah` |

⚠ The `Dfb` Köppen class on ONAQ is almost certainly wrong in the AmeriFlux table — Onaqui is cold-desert sagebrush and should be B-class like its partner. Do not rely on that field; every other site in the network is `Bsk`.

#### Why these three

All three are `Bsk` semi-arid rangeland on a tight aridity gradient (288–407 mm MAP), with no cropland. They also give a graded ladder of transfer difficulty, which is the point:

| Pair | IGBP match | MAP difference | Transfer difficulty |
|---|---|---|---|
| **D15** — ONAQ / Rws | **OSH → OSH, exact** | **2 mm** (288 vs 290) | **Easiest.** Near-identical sagebrush steppe — the cleanest available Axis-1 baseline |
| **D13** — MOAB / Wjs | OSH → SAV | 42 mm | Moderate. Same biome, woody-encroached partner |
| **D14** — SRER / Wkg | OSH → GRA | 61 mm | Hardest. Transfer across structural type. `US-Whs` (Lucky Hills Shrub, OSH, 320 mm, 76 km) is the secondary option that isolates this variable by matching IGBP |

**D15 replaces D06 as the network's control pair** — see §2.1a.

Alternative within-domain pairs, should any of the above fail on data availability: **D14** `US-xJR` (JORN, OSH, 271 mm) / `US-Ses` (Sevilleta shrubland, OSH, 275 mm) — a second near-exact IGBP and MAP match, but it costs a domain by duplicating D14. **D13** `US-Mpj` (Mountainair Piñon-Juniper, WSA, 385 mm, 509 km) in place of `US-Wjs` — 20 km closer with a record running to 2026.

#### 2.1a D06 (KONZ) — retained, but its partner fails the hard constraint

`US-KFB` (Konza Prairie LTER 4B) is **not in the PLSP product**, so the co-located D06 pair as previously specified cannot be built. The only PLSP-available flux site in D06 is **`US-KFS`** (Kansas Field Station, GRA, Cfa, 1014 mm, record 2007–**2019**), **119 km from KONZ** — not co-located, and so it does not deliver the sensor-only control that justified D06 in the first place.

Consequences:
- **The "D06 is the co-located control" rationale is void.** D15 (ONAQ / Rws) takes over as the cleanest Axis-1 test, earning it on ecological similarity rather than co-location.
- D06 is retained as the **mesic contrast for Axis 2** — 870–1014 mm tallgrass prairie against 288–407 mm dryland — which is the role it uniquely fills.
- `US-xKZ` stays in the site list; **the D06 partner is unresolved**.

#### Two domains have no eligible partner at all

**D09 Northern Plains and D10 Central Plains contain zero non-NEON PLSP sites.** D10 holds only `US-xCP` (CPER) and `US-xSL` (North Sterling); D09 only `US-xDC`, `US-xNG`, `US-xWD`. No AmeriFlux test site exists in either under the hard constraint, so neither domain can be paired — this is not a shortage of good options, it is zero. **D11 Southern Plains** is likewise unusable: its only flux site, `US-ARM`, is **CRO** (cropland).

NEON sites carry AmeriFlux registrations under their `x`-prefixed IDs (`US-xSR`, `US-xNQ`, `US-xMB`, `US-xKZ`), so flux data is available on both sides of every pair.

**References**:
- NEON field site map and info — https://www.neonscience.org/field-site-map-and-info
- AmeriFlux site search / mapping tool — https://ameriflux.lbl.gov/sites/site-search/?mapping-tool

**Current scope**: SRER only. The other five sites are the roadmap; the pipeline must be built so adding one is a config file plus data, with zero code edits (R8).

> **Measured note — the SRER train and test blocks differ substantially in composition.** Fraction of each tile inside the CHM shrub band [0.7, 2.0) m:
>
> | Tile | Role | Shrub band | CHM >= 2 m |
> |---|---|---|---|
> | 511000_3527000 | train | 11.9% | 11.9% |
> | 511000_3528000 | train | 12.1% | 11.7% |
> | 511000_3529000 | train | 11.6% | 11.9% |
> | 515000_3530000 | **test** | **54.6%** | 8.1% |
> | 515000_3531000 | **test** | **37.3%** | 7.2% |
>
> The test block carries three to four times the shrub-band cover of the train block. The held-out tiles are therefore **not** a pure spatial-generalization check — they are simultaneously a **class-prior shift** test (R6). That is arguably more useful, since it previews the WKG transfer, but it must be reported as such: weak test-block performance may reflect composition shift rather than spatial overfitting, and the two cannot be separated with this split. Report per-class metrics on the test block, never overall accuracy alone.

> **The network above is PROVISIONAL, pending a combined data-availability analysis.** Site pairs will be reselected once QA2 scans for additional sites and the NEON / NAIP / 3DEP / phenocam timelines are assembled. MOAB is absent from the QA scan (§5.1a), KFB is not in the PLSP product at all (§2.1a), and the D15 pair is newly added and unscanned, and SRER's usable-year set is a single year — so the current pairing is not yet evidence-backed. Do not hard-code the site list anywhere; it lives in config (R8) precisely so this can change.

#### Site selection is an intersection problem, not a ranking

The binding constraint is **temporal alignment across four independent data streams**, not the quality of any one. A site with excellent PLSP QA but no NEON flight in that year is unusable, and so is a NEON flight year whose PLSP QA is 4%.

For a pair to qualify, all of the following must hold **in the same window**:

| Stream | Requirement |
|---|---|
| **PLSP QA2** | A high-quality year at the NEON site (`percent_good_pixels` high under `QA ∈ {1,2}`) |
| **NEON AOP** | A flight year matching or adjacent to that PLSP year — this pairing is what removes the temporal offset (§5.1a) |
| **NAIP** | At the AmeriFlux partner, near a high-QA PLSP year there, and **post-monsoon** (R7) |
| **3DEP** | Within 1 year of the NAIP imagery, or with verifiable alignment (§5 Step 8) |
| **Phenocam** | Present, for the independent phenology cross-check (§5 Step 5) |
| **Domain** | Both members in the same NEON domain (R1) |

**Score sites on the size of that intersection**, not on any single stream. A site scoring 98% on PLSP QA across all years but lacking a coincident NEON flight ranks below one at 85% where every stream lines up.

> **One case worth checking first.** SRER's only strong pre-2022 year is 2021 (91.1%), while its ground truth is the 2022-08 flight. **If NEON also flew SRER in 2021**, that pairing resolves the year-alignment problem immediately — no waiting on the PLSP 2022 product, no offset, no reliance on an unknown 2022 QA figure. Check this before treating 2022 as fixed.

### 2.2 What this network is designed to test

The pairs give two independent axes of transfer, which must be reported separately:

**Axis 1 — within-domain transfer (NEON → AmeriFlux).** Same domain, same vegetation class, different sensor. Isolates the **sensor and resolution** problem (AOP vs. NAIP) from the ecological one. This is §8 Step 8 Route 1 vs. Route 2.

**Axis 2 — cross-domain transfer.** D14 and D06 are **both GRA grasslands but ecologically very different** — hot semi-arid C4 desert grassland versus mesic tallgrass prairie. That contrast is the strongest available test of the working assumption that sites outside a shared ecoregion require independent end-member sets: same nominal vegetation class, very different phenology. D13 (SAV) supplies the woody-dominated contrast and is where tree and shrub signal is strongest.

### 2.3 Phenology calendars differ across domains — raw DOY is not comparable

This is the central cross-site obstacle, already identified in `instructions1.md` §§3 and 8:

| Domain | Growth driver | Approximate green-up |
|---|---|---|
| D14 (SRER, WKG) | North American monsoon | Jul–Sep |
| D15 (ONAQ, Rws) | Cool-season moisture + snowmelt | Spring — Apr–Jun, largely unimodal |
| D13 (MOAB, WJS) | Cool-season moisture + monsoon | Bimodal — spring and late summer |
| D06 (KONZ, KFB) | Spring warming + summer rainfall | Apr–Sep |

Consequences that must be handled before any cross-domain comparison:

1. **Raw DOY features cannot be compared across domains.** A grass end member at SRER and a grass end member at KONZ will differ in `OGI` by months for reasons that are climatic, not compositional. `instructions1.md` §8 specifies **climate-anchored** (site-relative and biome-relative anomaly) features for exactly this; that work is designed but not yet implemented and is a **prerequisite for Axis 2**, not for Axis 1.
2. **The `NumCycles == 1` strict filter is a live risk at D13 — see §2.4.** Bimodal cool-season-plus-monsoon systems legitimately produce two cycles, and the strict filter would discard those pixels wholesale.
3. **Acquisition-date checks are per site, not global** (R7). The monsoon-peak timing that makes SRER 2022-08 favorable has no equivalent meaning at D06.

### 2.4 Initial step — NumCycles distribution per site

**This runs first, before any other analysis, at every site in the network.** It is cheap (one layer, one histogram) and it decides whether the project's core 1-cycle constraint is viable at that site.

**QA policy — project-wide, not per-step.** All analysis uses **`QA ∈ {1, 2}`** (high + medium quality, layer 12). Wherever a second cycle is admitted, the same standard applies to it: **`QA_2 ∈ {1, 2}`** (layer 23). QA values 3–4 are dropped at both cycles. This rule is fixed once here and is not renegotiated downstream.

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

Outputs → `stage1_data_and_features/qa/numcycles_distribution_{SITE}_{YEAR}.json` plus a per-site histogram and map. **Run this for every site in the network as early as PlanetScope LSP data allows** — it is a prerequisite for scoping, not a per-site step to be deferred until that site's turn.

### 2.5 Site verification required before use

| Item | Status |
|---|---|
| `US-KFB` — Konza Prairie LTER (4B) | **Not in the PLSP product — fails the hard constraint (§2.1a).** D06 partner unresolved; `US-KFS` is the only PLSP alternative and is 119 km away |
| `US-xNQ` / `US-Rws` (D15) | Newly added (§2.1). Needs the same NEON AOP / NAIP / 3DEP / PLSP QA checks as the other pairs |
| `plsp_raw_id` geojson names for the four added sites | **Unverified** — `Onaqui-Ault_NEON`, `Reynolds_Creek_Wyoming_big_sagebrush`, `Moab_NEON`, `Willard_Juniper_Savannah` are inferred from the existing naming pattern; confirm against `/projectnb/planet/PLSP/geojson` |
| NEON AOP flight years available per site (xSR, xMB, xKZ) | Needed — governs which years §6's stability check can cover |
| NAIP acquisition years and dates per AmeriFlux site | Needed (R7) |
| 3DEP LiDAR year per AmeriFlux site and offset from imagery | Needed (§11 check #27) |
| UTM zone per site | Verify from file CRS; differs across domains |
| Ecoregion definition | **Resolved — NEON domain** (R1, §12 Q4). All pairs are within-domain, so all three are model-transfer tests |
| Pair separation distance and any EPA L3 boundary crossed | Record per pair for interpreting Route 1 outcomes (R1) |
| PlanetScope LSP coverage per site and year | Needed |
**Flight-year inventory — to be supplied by the user, one table per site**: NEON AOP flight years, NAIP flight years, and 3DEP LiDAR flight years. These three drive the §6 multi-year stability work, the R7 acquisition-phenology check, and the LiDAR-offset decision at Step 8, so none of that can be scoped until they land.

**PlanetScope LSP** is already confirmed at **full site coverage for a minimum of 2017–2021**, extending to 2022–2025 (§5.1).

---

## 2A. SRER — current site

**Site**: Santa_Rita_Experimental_Range_NEON (`SRER`), NEON Domain D14, AmeriFlux `US-xSR`
**Acquisition vintage**: 2022-08 (monsoon peak — grasses green; see §9 temporal note)
**CRS**: UTM (verify against actual file CRS; do not assume the zone)

**Confirm CRS, UTM zone, and the easting/northing origin by reading them from the files themselves** — never from the filename, the tile key, or an assumption about the standard NEON zone for the site. The tile key encodes an easting/northing but is not authoritative for the grid. Record all three per file in the Step 0 audit (§11 checks #6, #11, #12).

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

> **MEASURED — the two blocks are ecologically different, so this is not an ordinary spatial-CV split.** The §4.2 k=16 clustering (run 2026-08, all 5 tiles pooled) shows the train and test blocks occupying substantially different regions of feature space:
>
> | Cluster | % of train block | % of test block | Character (cluster-mean) |
> |---|---|---|---|
> | 11 | **21.1%** | 0.8% | mid SAVI, no canopy |
> | 9 | **14.4%** | 1.8% | mid SAVI, flat |
> | 14 | **12.6%** | 0.05% | mid SAVI, flat |
> | 6 | 0.3% | **15.2%** | low SAVI, bright (luma 197) |
> | 15 | 1.2% | **15.1%** | mid SAVI, CHM 1.5 m |
> | 16 | 0.3% | **15.0%** | low SAVI, bright (luma 178) |
> | 3 | 2.5% | **10.1%** | high SAVI, bright |
>
> About **55% of the test block sits in clusters that are under 3% of the train block**, and about 48% of the train block sits in clusters nearly absent from test. The test tiles are brighter, sparser terrain.
>
> **Consequences, all of which must be reported rather than absorbed:**
> 1. Step 6 test accuracy will **understate** performance on terrain resembling the training block, and errors will concentrate in clusters 3/6/15/16. Report accuracy stratified by cluster as well as by class.
> 2. This is broader than **R6**, which anticipates class-*prior* shift. This is a shift in the feature distribution itself, and it also weakens **R4**: one per-site `StandardScaler` fitted site-wide sits between two distinct populations.
> 3. It plausibly explains the other block-level anomalies already recorded — coregistration 0.8 m (train) vs 0.3 m (test) at §11.2 check 9, and shadow 3.1% vs 1.1% at Step 1c. Those now read as symptoms of genuinely different ground cover rather than sensor artifacts.
>
> **This does not require changing the tiles** — a hard held-out set has real value ahead of the WKG transfer — but the Step 6 number must never be presented as plain within-site spatial generalization.

### Tile selection for site-wide prediction — RESOLVED, and the method to reuse

**The problem**: prediction must eventually run on the full 10 km × 10 km footprint, roughly **100 tiles**. Training currently uses **3 contiguous tiles**, and testing **2 contiguous tiles** — a 5% sample drawn from two spots. The measured evidence above shows those two spots are not interchangeable, so there is no reason to expect five clustered tiles to span 100.

The symptoms are already visible in Step 1d (`results/stage3_1_results.md`): predicted grass falls to **0.3% and 0.1% on the test tiles** despite grass holding 21% of test label pixels, and median `prediction_quality` is lowest there (0.50–0.52 against 0.59–0.77 on train). A model trained on one neighbourhood is being asked about another and is visibly less sure.

**The reframe that makes this cheap**: **the full tile set has to be downloaded anyway.** Prediction needs the same feature stack as training — 10 cm RGB for texture, 1 m VI, 1 m CHM — for every tile it runs on. So acquiring all ~100 tiles is a prerequisite of Step 1 regardless of how training tiles are chosen. Once they are on disk, **selecting training tiles from the full set costs nothing in data**; the only scarce resource is labelling effort.

**Proposed method — stratify tiles the same way §4.2 stratifies labels, one level up:**

1. Download all ~100 tiles (required for prediction).
2. Compute cheap per-tile summary statistics from CHM and VI only — no 10 cm processing needed: CHM zero fraction, shrub-band fraction `[H_GRASS_MAX, H_TREE_MIN)`, canopy fraction `>= H_TREE_MIN`, and SAVI mean and percentiles.
3. Cluster the ~100 tiles in that summary space (k ≈ 6–8).
4. **Sample 2 tiles per cluster** for labelling, assigning train and test roles *within* each cluster so both blocks span the site's compositional range rather than one corner of it.

That yields roughly **12–16 labelled tiles** covering the range, instead of 5 covering two neighbourhoods.

**Labelling effort need not scale with tile count.** For generalization, between-tile variance matters more than within-tile variance: 40 polygons on each of 12 tiles beats 100 polygons on each of 5. The per-class gate is **per role, not per tile** (§4.2), so more tiles do not multiply it. And shrub is already largely automated by the CHM candidates (§4.2 Stage 1b), so the incremental hand cost per new tile is bare, grass and tree only.

**Trade-off to weigh**: the current test block is a genuinely hard, ecologically distinct hold-out, which has real value ahead of the WKG transfer. Stratified tile selection makes train and test more alike and so makes Step 6 look better without the map necessarily being better. **Keep both**: a stratified set for training and site-wide accuracy, plus the existing 515000 pair retained as a named *hard* hold-out reported separately.

#### Measured, 2026-08 — the original 5 tiles occupied 2 of 5 quintiles

CHM for all **70 tiles** covering the site was downloaded first (1 m, a few MB each) and one number computed per tile: the fraction inside the CHM shrub band `[H_GRASS_MAX, H_TREE_MIN)` = 0.7–2.0 m. That is a cheap, objective proxy for how woody a tile is, and it needs no 10 cm imagery — so it can be computed across a whole site *before* committing to a ~25 GB RGB download or to any labelling.

Split into quintiles across all 70 tiles:

| Quintile | Shrub-band cover | Tiles |
|---|---|---|
| Q1 | 0.116 – 0.143 | 14 |
| Q2 | 0.143 – 0.163 | 14 |
| Q3 | 0.163 – 0.208 | 14 |
| Q4 | 0.226 – 0.294 | 14 |
| Q5 | 0.295 – 0.546 | 14 |

The five original tiles fell in **Q1 and Q5 only** — the two extremes:

| Tile | Role | Shrub band | Quintile |
|---|---|---|---|
| 511000_3527000 | train | 0.119 | **Q1** |
| 511000_3528000 | train | 0.121 | **Q1** |
| 511000_3529000 | train | 0.116 | **Q1** |
| 515000_3531000 | test | 0.373 | **Q5** |
| 515000_3530000 | test | 0.546 | **Q5** |

Train sat on the site **floor**, test on the **ceiling**, and Q2–Q4 — 42 of 70 tiles, including the site median of 0.181 — had no labelled representation at all. The model was trained on the least woody ground, validated against the most woody, then asked to predict 70 tiles that are mostly neither.

**This explains symptoms already in the Step 1d results** (`results/stage3_1_results.md`): predicted grass collapsing to 0.3% and 0.1% on the test block despite grass holding 21% of test label pixels, and median `prediction_quality` lowest there (0.50–0.52 against 0.59–0.77 on train). Those were not model defects; they were the sampling design showing through.

#### The fix, and the selection rule to reuse at every site

Five tiles were added, chosen **one per quintile**, excluding any candidate within 2 km of an already-labelled tile, taking the most spatially isolated candidate within each stratum:

| Tile | Role | Shrub band | Quintile |
|---|---|---|---|
| 519000_3527000 | train | 0.120 | Q1 |
| 520000_3532000 | test | 0.146 | Q2 |
| 515000_3526000 | train | 0.172 | Q3 |
| 511000_3532000 | train | 0.270 | Q4 |
| 518000_3529000 | test | 0.271 | Q4 |

**Coverage went from quintiles {1, 5} to {1, 2, 3, 4, 5}.** Train now spans Q1/Q3/Q4, test spans Q2/Q4/Q5 — both blocks cross the site instead of occupying opposite tails.

> **MEASURED LATER, AND IT WEAKENS THIS CLAIM — two of the five added tiles are mostly outside the AOP flight box** (`run_stage4_1_aggregate_to_planet_blocks.py`, 2026-08-18; full numbers in `results/stage4_1_results.md` §2).
>
> | tile | quintile | shrub / flown | **flown area** |
> |---|---|---|---|
> | `511000_3532000` | **Q4 train** | 0.270 | **4.1 ha of 100** |
> | `520000_3532000` | **Q2 test** | 0.146 | **24.4 ha of 100**, ~14 ha inside the Planet footprint |
>
> The quintile statistic is computed as shrub ÷ **flown** area, which is the correct way to measure cover — these figures are not wrong. But **the entire Q4 training representation is a 4.1 ha sliver**, and `511000_3532000` contributed 38,684 of a possible 1,000,000 pixels to run 3, so its leave-one-tile-out fold is a 4 ha fold. The stratification is thinner than the table above implies.
>
> **Two selection criteria were missing and are now required for any tile added at any site:**
> 1. **Flight coverage must be ~100%** — test it on the **CHM**, which is the binding product (CHM nodata 95.91% on `511000_3532000` against 74.96% for RGB and 66.79% for SAVI).
> 2. **The tile must sit wholly inside the PlanetScope footprint**, or its ground truth has no Planet pixel to aggregate into.
>
> **Recommended**: replace both tiles with fully-flown alternatives from the same quintiles — 14 tiles were available per quintile — and re-run stage 3 and stage 4.

Note that a new tile is not redundant just because its shrub cover resembles an existing one. `519000_3527000` matches the train block on shrub (0.120 against 0.119) but carries **one third the tree cover** (3.6% against 11.9%) — the same shrub density in a very different canopy, a combination the model had never seen.

**Reuse this at WKG, MOAB, KONZ and any site added later.** The full site must be downloaded for prediction regardless, so selecting training tiles from the complete set costs nothing in data — only labelling effort is scarce. Download CHM first, compute per-tile shrub-band cover, stratify, then fetch RGB and VI only for the chosen tiles.

**Action required**: the existing `00_ground_truth_helpers.py` carries a different 3-tile set. Confirm all five tiles above are downloaded for all three products before Step 0 completes; download any missing tiles.

**Build a file-validation helper.** A single function that, given a site, walks the expected product paths and reports presence/absence for every required file — run before any processing, at every site, as the first action of Step 0.

**Produce a per-site checklist by file category** (RGB, vegetation indices, CHM, PlanetScope LSP, and later NAIP and 3DEP), listing expected count, found count, and the specific missing paths. The checklist is an output written to `stage1_data_and_features/qa/`, not console text, so a run's completeness is auditable after the fact. This is the machine-readable form of §11.1.

**Phenocam**: located within the 511000 train block. Use as an independent phenology reference in Step 5.

### Paths

- **Local (now)**: `.../Dropbox/planet/data/NEON/{SITE_NAME}/`, results under `/Dropbox/planet/results/v4
- **Cluster (SCC)**: `/projectnb/modislc/users/fache/data/NEON/{SITE_NAME}/`, results under `/projectnb/modislc/users/fache/results/v4`

Local paths are used for development. **All deep-learning tracks run on the SCC** (see §8 prerequisites). Path root must be a single configurable constant in the helpers module so the switch is one edit, not a search-and-replace.

### Output directory structure

All directories below hang off the same `{DATA_ROOT}`-derived results root as §5.1, so the local → SCC migration remains a single constant change (R8). Do not hard-code either root here.

```
results/
    stage1_data_and_features/qa/                          # per-tile QA, CHM noise floor, grid alignment reports
    stage3_classification/        # per-framework 1 m hard classification (A-E)
    stage4_aggregation/            # N x N m window % cover per class
    stage5_pure_endmembers/             # pure end-member windows + validation
    stage6_rf_phenology/                # PlanetScope fractional-cover model
    stage7_accuracy_assessment/         # shared sample set, manual labels, per-framework accuracy
    stage8_rap_comparison/              # RAP 10 m vs. ground truth vs. Planet
    stage9_transferability_wkg/         # WKG transfer test
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

#### Class labels and colours — LOCKED

Both mappings are fixed project-wide, alongside the codes. Every raster colour table, every figure, every legend, and every QGIS style file uses these and nothing else.

```python
CLASS_LABELS = {0: "bare", 1: "grass", 2: "shrub", 3: "tree"}
CLASS_COLORS = {0: "#c2b280", 1: "#7cb342", 2: "#8d6e63", 3: "#1b5e20"}
```

| Code | Label | Hex | RGB | Reads as |
|---|---|---|---|---|
| 0 | `bare` | `#c2b280` | (194, 178, 128) | Sand / bare soil |
| 1 | `grass` | `#7cb342` | (124, 179, 66) | Living herbaceous green |
| 2 | `shrub` | `#8d6e63` | (141, 110, 99) | Woody brown |
| 3 | `tree` | `#1b5e20` | (27, 94, 32) | Dark canopy green |

**Rendering conventions for the two non-terminal values:**
- **255 (nodata)** — fully transparent, `(0, 0, 0, 0)`. Never rendered as a colour, so it cannot be mistaken for a class.
- **4 (shadow)** — intermediate only and never present in a final product (§11 check #19a). Where an intermediate is rendered for inspection, use a neutral mid-grey, deliberately outside the earth-and-green family so it is unmistakable.

**Two properties worth preserving if these are ever revised:**
1. **Lightness decreases monotonically** bare → grass → shrub → tree. That is what keeps the palette legible in greyscale and under colour-vision deficiency, where hue separation degrades but lightness ordering survives.
2. **Colours are semantically intuitive** — sand, grass green, woody brown, dark canopy. Legends become nearly unnecessary, which matters for the colour-blend composite (§3.1 view B1).

> **One consequence to expect in the blend composite**: `bare` and `shrub` are both earth tones, so a bare/shrub mixture blends to a muddy mid-brown that is harder to read than, say, a grass/tree mixture. The margin view (B3) and the per-class probability panels (B2) are the fallback for inspecting that specific pair.

**`mixed` is not a class code.** It is a **continuous confidence layer** (§3.1), so a pixel always carries both its plurality class *and* how strongly it resembles that class — rather than losing the class label to an ambiguity code, or collapsing a graded quantity into a boolean.

> **Threshold values are provisional and will be refined** — the *codes* above are locked, the *rules* are not. Thresholds are a starting point for the first run, expected to iterate against visual inspection and Step 6 accuracy. Any code written against them must read config constants, never inline literals, so refinement is a config edit.

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| `H_TREE_MIN` | **2.0 m** | Locked per decision |
| `H_GRASS_MAX` | **0.7 m** (measured, no longer provisional) | Set from the CHM distribution, not assumed — see safeguard 2 |
| `SAVI_BARE_MAX` | **0.2** | Carried forward from v3 bare detection |
| `SHADOW_TREE_RADIUS` | **5 m** | Carried forward from v3 |

### Species context (SRER)

Velvet mesquite (*Prosopis velutina*) is the dominant woody species and straddles the shrub/tree boundary — many individuals sit at 1.5–3 m. Creosote (*Larrea*), burroweed (*Isocoma*), cholla and prickly pear are shrub. Lehmann lovegrass and native gramas are grass.

### Three required safeguards

1. **The 2.0 m cut splits mesquite.** Persist per-crown CHM statistics (min/mean/max/p90, pixel count) as attributes on the crown vector outputs, so the shrub/tree threshold can be re-cut analytically without re-running the pipeline.
2. **`H_GRASS_MAX` — RESOLVED by measurement, and not as anticipated.** The concern was that 0.3 m might sit inside CHM noise. Measured across all five SRER tiles, there is **no noise floor to find: NEON has already applied one.** The minimum non-zero CHM value is exactly **0.700 m on every tile**, and **no pixel anywhere falls between 0 and 0.7 m**. The distribution is 76% exact zeros, then nothing until 0.7 m.

   Three consequences:

   - **The old 0.3 m was inert.** Any value in (0, 0.7] produces an identical shrub band, so the parameter had no sensitivity at all. It is now set to **0.7 m** so the config states what the rule actually does.
   - **Woody vegetation below 0.7 m is structurally invisible to CHM** — recorded as 0, indistinguishable from bare ground. At SRER that means burroweed and young creosote below 0.7 m are **systematically missed by the reference path**, and will be labeled grass or bare by frameworks D/E.
   - **This is a reference-path error that propagates**, because `RF-A_D` is the standard `RF-A_A`–`RF-A_C` are measured against (§4.1). They may detect sub-0.7 m shrubs from RGB and texture and be *penalized* for it. Where a transferable variant and `RF-A_D` disagree on small shrubs, do not assume `RF-A_D` is right.

   The grass/bare split therefore cannot use CHM at all and rests entirely on SAVI — which is what §3 already specifies, so no rule changes. Measurement recorded in `stage1_data_and_features/qa/`.
3. **1 m pixels are mixed — handled as continuous confidence, not a boolean flag.** See §3.1.

### 3.1 Mixture as a confidence layer

1 m pixels in semi-arid rangeland are usually mixed. Rather than discarding that information into a boolean flag, **mixture is carried through the whole pipeline as a continuous confidence measure**: low mixture means high purity and strong class resemblance; a near-even mixture means low confidence in the predicted cover. It propagates into aggregation, end-member selection, model training, and accuracy reporting.

#### Two outputs per 1 m pixel, always produced together

| Output | Content | Purpose |
|---|---|---|
| **Hard label** | Plurality class, codes 0–3 (§3) | The discrete product; accuracy assessment; Step 3 counting |
| **Soft vector** | Per-class membership `p = [p_bare, p_grass, p_shrub, p_tree]`, sums to 1 | Confidence; soft aggregation; training weights |

#### What "hard label" means, precisely — three separate things

These are routinely conflated. They are not the same.

**(a) The *definition* of a 1 m pixel's true class.** The class occupying the **largest within-pixel area fraction** — plurality, or dominance. A pixel that is 40% grass, 35% bare, 25% shrub *is* grass. Ties break **tree > shrub > grass > bare**. This is a definition, not a computation: it is the rule a human labeler applies and the standard Step 6 validates against. The underlying area fractions are never computed directly at 1 m.

**(b) How each framework *estimates* (a).** Three mechanisms, all producing the same locked class codes (§3):

| Path | Mechanism |
|---|---|
| Reference (D/E rules) | Deterministic threshold cascade: CHM >= `H_TREE_MIN` → tree; `H_GRASS_MAX` <= CHM < `H_TREE_MIN` → shrub; else SAVI >= `SAVI_BARE_MAX` → grass; else bare |
| Learned (`RF-A_A`–`RF-A_D`) | **argmax of `p`** — for a random forest, the proportion of trees voting each class, i.e. a majority vote across the forest |
| Deep learning (`RF-A_DL2`, `RF-A_DL3`) | argmax of the softmax output |

**There is no regression at 1 m.** RF-A is a classifier; RF-B (§4.3) is the regressor. The only fractional quantity at 1 m is the conceptual area fraction in (a), which argmax estimates and manual labeling checks.

**(c) How cleanly (b) resolved** — the confidence measure below. The 40/35/25 pixel above is labeled grass *and* carries low confidence, so a weak plurality is recorded rather than hidden.

**All three are persisted.** `p` is never discarded on hardening, and `prediction_quality` is written to disk as its own raster at both the 1 m and block levels — Step 4 ranking, Step 5 `sample_weight`, and Step 6 stratification all read it from file rather than recomputing it, so the value used downstream is provably the value that was produced.

#### Where the soft vector comes from

- **Learned variants (`RF-A_A`–`RF-A_D`)**: the classifier's per-class probability output directly.
- **Rule-based reference path (§3 threshold rules)**: distance-to-threshold scaled to [0, 1], the convention already established in v3 (`instructions2.md` Phase 2, `bare_confidence` as linear distance below the SAVI threshold), then normalized across classes.
- **DL tracks**: softmax output (`RF-A_DL2`), or detection score (`RF-A_DL1`).

#### Confidence measure

Primary, for continuity with `instructions1.md` §3.5, which already uses Shannon entropy for cluster purity:

```
confidence = 1 − H(p) / log(K),    H(p) = −Σ p_c · log p_c,    K = 4
```

Range [0, 1]: **1.0 = pure** (all membership in one class), **0.0 = perfectly even** four-way mixture. This is the layer meant by "mixed" throughout this document.

Also record, as a secondary diagnostic, the **margin** `p_(1) − p_(2)` between the top two classes. Entropy and margin disagree in an informative way: a pixel split 50/50 between grass and shrub has moderate entropy but near-zero margin, and it is the margin that identifies the specific confusions worth investigating (grass/shrub, bare/senesced grass).

#### The probability simplex — what `p` actually is

A 4-class probability vector carries 4 numbers but only **3 degrees of freedom**, because `Σp = 1` fixes the fourth once three are known. So `p` does not fill 4-D space; it lives on a 3-dimensional surface embedded in it, the **3-simplex**.

Geometrically that surface is a **tetrahedron with one vertex per class**:

| Position | Meaning |
|---|---|
| **Vertex** | Pure pixel, e.g. `(1,0,0,0)` = all bare. Entropy 0, confidence 1 |
| **Edge** | Two-class mixture, e.g. 50/50 grass–shrub |
| **Face** | Three-class mixture |
| **Center** | `(0.25, 0.25, 0.25, 0.25)`, maximally mixed. Confidence 0 |

The familiar case is 3 classes, giving a triangle — the 2-simplex — which is exactly the ternary diagram used for soil texture (sand/silt/clay). Four classes is the same construction one dimension up. Confidence is therefore a scalar measure of **how far a pixel sits from the nearest vertex**.

#### Visualization helper

**These visualizations are not decoration — they are the primary means of making the mixture structure legible, and building them well matters.** The tetrahedron has parts that mean different things (vertices, edges, faces, interior volume), and a visualization that collapses them hides exactly the information the confidence layer exists to expose. Render each part explicitly.

**Group A — the simplex itself, showing every part of the tetrahedron:**

| # | View | Which part of the tetrahedron it shows |
|---|---|---|
| A1 | **3D tetrahedron scatter**, rotatable, vertices labelled by class, points coloured by hard label | The **whole volume** at once — the master view |
| A2 | **Four face ternary diagrams** — one per face, dropping one class and renormalizing the other three | Each **face** (3-class mixtures), read as a familiar ternary plot |
| A3 | **Six pairwise mixing histograms**, one per class pair | Each **edge** (2-class mixtures) — where grass/shrub and bare/senesced-grass confusion lives |
| A4 | **Vertex-proximity histogram** — distribution of distance from each pixel's `p` to its nearest vertex | How close the population sits to **purity** |
| A5 | **Volume-occupancy summary** — fraction of pixels near a vertex, along an edge, on a face, or in the deep interior | How much of the volume is actually **occupied** |

> **A5 is the scientifically loaded one.** A landscape of genuinely pure pixels clusters tightly at the four vertices; a heavily mixed landscape fills the interior. Occupancy therefore forecasts, *before Step 4 is ever run*, whether pure end members at the >= 90% threshold will be plentiful or scarce — and if the interior dominates, it warns that 1 m is simply too coarse to resolve pure cover at this site, which is a finding rather than a failure.

**Group B — spatial and per-class views:**

| # | View | Purpose |
|---|---|---|
| B1 | **Colour-blend composite map** — fixed colour per class code, per-pixel colour = `Σ_c p_c · colour_c` | The primary map. Pure pixels render saturated, mixed pixels desaturate toward grey, so purity reads directly with no legend |
| B2 | **Four-panel per-class probability maps** | Shows *which* class is contesting a given area |
| B3 | **Top-1 vs. top-2 probability scatter**, coloured by predicted class | Margin structure and the specific confusable pairs. Expect a visible grass/shrub arm |
| B4 | **`prediction_quality` histogram per class** | Feeds the §6.5 monotonicity check and sets the within-class median bin edges used in §6.2 |

Produce Group A and Group B at both the 1 m level and the block level (§3.1) — the tetrahedron structure at block scale is a direct picture of the Step 3 fraction estimates, and comparing the two levels shows how much mixture the N×N aggregation absorbs.

#### Secondary diagnostic — Mahalanobis distance to class centroids

Alongside the probability view above, compute a **4-vector of Mahalanobis distances** from each pixel to the four class centroids in feature space, using pooled per-class covariance with ridge regularization — the approach already proven in `instructions1.md` §3.2.

This is a genuinely different measurement, not a restatement:

| | Entropy of `p` (operational) | Mahalanobis distance vector (diagnostic) |
|---|---|---|
| Measures | How the *classifier* resolved the pixel | Raw feature-space geometry |
| Depends on the classifier | Yes | **No** |
| Comparable across `RF-A_*` variants | No — each has its own decision boundaries | **Yes** — same feature space, same centroids |
| Role | Weights RF-B, ranks end members, strata for Step 6 | Visualization and cross-framework comparison |

The two are related — `softmax(−d²/2)` maps distances back to something close to a QDA posterior — but they are not interchangeable. **Entropy of `p` remains the operational `prediction_quality`**; the distance vector is diagnostic only and must never be substituted into Step 3 weighting, Step 4 ranking, or Step 5 `sample_weight`.

#### Outputs

`prediction_quality_{framework}_{tile}_{YEAR}.tif` (float32, [0,1]) — the operational `prediction_quality`,
`margin_{framework}_{tile}_{YEAR}.tif` (float32),
`class_probability_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands, band order = class codes 0–3),
`mahalanobis_distance_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands, class-code order) — diagnostic only.

Retaining the full probability stack is what makes soft aggregation (Step 3) possible. Do not harden and discard.

#### This layer is `prediction_quality`

Throughout this document and in config, the §3.1 confidence layer is named **`prediction_quality`** — it measures how confident the *classification* is. It is distinct from **`data_quality`** (PlanetScope `numObs`, `QA`, `NumCycles`), which measures whether the *observations* were good enough to fit a phenology curve. The two are never blended into one score; see the table in §5.3.

#### Downstream use — four places

1. **Step 3 aggregation** — soft aggregation, and confidence weighting of the hard count.
2. **Step 4 end members** — a block that is 95% grass built from low-confidence pixels is a weaker end member than one that is 92% grass at high confidence. Rank by purity **and** confidence.
3. **Step 5 RF-B training** — block confidence becomes the regression `sample_weight`. Blocks with noisy targets contribute less.
4. **Step 6 accuracy** — report accuracy stratified by confidence, not only pooled.

### Transferability constraint

The CHM rules above define the **reference** labels only. CHM is not reliably available at WKG (NAIP-era LiDAR is offset 1–2 years). Transferable variants `RF-A_A`–`RF-A_C` must therefore reproduce these labels **from RGB + vegetation indices + texture alone**. This is the central design tension and is handled explicitly in §4.

---

## 4. Framework design

### 4.1 Deconfounding

#### Naming convention — LOCKED

Two distinct models exist (§4.3), and each has variants. Names carry the model first, then the variant, so a variant can never be mistaken for the other model's:

| Prefix | Model | Variants |
|---|---|---|
| **`RF-A_*`** | RF-A, the 1 m ground-truth classifier | `RF-A_A` … `RF-A_D` (inputs), `RF-A_DL1` … `RF-A_DL3` (deep learning) |
| **`RF-B_*`** | RF-B, the PlanetScope phenology fractional-cover regressor | `RF-B_*` — variants to be defined when Step 5 runs |

Bare letters (`A`, `B`, …) remain the **on-disk key** used in directory and file names written by `run_stage3_1_random_forest_ground_truth_classification.py`; `RF-A_A` is the reporting name. Both refer to the same thing.

#### The variants

`RF-A_A` through `RF-A_D` vary **input layers only**. The algorithm is held fixed so that any accuracy difference is attributable to the inputs, which is the actual research question.

**Fixed algorithm**: segmentation (SLIC at 1 m) + feature extraction + Random Forest classifier. *(As built, Step 1d trains per pixel rather than per segment — see `results/stage3_1_results.md` §1.2 for the measurement that forced the change.)*

| Variant | Inputs | Transferable to WKG / non-NEON sites? |
| --- | --- | --- |
| `RF-A_A` | RGB only | Yes |
| `RF-A_B` | RGB + vegetation indices (SAVI priority) | Yes |
| `RF-A_C` | RGB + VI + texture | Yes |
| `RF-A_D` | RGB + VI + texture + CHM | Degraded (offset LiDAR) |
| ~~`RF-A_E`~~ | ~~All layers + full feature set~~ | **RETIRED — see below** |

> **`RF-A_E` is RETIRED.** It was defined as *all layers + full feature set*, but `RF-A_D` already consumes every band in the feature stack, so E resolved to an **identical 20-feature set with nothing added**. Verified directly. Running it would have produced a duplicate of `RF-A_D` reading as an independent result.
>
> Its intended distinguishing element was never another input family — there are none left — but the **deep-learning track**, which is now carried explicitly by `RF-A_DL1`–`RF-A_DL3`. The letter is retired rather than redefined so that no future reader assumes a fifth input tier exists. The traditional track is complete at four variants.

**Priority order for feature emphasis**: vegetation indices and texture first; **LiDAR/CHM is a sanity check between classes, not a primary driver**. All available data is used, but the transferable variants (`RF-A_A`–`RF-A_C`) are the ones carried to WKG.

**Deep-learning tracks are a separate, smaller comparison** — not folded into the input-tier comparison:

| Track | Method | Notes |
|---|---|---|
| `RF-A_DL1` | DeepForest (`weecology/deepforest-tree`, pretrained, no fine-tuning) | RGB 10 cm, tree crowns only. **Role: independent tree-crown validation backup**, not a competing 4-class variant — trained on NEON RGB, so it is an outside opinion on crowns (§12 Q9). NAIP use requires the degrade-and-compare test in Q9a |
| `RF-A_DL2` | U-Net semantic segmentation, 4-class | Requires the hand-labeled training set from §4.2 |
| `RF-A_DL3` | SAM zero-shot segment proposals + color/texture classification | Run unconditionally — see below |

**`RF-A_DL3` is run unconditionally**, not held back as optional. SAM's zero-shot segmentation is worth evaluating on its own terms as an advanced, inherently transferable segmentation front end — it needs no training data and no site-specific tuning, which is exactly the property `RF-A_A`–`RF-A_C` are being engineered toward. Run it alongside DL1 and DL2 and report it whether or not the others leave gaps.

DL tracks output tree (`RF-A_DL1`) or full 4-class (`RF-A_DL2`/`RF-A_DL3`) maps evaluated on the same shared sample set. **`RF-A_DL1` is scored on the tree class only** — it is a cross-check on tree detection, not a candidate for variant selection (§12 Q9).

**Reference vs. candidate framing**: `RF-A_D` (CHM-bearing) is expected to be the most accurate at SRER, but is the *least* transferable. It therefore serves as the **reference/labeler** against which the transferable candidates (`RF-A_A`–`RF-A_C`, DL) are measured — the "best framework" selection in Step 6 selects among transferable candidates, judged by agreement with both D/E and the manual sample set.

### 4.2 Label source — hand-labeled, with clustering used to target the labeling

**Hand-labeled polygons are the ground truth.** Unsupervised clustering is **not** a competing label source; it runs first, as a targeting aid that decides *where* the human should label. The two stages are sequential, not parallel.

> **The cluster ID is never a label.** Clustering says *where to look*; the analyst says *what it is*. No cluster-to-class assignment is performed, and no cluster ID propagates into the training labels. This keeps the ground truth fully human-determined and avoids importing the clustering's errors into the labels it was meant to help collect.

#### Stage 1 — K-means to define labeling zones

Purpose: unguided hand labeling drifts toward the visually obvious. A labeler naturally picks clean, unambiguous patches, which starves the classifier of exactly the intermediate and difficult cases that decide Step 6 accuracy. Clustering the feature space first exposes the full range of what is actually present, so labeling can be allocated across it rather than across whatever caught the eye.

- **Simple K-means**, per decision — no GMM, no covariance-type sweep, no soft assignment at this stage. This is a targeting tool, and its output is a set of zones to inspect.
- **Input**: the 1 m feature stack, z-scored with a per-site `StandardScaler` (R4; matches `instructions1.md` §3). Use the **full stack including CHM** — this stage runs only at NEON sites and only to place labels, so the transferability constraint on `RF-A_A`–`RF-A_C` does not apply here.
- **Deliberately over-segment**: choose `k` well above the 4 classes — **k ≈ 15–20**. Over-segmentation is the *goal*. `instructions1.md` §6 found bare fragmenting across 4 clusters at k=10 (varying soil, litter), and that fragmentation is information: it marks distinct appearances of one class that the labeling must cover.
- **`k` is chosen once and does not need optimizing.** No elbow or silhouette sweep — a suboptimal `k` yields slightly redundant zones, which costs a little labeling effort and nothing else.
- **Seed** `BASE_SEED + year` (§12 Q8).

> **AS BUILT — `run_stage2_1_find_kmeans_cluster_labeling_zones.py`, SRER 2022, k = 16, seed 2028.** Three implementation facts that the wording above does not anticipate:
>
> **1. The feature stack is rank-deficient, so Mahalanobis cannot be applied to it directly.** The Step 1a stack is **rank 16 of 20**: `r+g+b = 1` exactly, and `ExG`, `ExR`, `ExGR` are exact linear combinations of `r,g,b`. Four eigenvalues sit at machine zero, the covariance is singular, and ridge-regularized whitening amplifies numerical noise into empty clusters (`cond = 2.4e5` even after dropping the four redundant bands). **Procedure**: drop `b`, `ExG`, `ExR`, `ExGR` — which loses no information, since Mahalanobis is affine-invariant on their span — then `StandardScaler` → **PCA whitening at 99% variance (10 components)** → Euclidean K-means. Euclidean distance on whitened components *is* Mahalanobis on the retained subspace. **Random Forest at Step 1d is unaffected by collinearity and keeps all 20 bands.**
>
> **2. Candidate sites are budgeted per cluster PER TILE ROLE, not per tile.** RF-A trains on train-block labels only, so train coverage must be guaranteed on its own — pooling across roles let clusters land with zero train sites while being well covered in test. **30 per cluster per role** (10/tile × 3 train, 15/tile × 2 test). Delivered: median 30 in both roles, 391 + 391 = 782 sites. Eleven cluster×role combinations fall short (minimum 3), in every case because that cluster barely exists in that role — a direct consequence of the §2A block divergence.
>
> **3. The 5×5 interior test needs a documented fallback.** Cluster 12 is 2.8% of the train block yet has **zero** pixels passing 5×5 ≥ 0.8 anywhere — it is pure speckle. Refusing to sample it would violate the coverage rule below, so the test relaxes stepwise (**5×5 ≥ 0.8 → 3×3 ≥ 0.7 → 3×3 ≥ 0.5**) and the level used is recorded per site in the GeoPackage. Only cluster 12 required relaxation. Sites carrying a relaxed `interior_level` sit in genuinely mixed terrain and will be the hardest calls — which is the point of zone-guided labeling, not a defect.

- **Distance metric: Mahalanobis**, not Euclidean. The 1 m features are strongly correlated and on different natural scales, so Euclidean distance would over-weight whichever direction happens to carry the most variance. Mahalanobis accounts for the covariance structure and matches the convention already proven in `instructions1.md` §3.2. Use a ridge-regularized covariance for numerical stability, as done there.

#### Stage 2 — Zone-guided hand labeling

- **Per-cluster interior-pixel sampling** to place candidate labeling sites: 5×5 majority filter to avoid boundary and speckle pixels, matching the protocol proven in `instructions1.md` §3.4. Emit **30 candidate sites per cluster per tile role** (superseding the original "~20 per cluster", which did not distinguish train from test — see the as-built note above), with a 15 m minimum separation between sites.
- Written as an **editable GeoPackage** for QGIS, with an empty nullable-integer `class_code` column (§3 codes 0–3), plus `cluster_id` retained as an attribute **for provenance only**.
- The analyst draws polygons at or around these sites on 10 cm RGB, with CHM and SAVI available as reference overlays.
- **Coverage requirement**: every cluster must receive labeled polygons, and the existing per-class minimum still holds — **>= 50 polygons per class per tile role** (train block / test block). Clusters constrain *where* labels are placed; they do not relax *how many* per class are needed.

> **The training label set is a UNION of two sources.** Every count in 2b, and every label Step 1d reads, is:
>
> | Source | File | Condition |
> |---|---|---|
> | hand-drawn | `training_polygons_*.gpkg` | all features with a valid `class_code` |
> | accepted candidates | `shrub_review_*.gpkg` | `reviewed = 1` and `rejected != 1` |
>
> An accepted CHM candidate **is** an ordinary hand-validated label — the analyst confirmed it against RGB, which is the same act as drawing one. Counting only the drawn file would make reviewing 750 candidates register as zero progress and defeat the accelerator entirely. **Step 1d must read the same union**; a single rule applied in two places, with no copying between files, so the two can never diverge. 2c reports the split in an `of which chm` column so the provenance stays visible.

**Polygon geometry convention.** A labeled polygon is a **set of 1 m pixels**, not an arbitrary free-hand shape — polygons are rasterized to the 1 m analysis grid, so pixel membership is what the label actually is. Draw and store them accordingly.

#### Minimum polygon area is CLASS-SPECIFIC

**Why a minimum exists at all**: very small polygons are dominated by edge pixels — the most mixed and least reliable — and are the most vulnerable to coregistration error, since a sub-metre shift can move the whole polygon off the object. A 3×3 polygon is the smallest with even one non-boundary pixel.

**Why it must not be uniform.** A single floor does not bind equally. Bare and grass form large contiguous patches; shrubs at SRER are isolated creosote and burroweed with crowns of roughly 1–3 m². A uniform 9 m² floor is therefore free for bare and grass and lands in the **middle of the shrub size distribution**, filtering shrub by size while leaving the abundant classes untouched.

Measured at SRER 2022 after the first labeling pass (`run_stage2_4_check_hand_labeling_progress.py`):

| Class | n | min m² | **median m²** | max m² | 9 m² floor binds? |
|---|---|---|---|---|---|
| bare | 25 | 2.5 | **64.0** | 1385.7 | No |
| grass | 20 | 11.8 | **80.9** | 314.3 | No |
| tree | 43 | 6.3 | **19.5** | 88.7 | Marginally |
| shrub | 15 | 2.4 | **9.4** | 35.8 | **Yes — at the median** |

Two failure modes follow, and the second is the more damaging:

1. **Size-biased exclusion.** Surviving shrub polygons are the atypically large ones. RF-A learns "shrub = large woody patch", misses isolated small shrubs — the majority of real shrub cover — and shrub fraction is biased low everywhere, in the class already weakest (`instructions1.md` §6: 40–70% purity).
2. **Label contamination.** A floor above typical object size pushes the analyst to *stretch* polygons to qualify, importing surrounding bare soil into shrub polygons. A shrub median of 9.4 m² against a 9 m² floor is the signature of exactly this. Contaminated labels are worse than missing ones: exclusion loses data, contamination teaches the wrong thing.

**Minima, per class:**

| Class | Minimum | Basis |
|---|---|---|
| bare | 9 m² | **Decided, stays at 9.** Non-binding at the median (64 m²). Small interstitial bare gaps between vegetation fall below it and are excluded deliberately — they are the most mixed pixels in the class and the least reliable as training data |
| grass | 9 m² | Non-binding; same reasoning |
| tree | 4 m² | Mesquite crowns from ~2 m diameter upward; 9 m² excludes small trees unnecessarily |
| **shrub** | **2 m²** | Typical creosote/burroweed crown is 1–3 m². Anything higher filters the class by size |

**General rule where a site's class sizes are unknown**, derive the floor from measured coregistration error (§11 check #9) rather than a round number:

```
minimum linear dimension ≈ 2 × coregistration_error + 1 pixel
```

At a measured 0.4 m offset that gives ~1.8 m → **4 m²**, not 9. This is a measurable basis and generally lands below a conventional default.

**Point labels for sub-minimum objects.** Where a genuine shrub crown falls below even the 2 m² floor, label the **single most-central pixel** as a point rather than discarding it. Standard practice in crown work, and it keeps the small-shrub population — precisely the part that matters — in the training set. Point labels are stored in the same GeoPackage with a `label_geometry` field of `polygon` or `point`, and count toward the per-class minimum.

**A below-floor polygon is IGNORED, not a fault.** It is excluded from the class totals, from cluster coverage, and from the area distribution, and it **does not block the gate**. It is listed separately in 2b for reference, and nothing is deleted from the GeoPackage — a polygon that is too small to train on is still a record of what the analyst saw, and a class drifting under its floor is worth noticing. This keeps the floor a *filter* rather than an error condition, so the analyst is never asked to redraw work merely to clear a gate.

**Record the polygon area distribution per class** (2c check 3) every run, so residual size bias stays visible rather than inferred.

> **Retroactive note**: polygons already drawn below the old uniform 9 m² floor are **not deleted**. Under the class-specific minima most become valid, and they represent real labeling effort. Re-validate rather than discard. At SRER 2022 the change took violations from **16 to 5**, clearing every shrub and tree case.
>
> The 5 remaining are all small interstitial **bare** patches. **The bare floor stays at 9 m².** The reasoning is that bare is the one class where a small polygon carries no benefit: bare is abundant, large clean patches are easy to find, and a 3 m² gap between shrubs is dominated by canopy-edge and shadow pixels — the least reliable training data available. Excluding them costs nothing and raises label purity.
>
> This is the opposite of the shrub case, and deliberately so. There, a low floor was necessary because small crowns *are* the class; here, small patches are merely the worst examples of an abundant class.

#### CHM-derived shrub candidates — Stage 1b (`run_stage2_2_find_chm_derived_shrub_candidates.py`)

Shrub is simultaneously the hardest class to hand-label (small, isolated, numerous) and the weakest in every prior result. Drawing it by hand is the bottleneck. **Generate shrub candidates automatically from CHM instead, and let hand labeling validate rather than draw.**

Shrub is *defined* by CHM in [`H_GRASS_MAX`, `H_TREE_MIN`) (§3), so the delineation is already specified — no new criterion is being introduced.

**Procedure**, per tile, reusing the crown machinery v3 built for trees (`instructions2.md` §4.1):

1. Threshold CHM to the shrub band: `H_GRASS_MAX <= CHM < H_TREE_MIN`.
2. Connected-component label the result; discard components below the shrub minimum area (2 m²) unless retained as point labels.
3. Emit each component as a **candidate shrub polygon** into an editable GeoPackage with `class_code` pre-filled as 2, a `source` field of `chm_candidate`, and `reviewed` / `rejected` flags at 0.
4. The analyst **accepts, edits, or rejects** each candidate against 10 cm RGB — a single click for the common case instead of drawing a crown outline:

   | Action | Fields |
   |---|---|
   | accept | `reviewed = 1`, leave `class_code = 2` |
   | edit | adjust geometry, `reviewed = 1`, keep `class_code = 2` |
   | reject | `reviewed = 1`, `rejected = 1`, clear `class_code` |

5. **`reviewed` exists so an accepted candidate is distinguishable from one never looked at.** A zero-click accept would make the two identical, which is unworkable at 150 per tile. Pending candidates render magenta — the same unlabelled convention the zone and polygon layers use — and recolour to shrub brown once reviewed.
6. Rejected candidates are retained with `class_code` cleared and `rejected = 1`, since a CHM-band object that is *not* shrub is itself informative about where the CHM rule fails.

**Isolation makes this easier, not harder** — well-separated crowns are exactly the case connected-component labeling handles cleanly, with none of the merged-canopy ambiguity that complicates tree delineation.

**The full candidate set is provenance; hand review works from a stratified subset.** Measured at SRER 2022, the shrub band yields **~20,000 components per train tile** (78,733 candidates across five tiles). Reviewing that is far more work than drawing shrubs by hand, which would defeat the purpose. So:

- Write the **full candidate set** as provenance, unreviewed.
- Draw a **review subset** of `review_sample_per_tile` candidates (150), **stratified by area** into `review_area_strata` bands (3), seeded `BASE_SEED + year`.
- Stratification is not optional. An unstratified draw is dominated by the smallest components, which reintroduces size bias — merely inverted — and size bias is the exact failure this mechanism exists to remove.
- Sub-minimum **point candidates are excluded from review**, since they are the least reliable population; they remain in the full set.

150 per tile against a 50-per-class-per-role gate leaves comfortable margin for rejections.

> **These counts are not an artifact.** The initial reading was that ~6,000 sub-2 m² components per tile must be CHM noise. **Measurement disproved that** (§3 safeguard 2): NEON pre-thresholds CHM at 0.7 m, so every component is a real object of at least that height. SRER genuinely carries on the order of 20,000 shrub-band objects per km², which is consistent with creosote and small mesquite density. The count is ecology, not error.

**Two constraints:**
- **Candidates are proposals, not labels.** An unreviewed candidate never enters training. The CHM rule is the reference-path definition (§3) and inherits every CHM error; hand validation is what makes it ground truth.
- **NEON sites only.** This is a labeling accelerator for sites with reliable CHM. It does not transfer to NAIP-only sites and creates no dependency in `RF-A_A`–`RF-A_C` — the labels it produces are ordinary hand-validated polygons, indistinguishable downstream.

Track accept/reject rate per tile: a low acceptance rate is direct evidence that `H_GRASS_MAX` or `H_TREE_MIN` needs revisiting (§3 safeguards 1–2).

#### Scripts for §4.2

**Run in this order.** The letter in each filename *is* the execution order. Scripts were renamed to make that true — shrub candidates and the QGIS project builder were both written after the progress check, so the letters no longer reflect the order they were authored in, and should not be read that way.

| Order | Script | What it does |
|---|---|---|
| 1 | `run_stage2_1_find_kmeans_cluster_labeling_zones.py` | K-means (k=16), cluster maps, candidate sites, empty polygon templates |
| 2 | `run_stage2_2_find_chm_derived_shrub_candidates.py` | CHM-derived shrub candidates, per-component CHM statistics, stratified review subset |
| 3 | `run_stage2_3_create_qgis_labeling_project.py` | builds the QGIS labeling project — styled, editable, saves the `.qgz` |
| 4 | `run_stage2_4_check_hand_labeling_progress.py` | progress and **gate check** — run repeatedly while labeling |
| — | `run_stage3_3_create_qgis_results_project.py` | builds the QGIS **results** project. Runs after `run_stage3_1_random_forest_ground_truth_classification.py`, not during labeling, so it sits outside the 1–4 sequence |

Settings live under `stage2_1_labeling_zones` (zones) and `stage2_2_shrub_candidates` (shrub candidates) in the site config.

> ### Which scripts write to your labels, and which do not
>
> | Script | Touches drawn work? |
> |---|---|
> | `run_stage2_3_create_qgis_labeling_project.py` | **No.** `project.clear()` clears the in-memory project only; `project.write()` writes the `.qgz`. GeoPackages are read-only data sources. Re-run it freely to pick up new layers — only project structure and styling are rebuilt |
> | `run_stage2_4_check_hand_labeling_progress.py` | **No**, except with `--fill`, which writes derived `tile` and `cluster_id` back |
> | `run_stage2_1_find_kmeans_cluster_labeling_zones.py` | **Yes** — writes an *empty* template over `training_polygons_*.gpkg`. **Guarded**: refuses to overwrite a file holding polygons unless `--force` |
> | `run_stage2_2_find_chm_derived_shrub_candidates.py` | **Yes** — rewrites `shrub_review_*.gpkg`. **Guarded**: refuses once the file holds any `reviewed` or `rejected` decision unless `--force`. The full `shrub_candidates_*` set carries no analyst input and is always rewritten |
>
> Both guards report what they kept and why, and skip that tile rather than failing. Refreshing zones or candidates is a normal thing to want; losing a week of labeling to it is not. **The one real way to lose work remains unsaved QGIS layer edits** — see the warning below.

#### How to run Stage 2 — `run_stage2_3_create_qgis_labeling_project.py`

Stage 1 (`run_stage2_1_find_kmeans_cluster_labeling_zones.py`) writes the layers; **`run_stage2_3_create_qgis_labeling_project.py` loads them into QGIS styled and ready to edit, and saves the project**. Run it once per site — after that, just reopen the saved `.qgz`.

It is a QGIS Python script rather than a hand-authored `.qgs` on purpose: the project XML schema is version-specific and degrades silently when it does not match the running QGIS, while the Python API does not. The `.qgz` is produced either way.

**1. Point it at the site config.** `CONFIG` at the top of the script is the only line to edit — every path derives from it. Update it when moving between sites, or between a local checkout and the SCC:

```python
CONFIG = os.path.expanduser(
    "~/Documents/GitHub/PLSP/code/code_ground_truth_land_cover/v4/config/srer_2022.json"
)
```

**2. Run it in QGIS.** `Plugins → Python Console → Show Editor → Open Script…` → select `run_stage2_3_create_qgis_labeling_project.py` → **Run**. QGIS must be its own conda environment (§7A) — it pulls a large Qt stack and must not contaminate the pipeline environment.

**3. Reopen later** from `results/v4/stage2_labeling/labeling_{SITE}_{YEAR}.qgz`. Re-running the script rebuilds the project from scratch and **clears the current one**, so reopen the `.qgz` rather than re-running once labeling has started.

**Layer tree** — one collapsed group per tile, layers in panel order top to bottom:

| Layer | State | Role |
|---|---|---|
| `POLYGONS … <- draw here` | editable, legend expanded | the deliverable — draw labelled polygons |
| `zones … (role)` | editable, legend expanded | the 30-per-cluster-per-role candidate sites |
| `shrub review …` | editable, legend expanded | the 150-per-tile CHM candidate subset — accept or reject. The full `shrub_candidates_*` set is **not loaded**: at ~14k features per tile it would make the project unusable |
| `clusters k16` | **off**, 75% opacity | provenance only — never map a cluster to a class |
| `CHM` | on, 75%, 0–5 m ramp | the shrub/tree call (`H_TREE_MIN` = 2.0 m) |
| `SAVI` | on, 75%, 0–0.6 ramp | the bare/grass call |
| `RGB 10cm` | on, bottom | the basemap actually being labelled from |

**Editing conventions the script sets up:**
- `class_code` is a **dropdown** (bare / grass / shrub / tree) via a ValueMap widget, so the analyst picks a name and QGIS stores the locked §3 integer — mistyped codes are not possible.
- Symbology uses the **locked §3 `CLASS_COLORS`**, with unlabelled features in magenta `#e6007e`, deliberately outside the earth-and-green family so unfinished work is obvious at a glance. Expanding the two editable layers' legends shows the per-class counts, which is the fastest way to track progress against the ≥ 50-per-class-per-role requirement.
- `tile` and `cluster_id` on the polygon layer are **auto-filled at draw time** by QGIS default-value expressions (`tile` a literal per layer; `cluster_id` = `raster_value(<cluster layer>, 1, point_on_surface($geometry))`, which recomputes if the polygon is reshaped). Both are set **read-only** in the form and shown in a hover map tip, so they can be checked while drawing but not typed over. Only `class_code` is yours to enter.
- Cluster maps load unchecked deliberately: having them visible while labeling invites exactly the cluster→class mapping this section forbids.

> ### ⚠️ Save layer edits, or nothing reaches disk
>
> **QGIS holds edits in an in-memory buffer and does not write to the GeoPackage until the layer is saved.** Toggle editing per layer, fill or draw, then **Save Layer Edits** (or toggle editing off and confirm). Saving the *project* (`.qgz`) does **not** save layer edits — they are separate actions.
>
> This has already bitten once: three polygons were drawn and `run_stage2_4_check_hand_labeling_progress.py` reported zero, because the features existed only in QGIS's buffer. The file's modification time had changed, which makes it look saved — it is not. **If 2c reports fewer polygons than you drew, the first thing to check is unsaved edits**, not the script.

#### Three conventions for drawing

**A zone is a place to look, not a quota of one. Draw as many polygons per zone as the neighbourhood supports.** If arriving at a candidate site reveals two or three further clean examples nearby, label them all — the zone has done its job by sending the analyst somewhere informative, and extra polygons there are free training data. **Zones and polygons are therefore one-to-many, and the two counts are deliberately decoupled**: 782 zones against a 400-polygon minimum, so neither number bounds the other. A tile showing more polygons than filled zones is expected and correct, not a double-count.

> Note the consequence for classes: the extra polygons near a zone **need not share the zone's class**. Visiting a tree zone and finding a clean bare patch beside it is a good reason to draw a bare polygon there. Only a polygon that actually *contains* the zone point should agree with that point's `class_code`.

**Fill `class_code` on the zone points as well as drawing polygons.** The polygons are the deliverable — the ≥ 50-per-class-per-role gate counts polygons, and a point contributes nothing to training. But setting the point's `class_code` costs one dropdown click and buys three things: an unambiguous visual worklist (points stay magenta until visited, so what remains in a tile is readable at a glance), a progress percentage via 2c's check 6, and an independent record of the call made at that location. As points are filled they recolour from magenta to their class colour and leave the `unlabelled` category in the legend. The point records the class **at the point**, which is why a nearby polygon of a different class is not a contradiction.

**Polygons must be class-pure. They need not be cluster-pure, and overlapping cluster boundaries is fine — often desirable.** The binding constraint is that every 1 m pixel inside a polygon is genuinely the same class, because the polygon is rasterized to the 1 m grid and *each pixel* becomes a training sample; a polygon straddling a crown edge or a bare/grass transition injects exactly the mixed pixels that are least reliable. Cluster boundaries carry no such weight — a polygon spanning three clusters that are all bare is not a mistake but **evidence**, and precisely the phenomenon `instructions1.md` §6 found when bare fragmented across four clusters at k=10. Constraining polygons to single clusters would quietly teach the classifier that the clustering was correct, which is the failure mode this section exists to prevent. **Draw to the object, not to the cluster.**

> One consequence to be aware of: the stored `cluster_id` is the **modal** cluster, so it is lossy for a polygon that spans several. Coverage accounting is unaffected — 2c's check 2 rasterizes each polygon and credits *every* cluster it touches — but the §4.2 free diagnostic (cross-tabulating `cluster_id` against `class_code`) should use the all-clusters-touched form rather than the stored modal field.

#### Checking progress — `run_stage2_4_check_hand_labeling_progress.py`

**Run this repeatedly while labeling, not once at the end.** It reads the GeoPackages the analyst is editing and reports whether Stage 2 is finished, so a shortfall is visible in time to fix cheaply rather than being discovered at Step 1d. It is read-only.

```bash
python run_stage2_4_check_hand_labeling_progress.py config/srer_2022.json          # console report
python run_stage2_4_check_hand_labeling_progress.py config/srer_2022.json --json   # also writes labeling_progress_{SITE}_{YEAR}.json
```

Six checks, the first five of which gate Step 1d:

| # | Check | Gate |
|---|---|---|
| 1 | ≥ 50 polygons per class **per tile role** | yes |
| 2 | every cluster has received at least one labeled polygon | yes |
| 3 | per-class size distribution, and a separate list of below-floor polygons | **no** — below-floor polygons are ignored, not faults |
| 4 | geometry validity, and polygons extending outside their tile | yes |
| 5 | `class_code` inside the locked §3 set 0–3 | yes |
| 6 | candidate-site fill rate per cluster and per role | reported |

It ends with `GATE PASSED` or `GATE NOT PASSED` plus exactly what is outstanding — how many polygons short per class per role, which clusters still have no label, and how many geometry or `class_code` faults to fix.

**Cluster coverage (check 2) is measured by rasterizing each polygon onto the 1 m analysis grid and reading the cluster map underneath**, not from the `cluster_id` attribute. The analyst draws freely and is not required to fill that field, and what matters is which part of feature space actually received labels.

Two behaviours worth knowing: a polygon extending outside its tile is flagged and fails the gate but **still counts** toward its class total — it is a real label, merely clipped, and the gate stops it passing silently. A polygon with a `class_code` outside 0–3 is excluded from the counts entirely, since there is no class to credit it to.

#### Outputs

- `labeling_zones_{tile}_{YEAR}.gpkg` — candidate sites with `cluster_id`, `class_code` (empty, to be filled)
- `cluster_map_{tile}_{YEAR}.tif` — the k-means map, retained for provenance and the diagnostic below
- `training_polygons_{tile}_{YEAR}.gpkg` — the delivered hand labels, attribute `class_code` (0–3)
- `training_labels_{tile}_{YEAR}.tif` — rasterized to 1 m (uint8, 255 = unlabeled)

#### A free diagnostic worth keeping

After labeling, cross-tabulate `cluster_id` against the assigned `class_code`. A cluster receiving a single class is a clean, separable region of feature space. A cluster receiving several classes is genuinely ambiguous **in the features the classifier will use** — which both predicts where the `RF-A_*` variants will struggle and independently corroborates the §3.1 confidence layer. Record the cross-tabulation; it costs nothing beyond labels already collected.

> ### Results to date
>
> Step 1d has been run at SRER 2022 for `RF-A_A` through `RF-A_D`, across two runs (baseline, then polygon subsampling). Full results, per-class scores, confusion matrices, the measured circularity in `RF-A_D`, and the data-input caveats that constrain all of it: **[`results/stage3_1_results.md`](../results/stage3_1_results.md)**. Metric definitions and how to read them: **[`results/stage3_1_definitions.md`](../results/stage3_1_definitions.md)**.
>
> Headline: texture is the decisive input (`RF-A_C` shrub F1 0.658 against 0.455 without it), and `RF-A_C` is the best transferable variant.

### 4.3 The two random forests — RF-A and RF-B

**These are two different models with different inputs, different targets, and different failure modes. Do not conflate them.** Both understandings below are confirmed and settled.

| | **RF-A** — ground truth | **RF-B** — phenology |
|---|---|---|
| Pipeline step | Step 1d | Step 5 |
| Input features | 1 m spectral / texture / VI / CHM stack (per `RF-A_*` variant) | 10 PlanetScope LSP timing metrics |
| Spatial unit | 1 m pixel (via SLIC segment) | One Planet pixel (**3 m, measured**, native) |
| Model form | **Classifier** — hard class per pixel | **Multi-output regressor** — 4 fractions per pixel |
| Fractional cover obtained by | Aggregating N×N = **9** hard 1 m labels to a Planet block (Step 3) | Predicted directly by the model |
| Training data | §4.2 labels (hand polygons or clustered+annotated) | **All valid Planet blocks — mixed and pure alike** |
| Role of pure end members | n/a | **Anchors and validation only** |

**RF-A — confirmed.** Classify at 1 m, then aggregate 9 × 1 m pixels into one Planet block for fractional cover. Fractions are counts of hard labels, so they are exact by construction. There is no calibration question here.

**RF-B — confirmed.** Planet's native pixel *is* the 3 m cell, so there is no sub-pixel level to classify and aggregate; the model maps one 3 m feature vector directly to four fractions. Training uses **all pixel types, not only pure ones** — Step 3 supplies true fractions for every valid block across all five tiles, so the regressor sees the full mixing range. **Pure end members serve as anchors and validation**, and as the transfer basis at Step 8 — they are not the training set.

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

> **"Same ecoregion" means "same NEON domain" — RESOLVED (§12 Q4).** The NEON domain is the operative unit throughout this project. Every NEON/AmeriFlux pair in §2.1 sits within one domain by construction, so **all pairs are model-transfer tests**, and cross-domain work (Axis 2) is pipeline-transfer only.
>
> | Pair | Domain | Transfer kind attempted |
> |---|---|---|
> | **D14** — SRER / WKG | D14 Desert Southwest | Model + end-member (Route 1), then pipeline (Route 2) |
> | **D15** — ONAQ / Rws | D15 Great Basin | Model + end-member (Route 1), then pipeline (Route 2) |
> | **D13** — MOAB / WJS | D13 S. Rockies & Colorado Plateau | Model + end-member (Route 1), then pipeline (Route 2) |
> | **D06** — KONZ / *partner unresolved* | D06 Prairie Peninsula | Blocked pending a PLSP-eligible partner (§2.1a) |
>
> One caveat to record rather than resolve: a NEON domain is a broad climate/vegetation envelope, coarser than an EPA Level III ecoregion, so a within-domain pair may still span a finer ecological boundary — D14's SRER/WKG separation crosses a Sonoran/Madrean/Chihuahuan semi-desert grassland transition. **Record the separation distance and any EPA Level III boundary crossed for each pair** in the run report, so that a Route 1 failure can be interpreted against how far the transfer actually reached. This does not change the design: domains define scope, the record explains outcomes. **D15 is now the cleanest test** — exact IGBP match and a 2 mm MAP difference, so a Route 1 failure there is attributable to sensor and resolution with ecology near-constant (§2.1). D06's co-location rationale is void (§2.1a).

#### R2 — Common analysis scale, fixed before feature extraction

NEON RGB is 10 cm; NAIP is 60 cm. **Texture is scale-dependent**: GLCM and LBP computed at 10 cm then aggregated to 1 m are *not the same feature* as the same statistics computed at 60 cm. A model trained on one and applied to the other is reading a different variable under the same column name.

**Rule**: define a single `TEXTURE_SCALE` constant, set to the coarsest resolution across all target sites (**0.6 m**, set by NAIP). At SRER, degrade 10 cm RGB to 0.6 m *before* computing texture. Never compute texture at native NEON resolution for any feature that feeds `RF-A_A`–`RF-A_C`.

Native 10 cm is retained only for (a) visual inspection and manual labeling, and (b) DeepForest (`RF-A_DL1`), which is inherently resolution-specific and is not claimed to be scale-transferable.

#### R3 — No absolute spectral thresholds in transferable frameworks

NEON vegetation indices come from an atmospherically corrected, BRDF-corrected, narrowband hyperspectral product. NAIP is uncorrected 8-bit broadband DN, mosaicked across dates and sun angles. **SAVI = 0.2 does not mean the same thing in both.** A hard-coded threshold will silently mis-segment at the transfer site.

**Rule**: absolute thresholds (`SAVI_BARE_MAX` etc.) are permitted **only** in the reference/labeling path at SRER (§3, `RF-A_D`). Transferable variants `RF-A_A`–`RF-A_C` must use either:
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

> **Action — user input required.** NAIP flyover dates per site must be supplied before any transfer site is processed; they cannot be inferred. Request them explicitly rather than proceeding on an assumed summer acquisition.

#### R8 — Site-specific values live in config, never in code

Planet pixel size, grid origin, UTM zone, tile lists, acquisition dates, class priors, and all thresholds are per-site config entries. Adding a site must be a new config file plus data, with zero code edits. This is the practical test of every rule above.

---

## 5. Processing pipeline

### Step 0 — Data audit and grid definition

**Run the full checklist in §11 before any other step.** No downstream step begins until §11 passes, or until a specific failure is explicitly waived in writing and recorded.

Step 0 finalizes four values that every later step reads from config: the Planet aggregation factor `N`, the Planet grid origin, `TEXTURE_SCALE`, and `H_GRASS_MAX`.

Outputs → `stage1_data_and_features/qa/`, as both a machine-readable `data_audit_{SITE}_{YEAR}.json` and a human-readable summary.

**Scripts, in execution order:**

| | Script | Produces |
|---|---|---|
| 0a | `run_stage1_1_download_neon_tiles.py` | NEON AOP tiles for the three products, filtered to the site AOI |
| 0b | `run_stage1_2_data_audit.py` | §11 checklist → `data_audit_{SITE}_{YEAR}.json` |
| 0c | `run_stage1_3_define_planet_grid.py` | Planet grid geometry and the nested 1 m grid → `planet_grid_{SITE}_{YEAR}.json` + `.gpkg` (§11 checks 10–12a) |
| 0d | `run_stage1_4_create_qgis_grid_verification_project.py` | QGIS project for the **mandatory visual alignment pass** (run in `LCSC_QGIS`) |

**0c and 0d are a pair and neither substitutes for the other.** 0c proves the grid is arithmetically self-consistent; only 0d proves it sits on the imagery. A CRS mislabelled in the LSP product, a centre/edge half-pixel error, or an LSP-to-AOP coregistration offset would each pass every numeric check in 0c and still place every block on the wrong ground.

### Step 1 — Per-pixel classification at 1 m

**1a. Feature construction.**

> **Order of operations is fixed and non-negotiable: compute every spectral index at 0.6 m, then aggregate to the 1 m analysis grid.** The indices below are nonlinear ratios, so computing at 0.6 m then averaging gives a *different number* from averaging to 1 m then computing. Both paths are defensible in isolation; only one can be used at both NEON and NAIP, because 0.6 m is the finest resolution NAIP can supply. Computing at native 10 cm would produce NEON features that NAIP can never reproduce, silently breaking R2 and every model-transfer claim (R1).

**Why RGB-only indices at all**: the NEON RGB camera (`DP3.30010.001`) is **3-band with no NIR**, so **framework A** must separate vegetation from soil using visible bands alone.

> **RESOLVED — the §4.1 table is authoritative.** `A = RGB`, `B = RGB + vegetation indices`, `C = RGB + vegetation indices + texture`. **NIR is permitted in `RF-A_A`–`RF-A_C`**: NAIP is 4-band, so a NIR-bearing feature *is* reproducible at the transfer site. An earlier draft of this section restricted NIR to B/D/E and stated that A and C use visible bands alone, which contradicted §4.1; that restriction applies to **`RF-A_A` only**.
>
> The visible-band indices below stay in `RF-A_A` — they are deterministic functions of RGB and add no input layer, so each framework adds exactly one input family to the one before it. That nesting is what makes the `RF-A_*` comparison a clean deconfounding of inputs (§4.1).
>
> **Caveat to carry into interpretation**: B–E inherit a cross-sensor mismatch that A does not. NEON NIR indices come from the imaging spectrometer — narrowband, BRDF- and atmospherically corrected, native 1 m — while NAIP NIR comes from an uncorrected broadband camera at 0.6 m. Same index name, different measurement, so **R3 is binding for B–E**: percentile or learned boundaries, never absolute thresholds. Framework A is the only one whose features come from the same kind of sensor at both sites, which makes an A-vs-B/C gap at the transfer site partly a sensor-provenance effect rather than purely an information effect.

**Chromatic coordinates** — the base for every index below:

```
r = R / (R+G+B),    g = G / (R+G+B),    b = B / (R+G+B)
```

Normalizing out total brightness is the point: it suppresses illumination and shadow effects, which at these resolutions in mesquite savanna are a first-order confuser. Indices are built on `r, g, b`, never on raw DN.

**Visible-band vegetation indices** — all exploit chlorophyll absorbing red and blue while reflecting green:

| Index | Definition | Reference | Why it is included |
|---|---|---|---|
| **ExG** | `2g − r − b` | Woebbecke et al. 1995 | Standard green-vegetation vs. soil separator in RGB |
| **ExR** | `1.4r − g` | Meyer & Neto 2008 | Not used alone; the excess-red term inside ExGR |
| **ExGR** | `ExG − ExR` | Meyer & Neto 2008 | Subtracting excess-red suppresses **reddish-brown litter and standing dead** — directly targets the bare vs. senesced-grass pair that `instructions1.md` §4 identifies as degenerate |
| **VARI** | `(g − r) / (g + r − b)` | Gitelson et al. 2002 | Built to resist atmospheric variation — the right choice for **uncorrected NAIP** |
| **GLI** | `(2G − R − B) / (2G + R + B)` | Louhaichi et al. 2001 | Developed on rangeland; normalized form, low illumination sensitivity |

Four indices rather than one because they fail differently — ExG is strongest on green biomass, ExGR on litter, VARI under variable atmosphere, GLI under variable illumination. The classifier resolves which matters where; do not pre-select.

**Brightness and saturation**: Rec. 709 luma `0.2126R + 0.7152G + 0.0722B` (shared with shadow detection, Step 1c) and HSV saturation. Soil and vegetation separate in saturation; shadow is low-value.

**NIR indices — frameworks B, C, D, E** (all except A): SAVI first, NDVI second. SAVI's soil-adjustment factor is what makes it the dryland choice, since NDVI saturates against bright soil background at low cover (§2A index priority).

**Texture** at 0.6 m, aggregated to 1 m: GLCM (contrast, homogeneity, entropy, correlation), LBP, moving-window standard deviation.

**CHM** at 1 m, `RF-A_D` only. **Native 10 cm RGB** is retained only for manual labeling and `RF-A_DL1`.

> **All absolute index thresholds are reference-path only** (R3). NEON RGB and NAIP differ in radiometry, bit depth, and correction, so an index value does not carry the same meaning at both. `RF-A_A`–`RF-A_C` use percentile thresholds or learned boundaries.

**1b. Segmentation.** SLIC at 1 m, ~100k segments/tile. Per-segment spectral, texture, shape, and context features.

**1c. Shadow detection** (reinstated from v3, `instructions2.md` Phase 3):
- Rec. 709 luma (`0.2126R + 0.7152G + 0.0722B`) thresholded at the **pooled 20th percentile** — percentile-based per R3, so it transfers — combined with a blue-shift rule (`B > R`). Computed at `TEXTURE_SCALE` (0.6 m) per R2, not at native 10 cm.
- Aggregate to 1 m via > 70% majority.
- **Shadow is its own class.** Resolution rule: shadow within `SHADOW_TREE_RADIUS` (5 m) of CHM >= `H_TREE_MIN` is **assigned to tree**; all remaining shadow is **ignored** (masked to nodata, excluded from training, from Step 3 aggregation denominators, and from accuracy assessment).
- At transferable variants (`RF-A_A`–`RF-A_C`) without CHM, the tree-proximity test uses the framework's own predicted tree mask.

**1d. Classification.** RF over segments, trained on the §4.2 hand labels, leave-one-tile-out CV within the train block, evaluated on the held-out test tiles.

**1e. Soft outputs.** Retain the classifier's per-class probability vector for every pixel and derive the confidence and margin layers per §3.1. **Do not harden and discard the probabilities** — Step 3 soft aggregation and Step 5 sample weighting both depend on them.

Outputs per framework → `stage3_classification/{framework}/`:
`classification_{framework}_{tile}_{YEAR}.tif` (uint8, 0–3, 255 nodata),
`class_probability_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands in class-code order),
`prediction_quality_{framework}_{tile}_{YEAR}.tif` (float32, [0,1]),
`margin_{framework}_{tile}_{YEAR}.tif` (float32),
`shadow_mask_{framework}_{tile}_{YEAR}.tif` (uint8).

### Step 2 — PlanetScope QA masking

Apply the PlanetScope QA layers to remove faulty Planet pixels: **`NumCycles == 1` strict (layer 1) AND `QA ∈ {1, 2}` (layer 12)**. Full layer specification and handling traps are in §5.2–5.3 — in particular, **fill (32767) must be masked to NaN and cast to float before any arithmetic**, or derived durations silently evaluate to 0 on fill pixels.

QA-failing Planet cells are excluded from Steps 3–5. Log retention statistics, and log the `numObs` (layer 24) distribution for the surviving pixels.

### Step 3 — Aggregation to PlanetScope scale

**N × N** blocks (N from Step 0, **measured N = 3 at SRER**) **aligned to the PlanetScope pixel grid**, over each framework's 1 m hard classification → percent cover per class per block.

> **RESOLVED — the grid is measured, not assumed. `run_stage1_3_define_planet_grid.py`, SRER, from `US-xSR_..._PLSP_2021.nc` (2026-08).**
>
> An earlier draft of this section said "expected 4". **That was wrong.** The PlanetScope LSP pixel is **3 m**, so **N = 3** and a Planet block is **9 one-metre cells, not 16**. Every downstream count that assumed 16 changes accordingly.
>
> | Quantity | Measured value |
> |---|---|
> | CRS | `EPSG:32612` — matches `expected_crs` |
> | Planet pixel | **3.0 m**, square, uniform to 1e-6 m |
> | `N` | **3** (`analysis_grid_m` = 1.0) |
> | Grid size | 3333 × 3334 Planet cells |
> | Origin (cell **edge**) | 510555.0, 3535548.0 |
> | Extent | x [510555.0, 520554.0], y [3525546.0, 3535548.0] |
>
> **Coordinates in the netCDF are cell CENTRES** (`x[0] = 510556.5`). The half-pixel shift to edges is applied once, in Step 0c, and nowhere else. Applying it twice — or not at all — offsets every block by 1.5 m and no numeric check would catch it.
>
> **Nesting passes**: edges land on whole metres and 3 divides evenly into 1 m cells, so the 1 m analysis grid nests exactly inside the Planet grid site-wide, with no fractional remainder anywhere.
>
> **NO SRER TILE IS CONGRUENT WITH THE PLANET GRID, and this is load-bearing.** NEON tiles are 1 km on a 1 km origin; 1000 is not divisible by 3, so tile origins fall at offsets of 0, 1 or 2 m modulo the Planet pixel. Measured, all ten:
>
> | tile | role | offset x | offset y |
> |---|---|---|---|
> | 511000_3527000 | train | 1 | 2 |
> | 511000_3528000 | train | 1 | 0 |
> | 511000_3529000 | train | 1 | 1 |
> | 511000_3532000 | train | 1 | 1 |
> | 515000_3526000 | train | 2 | 1 |
> | 519000_3527000 | train | 0 | 2 |
> | 515000_3530000 | test | 2 | 2 |
> | 515000_3531000 | test | 2 | 0 |
> | 518000_3529000 | test | 2 | 1 |
> | 520000_3532000 | test | 1 | 1 |
>
> `519000_3527000` is congruent in x and not in y, which still makes it non-congruent — **zero of ten tiles align in both axes.**
>
> **Consequence, binding on Step 3**: tile boundaries cut through Planet cells on all four edges of every tile. **Aggregate from a site-wide mosaic, or carry a one-cell halo per tile — never block each tile in isolation.** Doing so would emit partial blocks all around every tile, which would then fail the retention rule and be dropped, silently deleting the tile perimeter from the fraction product.
>
> **DO NOT SHIFT OR RESAMPLE THE TILES TO REMOVE THE OFFSET. This is the natural reading of the offset table and it is wrong.** The 1 m data is already aligned to the Planet grid, exactly: 1 m cells sit on whole-metre boundaries, Planet cell edges sit on whole metres divisible by 3, so **every 1 m cell lies wholly inside exactly one Planet cell, everywhere on the site, with no cell ever split.** Nothing is misaligned and there is nothing to correct.
>
> What the offset actually describes is **where the first whole block starts inside a tile's array** — not where the ground is. A tile with `offset_x = 1` has its first complete Planet column beginning at array column 2, and its leading 2 columns belonging to a Planet cell shared with the neighbouring tile. **That is an indexing fact, resolved by index arithmetic and a halo, not by moving pixels.**
>
> Shifting a tile by 1 or 2 m to make its corner land on a block boundary would translate the imagery **off its true ground position** by that amount. The grid arithmetic would then look tidy while every 1 m label sat 1–2 m from where it was observed, corrupting every fraction while passing every alignment check. **The offsets are a property of the NEON 1 km tiling scheme (1000 is not divisible by 3), not a defect in the data, and they require no correction at all.**
>
> **TILES ARE CROPPED TO THE PLANETSCOPE FOOTPRINT**, which is the smaller SRER focus area rather than the full AOP flight box. Ground truth outside it has no Planet pixel to aggregate into and cannot enter Steps 3–6, so carrying it forward would inflate a ground-truth area against a Planet denominator that does not exist. At SRER **`520000_3532000` (test) is 55.4% inside** and is the only tile affected; the remaining nine are wholly inside. The cropped extents are written as their own GeoPackage layer so the loss is visible rather than implied.
>
> Config: `stage1_3_planet_grid` block. Outputs: `stage1_data_and_features/qa/planet_grid_{SITE}_{YEAR}.json` and `.gpkg`. Visual verification: `run_stage1_4_create_qgis_grid_verification_project.py`, run in the `LCSC_QGIS` environment — **required before any Step 3 aggregation**, because the numeric checks prove the grid is self-consistent, not that it sits where the imagery sits.
>
> **VISUAL VERIFICATION PASSED — 2026-08-18, `run_stage1_4_create_qgis_grid_verification_project.py`, all ten tiles.** Confirmed in QGIS against the 10 cm RGB: the tabulated offsets are correct **in both direction and magnitude**, each reading as the stated whole number of 1 m pixels between the NEON tile boundary and the nearest Planet cell edge. The 1 m analysis cells nest three-across inside each 3 m Planet cell with no sliver, and the grid is not systematically displaced against the imagery — so the centre-to-edge half-pixel convention is applied correctly and there is no LSP-to-AOP coregistration offset at this scale.
>
> **This is the Step 3 gate, and it is now cleared** for the geometric checks. It does not need re-running unless the LSP product, its grid, or the tile list changes — if any of those change, both Step 0c and Step 0d run again before Step 3.
>
> **ADDED AFTER THE FIRST PASS — the grid is also verified against the real LSP data, not only against the cell outlines.** Step 0c exports one LSP layer (`stage1_3_planet_grid.verification_variable`, default **`EVIamp`**) as a GeoTIFF whose transform is built from the origin and pixel size **measured by Step 0c itself**, and Step 0d loads it beneath the outlines. This closes a gap the first pass could not: checks 1–4 compare grid outlines against the AOP imagery and against each other, so a systematic error in the *Planet* georeferencing would pass all of them. Ticking `EVIamp` on shows where the Planet data actually is — its pixel blocks must coincide exactly with the 3 m outlines, and toggling it against the RGB must put dark `EVIamp` over woody canopy and bright over bare ground.
>
> `EVIamp` is the right layer for this: near-complete coverage (121 fill pixels of 11,112,222 at SRER 2021, range 0.0115–0.5359 after scaling) and a spatial pattern that tracks woody against herbaceous cover, so it is legible against 10 cm RGB by eye. It is `Int16` with `scale = 0.0001` and fill `32767` — masked to NaN and scaled **before** any arithmetic, per §5.3 trap 1.

> **"Moving window" in the source documents means grid-aligned, one block per Planet pixel** — the window steps by N, landing exactly on Planet pixel boundaries. It does **not** mean a stride-1 sliding window. Blocks are therefore non-overlapping and stand in one-to-one correspondence with Planet pixels, which is what Step 5 requires: a stride-1 window would produce overlapping, non-independent samples that cannot be matched to Planet pixels.

> **This grid is built first, before any other processing — it is the spine of the whole pipeline.** Every 1 m product, every N×N block, and every fraction estimate is defined relative to it, so an error here propagates silently into all of them and cannot be corrected downstream.
>
> **Procedure**, run once per site:
> 1. Read the site's PlanetScope LSP netCDF and extract its CRS, pixel size, and grid origin (§11 checks #10–12). This file is the authority — nothing else defines the grid.
> 2. Generate the **1 m analysis grid** nested exactly inside the Planet grid, so that `N` whole 1 m cells tile each Planet pixel with no fractional remainder.
> 3. Export **both grids as GeoPackage** in the site's native UTM, so alignment can be confirmed by visual inspection in QGIS against the NEON tiles before anything is computed.
>
> **SRER runs first as the worked example**: the user supplies the Planet netCDF, and the 1 m grid and GeoPackage are generated from it and visually verified. No Step 1 processing begins until that inspection passes.

Shadow-masked and nodata 1 m pixels are excluded from the denominator.

> **RESOLVED — the retention rule is a COUNT, not a percentage: a block is kept only at 8 or 9 valid pixels of 9.** At most **one** masked pixel. Config: `stage4_1_aggregation.min_valid_pixels_per_block: 8`. This supersedes the earlier "< 75% valid" wording, which was written when N was assumed to be 4.
>
> **Why a count.** At N = 3 a percentage is misleading, because there are only three cuts available: 7, 8 or 9 of 9. 75% of 9 is 6.75, so the old rule quantised to **≥ 7 of 9** — it would admit a block with **two of nine pixels missing, 22% of its area unobserved**, while appearing to enforce a 75% standard. Stating the count removes the gap between what the rule says and what it does.
>
> **Why 8 and not 7.** These blocks are the ground-truth fractions that RF-B trains on and that Step 6 areas are estimated from. A 7-of-9 block's fractions are quantised in ninths of an *incomplete* denominator, and the missing pixels are not missing at random: shadow is 79% of masked pixels on the train tiles and clusters against woody canopy (`results/stage3_1_results.md` §1.7), so admitting sparse blocks preferentially biases shrub and tree — the two classes already weakest, and shrub is already under-predicted by 14.7%. Tightening to 8 costs blocks; admitting 7 costs correctness in exactly the place the product is most fragile.
>
> **Report the cost, do not hide it.** The 0–9 valid-pixel histogram (below) is required output precisely so the number of blocks lost at 8 versus 7 is visible and revisable. If retention at 8 proves severe enough to threaten sample size for shrub or tree, that is a finding to record and decide on — not a reason to loosen the threshold silently.

**Valid-block accounting — a required output, not a log line.** For every framework and tile, report the full distribution of valid 1 m pixels per block:

- **Histogram of valid-pixel count per block**, over the complete range **0 to N²** (**0–9 at the measured N = 3**) — every count gets a bin, including 0.
- **Percentage of blocks at each count**, so block composition is readable as a distribution rather than a single retention figure.
- Counts and percentages both before and after the drop rule, **and at both 8-of-9 and 7-of-9**, so the cost of choosing the stricter cut is a measured number rather than an assumption.

This exposes whether blocks are being lost uniformly or concentrated along tile edges, shadow, and nodata — which a single retention percentage would hide entirely.

> **KNOWN BIAS ENTERING THIS STEP — shrub is under-predicted by ~15%, and Step 3 inherits it whole.** At run 3 the best transferable variant `RF-A_C` shows a **shrub area bias of −14.7%**, with shrub→bare confusion at 0.185 (`results/stage3_1_results.md` §9.4). This is a **systematic** error, not a random one, so it does not cancel across the 9 pixels of a Planet block the way per-pixel noise does — every block's shrub fraction is low by roughly the same proportion, and the bias survives into the fractions RF-B trains on and into the Step 6 area estimates.
>
> Completing the labelling did **not** fix it — the bias barely moved between the partial-label smoke run (−16.2%) and the finished run 3 (−14.7%) — which is evidence that it is a **feature-space** problem, not a sample-size one. The remedy is multi-scale texture and context features, not more polygons.
>
> **Operating rule until it is resolved**: Step 3 output built on run 3 is **provisional**. Record the per-class area bias of the source classification in the Step 3 report alongside the fractions, so a later reader cannot mistake a biased fraction product for an unbiased one. Do not train RF-B on it as a final product, and do not report Step 6 areas from it without the bias stated.

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

Outputs → `stage4_aggregation/`, per framework:
`fraction_hard_count_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands, class-code order),
`fraction_soft_mean_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands),
`fraction_confidence_weighted_{framework}_{tile}_{YEAR}.tif` (float32, 4 bands),
`block_prediction_quality_{framework}_{tile}_{YEAR}.tif` (float32),
`valid_pixel_count_{framework}_{tile}_{YEAR}.tif` (uint8).

> **Naming convention — all output files use full words, never shorthand or invented abbreviations.** `fraction_hard_count`, not `frac_hard`. `class_probability`, not `class_prob`. `valid_pixel_count`, not `valid_count`. `prediction_quality`, not `conf`.
>
> The one exception is **established product and index names, which are proper nouns and are kept verbatim**: `RGB`, `NDVI`, `SAVI`, `EVI`, `CHM`, `QA`, `LSP`, `NAIP`, `SLIC`, `GLCM`, `LBP`. Expanding those would make filenames less recognizable, not more. The rule targets abbreviations invented for brevity, not domain terminology.

### Step 4 — Pure end-member identification

Flag blocks with **>= 90%** single-class cover (hard fraction, estimate (a)) as candidate pure end members.

**Purity and confidence are two different things, and both gate selection.** A block that is 95% grass assembled from low-confidence 1 m pixels is a weaker end member than one that is 92% grass at high confidence — the first is a block of ambiguous pixels that happened to fall the same way, the second is genuinely pure grass. Rank candidates by **purity × block confidence**, and record both separately so the trade-off stays visible rather than being buried in a single score.

- Require both `hard_fraction >= 0.90` **and** `block_confidence` above a threshold set from its observed distribution (§11 check #22).
- Expect strong class imbalance: pure bare and pure grass will be abundant; pure tree and pure shrub rare. Record per-class counts; if any class yields fewer than 30 blocks, report it rather than silently lowering either threshold.
- End-member blocks cluster spatially. Use **spatial** (block/tile) holdout, never random holdout, in Step 5.
- Export for manual validation as GeoPackage with class, hard fraction, soft fraction, block confidence, and tile attributes.

Outputs → `stage5_pure_endmembers/`.

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

#### Target year selection — per site, chosen not assumed

**Each site trains on its own "best" year**, selected rather than fixed by calendar. The best year is the most **climatically normal** year with the **highest count of high-quality QA pixels** — in practice not a drought year, since dry years both depress the phenological signal and inflate QA rejection.

- **Target window: 2022 ± 1 year** (2021–2023), keeping the phenology year as close as possible to the 2022-08 ground-truth flight.
- **Selection criterion**: the share of pixels passing `NumCycles == 1` and `QA ∈ {1, 2}` across the site footprint, cross-checked against precipitation for the year to confirm it is not anomalously dry.
- **Selected independently per site.** Different sites may land on different years, which is expected and acceptable — the ground truth and the phenology year must match *within* a site, not across sites.
- Record the selected year, its QA-pass fraction, and the runner-up years in the run report.

> This makes §2.4's multi-year NumCycles and QA scan a **prerequisite for choosing the training year**, not merely a bimodality check — see §2.4.

#### Observed QA quality per site — and what it forces

From the `QA2` scan (§12 Q9a source files), percent of pixels passing `QA ∈ {1,2}`, `PLSP_production_nc` unless noted:

| Site | 2017 | 2018 | 2019 | 2020 | 2021 | 2022+ |
|---|---|---|---|---|---|---|
| **SRER** (xSR) | 12.9% | 12.9% | 3.8% | **0.1%** | **91.1%** | not yet scanned |
| **WKG** | 97.1% | 61.4% | 37.5% | 3.9% | **98.1%** | 98.3% / 98.6% / 33.3% / 39.3% *(stage)* |
| **WJS** | 4.1% | 19.6% | 0.2% | 9.2% | **81.3%** | 44.8% / 1.8% / 47.8% / 34.6% *(stage)* |
| **KONZ** (xKZ) | 98.4% | 98.7% | 98.8% | 99.0% | **99.1%** | not yet scanned |
| **MOAB** (xMB) | — | — | — | — | — | **absent from the scan** |
| **KFB** | — | — | — | — | — | **absent from the scan** |

**Four consequences, all load-bearing:**

1. **2021 is the only usable pre-2022 year at SRER.** 2017–2020 range from 12.9% down to 0.1% — effectively unusable. This does not change the 2022 target (§5.1a), but it means SRER has **no fallback year other than 2021** if 2022 proves poor.

2. **The 2017–2021 interannual-variability characterization (§5.1a item 3) collapses at SRER.** It assumed a five-year series; SRER has one usable year. That analysis is **not available at SRER** and must be run at **KONZ instead**, which has 98–99% across all five years and is the only site in the network with a genuinely complete series. Retarget it accordingly rather than reporting a one-year "spread".

3. **Per-site target years differ, as §5.1a anticipated.** On current evidence: **WJS → 2021** (2022 is only 44.8%); **WKG → 2022 or 2023** (both ~98%); **KONZ → any year**; **SRER → 2022 if it scans well, else 2021**.

4. **MOAB and KFB are missing from the scan entirely.** For MOAB, either the PLSP product does not exist yet or it was not included in the run — **resolve before Phase 3** (§5 Step 8), since D13 depends on it. **KFB is a separate matter: it is not in the PLSP product at all** (§2.1a), so it is not a scan gap but an ineligible site. The newly added D15 pair (ONAQ, Rws) has also not been scanned.

> **Also note**: 2022–2025 products already exist for WKG and WJS in **`PLSP_stage_nc`**, not `PLSP_production_nc`. §5.1's path convention points at production. Confirm whether stage products are usable for analysis or are a pre-release staging area before relying on those years.

#### 5.1a Year alignment — 2022 is the target year (resolves Q1)

**Decision: PLSP 2022 is the year used for PlanetScope prediction and validation**, matching the 2022-08 SRER ground truth exactly. The 2022–2025 products will be generated per site; 2022 is the one Step 5 targets.

> **Explicit dependency**: PLSP 2022 **does not exist yet**. Only 2017–2021 are available today. Every instruction in Step 5 that names 2022 is conditional on that product being generated first. This is a scheduling dependency, not an unresolved methodological choice — the year is decided, the data is pending. Nothing in Step 5's design depends on the interim years, and the interim gap has its own specific use (item 3 below).

Consequences:

1. **Step 5 is gated on the 2022 product.** RF-B training and validation both use PLSP 2022 against 2022 ground truth — same year, no temporal offset, no stationarity assumption required. This is the clean case, and it removes the largest methodological caveat any interim-year substitute would have carried. **Do not begin Step 5 against 2017–2021 as a stand-in**; the earlier options of "use 2021 with a 1-year offset" or "train across five years as replicates" are **withdrawn**, not pending.
2. **Sequencing.** Steps 0–4 (ground truth, aggregation, end members) run to completion on the 2022 NEON flight without needing PLSP at all. Only Step 5 waits. Build and validate the ground-truth side first; the dependency is one-directional.
3. **Use 2017–2021 for feature characterization in the meantime.** Five years of LSP metrics over a landscape whose 2022 cover is known measures **interannual variability of the timing metrics for effectively unchanged cover**. That separates two things the project otherwise cannot distinguish: metrics varying because *cover* changed versus because *climate* varied. The per-pixel spread across 2017–2021 for stable-cover pixels estimates climatic noise in the features, and therefore sets a **floor on achievable RF-B accuracy** before a single 2022 model is fitted. This is characterization, not training — it produces an error bound, not a model. Run it while waiting; it is free information and it calibrates expectations for Step 5.
> **Deliverable — per-pixel phenology spread.** Compute and retain, for every pixel over the available year series: the **mean and standard deviation of each timing metric**, the **range**, and a per-pixel **coefficient of variation**. Map these, and summarize them per class using the ground-truth labels.
>
> This is the interannual-variability layer. Two uses: it quantifies how much of a metric's variance is climatic rather than compositional, and — since the same pixel's cover is effectively fixed across the series — the spread for stable-cover pixels is a **direct estimate of the noise floor** any RF-B model must work above. A class whose between-class separation is smaller than its within-class interannual spread cannot be separated by timing alone, and that is worth knowing before the model is fitted rather than after.

4. **§6's stability check remains required, with a narrower job.** With prediction and validation both on 2022, it no longer guards a temporal offset. It now answers whether end members and the RF-B model hold across years — the multi-year and cross-site ambitions in §2 and §6 still depend on it. SRER has documented mesquite encroachment, so interannual woody change is real, not hypothetical.

**Also run §2.4 (NumCycles distribution) across all available years**, not just the target year. Monsoon strength varies year to year, so the bimodality question may have a different answer in a wet year than a dry one, and a single-year check could mislead. The 2017–2021 series is directly useful here, and this check does not need to wait for 2022.

> **This runs at the front of the project, across all available years at once — not as a per-step check.** It is promoted to an initial scan because its results feed the target-year selection above: a year that looks attractive on the calendar may fail on cycle count or QA share, and **the scan can therefore change which year is chosen for training**. Run it for every site and every available year before committing to a training year (§2.4).
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

> **Rule: every fill, flagged, or QA-failing value becomes NaN in a float array, immediately on read, before any other operation.** No sentinel value (32767, −9999, 65535) is ever permitted to reach arithmetic, comparison, or aggregation. This applies to every data source in the project, not only the LSP product. The failure mode is silent — arithmetic on sentinels produces numbers that look valid and pass range checks — so masking is the first operation after load, never a later cleanup step.

2. **Timing metrics are extended DOY, not circular.** Valid range is **−181 to 548**, so a cycle may legitimately start in the previous calendar year (negative DOY) or end in the next (> 366). Two consequences: (a) **no circular statistics are needed** — treat these as ordinary continuous variables; (b) **a `value < 0` filter would silently discard valid early-onset pixels.** Filter on the fill value only, then range-check against the per-layer bounds in the table.

3. **`Peak` has a different valid range from every other timing layer** (1–366, not −181–548). It cannot cross a year boundary while its neighbours can. Range-check per layer, never with one shared bound, and be aware that a cycle spanning the year boundary will pair an out-of-calendar `OGI` with an in-calendar `Peak`.

4. **Scale factors are not uniform.** Timing layers are scale 1; the EVI layers are 0.0001 or 0.01. Since the design is timing-only, no scaling applies to any feature actually used — but **do not blanket-apply a single scale factor** across the stack, and re-check this if the amplitude layers are ever reinstated (see 5.3).

5. **Cycle-2 layers (13–23) are unused by construction.** The `NumCycles == 1` filter guarantees any surviving pixel has cycle-2 layers at fill. Do not read them; if they are read, assert they are fill as a QA check.

6. **NetCDF, not GeoTIFF — the read path is different.** Use `xarray`/`netCDF4`, not `rasterio`. Three specific consequences:
   - **Georeferencing is not guaranteed.** NetCDF carries CRS and grid via CF conventions (a `crs`/`spatial_ref` variable plus 1-D `x`/`y` coordinate arrays), not a GeoTIFF geotransform. Extract CRS and affine explicitly and verify against the NEON tiles; do not assume either is present.
   - **Layers may be named variables rather than a band axis.** This is an improvement — variable names should match the `short_name` column in §5.2, removing the band-order risk of check #30. **Verify the names match** rather than assuming positional order; if the file instead stores a single array with a layer dimension, positional order applies and check #30 stands in full.
   - **`xarray` may auto-apply `_FillValue` and `scale_factor`** from CF attributes on read (`mask_and_scale`, default `True`). This is convenient but must be **verified, not assumed** — if it fires, trap 1 is already handled; if it does not, fill arrives as raw 32767 and trap 1 is live. Check which, and record it. Applying scaling twice is equally damaging.

> **Standing rule for netCDF and every other data import: assume nothing, verify everything.**
>
> Read and record — do not infer — CRS, affine/geotransform, pixel size, grid origin, dtype, band or variable count and order, nodata and fill values, scale factors and offsets, and units. Every one of these has a silent failure mode: wrong values that raise no error and produce plausible output.
>
> **Where any parameter is ambiguous, undocumented, or contradicts the specification, stop and ask rather than choosing a default.** A wrong assumption at import propagates through every downstream step and is close to undetectable once the pipeline is running. Persist all verified parameters to the run report so a later run can be checked against them.

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

**Confirmed intent**: the question under test is whether **timing alone** can differentiate plant functional type classes. The amplitude layers are withheld so that the answer is unambiguous.

**If timing alone proves insufficient**, the EVI metrics are added — but to answer a *different* question: not "can PFTs be separated by timing," but "how strongly is fractional cover expressed at PlanetScope scale." That is a **separate problem, and an important one**, not a rescue of the first. Keep the two results distinct in reporting: a timing-only result that fails is a genuine finding about phenological separability, and folding amplitude in afterwards would erase it.

Sequence accordingly — run and report timing-only to completion first, then treat the amplitude extension as a new study with its own framing.

if successful just with timing, fractional cover maps at planetscale will still be produced as a final product

#### 5.5 Model form — multi-output regression, not classification

The ground-truth RF (Step 1) classifies at 1 m and aggregates 9 pixels to get fractions; that is a valid count. The phenology model is different: **Planet's native pixel *is* the 3 m cell**, so there is no sub-pixel level to classify and aggregate. The model must map one 3 m feature vector directly to four fractions.

Do **not** train a classifier on pure end members and read per-class vote proportions as fractional cover. A classifier trained only on pure pixels has seen fractions of 0 and 1 only; its vote share on a mixed pixel measures distance to a decision boundary, not mixing proportion. A genuinely 50/50 grass/shrub pixel routinely returns 0.85/0.15.

Instead:
- **X** = 10 timing metrics per Planet pixel. **y** = the Step 3 fractions — use the **soft mean, estimate (b)**, as the primary target, since sub-pixel mixture is precisely what the model is being asked to predict and the hard count discards it. Fit against the hard count (a) as well and report both; a large gap indicates the model is limited by label hardening rather than by the features.
- **`sample_weight` = `prediction_quality`** (block confidence, §3.1 and Step 3). Blocks whose targets were assembled from ambiguous 1 m pixels carry noisier `y` and should influence the fit less. This is the main reason the confidence layer is propagated this far. If `data_quality` is also folded in, keep it a separate named factor per §5.3 — never a single blended score.
- Train a **multi-output RF regressor on all valid blocks**, mixed and pure alike — Step 3 supplies true fractions across all five tiles, so no synthetic mixtures are needed at SRER.
- Pure end members (Step 4) serve as **anchors, validation, and the WKG transfer basis** — not as the sole training set.
- Constrain or post-normalize predictions to sum to 1 and lie in [0, 1].
- Cross-check the seasonal signal against the phenocam in the train block.

Outputs → `stage6_rf_phenology/`.

### Step 6 — Accuracy assessment

**Governing protocol**: Olofsson, Foody, Herold, Stehman, Woodcock & Wulder (2014), *Good practices for estimating area and assessing accuracy of land change*, Remote Sensing of Environment 148:42–57 — https://www.sciencedirect.com/science/article/pii/S0034425714000704

This is not a loose reference. **Accuracy is estimated area-weighted, and reported per land-cover class with confidence intervals**, following the estimators below. Raw sample counts are never reported as accuracy.

> **The ground truth itself is assessed for the training year.** This step validates the 1 m classification and the derived blocks **for the year the RF-B model is trained on** (§5.1a target-year selection) — not for an arbitrary year. Where ground truth is produced for more than one year (§6 multi-year work), each year carries its own accuracy assessment; accuracy from one year is never carried across to another, since class composition and acquisition conditions both change. Record the assessed year alongside every reported figure.

#### 6.1 Why area weighting is mandatory here

A stratified sample over-samples rare classes by design. At SRER, tree and shrub are rare and bare and grass are abundant (§11 check #20), so raw counts from a stratified sample **do not** reflect the map's actual composition. Computing accuracy from those counts silently assumes equal class areas and produces a number that describes the sample rather than the landscape. Every estimate below therefore reweights sample counts by the map-area proportion of the stratum they came from.

#### 6.2 Sampling design

**Strata = map class × `prediction_quality` bin** (§3.1). Crossing the two satisfies two requirements at once: Olofsson's requirement that strata be defined on the map being assessed, and the §3.1 requirement that low-confidence pixels be populated well enough to validate the confidence layer. Record `W_h`, the area proportion of every stratum `h`, from the full map — these are the weights that make the estimators area-weighted.

#### Which confidence layer, and where the bin edges go

`prediction_quality` exists at two spatial levels (§3.1). **Bin the layer that matches the assessment level:**

| Assessment level | Layer binned |
|---|---|
| 1 m pixel accuracy | Per-pixel `prediction_quality` — entropy of `p` for that pixel |
| N×N block / end-member accuracy | Block `prediction_quality` — mean of pixel values over the block |

**Bin edges are within-class quantiles, never global absolute thresholds.** This is the part that is easy to get wrong. Shrub sits at systematically lower confidence than bare across the entire map, so a single global cut would place nearly all shrub in the low bin and nearly all bare in the high bin. Class and confidence would become collinear, the cross-stratification would carry no independent information, and the §6.5 monotonicity check would be measuring class difficulty rather than confidence calibration. Cutting at **each class's own median** keeps the two axes independent.

**Use 2 bins, split at the within-class median** → 4 classes × 2 bins = **8 strata**. Three bins would give 12 strata and, at a 50-sample floor each, a 600-sample floor before the size formula is even applied; 8 strata costs 400. Record the actual median value per class in the run report — it is a derived quantity, not a fixed constant, and it will differ per site and per framework.

**Sample size**, replacing the earlier "X per class, TBD":

```
n = ( Σ_h W_h · S_h / S(Ô) )²        S_h = sqrt( U_h · (1 − U_h) )
```

**Reading the formula, right to left:**

- **`S(Ô)`** — how precisely overall accuracy needs to be known. **`S(Ô) = 0.02` is the value used** (decided), giving a 95% interval of about **±0.04**: "overall accuracy = 0.85 ± 0.04".
- **`U_h`** — a *guess*, made before sampling, at the user's accuracy of stratum `h`. Guessing is expected and standard; the value only sizes the sample and never enters the reported results.
- **`S_h = √(U(1−U))`** — the standard deviation of a coin flip with bias `U`, because each sampled pixel is simply right or wrong (a Bernoulli trial). It peaks at `U = 0.5` and shrinks toward 0 or 1, so **uncertain classes demand more samples**. Guessing `U` low is therefore the safe direction: it only oversamples.
- **`W_h`** — the stratum's share of map area, since accuracy is area-weighted (§6.1).
- **`Σ W_h S_h`** — the area-weighted average per-sample standard deviation. The whole expression then reduces to the ordinary `n = (σ / SE)²`.

Conjecture `U_h` ≈ 0.6–0.7 for shrub and for low-confidence bins, ≈ 0.8–0.9 for bare and grass at high confidence — deliberately pessimistic.

**Worked example, plausible SRER values:**

| Class | `W` | `U` (conjectured) | `S` | `W·S` |
|---|---|---|---|---|
| bare | 0.45 | 0.85 | 0.357 | 0.161 |
| grass | 0.35 | 0.80 | 0.400 | 0.140 |
| shrub | 0.15 | 0.65 | 0.477 | 0.072 |
| tree | 0.05 | 0.75 | 0.433 | 0.022 |
| | | | **Σ** | **0.394** |

- **`S(Ô) = 0.02` → `n = (0.394 / 0.02)²` ≈ 390 samples — the operative figure**
- `S(Ô) = 0.01` → `n = (0.394 / 0.01)²` ≈ 1,550 samples (rejected: ~4× the manual labeling for ±0.02 instead of ±0.04)

> **`S(Ô) = 0.02` — DECIDED.** Every sample is a manual label, and 0.01 would have cost roughly four times the labeling effort (~1,550 points versus ~390) to tighten the overall-accuracy interval from ±0.04 to ±0.02. Not worth it here, particularly since framework selection rests on **per-class** accuracy (§6.4) rather than on overall accuracy.
>
> Two consequences to hold in mind. First, the ~390 figure is a *total* from the formula, while the 8 strata each carry a 50-sample floor — so the binding constraint is the **400-sample floor**, not the formula. Expect ~400–500 labels in practice. Second, the `W_h` in the table above are conjectured, not measured: **recompute with the real class proportions from §11 check #20** before fixing the final allocation.

**Allocation**: proportional to `W_h`, then raised to a **floor of 50–100 samples in any rare stratum** (Olofsson §5.1.2). Without that floor, tree and shrub confidence intervals will be too wide to separate frameworks — which is the entire purpose of this step.

**Single shared sample set**, drawn **once** and evaluated by every framework, so comparisons are direct.

**One sample set assesses every framework.** The same labeled points are scored against every `RF-A_*` variant alike, so differences between variants reflect the variants and not sampling variation. No variant gets its own sample.

> **A tension to handle explicitly.** Olofsson stratifies on the map being assessed, but the `RF-A_*` variants produce *different* maps, so a per-variant stratification would give each variant a different sample and destroy comparability. Resolution: stratify **once** on the reference variant (`RF-A_D`, §4.1), draw a probability sample, and **record every inclusion probability**. The estimators below remain unbiased for every variant, because the sample is a valid probability sample of the whole area regardless of which map defines the strata. The cost is precision, not bias: for a variant whose rare-class map disagrees sharply with the stratification, rare-class intervals will be wider. Where that happens, augment per-variant and **recompute the weights** — never merge an augmented sample in at equal weight.

**Stratify once, for now.** The single reference-based stratification is used for all frameworks in this phase — the goal here is to compare *techniques*, and a shared stratification is what makes that comparison direct. Per-framework augmentation and reweighting stay specified above but are **deferred**: revisit only if a rare-class interval turns out too wide to separate frameworks, and never as a default.

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

#### 6.3a Required output — the full error matrix with all calculations shown

For every framework, produce a **user's vs. producer's accuracy matrix** with the complete arithmetic exposed, not just the summary figures:

- **Counts matrix** `n_ij` — raw sample counts, rows = map class, columns = reference class.
- **Area-proportion matrix** `p̂_ij` — the same matrix after weighting by `W_i`, with the weights printed alongside.
- **Row margins** `p̂_i·` and **column margins** `p̂_·j`.
- **User's accuracy** per class with standard error and 95% interval — from the row margins.
- **Producer's accuracy** per class with standard error and 95% interval — from the column margins.
- **Overall accuracy** with standard error and 95% interval.
- **Error-adjusted area** `Â_j` per class with 95% interval, shown next to the naive mapped pixel count so the bias is directly readable.

Emit both a machine-readable table and a formatted version for reporting. **Showing the intermediate quantities is deliberate**: the area-weighted estimators are easy to implement subtly wrong, and a matrix that prints only final accuracies gives no way to catch it.

#### 6.4 Framework selection

Select the best-performing **transferable** variant (`RF-A_A`–`RF-A_C`, DL) per §4.1 on **per-class user's and producer's accuracy with intervals**, not on overall accuracy.

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

Outputs → `stage7_accuracy_assessment/`: the sample set with inclusion probabilities and stratum weights, reference labels, the area-proportion error matrix per framework, all estimators with 95% intervals, error-adjusted areas, and the accuracy-vs-confidence plot.

### Step 7 — RAP comparison

**Pending**: RAP data path. Compare RAP (10 m fractional cover) vs. ground truth (1 m aggregated to the RAP grid) vs. the Planet-scale mask (Step 3). Aggregation to 10 m must align to the RAP grid, same rule as Step 0.

> **PAUSED — Step 7 is deferred to Phase 2.** This comparison runs **last, after site transferability (Step 8) is complete**, not in sequence between Steps 6 and 8. Nothing in Steps 1–6 or Step 8 depends on it.
>
> Keep the design recorded here so it stays in view, but do not build it, and do not let it gate anything. The RAP data path remains an open item (§10) and does not need resolving until Phase 2 begins.

### Step 8 — Transferability to WKG

- **Phase 1**: framework development on SRER train tiles, validated on SRER test tiles.
- **Phase 2**: transfer to WKG. Two routes, tested in order. SRER and WKG share NEON domain D14, so **Route 1 is in scope** (R1).

  **Route 1 — transfer the model and end members directly.** Apply the SRER-trained model and SRER-derived pure end members to WKG features and phenology, without rebuilding ground truth. This is the **within-ecoregion model-transfer claim** (R1) and the ideal outcome. Requires the persisted per-site scaler (R4) and identical feature definitions (R2); check per-class recall for prior shift (R6) before concluding anything.

  **Route 2 — build WKG its own ground truth layer** by running the transferable pipeline (`RF-A_A`–`RF-A_C`) on **NAIP imagery**. NAIP is 0.6 m RGB+NIR, so vegetation indices and texture are both computable and Steps 1–4 run unchanged; this yields native WKG fractional-cover blocks and therefore native WKG training data for RF-B.

  Route 2 is the **expected path across an ecoregion boundary** and the fallback if Route 1 fails within one; it is the reason `RF-A_A`–`RF-A_C` are constrained to RGB + VI + texture in §4.1. **It also removes the need for synthetic mixtures at WKG** (see §9 risk 1): with its own ground truth layer, WKG has true mixed-pixel fractions and RF-B can be trained there exactly as at SRER.

  Constraints at WKG: NAIP 0.6 m RGB+NIR; LiDAR offset 1–2 years, so CHM is a cross-check at best and frameworks D/E are not carried over.
> **LiDAR may carry over, conditionally.** CHM is usable at a transfer site — and frameworks D/E may then be carried over — if **either** the LiDAR acquisition is **within 1 year** of the imagery, **or** its alignment against the imagery can be positively verified (§11 check #9 coregistration test, plus a CHM-vs-imagery edge check on stable features).
>
> Where neither holds, CHM stays a cross-check only and D/E are not transferred. Record which condition was met, and the measured offset, in the run report — a carried-over CHM that later proves misaligned would corrupt the shrub/tree split (§3) with no visible symptom.

  Working assumption: sites outside a shared ecoregion require independent end-member sets. Comparing Route 1 against Route 2 is the direct test of that assumption.

- **Phase 3 — extend across the network (§2).** SRER → WKG is the D14 pair only, and tests **Axis 1** (within-domain, sensor transfer). Repeating the same two routes at D15 (ONAQ → Rws) and D13 (MOAB → WJS) completes Axis 1. **D15 is the cleanest model-transfer test in the network** — exact IGBP match, 2 mm MAP difference, so a Route 1 failure there is attributable to sensor and resolution with ecology near-constant (§2.1). D06 is blocked pending a PLSP-eligible partner (§2.1a).

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
- **Output filenames use full words, never invented shorthand** — `fraction_hard_count`, not `frac_hard`; `class_probability`, not `class_prob`. Established product and index names are proper nouns and kept verbatim: `RGB`, `NDVI`, `SAVI`, `EVI`, `CHM`, `QA`, `LSP`, `NAIP`, `SLIC`, `GLCM`, `LBP` (§5 Step 3).
- **Assume nothing on data import; verify and record every parameter** — CRS, transform, pixel size, origin, dtype, band/variable order, nodata, scale, offset, units. Ask rather than default when any of these is ambiguous (§5.3).
- **Mask fill and flagged values to NaN in float arrays immediately on read**, before any arithmetic. No sentinel value ever reaches a computation (§5.3).
- **Seeds**: `base_seed + year`, matching `instructions1.md` §3, applied to SLIC, RF, and all sampling.
- variable and function names should not be shortened, no acronyms
- for each step create py file for functions and separate runner jupyter file that calls a "run()" function so that output is easy to see
- for every file saved, print "Written: file"
- 

---

## 7A. Compute environment and packages

**A dedicated conda environment has been created for this project.** All work runs inside it; additional packages may be installed on request.

**Core stack — required:**

| Package | Used for |
|---|---|
| `rasterio` | GeoTIFF read/write, colour tables (§3 `CLASS_COLORS`), windowed reads |
| `xarray`, `netCDF4` | PlanetScope LSP `.nc` products (§5.1, §5.3 trap 6) |
| `numpy`, `pandas` | Arrays, tabular outputs, the QA/target-year tables |
| `geopandas`, `shapely`, `fiona` | GeoPackage I/O — labeling zones, training polygons, crown outlines, grids |
| `pyproj` | CRS and UTM handling (§11.2) |
| `scikit-learn` | K-means (§4.2), `StandardScaler` (R4), RF-A classifier, RF-B multi-output regressor |
| `scikit-image` | SLIC segmentation, GLCM and LBP texture (§5 Step 1) |
| `scipy` | Mahalanobis distance, watershed, general numerics |
| `joblib` | Model and scaler persistence (R1 model transfer, R4) |
| `matplotlib` | All diagnostics — 1:1 scatters, histograms, ternary and tetrahedron views (§3.1) |
| `gdal` | `gdal_translate -co TILED=YES` preprocessing (§8) |

**Deep-learning tracks — required only for `RF-A_DL1`–`RF-A_DL3`, and gated on the GPU blocker (§8):**

| Package | Used for |
|---|---|
| `torch`, `torchvision` | Backbone for DeepForest and U-Net |
| `deepforest` | `RF-A_DL1` tree-crown detection |
| `segment-anything` | `RF-A_DL3` SAM segmentation |

**Optional, decision-dependent:**

| Package | Used for |
|---|---|
| `qgis` (conda-forge) | Only if §12 Q10 mechanism 4 is built. **Install in a separate environment** — it pulls a large Qt stack and must not contaminate the pipeline environment |
| `mpltern` or equivalent | Ternary face plots (§3.1 view A2) if not hand-rolled in `matplotlib` |

**Convention**: pin and export the environment (`environment.yml`) into the repository, and record the resolved package versions in `report.json` per run (§7). A model persisted by one `scikit-learn` version and loaded by another is a silent failure mode, and R1 model transfer depends on exactly that operation working across machines.

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

| Item                                                                   | Status                                                                                                                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PlanetScope phenology data path                                        | **Resolved — §5.1**. NetCDF; 2017–2021 available now                                                                                                   |
| Which PLSP year for Step 5                                             | **Resolved — 2022** (§5.1a), matching the ground-truth flight year                                                                                     |
| PLSP 2022 product generation                                           | **Pending — gates Step 5 only.** Steps 0–4 proceed independently. Track as a schedule dependency                                                       |
| SRER 2022 QA quality                                                   | **Unknown — the key risk.** 2021 (91.1%) is SRER's only viable fallback; 2017–2020 are unusable (§5.1a QA table)                                       |
| MOAB PLSP product                                                      | **Absent from the QA scan.** Resolve before Phase 3 transfer testing (§5.1a QA table)                                                                  |
| D06 partner site                                                       | **KFB is not in the PLSP product (§2.1a).** Choose `US-KFS` (119 km, record ends 2019) or drop D06 to Axis-2 contrast only                              |
| D15 (ONAQ / Rws) data availability                                     | Newly added pair — run the NEON / NAIP / 3DEP / phenocam timeline and PLSP QA scan                                                                     |
| `PLSP_stage_nc` vs `PLSP_production_nc`                                | 2022–2025 exist for WKG/WJS in **stage** only. Confirm whether stage is analysis-grade before use                                                      |
| `numObs` minimum-observation threshold                                 | TBD — evaluate at Step 0 check #35. **Produce a histogram of `numObs`** over QA-passing pixels                                                                                       |
| RAP data path                                                          | Pending. Will follow the same `{DATA_ROOT}/...` convention as §5.1                                                                                                |
| Accuracy assessment sample size                                        | **Resolved — §6.2**. `S(Ô) = 0.02`, 8 strata (4 classes × 2 confidence bins), 50-sample floor per stratum → ~400–500 labels. Recompute allocation with measured `W_h` from check #20 |
| SAVI/EVI filename convention                                           | **Confirmed to mirror the NDVI pattern**; still verify against the actual files at Step 0                                                                                |
| UTM zone                                                               | Read from file CRS. **Never assume any file parameter — always verify** (§5.3 import rule)                                                        |
| WKG data paths                                                         | Needed at Step 8                                                                                                                                       |
| `NumCycles == 1` viability at D13 (and all sites)                      | **Run §2.4 first** — decides whether the 1-cycle constraint holds                                                                                      |
| NEON AOP flight years per site; NAIP and 3DEP years per AmeriFlux site | Needed — see §2.4                                                                                                                                      |
| Climate-anchored features for cross-domain (Axis 2) comparison         | Designed in `instructions1.md` §8, not implemented                                                                                                     |
| Presence of all 5 tiles for all 3 products                             | Verify at Step 0                                                                                                                                       |
| `H_GRASS_MAX` final value                                              | Provisional 0.3 m; set from measured CHM noise floor at Step 0                                                                                         |
| PlanetScope pixel size and grid origin                                 | **RESOLVED 2026-08-18** — measured by `run_stage1_3_define_planet_grid.py`: **3.0 m, N = 3**, origin 510555.0 / 3535548.0, `EPSG:32612`. Visually verified in QGIS. See §5 Step 3 |

---

## 11. Data checks (Step 0 checklist)

Every check writes a pass/fail plus the observed value into `stage1_data_and_features/qa/data_audit_{SITE}_{YEAR}.json`. Checks marked **BLOCKER** stop the pipeline; others are recorded and reviewed.

### 11.1 Inventory

| # | Check | Fail condition |
|---|---|---|
| 1 | All 5 tiles present for all 3 products (15 files/dirs) | **BLOCKER** if any missing |
| 2 | SAVI/EVI filenames match the NDVI pattern | **BLOCKER** — assumption from `instructions4.md`, never verified |
| 3 | RGB tile dimensions = 10000 × 10000 at 10 cm (1 km) | Report; partial tiles are usable but must be flagged |
| 4 | VI and CHM tile dimensions = 1000 × 1000 at 1 m | Report |
| 5 | No duplicate or overlapping tile footprints | Report |

### 11.2 Georeferencing and alignment

| #   | Check                                                                                                                                          | Fail condition                                                          |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 6   | CRS identical across all products and all tiles; UTM zone recorded explicitly                                                                  | **BLOCKER** on mismatch                                                 |
| 7   | VI and CHM 1 m grids share an identical origin (no half-pixel offset)                                                                          | **BLOCKER** — a silent half-pixel shift corrupts every class rule in §3 |
| 8   | RGB 10 cm grid nests exactly within the 1 m grid                                                                                               | **BLOCKER**                                                             |
| 9   | Coregistration test: pick a hard edge (road, building, large crown), cross-correlate RGB against CHM and against SAVI, report offset in metres | Report; > 1 m offset is a **BLOCKER**                                   |
| 10  | Planet LSP CRS matches, and its footprint is compared against every tile. Tiles are **cropped** to the footprint, so partial coverage is recorded, not fatal | **BLOCKER** only if a tile is wholly outside. **PASS** at SRER, `EPSG:32612`; `520000_3532000` at 55.4% is cropped |
| 11  | Planet pixel size read from the product file — **never hard-coded** as 3, 3.7, or 4 m. Record it; derive `N = round(planet_pixel_size / 1.0)`  | **BLOCKER** if `N` is non-integer within tolerance. **Double-check independently** before it propagates. **PASS** at SRER: 3.0 m, N = 3 — the "expected 4" in earlier drafts was wrong |
| 12  | Planet grid origin recorded **as cell edges, from centre coordinates**; confirm 1 m blocks tile it exactly with no fractional remainder        | **BLOCKER**. **PASS** at SRER: origin 510555.0, 3535548.0                |
| 12a | NEON tile origins tested for congruence with the Planet grid, modulo the Planet pixel                                                          | Report — but a non-congruent tile **forbids per-tile aggregation** at Step 3. **0 of 10 congruent** at SRER |

> Checks 10–12a are implemented by `run_stage1_3_define_planet_grid.py` and must be followed by the visual pass in `run_stage1_4_create_qgis_grid_verification_project.py`. Numeric checks prove the grid is internally consistent; only the visual pass proves it sits on the imagery.

### 11.3 Radiometry and value sanity

| #   | Check                                                                                                                                                            | Fail condition                                                                              |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 13  | **VI scale factor.** NEON VI rasters may be stored as scaled integers. Read dtype and any scale/offset metadata; confirm SAVI/NDVI land in [-1, 1] after scaling | **BLOCKER** — an unapplied scale factor silently breaks `SAVI_BARE_MAX` = 0.2 with no error |
| 14  | RGB band count (3 vs 4) and dtype (uint8 vs uint16); identify any alpha band                                                                                     | **BLOCKER** if unexpected — an alpha band read as blue corrupts the shadow rule             |
| 15  | Nodata / fill value recorded per product and confirmed actually applied (not left as a raw sentinel such as -9999 or 65535 entering arithmetic)                  | **BLOCKER**                                                                                 |
| 16  | CHM range: no negatives, and max plausible for SRER (values > ~15 m are suspect)                                                                                 | Report                                                                                      |
| 17  | CHM noise floor: distribution over visually-confirmed bare areas; set `H_GRASS_MAX` above it (§3 safeguard 2)                                                    | Report — sets a config value                                                                |
| 18  | Per-tile percentage of nodata in the VI mosaic — bidirectional mosaics carry flightline gaps                                                                     | Report; > 5% flags the tile                                                                 |
| 19  | Visual scan for cloud, cloud shadow, and mosaic seams in RGB                                                                                                     | Report                                                                                      |

### 11.4 Class and sampling sanity

| #   | Check                                                                                                                                                                    | Fail condition                                                                                                         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| 19a | All class-valued rasters use the locked §3 codes (0–3 terminal, 4 shadow intermediate only, 255 nodata); assert no value outside that set, and no 4 in any final product | **BLOCKER**                                                                                                            |
| 20  | Per-class pixel counts under the §3 reference rules, per tile                                                                                                            | Report — establishes the class prior for R6                                                                            |
| 21  | Both train and test blocks contain all four classes                                                                                                                      | **BLOCKER** if a class is absent from either                                                                           |
| 22  | Expected pure end-member counts per class at the 90% threshold (§5 Step 4)                                                                                               | Report; < 30 for any class is flagged, not silently worked around                                                      |
| 22a | Distribution of the 1 m confidence layer and of block confidence (§3.1), per class and per tile — sets the Step 4 confidence threshold                                   | Report — sets a config value                                                                                           |
| 22b | Agreement between hard (a) and soft (b) fraction estimates per block (§5 Step 3); map the disagreement                                                                   | Report — large systematic divergence means the site is mixture-dominated at 1 m and Step 5 targets should be revisited |
| 23  | Phenocam location falls inside the 511000 train block; record its footprint                                                                                              | Report. **Phenocam coordinates to be provided by user**                                                                               |

### 11.5 Cross-site checks (run at every transfer site)

> **Action — user input required before any transfer site is processed:**
> - **3DEP LiDAR acquisition dates per site** — needed for the carry-over decision in Step 8 and check #27.
> - **NAIP data and full metadata per site** — acquisition date, band order, native resolution, and radiometric/processing notes, covering checks #24–26.
>
> These are blocking inputs for §11.5, not background information. Request them explicitly rather than proceeding on assumed values.

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

**Decision: use an edge margin, not neighbour-tile buffer reads.** Mask a margin equal to the largest window radius used anywhere in the pipeline (texture window, SLIC neighbourhood, `SHADOW_TREE_RADIUS`) around every tile edge, and exclude those pixels from all products and from Step 6 sampling.

Losing a few edge pixels is acceptable and is the right trade: buffer reads add cross-tile I/O dependencies that break tile-level parallelism and fail silently at the site boundary, where no neighbour exists. A fixed margin behaves identically at interior and boundary tiles.

Record the margin width in config and the count of pixels dropped per tile — the loss should be small, and a large number signals a window radius that has grown beyond what the tiling supports.

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

### Q3 — How is RF-B evaluated? — **RESOLVED**

Fractional-cover regression needs different metrics from classification (§6.6 bars forcing fractional output into an error matrix). **Overall R² alone is not reportable** — it hides class-specific failure, and shrub is exactly where failure is expected.

**Required metrics, all computed per class on held-out blocks** (spatial holdout, never random — §5 Step 4):

| Metric | What it catches |
|---|---|
| **RMSE** per class | Overall error magnitude, penalising large misses |
| **MAE** per class | Typical error, robust to outliers. Report alongside RMSE — a large RMSE/MAE gap means a few severe failures rather than uniform noise |
| **Mean signed error** per class | **Systematic bias** — whether a class is consistently over- or under-predicted. The one an unsigned metric cannot show |

**Required plot — 1:1 predicted-vs-true scatter, one panel per class, four panels:**

- Predicted fraction on the y-axis, true (Step 3) fraction on the x-axis, both on `[0, 1]`.
- **1:1 line drawn** on every panel.
- Per-class **RMSE, MAE, and mean signed error annotated** in-panel.
- Points **coloured by block `prediction_quality`** (§3.1), so label-noise effects are visible rather than confounded with model error.
- Equal axes and identical scaling across all four panels, so the classes are directly comparable.

**Why the scatter is required and not optional**: three failure modes matter here and none is visible in a summary number — **saturation** at high cover (predictions compressing below the 1:1 line as true fraction approaches 1), a **floor** at low cover (a class never predicted below some value, common when a rare class is regularised toward its mean), and **systematic offset** of one class. All three are obvious at a glance in the scatter and invisible in R².

**Also report**: whether predictions were constrained or post-normalized to sum to 1 (§5 Step 5), and the metrics both before and after that step — normalization redistributes error between classes, so a class can appear to improve purely because another absorbed its excess.

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

DeepForest (`weecology/deepforest-tree`) stays as **`RF-A_DL1`**, repositioned: it is **not** a competing variant for the 4-class map, but an **independent cross-check on the tree class**. Its value is precisely that it was trained on NEON RGB — it is an outside opinion on crowns, derived from the same imagery type but a completely different method from the CHM/SLIC/RF path.

**Role**: run at NEON sites on native 10 cm RGB; compare crown count, crown area, and spatial pattern against the Step 1 tree class and the CHM-derived crowns. Agreement raises confidence in the tree class; systematic disagreement localizes where the tree detection is wrong. It contributes to Step 6 as a **tree-class-only** comparison, not as an input-tier variant.

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

This yields a quantified answer — *DeepForest at NAIP resolution recovers X% of crowns above Y m diameter* — over a scene where the true answer is already known from CHM. It is cheap, needs no new data, and converts an assumption into a measured degradation curve that also tells you the **minimum detectable crown size** at NAIP resolution. If step 3 shows recovery is adequate for the crown sizes that matter at WKG, use it there; if not, `RF-A_DL1` stays a NEON-only cross-check and the AmeriFlux tree class rests on `RF-A_A`–`RF-A_C`.

**Optional extension if the degradation proves too severe**: fine-tune DeepForest on 0.6 m simulated-NAIP chips using the native-10 cm detections as labels. This is well-posed — NEON supplies both the imagery and, via CHM, an independent crown check — and produces a NAIP-resolution crown detector without any manual crown digitizing. Treat as a follow-on, not part of the initial run.

> **Note for target-year selection (§5.1a).** The choice is grounded in an **existing per-site, per-year high-quality-pixel scan**, already produced. Selection rule: take one of the **high-quality years** for each site, choosing the one **closest to 2022**. Use this as the primary evidence rather than regenerating the statistic, and record the selected year and the runner-up per site.
>
> **Source files** (produced by `code/code_lsp_analysis/run_check_site_pixel_quality.py`):
> ```
> code/code_lsp_analysis/check_site_pixel_quality_results_QA2.csv
> code/code_lsp_analysis/check_site_pixel_quality_results_QA2.png
> ```
> `QA1`/`QA2`/`QA3` variants exist for different QA thresholds. **Use the `QA2` variant** — it matches the project-wide `QA ∈ {1, 2}` policy (§2.4).
>
> **Columns**: `file, site, year, good_pixels, total_pixels, percent_good_pixels`.
>
> **Three handling notes.** `percent_good_pixels` is a **string with a `%` suffix** and must be parsed, not cast. Rows appear for **both `PLSP_production_nc` and `PLSP_stage_nc`**, giving duplicate `(site, year)` pairs with slightly different values — **dedupe on `(site, year)` preferring `PLSP_production_nc`**, the path this specification uses (§5.1). And the `file` column carries the full SCC path, so it doubles as an inventory of which products actually exist.

### Q10 — Can QGIS layer loading and symbology be generated automatically? — **feasibility assessed, not implemented**

**Goal**: eliminate the manual clicking involved in loading each run's outputs into QGIS with correct paths and consistent styling. At 5 frameworks × 5 tiles × multiple pipeline steps, this is repeated often enough to be worth automating.

**Answer: yes, by four mechanisms of increasing effort.** None is implemented; this records the options and the trade-offs.

| # | Mechanism | Best for | Cost |
|---|---|---|---|
| 1 | **Colour table embedded in the GeoTIFF** — `rasterio` `dst.write_colormap()` | Categorical rasters (`classification_*`, masks) | A few lines at write time |
| 2 | **Sidecar `.qml` style files**, basename-matched so QGIS auto-loads them | Continuous rasters (`prediction_quality_*`, `fraction_*`) | Small, one template per layer type |
| 3 | **`.qlr` layer-definition files** — sources plus symbology for a group of layers | One file per pipeline step | Moderate |
| 4 | **Generated QGIS project via PyQGIS** — `QgsProject.write('site_run.qgz')`, headless | Whole-run loading, grouped by step | Largest; adds a dependency |

**Why 1 and 2 are the high-value pair**: symbology travels with the data and is honoured outside QGIS too (GDAL, ArcGIS). Mechanism 2 additionally pins a **fixed stretch** — QGIS defaults to a per-layer min/max stretch, so without it two frameworks' `prediction_quality` maps would render on different scales and appear to differ when they do not. That is a real interpretation hazard given the framework comparison in Step 6.

**Constraints to respect if implemented:**
- **PyQGIS requires the QGIS Python environment** (conda-forge `qgis`), not the pipeline env. Keep it isolated to a project-generation helper so the pipeline gains no dependency.
- **Save relative paths.** With the local/SCC dual root (§5.1, R8), an absolute-path project breaks on the other machine.
- **Prefer `.qgs` (plain XML) over `.qgz` (zipped)** for a generated artifact, so it is diffable in version control.
- Colour tables are **uint8/paletted only** — they cover the categorical outputs, not the float layers, which is exactly the split between mechanisms 1 and 2.

**Dependency**: the locked class colour table (§3, `CLASS_COLORS`) — now defined, so categorical symbology is identical across every framework, tile, site, and figure. Mechanism 1 writes exactly that table into the GeoTIFF.
