# Stage 4_1 — aggregation to PlanetScope 3 m blocks

Produced by `run_stage4_1_aggregate_to_planet_blocks.py`, which implements **`instructions5.md` §5 Step 3**. File named for its execution stage; text refers to the spec step — see `instructions5.md` §0.1.

**Site**: SRER 2022 · **Source**: run 3, `RF-A_C` · **Grid**: 3 m, N = 3, `EPSG:32612`, origin 510555.0 / 3535548.0 · **Retention**: ≥ 8 of 9 valid pixels

> **These outputs are PROVISIONAL.** The source classification under-predicts shrub by 14.7% and that bias is systematic, so it does not cancel on aggregation — every block's shrub fraction inherits it. Do not train RF-B on these as final. See §4.

---

# 1. What this stage does and does not mask

A block loses pixels **only to the ground-truth side**: unflown ground, shadow, tile edge margin, and the PlanetScope footprint crop. **No PlanetScope QA masking happens here.** The PLSP QA layers (`NumCycles`, `QA`) describe defects in the *phenology* product and are applied later, at RF-B training. Applying them here would discard perfectly good ground truth because a different dataset is faulty.

---

# 2. Flight coverage — the finding that changes the tile set

**Two of the ten tiles are largely outside the AOP flight box.** This was invisible until the aggregation separated *unflown* from *shadowed*; both are nodata in the classification, but they mean opposite things — unflown ground was never observed, shadowed ground was observed and then discarded.

**CHM is the binding product.** On `511000_3532000` CHM is nodata over 95.91% of the tile against 74.96% for RGB and 66.79% for SAVI, so a pixel is lost to CHM before any other band. Stage 3 requires every band valid, so classification nodata tracks CHM.

| tile | role | flown | used (of flown) | **shadow lost** | shadow→tree | edge margin | outside footprint |
|---|---|---|---|---|---|---|---|
| 511000_3527000 | train | 100.00% | 97.04% | **0.22%** | 2.96% | 8,363 | 0 |
| 511000_3528000 | train | 100.00% | 97.05% | **0.16%** | 2.98% | 8,380 | 0 |
| 511000_3529000 | train | 100.00% | 97.01% | **0.20%** | 3.02% | 8,283 | 0 |
| 515000_3526000 | train | 100.00% | 97.41% | **0.09%** | 2.98% | 8,082 | 0 |
| 519000_3527000 | train | 100.00% | 97.13% | **0.98%** | 3.12% | 7,876 | 0 |
| 515000_3530000 | test | 100.00% | 99.01% | **0.05%** | 0.63% | 7,960 | 0 |
| 515000_3531000 | test | 100.00% | 99.03% | **0.05%** | 0.60% | 7,954 | 0 |
| 518000_3529000 | test | 100.00% | 96.58% | **1.02%** | 2.02% | 8,691 | 0 |

*(`517000_3531000` and `516000_3528000` are in the tile set but have no stage 3 outputs yet — they need labelling first. The two retired tiles are gone; see §2.2.)*

> **CORRECTED 2026-08-18 — the first version of this table overstated shadow loss by roughly ten times.** `run_stage1_6_detect_shadows.py` writes three codes: **0 not shadow, 1 resolved to tree, 2 masked to nodata**. `run_stage4_1_aggregate_to_planet_blocks.py` was counting **code 1** as loss, but code 1 is shadow within `SHADOW_TREE_RADIUS` of CHM ≥ `H_TREE_MIN`, which §5 Step 1c **assigns to the tree class** — those pixels are classified, not discarded. Only code 2 is a loss. The script now reports both columns separately.

**True shadow loss is 0.05–1.02% of flown area.** Shadow is not a meaningful source of block attrition at this site, and the earlier framing of it as the dominant mask term was wrong. `stage2_results.md` §1.7 cites shadow as "79% of masked pixels" — that is a *share of masked pixels*, and once the resolved-to-tree portion is separated out, the ground actually lost to shadow is well under 1% on seven of eight tiles. **The §1.7 concern that `luma` may separate classes via shadow structure still stands** — that is about the *feature*, not about pixel loss — but shadow is not why blocks are dropped.

**The dominant loss is the edge margin**: a uniform ~8,000 px per tile, which is the 2 m margin around a 1000 × 1000 tile. That is deterministic geometry, not data quality, and it is why the valid-pixel histogram is bimodal (§3).

## 2.1a Adding two tiles re-cut shadow everywhere, as predicted

`run_stage1_6_detect_shadows.py` pools its luma threshold across all tiles, so changing the tile set shifts the site-wide 20th percentile and re-cuts shadow on **every** tile. Measured after `517000_3531000` and `516000_3528000` entered the pool (`pooled_threshold` 111.43, blue threshold 0.2966):

| tile | shadow→tree before | after |
|---|---|---|
| 511000_3527000 | 2.02% | 2.96% |
| 519000_3527000 | 1.70% | 3.12% |
| 515000_3530000 | 0.18% | 0.63% |

Shadow extent rose by half to double across the board. The new tiles are more wooded — `517000_3531000` is 11.6% shadow at 0.6 m, the highest of any tile — and the two retired tiles contributed almost no valid pixels to the pool, so the percentile moved. **This is a confound for the next classification run**, exactly as recorded for run `3_smoke`: the next run changes the tile set *and* the shadow masks on unchanged tiles at the same time.

## 2.1 The stratification claim is weaker than recorded

`instructions5.md` §2A selected the five added tiles by CHM shrub-band cover quintile. That statistic was computed as **shrub ÷ flown area**, verified by recomputation:

| tile | quintile | shrub / flown | shrub / whole tile | flown area |
|---|---|---|---|---|
| 519000_3527000 | Q1 | 0.120 | 0.120 | 100 ha |
| 515000_3526000 | Q3 | 0.172 | 0.172 | 100 ha |
| 518000_3529000 | Q4 test | 0.271 | 0.271 | 100 ha |
| **511000_3532000** | **Q4 train** | **0.270** | 0.011 | **4.1 ha** |
| **520000_3532000** | **Q2 test** | **0.146** | 0.035 | **24.4 ha** |

Excluding nodata is the right way to compute cover, so these numbers are not wrong. But **the entire Q4 *training* representation is a 4.1 ha sliver**, and Q2 test is 24 ha of which only ~14 ha falls inside the Planet footprint. Train nominally spans Q1/Q3/Q4; in practice Q4 contributes about 4% of one tile.

**This also revises how run 3 should be read.** `511000_3532000` is counted as one of six training tiles in `stage2_results.md` §9, but contributed 38,684 of 1,000,000 possible pixels. Its leave-one-tile-out fold is a 4 ha fold.

**Recommendation, not yet acted on**: replace both tiles with fully-flown alternatives from the same quintiles — 14 tiles per quintile were available — and add two selection criteria that were missing: **require ~100% flight coverage, and require the tile to sit wholly inside the PlanetScope footprint.** Neither was checked when the five tiles were chosen.

## 2.2 Replacement tiles — determined

All 70 CHM tiles are already on disk, so every candidate was evaluated without downloading anything. **60 of 70 tiles are fully flown, 63 sit wholly inside the Planet footprint, 54 satisfy both.** The entire northern row at northing 3532000 — all ten tiles — is 4–24% flown, which is why both bad tiles came from it.

Quintile edges recomputed over the 54 eligible tiles: **0.116 / 0.132 / 0.157 / 0.188 / 0.291 / 0.546**. Every kept tile holds its previous quintile under this recomputation, so only the two replacements move.

| replaces | new tile | role | shrub | to opposite role | to same role |
|---|---|---|---|---|---|
| `511000_3532000` (Q4 train) | **`517000_3531000`** | train | 0.273 | 2.00 km | 4.47 km |
| `520000_3532000` (Q2 test) | **`516000_3528000`** | test | 0.138 | **2.24 km** | 2.24 km |

Candidates were ranked by **distance to the opposite role** — a train tile far from test tiles and vice versa — because that is what protects the leave-one-tile-out split from spatial autocorrelation. `516000_3528000` is the only Q2 test candidate exceeding 2 km from any train tile. Full criteria and runners-up are in `instructions5.md` §2A.

Only RGB and the vegetation indices need downloading:

```
python run_stage1_1_download_neon_tiles.py --site SRER --year 2022 --tiles-to-download 517000_3531000 516000_3528000
```

---

# 3. Block retention

**885,024 blocks touched, 847,059 retained** at ≥ 8 of 9 (95.71%). Counts are lower than the first run because the two retired tiles no longer contribute and the two replacements are not yet labelled.

| valid pixels | blocks | share |
|---|---|---|
| 0 | 0 | 0.00% |
| 1 | 354 | 0.04% |
| 2 | 621 | 0.07% |
| 3 | 4,184 | 0.47% |
| 4 | 2,019 | 0.23% |
| 5 | 3,832 | 0.43% |
| 6 | 10,575 | 1.19% |
| 7 | 16,535 | 1.87% |
| **8** | **41,193** | **4.65%** |
| **9** | **805,866** | **91.06%** |

**The distribution is strongly bimodal**: 90.9% of blocks are complete, and everything else is a thin tail. That is the signature of losses concentrated at *edges* — tile perimeters, flight-line boundaries, the footprint crop — rather than spread through the interior. Interior shadow at ~2% would otherwise have produced a fat 8-of-9 shoulder, and 4.7% is modest.

**Cost of the retention cut**, measured rather than assumed:

| cut | blocks retained | vs 8-of-9 |
|---|---|---|
| 9 of 9 | 805,866 | −41,193 |
| **8 of 9 (chosen)** | **847,059** | — |
| 7 of 9 | 863,594 | +16,535 |

Relaxing to 7 buys 16,535 blocks, **2.0%**, at the price of admitting blocks with 22% of their area unobserved. Tightening to 9 costs 41,193, 4.9%. **8 is the right cut**: the gain from relaxing is small, and the tail below 8 is dominated by edge blocks whose missing pixels are not missing at random.

---

# 4. Fractions, and the bias they inherit

Mean hard-count fraction over retained blocks:

| class | mean fraction |
|---|---|
| bare | 0.1765 |
| grass | 0.2453 |
| **shrub** | **0.4310** |
| tree | 0.1472 |

**These are not a site estimate and must not be quoted as one.** The ten tiles were deliberately quintile-stratified to span the site's extremes, not sampled at random, and two of them contribute slivers (§2.1). A site-level cover estimate requires the Step 6 probability sample.

**Area bias carried in from run 3 `RF-A_C`**, computed from its confusion matrix (rows true, columns predicted):

| class | area bias |
|---|---|
| bare | +3.58% |
| grass | +1.67% |
| **shrub** | **−14.66%** |
| tree | +4.54% |

Shrub is under-predicted by 14.7%. **Random per-pixel error partly cancels across the 9 pixels of a block; systematic bias does not** — every block's shrub fraction is low by roughly this proportion, and the error propagates intact into the RF-B training target and the Step 6 areas. Mean shrub cover of 0.431 therefore implies something nearer 0.50 on the labelled tiles, which is high enough to be worth checking against RAP at Step 7 before it is believed.

---

# 5. Open items

1. **Download and hand-label `517000_3531000` and `516000_3528000`** (§2.2), then re-run stage 3 and stage 4. This is the highest-value fix here, because it affects run 3's training set as well as this stage. The labelling is the real cost; the download is two tiles of RGB and VI.
2. **Shrub bias remains the binding problem** and is a feature-space issue, not a sample-size one (`stage2_results.md` §9.4). Outputs stay provisional until it is addressed.
3. **Only `RF-A_C` has been aggregated.** Running `RF-A_D` as well would measure what the CHM circularity does to the fractions, which is cheap and worth knowing.
4. **The soft-mean and confidence-weighted estimates have not yet been compared** against the hard count. Blocks where they diverge are mixture- or ambiguity-dominated and are the ones worth inspecting first.
5. **No visual verification of the blocks yet.** A stage 4_2 QGIS project loading the fraction bands over the 1 m classification and the `EVIamp` layer is the next check.
