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

| tile | role | flown | used (of flown) | shadow (of flown) | outside footprint |
|---|---|---|---|---|---|
| 511000_3527000 | train | 100.00% | 97.04% | 2.02% | 0 |
| 511000_3528000 | train | 100.00% | 97.05% | 2.04% | 0 |
| 511000_3529000 | train | 100.00% | 97.01% | 2.05% | 0 |
| 515000_3526000 | train | 100.00% | 97.41% | 1.76% | 0 |
| 519000_3527000 | train | 100.00% | 97.13% | 1.70% | 0 |
| 515000_3530000 | test | 100.00% | 99.01% | 0.18% | 0 |
| 515000_3531000 | test | 100.00% | 99.03% | 0.16% | 0 |
| 518000_3529000 | test | 100.00% | 96.58% | 1.84% | 0 |
| **511000_3532000** | **train** | **4.09%** | 94.53% | 0.29% | 0 |
| **520000_3532000** | **test** | **24.37%** | 58.67% | 2.46% | 89,530 |

**Shadow is 0.16–2.46% of flown area on every tile** — an order of magnitude smaller than the "79% of masked pixels" figure in `stage3_1_results.md` §1.7 suggested at a glance. That figure was a *share of masked pixels*, not a share of ground, and on fully-flown tiles it corresponds to about 2% of the tile. The §1.7 concern about shadow clustering against woody canopy still stands; its magnitude does not.

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

**This also revises how run 3 should be read.** `511000_3532000` is counted as one of six training tiles in `stage3_1_results.md` §9, but contributed 38,684 of 1,000,000 possible pixels. Its leave-one-tile-out fold is a 4 ha fold.

**Recommendation, not yet acted on**: replace both tiles with fully-flown alternatives from the same quintiles — 14 tiles per quintile were available — and add two selection criteria that were missing: **require ~100% flight coverage, and require the tile to sit wholly inside the PlanetScope footprint.** Neither was checked when the five tiles were chosen.

---

# 3. Block retention

906,177 blocks touched, **866,240 retained** at ≥ 8 of 9 (95.59%).

| valid pixels | blocks | share |
|---|---|---|
| 0 | 0 | 0.00% |
| 1 | 368 | 0.04% |
| 2 | 645 | 0.07% |
| 3 | 4,407 | 0.49% |
| 4 | 2,168 | 0.24% |
| 5 | 4,061 | 0.45% |
| 6 | 11,048 | 1.22% |
| 7 | 17,240 | 1.90% |
| **8** | **42,628** | **4.70%** |
| **9** | **823,612** | **90.89%** |

**The distribution is strongly bimodal**: 90.9% of blocks are complete, and everything else is a thin tail. That is the signature of losses concentrated at *edges* — tile perimeters, flight-line boundaries, the footprint crop — rather than spread through the interior. Interior shadow at ~2% would otherwise have produced a fat 8-of-9 shoulder, and 4.7% is modest.

**Cost of the retention cut**, measured rather than assumed:

| cut | blocks retained | vs 8-of-9 |
|---|---|---|
| 9 of 9 | 823,612 | −42,628 |
| **8 of 9 (chosen)** | **866,240** | — |
| 7 of 9 | 883,480 | +17,240 |

Relaxing to 7 buys 17,240 blocks, **2.0%**, at the price of admitting blocks with 22% of their area unobserved. Tightening to 9 costs 42,628, 4.9%. **8 is the right cut**: the gain from relaxing is small, and the tail below 8 is dominated by edge blocks whose missing pixels are not missing at random.

---

# 4. Fractions, and the bias they inherit

Mean hard-count fraction over retained blocks:

| class | mean fraction |
|---|---|
| bare | 0.1775 |
| grass | 0.2473 |
| **shrub** | **0.4285** |
| tree | 0.1467 |

**These are not a site estimate and must not be quoted as one.** The ten tiles were deliberately quintile-stratified to span the site's extremes, not sampled at random, and two of them contribute slivers (§2.1). A site-level cover estimate requires the Step 6 probability sample.

**Area bias carried in from run 3 `RF-A_C`**, computed from its confusion matrix (rows true, columns predicted):

| class | area bias |
|---|---|
| bare | +3.58% |
| grass | +1.67% |
| **shrub** | **−14.66%** |
| tree | +4.54% |

Shrub is under-predicted by 14.7%. **Random per-pixel error partly cancels across the 9 pixels of a block; systematic bias does not** — every block's shrub fraction is low by roughly this proportion, and the error propagates intact into the RF-B training target and the Step 6 areas. Mean shrub cover of 0.43 therefore implies something nearer 0.50 on the labelled tiles, which is high enough to be worth checking against RAP at Step 7 before it is believed.

---

# 5. Open items

1. **Replace the two part-flown tiles** and add flight-coverage and footprint-containment gates to tile selection (§2.1). This is the highest-value fix here, because it affects run 3's training set as well as this stage.
2. **Shrub bias remains the binding problem** and is a feature-space issue, not a sample-size one (`stage3_1_results.md` §9.4). Outputs stay provisional until it is addressed.
3. **Only `RF-A_C` has been aggregated.** Running `RF-A_D` as well would measure what the CHM circularity does to the fractions, which is cheap and worth knowing.
4. **The soft-mean and confidence-weighted estimates have not yet been compared** against the hard count. Blocks where they diverge are mixture- or ambiguity-dominated and are the ones worth inspecting first.
5. **No visual verification of the blocks yet.** A stage 4_2 QGIS project loading the fraction bands over the 1 m classification and the `EVIamp` layer is the next check.
