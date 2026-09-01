# Stage 4_1 — aggregation to PlanetScope 3 m blocks

Produced by `run_stage4_1_aggregate_to_planet_blocks.py`, which implements **`instructions5.md` §5 Step 3**. File named for its execution stage; text refers to the spec step — see `instructions5.md` §0.1.

**Site**: SRER 2022 · **Source**: **run 5**, `RF-A_C` (§8) · **Grid**: 3 m, N = 3, `EPSG:32612`, origin 510555.0 / 3535548.0 · **Retention**: ≥ 8 of 9 valid pixels

**Sections 1–4 describe the run-3 aggregation** and are kept because the flight-coverage and shadow findings in them are what forced the tile replacement. **§6 is the current product, built on run 4.**

> **Outputs are run-scoped**: `stage4_aggregation/run{N}/`. A fraction product is only meaningful against the classification run that produced it — run 3 carried a −14.7% shrub area bias and run 4 carries −4.7% — so an unscoped directory would let whichever ran last silently replace the other. An earlier version did exactly that and overwrote the run-3 fractions; only run 4 survives on disk.

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

**True shadow loss is 0.05–1.02% of flown area.** Shadow is not a meaningful source of block attrition at this site, and the earlier framing of it as the dominant mask term was wrong. `stage3_results.md` §1.7 cites shadow as "79% of masked pixels" — that is a *share of masked pixels*, and once the resolved-to-tree portion is separated out, the ground actually lost to shadow is well under 1% on seven of eight tiles. **The §1.7 concern that `luma` may separate classes via shadow structure still stands** — that is about the *feature*, not about pixel loss — but shadow is not why blocks are dropped.

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

**This also revises how run 3 should be read.** `511000_3532000` is counted as one of six training tiles in `stage3_results.md` §9, but contributed 38,684 of 1,000,000 possible pixels. Its leave-one-tile-out fold is a 4 ha fold.

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
2. **Shrub bias remains the binding problem** and is a feature-space issue, not a sample-size one (`stage3_results.md` §9.4). Outputs stay provisional until it is addressed.
3. **Only `RF-A_C` has been aggregated.** Running `RF-A_D` as well would measure what the CHM circularity does to the fractions, which is cheap and worth knowing.
4. **The soft-mean and confidence-weighted estimates have not yet been compared** against the hard count. Blocks where they diverge are mixture- or ambiguity-dominated and are the ones worth inspecting first.
5. **No visual verification of the blocks yet.** A stage 4_2 QGIS project loading the fraction bands over the 1 m classification and the `EVIamp` layer is the next check.

---

# 6. Run 4 aggregation — the current product

Built on run 4 `RF-A_C`, 2026-08-26. **All ten tiles contributed for the first time**: the run-3 aggregation skipped `517000_3531000` and `516000_3528000`, which had no classification yet.

```
python run_stage4_1_aggregate_to_planet_blocks.py config/srer_2022.json --run 4 --frameworks C
```

## 6.1 The bias that propagates is three times smaller

| class | run 3 source | **run 4 source** |
|---|---|---|
| bare | +3.58% | **−4.49%** |
| grass | +1.67% | +1.89% |
| **shrub** | **−14.66%** | **−4.65%** |
| tree | +4.54% | +5.67% |

This is the single most important number in the stage, because a systematic bias does not cancel on aggregation — every block's shrub fraction inherits it whole. **Shrub fractions are now low by roughly 5%, not 15%.**

The cause was tile-set composition, not features or sample size (`stage3_results.md` §11.4). Two part-flown tiles were dragging shrub toward bare; removing them fixed the bias even though shrub training pixels halved.

## 6.2 Coverage and retention

**1,104,989 blocks touched, 1,031,535 retained** at ≥ 8 of 9 valid — **93.35%**.

| valid pixels | blocks | share |
|---|---|---|
| 0 | 0 | 0.00% |
| 1 | 842 | 0.08% |
| 2 | 1,573 | 0.14% |
| 3 | 6,849 | 0.62% |
| 4 | 4,775 | 0.43% |
| 5 | 8,477 | 0.77% |
| 6 | 18,997 | 1.72% |
| 7 | 31,941 | 2.89% |
| **8** | **74,812** | **6.77%** |
| **9** | **956,723** | **86.58%** |

Still strongly bimodal, so losses remain edge-concentrated rather than spread through the interior.

**Retention fell against the run-3 aggregation, 95.71% → 93.35%, and that is expected.** Run 4's classification was produced against the *re-cut* shadow masks, which are more extensive: on `511000_3527000`, classification nodata went from 29,608 px to 39,609 px, an extra 1.0% of the tile. Shadow at 1 m is now 3.18% there against roughly 2.2% before. **The shadow re-cut costs about one percent of pixels per tile and about 2.4 points of block retention.** That is the price of change 3 in the run-4 note, and it is worth knowing before anyone reads the retention drop as a defect.

Cost of the retention cut, measured:

| cut | blocks | vs 8-of-9 |
|---|---|---|
| 9 of 9 | 956,723 | −74,812 |
| **8 of 9 (chosen)** | **1,031,535** | — |
| 7 of 9 | 1,063,476 | +31,941 |

Relaxing to 7 buys 3.1%; tightening to 9 costs 7.3%. **8 remains the right cut**, and the margin is wider than in the run-3 aggregation because there is more shadow to admit or exclude.

Per-tile usable share of flown ground runs **93.92% to 98.55%**. `517000_3531000` is lowest, as expected: it carries both the highest shadow loss (1.30%) and the only unflown RGB (§2).

## 6.3 Mean fractions over retained blocks

| class | fraction |
|---|---|
| bare | 0.1769 |
| grass | 0.2545 |
| **shrub** | **0.4239** |
| tree | 0.1447 |

**Not a site estimate.** The ten tiles are quintile-stratified to span the site's extremes, not randomly sampled. A site-level cover figure requires the Step 6 probability sample.

Barely moved from the run-3 aggregation (shrub 0.4310 → 0.4239) despite the bias improving threefold — because both aggregations cover the same stratified tiles, and the bias correction and the tile swap partly offset in the mean.

## 6.4 Output verification

Every check passed on `fraction_hard_count_C_SRER_2022.tif`:

- **Fractions sum to exactly 1.000000** over all 1,031,535 retained blocks, min and max alike.
- **Grid matches the Planet grid exactly**: 3.0 m, origin 510555.0 / 3535548.0, 3333 × 3334, `EPSG:32612`. The arrays align 1:1 with the LSP netCDF.
- **`valid_pixel_count` spans 0–9** as designed, recording the raw count before the retention rule.
- **`block_prediction_quality` median 0.580**, range 0.113–1.000.
- **Soft mean differs from hard count on shrub by 0.092 mean absolute.** That gap is the sub-pixel mixture signal that hardening throws away, and it is why the soft estimate is the better RF-B target (§5 Step 3).

## 6.5 What is still outstanding

1. ~~Only `RF-A_C` is aggregated.~~ **Done — see §7.** `RF-A_D` diverges from `RF-A_C` by 20% on mean shrub fraction and by 40% on pure shrub end-member count, which is far larger than the pixel-level scores imply.
2. **The remaining shrub bias is −4.65%**, better but not zero, and it still propagates. Sweeping the shrub candidates on `517000_3531000` (0 of 150 reviewed) and `516000_3528000` (50 of 150) is the cheapest way to attack what is left.
3. ~~No visual verification of the blocks yet.~~ **`run_stage4_2_create_qgis_aggregation_project.py` written** — 47 layers, both frameworks, over the 1 m classification and `EVIamp`. Run it in `LCSC_QGIS`; the review is still outstanding.
4. **Block retention by predicted class is not yet reported.** Shadow clusters against woody canopy, so tree- and shrub-adjacent blocks are dropped at a higher rate than open ground (`instructions5.md` §5 Step 1c). The size of that gap should be measured before Step 6.

---

# 7. RF-A_D aggregation — what CHM does to the fractions

Run 4 `RF-A_D` aggregated 2026-08-26, same ten tiles, same grid, same retention rule. **Retention is identical to `RF-A_C` at 1,031,535 of 1,104,989 blocks**, which is the expected result and a useful check: stage 3 computes pixel validity over the full 20-band stack rather than each framework's subset, precisely so every framework sees the same pixels (§4.1 deconfounding). Any difference between C and D is therefore attributable to the inputs, not to different masking.

## 7.1 The two frameworks disagree far more at block scale than at pixel scale

Mean hard-count fraction over the 1,031,535 blocks both retained:

| class | `RF-A_C` | `RF-A_D` | difference | relative | mean abs error | correlation |
|---|---|---|---|---|---|---|
| bare | 0.1769 | 0.2054 | +0.0285 | +16.1% | 0.0388 | 0.953 |
| grass | 0.2545 | 0.3417 | +0.0873 | **+34.3%** | 0.1061 | 0.893 |
| **shrub** | **0.4239** | **0.3376** | **−0.0863** | **−20.4%** | **0.1509** | **0.793** |
| tree | 0.1447 | 0.1152 | −0.0295 | −20.4% | 0.0582 | 0.874 |

**Shrub is the least correlated class at 0.793 and carries the largest mean absolute error at 0.151** — roughly 1.4 of the 9 pixels in a block. Per-block disagreement on shrub has a median of 0.111, a 90th percentile of 0.444, and **19.5% of blocks differ by more than 0.33, which is three or more pixels out of nine.**

## 7.2 This contradicts what the pixel-level scores imply, and that is the finding

On the labelled sample, D is simply better at shrub and both under-predict it slightly:

| | shrub recall | shrub precision | shrub F1 | shrub area bias |
|---|---|---|---|---|
| `RF-A_C` | 0.621 | 0.652 | 0.636 | −4.65% |
| `RF-A_D` | 0.753 | 0.777 | 0.765 | −3.07% |

Both under-predict shrub by a few percent **on training polygons**. Yet across the full site D maps **20% less shrub than C**. Those two statements can only both be true if **C and D diverge mainly on ground the labels do not cover.**

That is exactly what should be expected. The labelled polygons were drawn where the analyst was confident; the disagreement lives in the ambiguous ground between grass and shrub, which is under-represented in the labels by construction (`stage3_definitions.md` §7). C must infer woody structure from spectra and texture; D reads height directly. Where the two part company, D has physical evidence and C has a correlate.

**The consequence is a warning about the area-bias number itself.** §6.1 reports the source classification's area bias from its confusion matrix, and that figure is computed on training polygons. **It does not capture site-wide divergence of this size.** A −4.65% bias measured on labels sits alongside a 20% between-framework disagreement over the mapped area. The bias number is still worth carrying, but it is a lower bound on uncertainty, not a measure of it.

**Neither product can be declared correct from what is on disk.** That requires the Step 6 independent probability sample, and this section is the clearest argument yet that Step 6 is not optional bookkeeping.

## 7.3 Step 4 end members are affected most

> **The "≥ 90%" purity rule was broken at N = 3, and it is now a count.** The achievable hard-count fractions are multiples of 1/9, and **8/9 = 0.889 < 0.90**, so a 90% threshold admits **only perfectly uniform 9-of-9 blocks** and silently discards every 8-of-9 one. This is the same trap the retention rule fell into (§3), with the same fix: state the count. Purity is now `stage4_1_aggregation.min_pure_pixels_per_block: 8`, and `run_stage4_1_aggregate_to_planet_blocks.py` writes `pure_endmember_{fw}_{SITE}_{YEAR}.tif` as a first-class product rather than leaving purity as a statistic in these notes.

Pure end-member blocks, **≥ 8 of 9 pixels sharing one class**:

| class | `RF-A_C` | of which 9-of-9 | `RF-A_D` | of which 9-of-9 | C vs D |
|---|---|---|---|---|---|
| bare | 65,196 | 40,269 | 74,101 | 45,367 | +13.7% |
| grass | 130,215 | 88,814 | 195,691 | 135,506 | +50.3% |
| **shrub** | **172,028** | 90,111 | **111,491** | 52,590 | **−35.2%** |
| tree | 37,658 | 16,831 | 27,746 | 12,139 | −26.3% |

**Correcting the threshold roughly doubles the pool.** Shrub under `RF-A_C` goes from 90,111 to 172,028 blocks, a gain of 81,917; under `RF-A_D` from 52,590 to 111,491. Every class gains 50–120%.

**The choice of framework still changes the pure shrub pool by 35%**, and the pure grass pool by half. Since end members anchor RF-B and are the transfer basis at Step 8, this is not a downstream detail. All four classes clear the 30-block floor by four orders of magnitude under either framework, so the gate in §5 Step 4 would pass either way and would **not** flag the divergence — the floor was written for a much smaller site.

## 7.4 What to do about it

1. **Keep `RF-A_C` as the operational product.** It is the transferable one: CHM needs lidar, and NAIP has none, so D cannot move to a transfer site at all (R1, §4.4). This is a constraint, not a quality judgement.
2. **Carry D as the structural reference.** Where C and D disagree by more than about a third of a block on shrub, that block is genuinely ambiguous, and the disagreement is a better uncertainty estimate for those blocks than either model's own confidence.
3. **Consider a C-versus-D disagreement raster as a Step 6 stratification variable.** The 19.5% of blocks that differ by ≥ 3 pixels are exactly where an independent sample buys the most information, and stratifying on disagreement would spend the sample where it settles the question.
4. **Do not read D's better shrub F1 as better shrub mapping.** Most shrub labels came from a CHM threshold and D receives CHM, so its shrub score is partly circular (§1.3). The circularity inflates the score; it does not by itself make the map wrong, and §7.1 shows D is the more conservative mapper, not the more generous one.

---

# 8. Run 5 aggregation — the current product

Built on run 5 `RF-A_C`. Run 5 changed one thing against run 4, the shrub candidate review (`stage3_results.md` §12), so this is the first stage 4 comparison where a single variable moved.

## 8.1 Retention is identical, to the block

| | run 4 | run 5 |
|---|---|---|
| blocks touched | 1,104,989 | 1,104,989 |
| blocks retained | 1,031,535 | 1,031,535 |
| retention | 93.35% | 93.35% |
| 9-of-9 / 8-of-9 / 7-of-9 | 86.58% / 6.77% / 2.89% | 86.58% / 6.77% / 2.89% |

**Not approximately equal — identical.** This is a strong consistency check rather than a coincidence: block validity comes from classifier nodata, which is driven by shadow masks and the edge margin, and run 5 touched neither. Labels changed which class a pixel gets, never whether it has one. Any difference here would have meant something unintended had moved.

## 8.2 The bias is nearly out of the product

| class | run 4 | **run 5** |
|---|---|---|
| bare | −4.49% | −4.70% |
| grass | +1.89% | +1.62% |
| **shrub** | **−4.65%** | **−1.99%** |
| tree | +5.67% | **+5.41%** |

**Shrub area bias is now −1.99%**, down from −14.66% at run 3. Across three runs the quantity that propagates into every block fraction has fallen sevenfold.

**Tree at +5.41% is now the largest bias of any class**, and it has never been investigated. It has sat between +4.5% and +5.7% across all three runs, unmoved by the tile swap or by the shrub labelling, which suggests a systematic cause rather than a sampling one. It is the obvious next target now that shrub is under control.

## 8.3 Fractions moved in the direction the bias predicts

| class | run 4 | **run 5** | change |
|---|---|---|---|
| bare | 0.1769 | 0.1749 | −0.0020 |
| grass | 0.2545 | 0.2456 | −0.0088 |
| **shrub** | **0.4239** | **0.4445** | **+0.0205** |
| tree | 0.1447 | 0.1350 | −0.0097 |

Mean shrub cover rose 2.1 points and the other three fell to pay for it, grass most of all. That is exactly what a shrub under-prediction being corrected should look like, and it matches the pixel-level confusion where shrub→grass fell from 0.196 to 0.176.

**These remain stratified-sample means, not site estimates.** The ten tiles were chosen to span the CHM shrub quintiles, so the mean is not the site's cover.

## 8.4 Pure end members

| class | run 4 | **run 5** | change |
|---|---|---|---|
| bare | 65,196 | 64,834 | −0.6% |
| grass | 130,215 | 126,100 | −3.2% |
| **shrub** | **172,028** | **196,183** | **+14.0%** |
| tree | 37,658 | 34,377 | −8.7% |

**24,155 more pure shrub blocks**, a 14% gain, from 180 extra training pixels. End members anchor RF-B and are the transfer basis at Step 8, so this is the most consequential single number in the run: the shrub anchor set is now both larger and less biased.

Tree lost 8.7% of its pure blocks, which is worth watching alongside its +5.41% bias — the two are consistent with tree being over-predicted in ambiguous blocks and losing purity as shrub reclaims ground it had taken.

## 8.5 What is still outstanding

1. **`RF-A_D` has not been aggregated on run 5.** The §7 C-versus-D comparison is against run 4's D. Re-run it if that comparison is to be used for Step 6 stratification.
2. **Tree bias +5.41%** is now the largest and is unexamined.
3. **`515000_3526000`** is isolated as a tile problem, not a label problem (`stage3_results.md` §12.4). It still enters these fractions.
4. **The visual review in QGIS has still not been done.** `run_stage4_2_create_qgis_aggregation_project.py` reads `RUN` at the top of the file and will need it set to 5.
