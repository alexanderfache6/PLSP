# Stage 3_1 — RF-A per-pixel classification results

Produced by `run_stage3_1_random_forest_ground_truth_classification.py`, which implements **`instructions5.md` §5 Step 1d**. The text refers to the spec step throughout — see `instructions5.md` §0.1 for why the spec's step numbers and the scripts' stage numbers differ.

**Site**: SRER 2022 · **Seed**: 6 + 2022 = 2028 · **Runs**: 1 (baseline), 2 (polygon subsampling), `3_smoke` (smoke test), 3 (10 tiles), 4 (10 fully-flown tiles), **5 (current reference)**, 6 (void, §14), 7 (§14.4), 8 (§16) · **Variants**: RF-A_A … RF-A_D

## Naming

Every framework on this page is a variant of **RF-A**, the ground-truth classifier that labels 1 m pixels (`instructions5.md` §4.3). They are named `RF-A_{framework}` so they can never be confused with **RF-B**, the separate PlanetScope phenology regressor that predicts fractional cover at Planet scale and will have its own variants.

| Name | Directory on disk | Input feature groups |
|---|---|---|
| `RF-A_A` | `stage3_classification/A/` | RGB only |
| `RF-A_B` | `.../B/` | RGB + vegetation indices |
| `RF-A_C` | `.../C/` | RGB + VI + texture |
| `RF-A_D` | `.../D/` | RGB + VI + texture + CHM |
| `RF-A_E` | `.../E/` | full combined |

**Directory and filename stems keep the bare letter** (`A`, `B`, …) — those are set by `run_stage3_1_random_forest_ground_truth_classification.py` and are not renamed here. `RF-A_*` is the reporting name; the letter is the on-disk key.

> **These are DIAGNOSTIC results, not map accuracy.** Every number below comes from leave-one-tile-out cross-validation over *training polygons*. A Step 6 accuracy assessment requires the Olofsson area-weighted protocol on an independent probability sample (`instructions5.md` §6), which is a different quantity. Do not quote a figure from this document as map accuracy.

## Files

**All Step 1d outputs are run-scoped** under `stage3_classification/run{N}/`. `features/`, `segments/` and `shadow/` are Step 1a–1c outputs shared by every run and are deliberately not run-scoped.

| Artifact | Path (relative to `results_root/stage3_classification/run{N}/`) |
|---|---|
| Machine-readable report | `stage3_1_report_SRER_2022.json` |
| Per-variant diagnostics | `stage3_1_diagnostics_{A,B,C,D}_SRER_2022.png` |
| Variant comparison | `stage3_1_framework_comparison_SRER_2022.png` |
| Classification rasters | `{framework}/classification_{framework}_SRER_{tile}_2022.tif` |
| Class probabilities | `{framework}/class_probability_{framework}_SRER_{tile}_2022.tif` |
| Prediction quality (§3.1) | `{framework}/prediction_quality_{framework}_SRER_{tile}_2022.tif` |
| Margin (§3.1) | `{framework}/margin_{framework}_SRER_{tile}_2022.tif` |
| QGIS project | `results_SRER_2022.qgz` — 105 layers in 5 groups (A–D plus RGB) |

`results_root` = `~/Dropbox/planet/results/v4` (local) — see `config/srer_2022.json`. Paths are given relative to it rather than as links, since results live outside the repository and the relative depth would break on the SCC.

**Definitions**: every metric on this page — recall, precision, F1, support, macro-F1, folds, feature importance, `prediction_quality`, `margin` — is defined in [`stage3_definitions.md`](stage3_definitions.md), with what to look for and the trap attached to each.

**Scripts**, all taking `--run N`:

```
python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 2
python run_stage3_2_generate_ground_truth_classification_plots.py config/srer_2022.json --run 2
python run_stage3_3_create_qgis_results_project.py --run 2 # or set RUN in the QGIS console
```

A completed run is **frozen**: `run_stage3_1_random_forest_ground_truth_classification.py` refuses to write into a run directory that already holds a report, and names the next free run instead. `--force` overrides, deliberately.

## Runs

| Run | Tiles | Change | Subsampling | Train pixels (bare/grass/shrub/tree) |
|---|---|---|---|---|
| **1** | 5 | Baseline | off | 8,200 / 4,102 / 844 / 1,190 |
| **2** | 5 | Polygon subsampling | max 100 px/polygon | 2,159 / 2,824 / 844 / 1,190 |
| **`3_smoke`** | **10** | run 3 smoke test — quintile-stratified tiles, advanced shrub review | max 100 px/polygon | 2,671 / 3,218 / **1,531** / 1,530 |
| **3** | **10** | the finished run — gate passed | max 100 px/polygon | **3,601 / 4,717 / 2,087 / 2,158** |
| **4** | **10** | two part-flown tiles replaced, RGB unflown ground masked, shadow and clusters refit (§11) | max 100 px/polygon | 3,361 / 4,752 / 1,075 / 1,959 |
| **5** | **10** | shrub candidate review only, one variable (§12) | max 100 px/polygon | 3,361 / 4,752 / 1,255 / 1,959 |
| **6** | **10** | hand-drawn labels on `515000_3526000` — **VOID as a test, §14.4** | max 100 px/polygon | 3,361 / 4,788 / 1,685 / 1,994 |
| **7** | **10** | more shrub on the same tile, labels still contradicted the 2 m rule (§14.4) | max 100 px/polygon | 3,361 / 4,788 / 1,738 / 1,994 |
| **8** | **10** | six shrub polygons reclassified to tree (§16) | max 100 px/polygon | 3,361 / 4,788 / 1,515 / 2,217 |

Labels are **identical between runs 1 and 2** (shrub 844 px in both), so run 2 isolates the subsampling change with nothing else moving.

**Run `3_smoke` changes four things at once** and must not be read as a clean comparison against run 2 — see §8. Its scores are lower and that is expected: it validates on a harder, representative tile set. **`3_smoke`, not run 2, is the baseline for run 3 and everything after.**

**On probability rasters**: `predict_proba` always sums to 1 for a classified pixel, so there is no minimum-probability rule and an all-zero pixel can only ever be masked. Masked pixels are written as **NaN**, matching the declared nodata, so they render transparent rather than as a legitimate probability of zero. Bands are named `p_bare`, `p_grass`, `p_shrub`, `p_tree`.

**On weak predictions**: the hard label is `argmax` regardless of how weak the winner is — by design (§3.1). On `RF-A_A`, max probability across classified pixels runs min 0.329, median 0.767, with 0.83% below 0.40 and none below 0.30 (chance is 0.25). Strength of the call is carried in `prediction_quality`, not by refusing to classify; filter on confidence downstream instead.

---

> **Sections 1–6 describe run 1** (the baseline, no subsampling). **Section 7** covers run 2 and the change it made. Where a number appears without a run label, it is run 1.

---

# 1. Data input considerations

Everything in this section constrains how the per-framework results should be read. **Read it before the scores.**

## 1.1 Training pixels per class — the dominant constraint

RF-A trains **per pixel**, so pixel counts are what matter, not polygon counts. These disagree sharply.

**Train block** (3 tiles):

| class | polygons | **pixels** | % of pixels | median polygon m² | from CHM candidates |
|---|---|---|---|---|---|
| bare | 34 | 8,205 | **54.5%** | 90.4 | 0 |
| grass | 39 | 4,130 | 27.5% | 88.3 | 0 |
| **shrub** | **125** | **918** | **6.1%** | **5.0** | 108 |
| tree | 59 | 1,791 | 11.9% | 32.8 | 0 |

**Test block** (2 tiles):

| class | polygons | pixels | % of pixels | median polygon m² | from CHM candidates |
|---|---|---|---|---|---|
| bare | 17 | 3,418 | 57.3% | 152.9 | 0 |
| grass | 16 | 1,253 | 21.0% | 80.3 | 0 |
| shrub | 88 | 846 | 14.2% | 6.0 | 88 |
| tree | 8 | 450 | 7.5% | 50.8 | 0 |

**Shrub has the most polygons and the fewest pixels.** Median shrub polygon is 5 m² against 90 m² for bare — an 18× difference. The `≥50 polygons per class per role` gate is therefore **not** a guarantee of balanced training data, and shrub sits at 6% of training pixels while bare sits at 55%.

Mitigated with `class_weight="balanced"` in the forest, but weighting cannot manufacture information: 918 pixels is a small sample for a spectrally ambiguous class.

## 1.2 Why training is per-pixel and not per-segment

`instructions5.md` §5 Step 1d originally specified RF over SLIC segments. Measured, that is unworkable here:

- SLIC segments are a uniform **9 px**; the median accepted shrub polygon is **5 px** — 0.56 of one segment.
- Segments claimable in the train block, by minimum coverage:

| class | any overlap | ≥50% | ≥70% |
|---|---|---|---|
| bare | 1302 | 897 | 738 |
| grass | 703 | 441 | 339 |
| **shrub** | 334 | 65 | **22** |
| tree | 389 | 178 | 104 |

At the ≥70% rule v3 used, shrub yields **22 training segments against bare's 738** — a 34:1 imbalance that would make shrub effectively unpredictable. That is the v3 failure in mirror image (`instructions2.md` §4.5: a starved training set produced 93–97% tree cover).

Per-pixel training gives shrub 918 samples instead of 22. Segment rasters are still produced and remain available for a segment-level comparison later.

## 1.3 Label provenance — and where it is circular

Labels are the **union** of hand-drawn polygons and accepted CHM shrub candidates, matching what `run_stage2_4_check_hand_labeling_progress.py` counts.

**All 108 train and 88 test shrub labels are CHM-derived.** Two consequences:

1. **`RF-A_D` and `RF-A_E` will be partly circular for shrub** — they receive CHM as a feature while the shrub labels came from a CHM threshold. Their shrub score will be inflated and should not be compared like-for-like with A–C.
2. **`RF-A_A` through `RF-A_C` carry the meaningful test**: can RGB, vegetation indices and texture recover CHM-defined shrub *without* CHM?

There is also a selection effect: accepted candidates are structurally the *detectable* shrubs. Shrubs below the CHM floor never became candidates, so they are absent from training and from this validation.

## 1.4 CHM is blind below 0.7 m

Measured across all five tiles, NEON CHM is pre-thresholded: **minimum non-zero value is exactly 0.700 m**, with no pixel anywhere between 0 and 0.7. Woody vegetation below 0.7 m is recorded as 0 and is indistinguishable from bare ground.

`H_GRASS_MAX` is therefore set to **0.7 m** (measured, not assumed — `instructions5.md` §3 safeguard 2). Small shrubs below that height are missing from the reference path entirely, which caps what any framework can be scored against.

## 1.5 Every framework sees identical pixels

Validity is computed over the **full 20-band stack**, not each framework's subset. A subset-derived mask would break the §4.1 deconfounding: a framework adding a band with more NaN would train on fewer pixels and could score differently for that reason alone.

Verified at SRER: all 20 bands carry the same 0.80% NaN (the 2 m edge margin), so A and B see identical masks — but the guarantee is now structural rather than coincidental.

Shadow pixels are excluded from training: §5 Step 1c resolves shadow to tree or to nodata, so a shadowed pixel carries no reliable class.

## 1.6 Validation design

- **Leave-one-tile-out** over the 3 train tiles.
- The 2 test tiles are **not** used for validation here — they carry only shrub labels in quantity and would not support a per-class estimate.
- Training pixels come from polygons the analyst chose as *clean examples*, so ambiguous pixels are under-represented and **all scores are optimistic**, shrub most of all.

## 1.7 Masked pixels — shadow and edge margin

Roughly 2–4% of every tile is excluded from both training and prediction, and carries NODATA (255) in the classification and NaN in the probability rasters.

| tile | role | masked | shadow | edge/NaN | shadow share of masked |
|---|---|---|---|---|---|
| 511000_3527000 | train | 3.86% | 3.09% | 0.80% | 79% |
| 511000_3528000 | train | 3.89% | 3.10% | 0.80% | 79% |
| 511000_3529000 | train | 3.92% | 3.15% | 0.80% | 80% |
| 515000_3530000 | test | 1.93% | 1.14% | 0.80% | 59% |
| 515000_3531000 | test | 2.00% | 1.21% | 0.80% | 60% |

Two independent causes, overlapping by only 0.01–0.02%:

- **Shadow** — 1.1–3.2% of a tile. §5 Step 1c resolves shadow to tree where it sits within `SHADOW_TREE_RADIUS` of a tall CHM return, and masks the rest; a shadowed pixel carries no reliable class. Train tiles carry roughly 3× the shadow of test tiles, consistent with their higher tree cover.
- **Edge margin** — a flat **0.80% on every tile**, the 2 m buffer where texture windows would run off the tile edge (§11.7). Constant by construction, identical across all 20 feature bands, which is what makes the §1.5 identical-pixels guarantee hold at this site.

**Consequences to carry forward:**

1. **Step 3 aggregation must exclude these from the denominator**, or a block bordering a tree will report a depressed total rather than the correct fraction over its valid pixels. The `< 75% valid` block-drop rule (§5 Step 3) is what keeps a heavily shadowed block out of the fraction estimates entirely.
2. **Shadow is not missing at random.** It clusters against the north side of woody canopy, so masking it removes pixels that are disproportionately *adjacent to tree and shrub*. Any bias this introduces runs against the woody classes, not the open ones.
3. **Class shares printed by `run_stage3_1_random_forest_ground_truth_classification.py` are fractions of the whole tile**, so they sum to about 96–98% rather than 100%. The remainder is this mask.

---

# 2. RF-A_A — RGB only

**10 features**: `r, g, b, ExG, ExR, ExGR, VARI, GLI, luma, saturation`
**macro-F1 0.788 · overall 0.913**

| class | recall | precision | F1 | support |
|---|---|---|---|---|
| bare | 0.988 | 0.995 | **0.991** | 8,200 |
| grass | 0.860 | 0.912 | 0.885 | 4,102 |
| **shrub** | 0.454 | 0.371 | **0.409** | 844 |
| tree | 0.903 | 0.833 | 0.866 | 1,190 |

**Confusion** (rows true, columns predicted):

| | bare | grass | shrub | tree |
|---|---|---|---|---|
| **bare** | 8,103 | 16 | 81 | 0 |
| **grass** | 8 | 3,529 | **476** | 89 |
| **shrub** | 36 | **299** | 383 | 126 |
| **tree** | 0 | 25 | 91 | 1,074 |

**Top features**: `luma` 0.179, `ExR` 0.149, `g` 0.101, `VARI` 0.100, `r` 0.090

**Per fold**: macro-F1 0.812 / 0.783 / 0.762 — stable. Overall swings wider (0.845 / 0.942 / 0.916) because tile composition differs.

**Reading**: bare/vegetated separation is essentially solved from visible bands alone. Tree at F1 0.87 with no height and no NIR is a genuinely good sign for the NAIP transfer. Shrub fails, and the confusion names the mechanism: **grass↔shrub is bidirectional** — 299 shrub pixels go to grass, 476 grass pixels come back as shrub. That is exactly the pair `instructions1.md` §4 identified as the weakest link.

`luma` ranking first is worth watching: it is also the channel shadow detection keys on, so part of its importance may be shadow structure rather than vegetation.

---

# 3. RF-A_B — RGB + vegetation indices

**13 features**: `RF-A_A` + `SAVI, NDVI, EVI`
**macro-F1 0.808 · overall 0.921**

| class | recall | precision | F1 | Δ F1 vs `RF-A_A` |
|---|---|---|---|---|
| bare | 0.989 | 0.995 | 0.992 | +0.001 |
| grass | 0.869 | 0.914 | 0.891 | +0.006 |
| **shrub** | 0.500 | 0.418 | **0.455** | **+0.046** |
| tree | 0.924 | 0.866 | 0.894 | +0.028 |

**Top features**: `NDVI` 0.142, `luma` 0.116, `ExR` 0.093, `SAVI` 0.089, `VARI` 0.086

**Reading**: adding NIR indices buys **+0.020 macro-F1**, and it is concentrated where it matters — shrub +0.046 and tree +0.028, while the already-solved bare barely moves. `NDVI` immediately takes the top importance slot and displaces `luma`, which is the expected result: a real near-infrared vegetation signal outperforms a visible-band proxy for it.

Shrub at F1 0.455 is still the weakest class by a wide margin. Vegetation indices help but do not resolve grass↔shrub, which is consistent with the two classes differing more in **structure** than in greenness — the argument for texture in `RF-A_C`.

---

# 4. RF-A_C — RGB + vegetation indices + texture

**19 features**: `RF-A_B` + `glcm_contrast, glcm_homogeneity, glcm_correlation, glcm_entropy, lbp_nonuniform, std`
**macro-F1 0.884 · overall 0.955**

| class | recall | precision | F1 | Δ F1 vs `RF-A_B` |
|---|---|---|---|---|
| bare | 0.992 | 0.996 | 0.994 | +0.002 |
| grass | 0.936 | 0.955 | 0.945 | +0.054 |
| **shrub** | 0.687 | 0.632 | **0.658** | **+0.203** |
| tree | 0.950 | 0.926 | 0.938 | +0.044 |

**Confusion** (rows true, columns predicted):

| | bare | grass | shrub | tree |
|---|---|---|---|---|
| **bare** | 8,138 | 22 | 40 | 0 |
| **grass** | 6 | 3,839 | **244** | 13 |
| **shrub** | 30 | **156** | 580 | 78 |
| **tree** | 0 | 5 | 54 | 1,131 |

**Top features**: `NDVI` 0.129, `ExR` 0.082, `glcm_entropy` 0.076, `luma` 0.070, `glcm_contrast` 0.068, `VARI` 0.063, `std` 0.062

**Per fold**: macro-F1 0.910 / 0.909 / 0.852.

**Reading — texture is the decisive input.** Shrub F1 jumps **+0.203** in a single step, roughly four times the entire gain from adding vegetation indices, and grass gains +0.054 alongside it. Both directions of the grass↔shrub confusion collapse: shrub→grass falls 299 → 156, grass→shrub falls 476 → 244.

This is the hypothesis in `instructions5.md` §4.1 confirmed on data: grass and shrub differ in **structure** far more than in greenness, so a spectral feature set cannot separate them however many bands it has, and a texture feature set can. Three texture measures land in the top seven (`glcm_entropy`, `glcm_contrast`, `std`), and they displace pure-colour features rather than supplementing them.

**Two cautions.**

Fold three is materially weaker (macro-F1 0.852 against 0.910 and 0.909), driven by shrub precision collapsing to 0.425 when `511000_3529000` is held out. Shrub is the least stable class across folds even at its improved score.

Predicted grass on the **test** tiles falls to 0.3% and 0.1%, from 2.0% and 1.8% under `RF-A_A`. The test block is genuinely shrub-dominated (§2.1: 54.6% and 37.3% CHM shrub-band cover against ~12% on train), so some of this is real — but a near-total absence of a class that holds 21% of test *label* pixels deserves checking before it is accepted. Test labels are not yet complete enough to test this properly.

---

# 5. RF-A_D — RGB + vegetation indices + texture + CHM

**20 features**: `RF-A_C` + `CHM`
**macro-F1 0.922 · overall 0.968**

| class | recall | precision | F1 | Δ F1 vs `RF-A_C` | circular? |
|---|---|---|---|---|---|
| bare | 0.995 | 0.997 | 0.996 | +0.002 | no |
| grass | 0.949 | 0.964 | 0.956 | +0.011 | no |
| **shrub** | 0.795 | 0.742 | **0.768** | **+0.110** | **YES** |
| tree | 0.975 | 0.963 | 0.969 | +0.031 | no |

**Confusion** (rows true, columns predicted):

| | bare | grass | shrub | tree |
|---|---|---|---|---|
| **bare** | 8,161 | 30 | 9 | 0 |
| **grass** | 14 | 3,892 | 194 | 2 |
| **shrub** | 14 | 116 | 671 | 43 |
| **tree** | 0 | 0 | 30 | 1,160 |

**Top features**: `CHM` **0.193**, `NDVI` 0.102, `luma` 0.077, `SAVI` 0.066, `glcm_entropy` 0.064

**Per fold**: macro-F1 0.916 / 0.971 / 0.899 — wider spread than `RF-A_C`, and shrub precision swings hard: 0.959 / 0.919 / **0.554**.

## 5.1 The shrub score is circular — measured, not assumed

All 196 shrub labels came from CHM-derived candidates (§1.3), and `CHM` is this model's top feature at 0.193. So its shrub score partly measures *can a model with CHM reproduce a CHM threshold*. The test is to rasterize the labelling rule and measure agreement with the prediction (`stage3_definitions.md` §13):

**IoU between predicted shrub and the CHM band [0.7, 2.0) m:**

| tile | `RF-A_C` (no CHM) | `RF-A_D` (CHM) |
|---|---|---|
| 511000_3527000 | 0.157 | **0.407** |
| 511000_3528000 | 0.181 | **0.530** |
| 511000_3529000 | 0.172 | **0.508** |
| 515000_3530000 | 0.423 | **0.572** |
| 515000_3531000 | 0.417 | **0.578** |

Agreement with the labelling rule **roughly triples on the train tiles** the moment CHM enters. `RF-A_D` is not using height as one signal among twenty — its shrub map converges on the threshold that generated its own labels.

## 5.2 What CHM actually buys, read from the clean classes

Bare, grass and tree are hand-labelled, so their gains are not circular and measure what height genuinely contributes:

| class | Δ F1 `RF-A_C` → `RF-A_D` | interpretation |
|---|---|---|
| tree | **+0.031** | real, and expected — height is the definition of tree |
| grass | +0.011 | small, real |
| bare | +0.002 | nothing left to gain at 0.996 |
| **shrub** | **+0.110** | 3.5× the largest clean-class gain — the leakage signature |

So CHM is worth roughly **+0.03 on tree** honestly. The +0.110 on shrub is not comparable to that, and should not be read as a like-for-like improvement over `RF-A_C`.

## 5.3 Why this does not make `RF-A_D` the winner

1. **The advantage cannot transfer.** CHM is unavailable or 1–2 years offset at the AmeriFlux sites (§1.4, `instructions5.md` R1), so any `RF-A_D` shrub edge does not travel to WKG — which is the point of the exercise.
2. **`instructions5.md` §4.1 already assigns it a different job.** `RF-A_D`/`RF-A_E` are the **reference/labeller**, not transfer candidates; selection happens among `RF-A_A`–`RF-A_C`. The measurement above is the empirical justification for that split.
3. **Fold instability worsens.** Shrub precision ranges 0.554–0.959 across folds, a wider spread than `RF-A_C`, which is not what a genuinely stronger model looks like.

**Also note**: predicted shrub on test tile `515000_3530000` reaches **62.0%**, against a CHM shrub-band cover of 54.6% on that tile — the prediction is tracking the band. Predicted grass falls to 0.7% and 0.3% on the test tiles, continuing the trend flagged for `RF-A_C`.

---

# 6. Framework comparison

**Per-class F1, all four classes:**

| framework | features | bare | grass | shrub | tree | **macro-F1** | overall |
|---|---|---|---|---|---|---|---|
| `RF-A_A` — RGB | 10 | 0.991 | 0.885 | 0.409 | 0.866 | 0.788 | 0.913 |
| `RF-A_B` — + VI | 13 | 0.992 | 0.891 | 0.455 | 0.894 | 0.808 | 0.921 |
| **`RF-A_C`** — + texture | 19 | 0.994 | **0.945** | **0.658** | 0.938 | **0.884** | 0.955 |
| `RF-A_D` — + CHM ⚠ | 20 | 0.996 | 0.956 | *0.768* | 0.969 | *0.922* | 0.968 |
| `RF-A_E` — full combined | 20 | *not runnable — see §6.1* | | | | | |

⚠ `RF-A_D` shrub and macro-F1 are **italicised because they are circular** — see §5.1. Not comparable like-for-like with `RF-A_A`–`RF-A_C`. **`RF-A_C` is the best transferable variant.**

## 6.1 `RF-A_E` was not run — it is currently identical to `RF-A_D`

`FRAMEWORK_GROUPS` gives `E` the same six feature groups as `D`, and the feature stack holds 20 bands which `D` already uses in full. Verified directly: the two resolve to **identical 20-feature sets, with nothing added**. With the same seed, same pixels and same features, `RF-A_E` would fit an identical model and report identical numbers — a duplicate row that reads as an independent result.

**What `RF-A_E` is supposed to be.** `instructions5.md` §4.1 defines E as the *full combined model — all layers above, same dual-track (traditional + DL) approach as A*. Its distinguishing element is therefore **not another feature group** — there are none left — but the **deep-learning track**. That track is gated on the unresolved SCC GPU blocker (§8), so E cannot yet be run as anything other than a copy of D.

**Three ways to resolve it**, none of which should be chosen silently:

1. **Leave E undefined until the DL track exists.** Honest, and matches the §4.1 intent. The A–E comparison is complete at four variants for the traditional track.
2. **Redefine E as a genuinely different traditional model** — for example D's features with a different algorithm, or D plus segment-level context features. That changes the §4.1 deconfounding contract, which fixes the algorithm and varies only inputs, so it would need recording as a deliberate departure.
3. **Drop E from the traditional track entirely** and reserve the name for the DL-combined model.

Recommend (1): the letter stays reserved, nothing misleading enters the comparison, and the DL blocker is what unblocks it.


**Change at each step:**

| step | bare | grass | shrub | tree | macro-F1 |
|---|---|---|---|---|---|
| `RF-A_A` → `RF-A_B` (add VI) | +0.001 | +0.006 | +0.046 | +0.028 | +0.020 |
| **`RF-A_B` → `RF-A_C`** (add texture) | +0.002 | **+0.054** | **+0.203** | **+0.044** | **+0.076** |
| `RF-A_A` → `RF-A_C` (total) | +0.003 | +0.060 | **+0.249** | +0.072 | +0.096 |
| `RF-A_C` → `RF-A_D` (add CHM) ⚠ | +0.002 | +0.011 | *+0.110* | +0.031 | *+0.038* |

Nothing regresses at either step, so both additions are clean gains rather than trades.

**Texture is worth roughly four times what vegetation indices are.** Adding NIR bought +0.020 macro-F1; adding texture bought +0.076, and the difference is concentrated almost entirely in shrub (+0.046 versus +0.203).

The pattern across the three frameworks is consistent and interpretable:

- **bare** was solved from visible bands alone (0.991) and never moves — nothing to gain.
- **tree** improves modestly with each addition, ending at 0.938 with no height information at all.
- **grass and shrub** are the pair that carries the whole story. They barely respond to more spectral bands and respond strongly to texture, which is the signature of two classes separated by **structure rather than colour**.

**Overall accuracy remains a misleading summary**: 0.913 → 0.921 → 0.955 understates a shrub gain of +0.249, because bare is 55% of the pixels. Judge on per-class, per §6.4.

The `RF-A_*` variants differ by input feature group **only** — same labels, same pixels, same seed, same algorithm — so any difference is attributable to the inputs (§4.1).

**Overall accuracy is a misleading summary here** (0.913 → 0.921 hides a shrub gain of +0.046) because bare is 55% of the pixels. Judge on per-class, per §6.4.

---

# 7. Run 2 — polygon subsampling

**Change**: `max_pixels_per_polygon = 100`. Nothing else differs from run 1 — same labels, same seed, same features, same variants.

## 7.1 Why

Pixels inside one polygon are near-duplicates: same object, same lighting, adjacent ground. Run 1 let every polygon donate all of its pixels, so a 1,386 m² bare polygon contributed 1,386 highly autocorrelated samples while a 5 m² shrub polygon contributed 5. Bare averaged **241 px/polygon against shrub at 7.4**, meaning the forest saw roughly 34 independent bare observations dressed as 8,200 samples.

The nominal 8.9:1 bare:shrub imbalance was therefore **real in weight but largely fake in information**, and `class_weight="balanced"` was compensating for redundancy rather than for genuine scarcity. Capping per polygon attacks the cause.

| class | run 1 px | run 2 px | polygons capped | run 1 share | run 2 share |
|---|---|---|---|---|---|
| bare | 8,207 | 2,159 | 12 of 34 | 54.5% | **28.2%** |
| grass | 4,131 | 2,824 | 15 of 39 | 27.5% | 36.7% |
| **shrub** | **920** | **920** | **0 of 125** | 6.1% | **12.0%** |
| tree | 1,792 | 1,775 | 1 of 59 | 11.9% | 23.1% |

**Shrub loses nothing** — no shrub polygon exceeds 100 px — so the cap strips redundancy only from the abundant classes. bare:shrub falls from 8.9:1 to 2.4:1 on roughly half the total pixels.

## 7.2 Effect, per variant

| variant | macro-F1 Δ | **shrub F1 Δ** | grass Δ | bare Δ | tree Δ |
|---|---|---|---|---|---|
| RF-A_A | −0.006 | **+0.032** | −0.049 | −0.013 | +0.006 |
| RF-A_B | −0.002 | **+0.042** | −0.044 | −0.011 | +0.004 |
| **RF-A_C** | **+0.005** | **+0.039** | −0.011 | −0.010 | +0.001 |
| RF-A_D | +0.001 | +0.020 | −0.008 | −0.007 | −0.000 |

**Shrub gains +0.02 to +0.04 in every variant**, and it is a real trade rather than a free lunch: bare and grass give up a little, having had redundant pixels removed. Since shrub is the binding constraint for the unmixing that consumes this map, the trade is the right direction.

Two patterns worth noting:

- **The gain is larger on the weaker variants** (A +0.032, B +0.042) than on D (+0.020). Once texture and CHM supply real signal, redundancy matters less — so redundancy was partly *masking weak features* rather than being the fundamental limit.
- **Grass pays most in A and B** (−0.049, −0.044) but barely in C (−0.011). Without texture, grass was leaning on sheer sample volume; with texture it has a feature that actually separates it.

## 7.3 Bias — the quantity Step 3 inherits

Per-class area bias for `RF-A_C`, the figure that propagates into Step 3 fractions:

| class | run 1 | run 2 |
|---|---|---|
| bare | −0.3% | −0.2% |
| grass | −2.0% | −2.9% |
| **shrub** | **+8.8%** | **+6.9%** |
| tree | +2.7% | +2.3% |

**Shrub over-prediction falls from +8.8% to +6.9%.** Random per-pixel errors partly cancel when the 9 one-meter pixels of a Planet block aggregate (N = 3, measured 2026-08-18 - these sections originally said 16); systematic ones do not, so this is the number RF-B actually inherits. Still positive, so shrub fraction remains biased high — but less so.

## 7.4 `RF-A_C` at run 2 — the current best transferable variant

**macro-F1 0.889 · overall 0.921** · bare 0.984, grass 0.934, **shrub 0.698**, tree 0.939

**Top features**: `NDVI` 0.129, `ExR` 0.090, `VARI` 0.076, `glcm_entropy` 0.074, `EVI` 0.069, `std` 0.065
**Per fold**: 0.914 / 0.910 / 0.859 — same pattern as run 1, third fold still weakest.

---

# 8. Run 3_smoke — the run 3 smoke test, on 10 quintile-stratified tiles

**Run `3_smoke` is run 3 in all but the label.** It was executed as a plumbing check before labelling was complete, using a throwaway run number so `run3/` stays reserved for the finished run. Its configuration *is* run 3's: 10 tiles, subsampling at 100 px/polygon, the advanced shrub review.

> **Read this comparison in one direction only.** Run `3_smoke` scores lower than run 2 on every headline number, and that is **not** a regression. `3_smoke` solves a harder, more honest problem: it validates against five held-out tiles spanning the site's full compositional range instead of three tiles all sitting at the site floor. Run 2's numbers were flattered by an unrepresentative validation set. **The right baseline for future runs is `3_smoke`, not run 2.**

## 8.1 What changed

| | run 2 | `3_smoke` |
|---|---|---|
| tiles | 5 | **10** (quintile-stratified, §2A) |
| CHM shrub-band quintiles covered | {1, 5} | **{1, 2, 3, 4, 5}** |
| leave-one-tile-out folds | 3 | **5** |
| training pixels | 7,017 | **8,950** |
| shrub training pixels | 844 (12.0%) | **1,531 (17.1%)** |

## 8.2 Headline scores — `RF-A_C`

| class | run 2 recall | `3_smoke` recall | run 2 prec | `3_smoke` prec | run 2 F1 | `3_smoke` F1 | Δ |
|---|---|---|---|---|---|---|---|
| bare | 0.983 | 0.936 | 0.985 | 0.886 | 0.984 | 0.910 | −0.073 |
| grass | 0.920 | 0.922 | 0.948 | 0.908 | 0.934 | 0.915 | −0.019 |
| **shrub** | 0.722 | 0.636 | **0.675** | **0.759** | 0.698 | 0.692 | −0.005 |
| tree | 0.950 | 0.925 | 0.929 | 0.897 | 0.939 | 0.911 | −0.028 |
| **macro-F1** | | | | | **0.889** | **0.857** | −0.031 |

Shrub is essentially unchanged (−0.005) and its **precision improved** (0.675 → 0.759) while recall fell — the model became more conservative about calling shrub, not worse at finding it.

## 8.3 Fold spread — the finding that matters most

| variant | run 2 folds | sd | `3_smoke` folds | sd |
|---|---|---|---|---|
| `RF-A_A` | 0.791 / 0.783 / 0.762 | 0.012 | 0.761 / 0.818 / 0.757 / 0.722 / 0.741 | **0.032** |
| `RF-A_B` | 0.806 / 0.801 / 0.793 | 0.005 | 0.782 / 0.847 / 0.791 / 0.725 / 0.702 | **0.051** |
| `RF-A_C` | 0.914 / 0.910 / 0.859 | 0.025 | 0.894 / 0.919 / 0.860 / **0.744** / **0.744** | **0.074** |

**Fold-to-fold variance roughly tripled, and that is the point.** Run 2's tight spread was an artifact: all three of its train tiles were near-identical (Q1, shrub cover 0.116–0.121), so holding one out tested almost nothing. Two of `3_smoke`'s folds score **0.744**, far below anything run 2 ever produced.

`3_smoke` is therefore the first honest estimate of generalization at this site, and it says the model is roughly **0.15 macro-F1 weaker on unfamiliar terrain** than run 2 implied.

## 8.4 Shrub's failure mode inverted — and so did its bias

`RF-A_C` shrub row, row-normalized:

| | → bare | → grass | → shrub | → tree |
|---|---|---|---|---|
| run 2 | 0.037 | **0.149** | 0.722 | 0.092 |
| `3_smoke` | **0.185** | 0.108 | 0.636 | 0.071 |

**Shrub→bare confusion rose 5× (0.037 → 0.185)** while shrub→grass fell. The per-class area bias reversed sign:

| class | run 2 bias | `3_smoke` bias |
|---|---|---|
| bare | −0.2% | +5.6% |
| grass | −2.9% | +1.5% |
| **shrub** | **+6.9%** | **−16.2%** |
| tree | +2.3% | +3.2% |

**This matters more than the F1 change.** Area bias is what Step 3 fractions inherit — random per-pixel errors partly cancel when the 9 pixels of a Planet block aggregate, systematic ones do not. Shrub moved from +6.9% over-predicted to **−16.2% under-predicted**, and that propagates directly into RF-B.

The cause is in the tiles: the added tiles are far more bare-dominated (`519000_3527000` is 84% CHM-zero, `511000_3532000` 71%) than the original train block. Shrub must now be separated from **bare**, a problem the original five tiles barely posed.

## 8.5 Feature importance shifted to match

| group | run 2 | `3_smoke` | Δ |
|---|---|---|---|
| texture | 0.259 | 0.287 | +0.028 |
| **brightness** | 0.088 | **0.158** | **+0.069** |
| **NIR indices** | 0.254 | **0.183** | **−0.071** |
| colour | 0.399 | 0.373 | −0.026 |

**`luma` moved from rank 10 to rank 1** (0.056 → 0.116), displacing `NDVI`, which fell to 4th. `std` rose from 6th to 3rd.

Coherent with everything above: on tiles that are 70–85% bare ground a greenness index has little to work with, and brightness plus texture carry the separation instead. It also sharpens the §1.7 concern — `luma` is the channel shadow detection keys on, and shadow clusters against woody canopy, so `luma` topping the list is a reason to verify it is not separating classes via shadow structure.

## 8.6 Status

Labelling continues per `run_stage2_4_check_hand_labeling_progress.py`. Three of the ten tiles still hold **zero** polygons, so `3_smoke`'s new folds are training on shrub candidates with almost no bare, grass or tree from those tiles. Two things to re-check once the gate passes:

1. whether shrub's **−16.2% area bias** closes as the new tiles receive real bare/grass/tree polygons;
2. whether `luma`'s rise to first survives, or was an artifact of the unlabelled tiles.

---

# 9. Run 3 — the finished run on 10 tiles

Same configuration as `3_smoke` — 10 quintile-stratified tiles, subsampling at 100 px/polygon — executed once labelling passed the §4.2 gate. **`3_smoke` was the plumbing check; run 3 is the result.**

> **Run 3 confirms `3_smoke` rather than reversing it.** Every pattern the smoke test showed survived the completed labelling: lower headline scores than run 2, wider fold spread, shrub confused with bare, and `luma` displacing `NDVI`. Those were not artifacts of incomplete labels.

## 9.1 Training data — the best balance yet

| class | run 1 | run 2 | `3_smoke` | **run 3** | run 3 share |
|---|---|---|---|---|---|
| bare | 8,200 | 2,159 | 2,671 | **3,601** | 29% |
| grass | 4,102 | 2,824 | 3,218 | **4,717** | 38% |
| **shrub** | 844 | 844 | 1,531 | **2,087** | **17%** |
| tree | 1,190 | 1,190 | 1,530 | **2,158** | 17% |
| total | 14,336 | 7,017 | 8,950 | **12,563** | |
| folds | 3 | 3 | 5 | **6** | |
| tiles | 5 | 5 | 10 | **10** | |

**Shrub training pixels are 2.5× run 2's**, and the class balance is the healthiest of any run — 29/38/17/17 against run 1's 57/29/6/8. The polygon subsampling and the added tiles both contributed; neither alone would have done it.

## 9.2 Per-class F1, all variants

| variant | class | run 1 | run 2 | `3_smoke` | **run 3** |
|---|---|---|---|---|---|
| `RF-A_A` | macro | 0.788 | 0.782 | 0.775 | **0.729** |
| `RF-A_B` | macro | 0.808 | 0.806 | 0.789 | **0.757** |
| **`RF-A_C`** | **macro** | 0.884 | 0.889 | 0.857 | **0.849** |
| `RF-A_D` ⚠ | macro | 0.922 | 0.923 | 0.909 | *0.899* |

`RF-A_C` per class:

| class | run 2 | `3_smoke` | **run 3** |
|---|---|---|---|
| bare | 0.984 | 0.910 | 0.898 |
| grass | 0.934 | 0.915 | 0.912 |
| **shrub** | 0.698 | 0.692 | **0.676** |
| tree | 0.939 | 0.911 | 0.908 |

`RF-A_C` remains **the best transferable variant** at macro 0.849, shrub 0.676. `RF-A_D` scores higher but its shrub number stays circular (§5.1) and cannot transfer.

**One telling detail**: `RF-A_D` shrub is the *only* cell in the entire run 2 → run 3 comparison that improved (0.787 → 0.798). That is consistent with the circularity rather than against it — more CHM-derived shrub labels make a CHM threshold easier to re-derive. It is not a reason to prefer D.

## 9.3 Fold spread — the number that keeps growing, and should

| | folds | macro-F1 per held-out tile | sd | worst |
|---|---|---|---|---|
| run 2 | 3 | 0.914 / 0.910 / 0.859 | 0.025 | 0.859 |
| `3_smoke` | 5 | 0.894 / 0.919 / 0.860 / 0.744 / 0.744 | 0.074 | 0.744 |
| **run 3** | **6** | 0.890 / 0.912 / 0.851 / 0.785 / **0.660** / 0.745 | **0.087** | **0.660** |

**A 0.25 macro-F1 spread between the best and worst held-out tile.** Run 2's sd of 0.025 was measuring three near-identical tiles (all Q1, shrub cover 0.116–0.121); it described tile similarity, not model stability. Run 3's 0.087 across six tiles spanning all five quintiles is what generalization at SRER actually looks like.

**Read this as the headline result, not the macro-F1.** A single pooled number hides a model that scores 0.912 on familiar terrain and 0.660 on unfamiliar.

## 9.4 Shrub bias improved, but remains the binding problem

| class | run 2 | `3_smoke` | **run 3** |
|---|---|---|---|
| bare | −0.2% | +5.6% | +3.6% |
| grass | −2.9% | +1.5% | +1.7% |
| **shrub** | **+6.9%** | **−16.2%** | **−14.7%** |
| tree | +2.3% | +3.2% | +4.5% |

More labels moved shrub 1.5 points toward zero — the right direction — but shrub is still **under-predicted by ~15%**, and area bias is what Step 3 fractions inherit (§7.3). Random per-pixel error partly cancels across the 9 pixels of a Planet block; this does not.

`RF-A_C` confusion, run 3, row-normalized:

| | → bare | → grass | → shrub | → tree |
|---|---|---|---|---|
| bare | 0.914 | 0.051 | 0.034 | 0.000 |
| grass | 0.010 | 0.920 | 0.052 | 0.017 |
| **shrub** | **0.185** | 0.107 | 0.627 | 0.081 |
| tree | 0.001 | 0.022 | 0.048 | 0.929 |

**Shrub→bare sits at 0.185**, five times run 2's 0.037 and unchanged from `3_smoke`. On bare-dominated tiles the model gives shrub away to bare. Completing the labelling did not fix this, which means it is a feature-space problem rather than a sample-size one — the argument for multi-scale texture and context features (§9 open items).

## 9.5 Feature importance has stabilised

`RF-A_C` top five:

| run | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| run 2 | NDVI 0.129 | ExR 0.090 | VARI 0.076 | glcm_entropy 0.074 | EVI 0.069 |
| `3_smoke` | **luma 0.116** | ExR 0.102 | std 0.094 | NDVI 0.093 | glcm_entropy 0.077 |
| **run 3** | **luma 0.116** | ExR 0.099 | NDVI 0.096 | std 0.090 | glcm_entropy 0.081 |

`luma` sits at **0.116 in both** `3_smoke` and run 3 — identical to three decimals, having displaced `NDVI` from run 2's top slot. That is now a stable property of the 10-tile set rather than a smoke-test artifact.

It also sharpens the §1.7 concern: `luma` is the channel shadow detection keys on, shadow clusters against woody canopy, and shrub is the class being lost. **Checking whether `luma` separates classes via shadow structure is now a priority, not a curiosity.**

## 9.6 What run 3 hands to Step 3, now that the grid is measured

The PlanetScope grid was measured on 2026-08-18 by `run_stage1_3_define_planet_grid.py` and visually verified in QGIS. Three consequences land directly on the numbers above.

**A Planet block is 9 pixels, not 16.** The LSP pixel is **3 m**, so **N = 3** — `instructions5.md` previously said "expected 4" and that was wrong. Every 1 m classification error now averages over 9 samples instead of 16, so **random per-pixel error cancels less** than the earlier design assumed. The per-pixel scores in §9.2 translate into noisier block fractions than a 16-pixel block would have given.

**The −14.7% shrub area bias does not cancel at all.** §9.4 established it is systematic. Averaging 9 pixels instead of 16 changes nothing about that — a systematic bias survives any amount of aggregation. **Every Planet block's shrub fraction will be low by roughly 15%**, and that propagates into the RF-B training target and into Step 6 areas. Step 3 output built on run 3 is therefore **provisional**, and the per-class area bias must be recorded in the Step 3 report beside the fractions so a later reader cannot mistake a biased fraction product for an unbiased one.

**The valid-block rule is now a count: 8 or 9 of 9.** The old "< 75% valid" wording quantised to ≥ 7 of 9 at N = 3 — it would have admitted blocks with 22% of their area unobserved while appearing to enforce 75%. The rule is now **at most one masked pixel per block** (`stage4_1_aggregation.min_valid_pixels_per_block: 8`). This matters here because masked pixels are not missing at random: shadow is 79% of them on the train tiles (§1.7) and clusters against woody canopy, so a loose rule would preferentially admit degraded blocks over shrub and tree — the two classes already weakest in §9.2, one of which is already under-predicted by 14.7%. **The 0–9 valid-pixel histogram, reported at both 8-of-9 and 7-of-9, is the first Step 3 output to read**: it makes the retention cost of the stricter cut a measured number rather than an assumption.

One tile is also cropped: **`520000_3532000` (test) is 55.4% inside the PlanetScope footprint**, and the remainder has no Planet pixel to aggregate into. Its contribution to any Step 3 or Step 6 total is roughly half what its 1 km extent suggests.

---

# 10. Open items

1. **Run 3 should add the completed shrub candidate review** — 349 candidates pending, worth roughly 160 more shrub polygons at the observed 46% accept rate, which would roughly double shrub training pixels. Run 2 changed only the subsampling; the review is the other half of the plan and the two have not yet been combined.
2. **Define `RF-A_E`, or retire the letter** — it is currently identical to `RF-A_D` and cannot be run meaningfully (§6.1). Its intended distinguishing element is the DL track, blocked on the GPU issue (`instructions5.md` §8). Bounded by the circularity in §1.3 — most shrub labels came from a CHM threshold and D receives CHM as a feature — so its shrub score is not comparable like-for-like with `RF-A_A` through `RF-A_C`.
3. **Shrub remains the weakest class** at F1 0.658 — small support (918 px), a selection effect toward detectable shrubs, and the least fold-to-fold stability of any class. Texture closed most of the gap; it did not close all of it.
4. **Test-block validation is not yet possible** — test carries shrub labels but few bare, grass or tree. Completing test labeling unlocks a true held-out spatial check.
5. **Predicted grass nearly vanishes on the test tiles under C** (0.3% and 0.1%). Partly real — the test block is shrub-dominated — but worth confirming once test labelling supports a per-class check.
6. **`luma` importance** should be checked against the shadow mask to confirm it is not tracking shadow structure. §1.7 makes this more pressing: shadow is 79% of masked pixels on train tiles and clusters against woody canopy, exactly where `luma` would separate tree from grass for the wrong reason.
7. **`mahalanobis_distance`** (§3.1 secondary diagnostic) is specified but not yet written by `run_stage3_1_random_forest_ground_truth_classification.py`. It needs class centroids and a pooled covariance.
8. **Median `prediction_quality` per tile** ranges 0.50–0.78 (`RF-A_A`). `515000_3530000` is lowest at 0.52 — the same tile with the 32% shrub-candidate accept rate and 54.6% CHM shrub-band cover. Three independent signals point at that tile being atypical; worth understanding before it enters Step 6.
9. **These figures never become Step 6 accuracy.** That requires the independent probability sample and area-weighted estimators in §6.
10. **Step 3 output from run 3 is provisional until the shrub bias is addressed** (§9.6). The remedy is multi-scale texture and context features, not more labels — the bias moved only 1.5 points between the partial-label smoke run and the fully labelled run 3, which is what rules out sample size as the cause. Build the Step 3 machinery on run 3 regardless, since the block distributions are needed to size everything downstream, but tag the fractions and do not train RF-B on them as final.
11. **Aggregation must run on a mosaic, not per tile.** No SRER tile is congruent with the Planet grid in both axes (offsets of 0, 1 or 2 m; `instructions5.md` §5 Step 3). Blocking each tile in isolation emits partial blocks on all four edges of all ten tiles, which then fail the valid-pixel rule and silently delete every tile perimeter from the fraction product.

---

# 11. Run 4 — the current run, on ten fully-flown tiles

§11.1 to §11.3 were written **before** the run, so the changes are on record independently of the scores. §11.4 holds the results.

```
python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 4 --frameworks A B C D
```

> **RUN 4 CHANGES SIX THINGS AT ONCE. It is not a clean comparison against run 3, and no single score movement can be attributed to a single cause.** Three of the six are corrections to defects that were present in run 3, so run 3's numbers are not a clean baseline either — they were computed over partly-unusable ground. Run 4 replaces run 3 as the reference; do not average or interpolate between them.

## 11.1 The six changes

**1. Two part-flown tiles replaced** (`instructions5.md` §2A, `stage4_results.md` §2.2).

| out | flown | in | flown | role | quintile |
|---|---|---|---|---|---|
| `511000_3532000` | **4.09%** | `517000_3531000` | 99.92% | train | Q4 |
| `520000_3532000` | **24.37%** | `516000_3528000` | 100.00% | test | Q2 |

Every tile in run 4 is ~100% flown *and* wholly inside the PlanetScope footprint — the first run for which that is true. In run 3, `511000_3532000` contributed 38,684 of a possible 1,000,000 pixels while counting as one of six training tiles, so **its leave-one-tile-out fold was a 4 ha fold**. Run 4 has six genuine folds.

**2. Unflown RGB ground is masked.** NEON RGB declares no nodata and writes unflown ground as all-zero, which every downstream test read as ordinary dark ground. `helpers.read_rgb_at_scale` now masks all-zero RGB to NaN. Affects **`517000_3531000` only, 1.137% of the tile**; all nine other tiles are 0.000%. Usable fraction there is now 98.105%.

**3. Shadow masks re-cut on every tile.** The pooled luma threshold is site-wide, so any change to the tile set or to RGB masking shifts it. Now **111.518** (p20), blue-fraction p80 **0.2966**. Shadow extent rose substantially against run 3:

| tile | shadow→tree, run 3 | run 4 |
|---|---|---|
| 511000_3527000 | 2.02% | 2.96% |
| 519000_3527000 | 1.70% | 3.13% |
| 515000_3530000 | 0.18% | 0.63% |

**4. k-means refit, clusters renumbered.** Cluster identities from run 3 do not carry over. The refit also eliminated the artifact cluster: run 3's **cluster 4 was 100% unflown RGB** on `517000_3531000` — 10,173 px of unphotographed ground that k-means had made into its own stratum. Clusters below the 0.1% floor are now **6 (0.02%) and 7 (0.06%)**, previously 8 (0.068%).

**5. Label set changed in both directions.** Net **fewer** labels, because the retired tiles carried more reviewed shrub candidates than the new tiles do yet:

| | run 3 tile set | run 4 tile set |
|---|---|---|
| drawn polygons | 423 | **449** (+26) |
| accepted shrub candidates | 350 | **242** (−108) |
| shrub review progress | 678/1500 (45%) | 569/1500 (38%) |

**Expect shrub training pixels to fall relative to run 3's 2,087.** `517000_3531000` has **0 of 150** candidates reviewed and `516000_3528000` has 50. This is the one change likely to push a score *down*, and it is the first thing to check if shrub F1 drops.

**6. Gate logic corrected** (no effect on training data). The cluster-coverage check required every above-floor cluster in **both** roles, but k-means is fit site-wide and a cluster can be confined to one tile — which made the gate unsatisfiable. It now requires a cluster only in roles where it is actually present (`cluster_presence_min_pixels: 1000`).

Also non-behavioural: `run_stage3_1_*.py` was refactored for readable names and a `TileData` NamedTuple, and a `config`-shadowing bug in `run_stage1_6_detect_shadows.py` was fixed (it had become unrunnable).

## 11.2 Gate state entering run 4

**PASSED.** Polygons per class per role, against a floor of 50:

| class | test | train |
|---|---|---|
| bare | 60 | 53 |
| grass | 51 | 71 |
| shrub | 135 | 150 |
| tree | 65 | 93 |

Cluster coverage 14/14 required in both roles. Geometry and `class_code` problems: 0.

## 11.3 What to compare against, and what not to

- **Compare run 4 to run 3 for direction only, never for attribution.** Six changes moved together.
- **Run 4 is the new reference.** Runs 1, 2 and `3_smoke` are superseded; run 3 is superseded as a baseline because it included two part-flown tiles and an unflown-ground cluster.
- **The interesting number is fold spread, not macro-F1.** Run 3's folds ran 0.890 / 0.912 / 0.851 / 0.785 / 0.660 / 0.745 (sd 0.087) — and one of those folds was the 4 ha tile. Run 4's spread over six real folds is the first honest read on tile-to-tile stability.
- **Shrub area bias is the number that propagates.** Run 3 `RF-A_C` sat at −14.7% (§9.4), and Stage 4_1 fractions inherit it whole. Watch whether replacing the tiles moves it; the working hypothesis (§10 item 10) is that it will not, because the cause is feature-space, not sample-size.
- **`RF-A_D` shrub remains circular** and is not comparable like-for-like (§1.3).

## 11.4 Results

**Run 4 completed 2026-08-26, all four variants, six folds each.** Read this against §11.1: six things moved together, so nothing below is attributable to one cause.

### The headline: shrub area bias fell by two thirds

`RF-A_C`, computed from the confusion matrix (rows true, columns predicted):

| class | run 3 | run 4 |
|---|---|---|
| bare | +3.58% | **−4.49%** |
| grass | +1.67% | +1.89% |
| **shrub** | **−14.66%** | **−4.65%** |
| tree | +4.54% | +5.67% |

**This is the number Stage 4_1 fractions inherit**, and it improved by a factor of three. It also **refutes the working hypothesis in §10 item 10**, which held that the shrub bias was a feature-space problem that more or better labels would not fix. Run 4 has *half* the shrub training pixels and a much smaller bias, so the cause was not sample size and not features — it was the **composition of the tile set**. The two part-flown tiles were dragging shrub toward bare.

Note that bare's bias flipped sign, +3.58% to −4.49%. Removing unflown RGB (§11.1 change 2) took away black pixels that had been classified as bare.

### Per-class F1, all variants

| variant | run | bare | grass | shrub | tree | macro | overall |
|---|---|---|---|---|---|---|---|
| `RF-A_A` | 3 | 0.883 | 0.800 | 0.383 | 0.850 | 0.729 | 0.766 |
| | **4** | **0.945** | 0.806 | **0.293** | 0.853 | 0.724 | 0.798 |
| `RF-A_B` | 3 | 0.892 | 0.813 | 0.451 | 0.873 | 0.757 | 0.787 |
| | **4** | **0.943** | 0.827 | **0.332** | 0.878 | 0.745 | 0.817 |
| `RF-A_C` | 3 | 0.898 | 0.912 | 0.676 | 0.908 | 0.849 | 0.871 |
| | **4** | **0.947** | 0.916 | **0.636** | 0.914 | **0.853** | 0.899 |
| `RF-A_D` | 3 | 0.924 | 0.939 | 0.798 | 0.936 | 0.899 | 0.912 |
| | **4** | 0.948 | 0.939 | 0.765 | 0.954 | **0.901** | 0.928 |

**Bare gained on every variant** (+0.024 to +0.062) and **shrub lost on every variant** (−0.033 to −0.119). Macro-F1 is flat: C and D rose slightly, A and B fell slightly. Overall accuracy rose everywhere, which is the usual warning that bare and grass are carrying it (§6).

**Shrub F1 fell as §11.1 change 5 predicted.** Training shrub pixels went 2,087 → 1,075, roughly half, because the retired tiles carried 159 reviewed shrub candidates and the new tiles carry 50, with `517000_3531000` at 0 of 150 reviewed. **This is a labelling-coverage effect, not a model regression.**

| class | run 3 px | run 4 px | ratio |
|---|---|---|---|
| bare | 3,601 | 3,361 | 0.93 |
| grass | 4,717 | 4,752 | 1.01 |
| **shrub** | **2,087** | **1,075** | **0.52** |
| tree | 2,158 | 1,959 | 0.91 |

### Shrub errors moved, they did not shrink

`RF-A_C` confusion, row-normalized. Shrub recall is essentially unchanged (0.627 → 0.621), but **where shrub leaks changed completely**:

| shrub is called | run 3 | run 4 |
|---|---|---|
| bare | **0.185** | **0.040** |
| grass | 0.107 | **0.196** |
| tree | 0.081 | **0.142** |

**Shrub→bare collapsed by a factor of four and shrub→grass nearly doubled.** §9.4 identified shrub→bare at 0.185 as the model's defining problem; that problem is gone, and a different one has taken its place. Shrub is now confused with the other *vegetated* classes rather than with soil, which is a more defensible error and consistent with the bias improvement, but it is not a smaller error.

### Fold spread barely moved, and the worst fold is unchanged

`RF-A_C` macro-F1 per held-out tile:

| run | folds | sd | spread |
|---|---|---|---|
| 3 | 0.890 / 0.912 / 0.851 / 0.785 / 0.660 / 0.745 | 0.087 | 0.252 |
| **4** | 0.900 / 0.900 / 0.860 / 0.809 / **0.669** / 0.770 | 0.081 | 0.231 |

**`515000_3526000` remains the worst fold at 0.669**, essentially unchanged from run 3's 0.660. It was never the suspect tile — the two retired tiles scored 0.745 and were mid-pack — so replacing them did not address the real source of instability. **Tile-to-tile instability is still the open problem**, and it now has a specific address: `515000_3526000`, the Q3 train tile.

The new train tile `517000_3531000` contributes a fold of only **374 labelled pixels**, against 1,219–4,010 for the others. So run 4 still has one small fold — this time from thin labelling rather than from flight coverage.

### `luma` no longer ranks first

`RF-A_C` top five:

| run | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| 3 | **luma 0.116** | ExR 0.099 | NDVI 0.096 | std 0.090 | glcm_entropy 0.081 |
| **4** | **NDVI 0.122** | luma 0.096 | ExR 0.083 | glcm_entropy 0.069 | VARI 0.069 |

`NDVI` displaced `luma`, and `luma` fell from 0.116 to 0.096. This **partly relieves the §9.5 / §10 item 6 concern** that `luma` was separating classes via shadow structure: with shadow re-cut and unflown RGB removed, its importance dropped and a genuine vegetation index took the top slot. It does not close the question — `luma` is still rank 2 — but it is no longer the dominant feature.

### What this changes downstream

1. **Re-run Stage 4_1 on run 4.** The fractions currently on disk carry run 3's −14.7% shrub bias; run 4's is −4.7%.
2. **`RF-A_C` remains the operational choice** at macro 0.853, now marginally *better* than run 3 despite half the shrub labels. `RF-A_D` stays circular on shrub (§1.3).
3. **Sweep the shrub candidates on `517000_3531000` (0/150) and `516000_3528000` (50/150).** This is the cheapest available gain: it should recover most of the shrub F1 lost between runs, and it is a labelling task, not a modelling one.
4. **Investigate `515000_3526000`** before Step 6. It is the worst fold in both runs and was untouched by everything run 4 changed.

---

# 12. Run 5 — one variable: the shrub candidate review

§12.1 to §12.3 were written **before** the run, so the change is on record independently of the result. §12.4 holds the results.

```
python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 5 --frameworks C --max-pixels-per-polygon 100
```

> **RUN 5 CHANGES EXACTLY ONE THING. It is the first clean single-variable comparison since run 2.** Tiles, shadow masks, k-means clusters, features, subsampling and code are all identical to run 4. Only the shrub candidate review advanced.

## 12.1 The change

| | run 4 | run 5 |
|---|---|---|
| candidates reviewed | 569 of 1500 (37.9%) | **850 of 1500 (56.7%)** |
| accepted | 242 | **296** |
| shrub polygons, train | 150 | **183** |
| shrub polygons, test | 135 | **156** |

`517000_3531000` is the biggest mover: **0 reviewed → 77**, contributing 28 shrub polygons instead of 2. It was the tile whose empty review was named in §11.1 as the reason shrub fell.

Other classes are untouched — bare 60/53, grass 51/71, tree 65/93, exactly as in run 4.

## 12.2 The question it answers

Run 4's shrub F1 fell to **0.636** from run 3's 0.676. §11.1 change 5 attributed that to shrub training pixels halving, 2,087 → 1,075, when the two retired tiles took their 159 reviewed candidates with them.

**If that attribution was right, shrub F1 recovers here. If it does not move, the cause was the tile swap itself, not the label count.** Either answer is worth having, and this is the only run so far able to give one.

Baseline to beat, `RF-A_C` run 4:

| quantity | run 4 |
|---|---|
| shrub F1 | 0.636 |
| shrub recall / precision | 0.621 / 0.652 |
| macro-F1 | 0.853 |
| shrub area bias | −4.65% |
| fold macro-F1 | 0.900 / 0.900 / 0.860 / 0.809 / **0.669** / 0.770 |

Watch `515000_3526000`, the 0.669 fold. It gained only 4 shrub polygons (3 → 7) because its accept rate is the lowest on the site at 7%, so it is the tile least helped by this run. If the other folds rise and it does not, that isolates it as a tile problem rather than a label problem — which is §11.4's open question.

## 12.3 A code change that is not a behaviour change

The review rule was tightened just before this run: the "swept tile" shortcut was removed, so a candidate is a label **only** where `reviewed = 1` and `rejected != 1`. Unmarked is never accepted.

**This does not affect comparability with runs 1–4.** `run_stage3_1_random_forest_ground_truth_classification.py` always applied the strict rule independently and never implemented the sweep path, so no previous run ever counted an unmarked candidate. The shortcut existed only in the progress checker and in a config key that was never populated. Had it been used, it would have promoted 650 unmarked proposals to labels in one edit — tripling accepted shrub from 296 to 946 — which is why it was removed rather than left available.

## 12.4 Results

**Run 5 completed, `RF-A_C`, six folds. The attribution in §11.1 was correct.** Shrub was the only class whose training data changed (+180 pixels, 1,075 → 1,255) and shrub was the only class whose score moved:

| class | run 3 | run 4 | **run 5** | 4→5 |
|---|---|---|---|---|
| bare | 0.898 | 0.947 | 0.945 | −0.001 |
| grass | 0.912 | 0.916 | 0.913 | −0.003 |
| **shrub** | 0.676 | **0.636** | **0.658** | **+0.021** |
| tree | 0.908 | 0.914 | 0.902 | −0.012 |
| macro | 0.849 | 0.853 | **0.855** | +0.001 |

Bare, grass and tree training pixels were **identical** to run 4 — 3,361 / 4,752 / 1,959 — and their F1 moved by at most 0.012, which is fold noise. **This is the cleanest causal statement the project has produced: shrub labels move shrub score, and nothing else.**

### The area bias is now nearly gone

| class | run 3 | run 4 | **run 5** |
|---|---|---|---|
| bare | +3.58% | −4.49% | −4.70% |
| grass | +1.67% | +1.89% | +1.62% |
| **shrub** | **−14.66%** | **−4.65%** | **−1.99%** |
| tree | +4.54% | +5.67% | +5.41% |

**Shrub area bias fell from −14.66% to −1.99% across runs 3 to 5**, a sevenfold reduction. This is the quantity Stage 4 fractions inherit, and it is now small enough that it is no longer the dominant source of error in the fraction product. Tree at +5.41% is now the largest bias of any class.

The mechanism is visible in the confusion. Shrub recall rose 0.621 → 0.651 while precision moved only 0.652 → 0.664: the model is **finding** more shrub rather than merely being more cautious about it.

| shrub is called | run 3 | run 4 | **run 5** |
|---|---|---|---|
| bare | 0.185 | 0.040 | **0.033** |
| grass | 0.107 | 0.196 | **0.176** |
| shrub | 0.627 | 0.621 | **0.651** |
| tree | 0.081 | 0.142 | 0.140 |

Shrub→grass fell from 0.196 to 0.176, so the error introduced in run 4 is partly reversing. It remains the largest off-diagonal cell and the model's defining problem.

### `515000_3526000` is a tile problem, not a label problem

This was the explicit watch item in §12.2, and the answer is unambiguous:

| held-out tile | run 3 | run 4 | **run 5** | 4→5 |
|---|---|---|---|---|
| 511000_3527000 | 0.890 | 0.900 | 0.895 | −0.005 |
| 511000_3528000 | 0.912 | 0.900 | 0.889 | −0.012 |
| 511000_3529000 | 0.851 | 0.860 | 0.863 | +0.003 |
| **515000_3526000** | 0.660 | 0.669 | **0.668** | **−0.002** |
| **517000_3531000** | — | 0.770 | **0.815** | **+0.045** |
| 519000_3527000 | 0.785 | 0.809 | 0.805 | −0.003 |

**`517000_3531000` gained the most labels and gained the most score** (+0.045, from 0 reviewed candidates to 77). **`515000_3526000` gained 4 shrub polygons and did not move at all** — 0.669 → 0.668, after 0.660 in run 3.

Three runs, three tile sets, three label states, and that fold has sat at 0.66–0.67 throughout.

> **CORRECTED in §13.** I concluded here that "whatever is wrong with `515000_3526000` is not a shortage of labels". **That was too strong and is wrong.** It is a shortage of *shrub* labels specifically, on that tile: it carries **7 shrub labels against 20–52 on every other tile**, and run 5 added only 4 of them, which was never enough to move a fold. §13 has the measurement.

### Feature importance is stable

| run | 1st | 2nd | 3rd |
|---|---|---|---|
| 3 | luma 0.116 | ExR 0.099 | NDVI 0.096 |
| 4 | NDVI 0.122 | luma 0.096 | ExR 0.083 |
| **5** | **NDVI 0.115** | luma 0.096 | ExR 0.077 |

`NDVI` holds first place and `luma` is unchanged at 0.096 across runs 4 and 5. The §9.5 concern that `luma` was separating classes via shadow structure is now settled as far as importance can settle it: it is a stable second-place feature, not a dominant one.

### Where this leaves the product

`RF-A_C` at **macro 0.855, shrub F1 0.658, shrub area bias −1.99%** is the best state the ground truth has been in. Shrub F1 is still below run 3's 0.676, but run 3 reached that with 2,087 shrub training pixels against run 5's 1,255, and with a −14.66% area bias. **Run 5 is more accurate per label and far less biased**, which is what matters downstream.

**Stage 4_1 should be re-run on run 5.** The fractions on disk carry run 4's −4.65%.

---

# 13. Why tree is biased and why one fold is bad — the same answer

Both open questions after run 5 turn out to have one cause. Measured on run 5 `RF-A_C`.

## 13.1 Tree's +5.41% bias is shrub leaking into tree

Tree was the largest remaining area bias. It is not a tree problem:

| true class | pixels called tree | share of all tree predictions |
|---|---|---|
| bare | 0 | 0.0% |
| grass | 74 | 3.6% |
| **shrub** | **176** | **8.5%** |
| tree | 1,815 | 87.9% |

**Bare never becomes tree, and grass almost never does. 70% of tree's over-prediction is shrub** — `shrub → tree` runs at 0.140, so one shrub pixel in seven is called tree.

**This is the `H_TREE_MIN` = 2 m boundary, and `RF-A_C` has no CHM with which to see it.** It must infer a height break from spectra and texture alone. `RF-A_D`, which receives CHM, halves the error: `shrub → tree` falls 0.140 → 0.086 and tree bias falls +5.67% → +2.76% (run 4 figures, the last run where both were fitted).

**Tree bias and shrub bias are one problem seen from two sides**, and the woody continuum is where the remaining error lives. Fixing `shrub → tree` fixes both.

## 13.2 Tree over-prediction explains fold instability almost exactly

Per tile: CHM tree band (ground truth for ≥ 2 m cover) against predicted tree, as percentage points of over-prediction.

| tile | CHM shrub | CHM tree | shrub labels | tree over-prediction | fold macro-F1 |
|---|---|---|---|---|---|
| 511000_3527000 | 11.9% | 11.9% | 49 | **−3.6 pp** | **0.895** |
| 511000_3528000 | 12.1% | 11.7% | 28 | −3.4 pp | 0.889 |
| 511000_3529000 | 11.6% | 11.9% | 51 | −1.0 pp | 0.863 |
| 519000_3527000 | 12.0% | 3.6% | 20 | +5.0 pp | 0.805 |
| 517000_3531000 | 27.3% | 4.5% | 28 | +6.4 pp | 0.815 |
| **515000_3526000** | 17.2% | **13.5%** | **7** | **+18.2 pp** | **0.668** |

**Correlation between tree over-prediction and fold macro-F1 is −0.992 across the six training folds.** One variable accounts for essentially all of the fold spread that has been the headline concern since §9.3.

The three tiles that do **not** over-predict tree are the three `511000_*` tiles, which are also the three highest folds. They share a property: CHM shrub band and CHM tree band are almost equal (about 12% each). Where shrub and tree are balanced, the model separates them. Where shrub dominates, it leaks into tree.

## 13.3 `515000_3526000` is a shrub-label famine, and §12.4 got this wrong

The tile carries **7 shrub labels — 0 hand-drawn and 7 accepted candidates** — against 20 to 52 on every other tile. It simultaneously has the site's **highest CHM tree band at 13.5%** and a substantial shrub band at 17.2%. So it is the tile with the most tall woody vegetation and almost no shrub training data, and when it is held out the model assigns its woody cover to tree: **31.7% predicted tree against 13.5% actual.**

§12.4 read run 5's flat fold score as proof that labels were not the issue. **That inference was wrong**: run 5 added 4 shrub labels to this tile, taking it from 3 to 7. Four labels cannot move a fold, so the flat result carried no information about whether labelling would help.

**Why the usual accelerator fails here.** The CHM shrub-candidate accept rate on this tile is **7%, the lowest on the site** — the candidate generator proposes mostly non-shrub, so reviewing candidates cannot supply the labels. **This tile needs hand-drawn shrub polygons**, which is exactly the work the candidate mechanism was built to avoid, and it is the only tile where that mechanism does not work.

## 13.4 What follows

1. **Hand-draw shrub polygons on `515000_3526000`.** Target 30 to 50, matching the other tiles. It is the single highest-value labelling task remaining, and it is not substitutable by candidate review.
2. **Expect tree bias to fall with it.** If §13.1 and §13.2 are right, shrub labels on the woodiest tile reduce `shrub → tree` and therefore tree's area bias, without touching tree labels at all. That is a testable prediction for run 6.
3. **Do not chase tree directly.** Adding tree labels would address the symptom. The error is shrub being unavailable as an alternative, not tree being poorly defined.
4. **The `RF-A_C` versus `RF-A_D` gap is now explained.** D's advantage is concentrated at the 2 m boundary, which is exactly where CHM carries information that spectra do not. That is a real information advantage on top of the labelling circularity in §1.3, not a substitute for it.

---

# 14. Run 6 — one tile, one class, and a falsifiable prediction

**Status: not yet executed.** Written before the run so the prediction is on record and can be wrong.

```
python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 6 --frameworks C --max-pixels-per-polygon 100
```

> **RUN 6 CHANGES ONE THING: hand-drawn shrub polygons on `515000_3526000`.** Tiles, shadow masks, clusters, features, subsampling and code are identical to run 5. No other tile is touched. No other class gains labels. This is a narrower change than run 5, which moved shrub labels across all ten tiles.

## 14.1 Why this tile, and why by hand

`515000_3526000` carries **7 shrub labels — 0 hand-drawn, 7 accepted candidates** — against 20 to 52 on every other tile, while holding the site's **highest CHM tree band at 13.5%**. Most tall woody vegetation on the site, almost no shrub training data.

Held out, the model gives that woody cover to tree: **31.7% predicted tree against 13.5% actual, a +18.2 pp over-prediction**, and its fold is the worst on the site at 0.668. Across the six folds, tree over-prediction correlates with fold macro-F1 at **−0.992** (§13.2).

**The candidate accelerator cannot fix this tile.** Its CHM shrub-candidate accept rate is **7%, the lowest on the site** — the generator proposes mostly non-shrub there — so reviewing candidates does not yield labels. It is the only tile where the mechanism fails, and the only tile requiring hand-drawn shrub.

**Target 30 to 50 shrub polygons**, matching the other tiles. **As drawn: 14 hand-drawn polygons**, taking the tile from 7 shrub labels to 21. That is a threefold increase and puts it level with `519000_3527000` (20 labels, fold 0.805), but at the bottom of the 20 to 52 band rather than the middle. If the fold moves only partway, the label count may need to reach the 40 to 50 of the strongest tiles rather than merely clearing the famine.

## 14.2 The prediction, stated before the run

Run 5 baseline, `RF-A_C`:

| quantity | run 5 |
|---|---|
| macro-F1 | 0.855 |
| shrub F1 | 0.658 |
| tree F1 | 0.902 |
| shrub area bias | −1.99% |
| **tree area bias** | **+5.41%** |
| shrub → tree | 0.140 |
| shrub → grass | 0.176 |
| `515000_3526000` fold | **0.668** |
| train px | 3,361 / 4,752 / 1,255 / 1,959 |

Three predictions, in increasing order of how much they would tell us:

1. **The `515000_3526000` fold rises** from 0.668 toward the 0.80–0.89 band of the other folds. Weak prediction — more labels on a tile usually help its own fold.
2. **`shrub → tree` falls** from 0.140. Moderate.
3. **Tree area bias falls from +5.41% without a single tree label being added.** **This is the falsifiable one.** §13.1 measured that 70% of tree's over-prediction is shrub leaking into it, and bare contributes literally zero. If that diagnosis is right, supplying shrub where shrub is most abundant should reduce tree's commission as a side effect.

**If tree bias does not move, §13.1 is wrong** and tree needs its own investigation rather than being treated as a shadow of the shrub problem.

## 14.3 What would make this run uninterpretable

- **Labelling any class other than shrub on this tile**, or shrub on any other tile. Either would reintroduce the multi-variable confound that runs 3 and 4 suffered from.
- **Re-running stage 1_6 or 2_1.** Both pool site-wide and would re-cut shadow or renumber clusters across all ten tiles (`instructions5.md` §2A). Neither is needed: adding polygons to an existing GeoPackage changes no pooled statistic.
- Running `run_stage2_4_check_hand_labeling_progress.py --fill` is safe and expected; it only writes derived attributes on the polygons.

## 14.4 Results — the run is VOID as a test, and it found something better

**Run 6 completed and every headline number moved the wrong way**: macro 0.855 → 0.832, tree F1 0.902 → 0.835, tree area bias **+5.41% → +10.78%**, shrub area bias −1.99% → **−6.94%**, `shrub → tree` 0.140 → **0.227**. The target fold barely moved, 0.668 → 0.675, and four of the other five folds fell slightly.

**Do not read this as falsifying §13.1.** The run is void as a test of that prediction, for two independent reasons, and the second one is the finding.

### Void reason 1: the isolation specified in §14.3 was not held

§14 required shrub labels only. Bare, grass and tree labels were also added to `515000_3526000`:

| class | run 5 px | run 6 px | change |
|---|---|---|---|
| bare | 3,361 | 3,361 | +0 |
| grass | 4,752 | 4,788 | +36 (+0.8%) |
| **shrub** | **1,255** | **1,685** | **+430 (+34.3%)** |
| tree | 1,959 | 1,994 | +35 (+1.8%) |

The prediction was specifically that **tree bias would fall without a single tree label being added**. Tree labels were added. They are few — 35 pixels, 1.8% — but they sit on the tile with the site's highest tree-height cover, which is exactly where the shrub/tree boundary is contested, so their influence is not proportional to their count.

### Void reason 2: the new shrub labels contradict the class definition

**45.7% of the hand-drawn shrub pixels on `515000_3526000` sit on CHM ≥ 2 m** — tree, by §3's own definition — with a median CHM of 1.44 m. Every other tile:

| tile | shrub polygons | shrub px on CHM ≥ 2 m |
|---|---|---|
| 511000_3527000 | 12 | 5.9% |
| 511000_3528000 | 2 | 0.0% |
| 511000_3529000 | 3 | 0.0% |
| 515000_3530000 | 5 | 8.4% |
| 519000_3527000 | 4 | 4.8% |
| 517000_3531000 | 2 | 7.3% |
| 516000_3528000 | 15 | 10.3% |
| **515000_3526000** | **22** | **45.7%** |

Site-wide, 19.2% of hand-drawn shrub pixels sit above 2 m, and **this one tile supplies 282 of those pixels on its own.** Its shrub labels are four to eight times more likely to fall on tree-height canopy than any other tile's.

**That fully accounts for the damage.** Teaching the classifier that ≥ 2 m canopy is shrub, on the tile with the most ≥ 2 m canopy, must blur the shrub/tree boundary — which is precisely what `shrub → tree` rising from 0.140 to 0.227 records. The classifier did not get worse; it learned what it was shown.

### The real finding: the analyst's shrub and the CHM's shrub disagree on this tile

This also re-reads §13.3. That section explained the tile's 7% shrub-candidate accept rate as "the candidate generator proposes mostly non-shrub". **The opposite is at least as likely**: on this tile the CHM proposes 0.7–2 m objects as shrub and the analyst rejects them, then draws shrub on 2 m+ canopy instead. Two definitions of shrub, disagreeing systematically, on one tile.

That is not a labelling error to be tidied away. `515000_3526000` has the site's highest tree-band cover at 13.5%, and if its mesquite is genuinely multi-stemmed woody vegetation that reads as shrub by eye while exceeding 2 m in height, then **`H_TREE_MIN` = 2 m is the thing that is wrong**, not the labels.

**This requires a decision, and it is not mine to make:**

- **(a) The 2 m boundary stands.** The 22 shrub polygons on this tile need redrawing or reclassifying to respect it, and the tile's woody cover is genuinely tree.
- **(b) The 2 m boundary is wrong for SRER mesquite.** `H_TREE_MIN` is revisited against the CHM distribution and the imagery, which changes the class definition site-wide and invalidates every run to date.

Option (b) is expensive but should not be dismissed: `H_GRASS_MAX` was already found to be inert at 0.3 m and corrected to 0.7 m by measurement (§3 safeguard 2), so a height threshold in this project has been wrong before.

### What run 6 is good for

Nothing as a test of §13.1, which remains **untested**. It is a clean demonstration that **label-definition consistency matters more than label count**: 430 new shrub pixels, a 34% increase, made every shrub and tree metric worse because nearly half of them were mislabelled by the project's own rule.

**Run 6 should not be used as a baseline.** Run 5 remains the reference until the boundary question is settled.

---

# 15. `H_TREE_MIN` = 2 m is correct, and the tile is wrong

The §14.4 decision — whether the 2 m shrub/tree boundary is wrong for SRER mesquite, or whether `515000_3526000`'s labels are wrong — is settled by measurement.

## 15.1 Method

Take every hand-drawn shrub and tree polygon, read the CHM underneath, and ask which height threshold best separates the analyst's own two classes. If the analyst's visual notion of shrub and tree separates cleanly at some height, that height is the empirical boundary, whatever the config says.

CHM = 0 pixels are excluded from the threshold question — the NEON CHM is pre-thresholded at 0.7 m, so a zero means the object is invisible to lidar, not that it is flat ground (§3 safeguard 2). That removes 38.5% of shrub pixels and 5.0% of tree pixels, which is itself a reminder that CHM under-sees shrub.

## 15.2 The answer: 2.00 m, exactly

| threshold | shrub correctly < th | tree correctly ≥ th | Youden J |
|---|---|---|---|
| 1.5 m | 36.1% | 92.8% | 0.290 |
| **2.0 m** | **68.8%** | **79.8%** | **0.487** |
| 2.5 m | 77.8% | 62.6% | 0.404 |
| 3.0 m | 86.6% | 44.1% | 0.307 |
| 3.5 m | 94.3% | 26.7% | 0.209 |

**The optimum is 2.00 m**, the configured value, and J falls away on both sides. Scanned at 0.05 m steps from 0.8 to 5.0 m, no other threshold does better.

**`H_TREE_MIN` needs no change.** Option (b) in §14.4 is closed.

## 15.3 `515000_3526000` does not separate shrub from tree at any height

Per tile, on CHM-visible pixels:

| tile | shrub px | tree px | shrub median | tree median | best threshold | J |
|---|---|---|---|---|---|---|
| 511000_3529000 | 24 | 903 | 1.36 | 3.56 | 1.95 | **0.925** |
| 511000_3527000 | 40 | 578 | 1.32 | 3.07 | 2.65 | 0.651 |
| 515000_3530000 | 485 | 319 | 1.55 | 2.19 | 1.90 | 0.572 |
| 519000_3527000 | 20 | 435 | 1.47 | 2.51 | 2.40 | 0.547 |
| 517000_3531000 | 28 | 255 | 1.68 | 2.07 | 2.10 | 0.494 |
| 516000_3528000 | 38 | 688 | 2.04 | 2.48 | 1.45 | 0.296 |
| **515000_3526000** | **265** | **358** | **2.86** | **2.94** | 1.50 | **0.145** |

**On `515000_3526000` the drawn shrub median is 2.86 m and the drawn tree median is 2.94 m — a difference of 8 cm.** The two classes are the same height distribution. No threshold separates them, which is what J = 0.145 means: the labels carry essentially no height information.

Every other tile puts shrub between 1.32 and 2.04 m and tree between 2.07 and 3.56 m.

## 15.4 Removing that one tile sharpens the site-wide boundary

| | shrub px | tree px | best threshold | J at optimum | shrub < 2 m |
|---|---|---|---|---|---|
| all ten tiles | 905 | 5,016 | **2.00 m** | 0.487 | 68.8% |
| **excluding `515000_3526000`** | 640 | 4,658 | **2.00 m** | **0.670** | **87.8%** |

The optimum stays at exactly 2.00 m either way — the tile does not shift the boundary, it just blurs it. Excluding it, **87.8% of drawn shrub falls below 2 m** against 68.8% with it, and J rises from 0.487 to 0.670.

**One tile out of ten is responsible for a third of the site's shrub/tree height confusion.**

## 15.5 Conclusion and action

**§14.4 option (a) holds: the 2 m boundary stands and the labels on `515000_3526000` do not respect it.** The evidence is that the boundary is optimal both with and without the tile, and that no other tile shows this pattern.

1. **Redraw or reclassify the 22 shrub polygons on `515000_3526000`.** Those sitting on CHM ≥ 2 m are tree by the project's definition. 45.7% of their pixels are.
2. **`516000_3528000` is worth a look too** — J 0.296, shrub median 2.04 m. Milder, same direction.
3. **Then re-run as run 7**, with shrub labels only and no other class touched, which is what run 6 was supposed to be.
4. **Add this as a stage 2 gate check.** A per-tile Youden J between drawn shrub and drawn tree against CHM would have caught this at labelling time rather than three runs later. Any tile below about J = 0.3 is labelling shrub and tree as the same thing.

## 15.6 A caveat that survives all of this

At the optimum, 20% of drawn tree still sits below 2 m and 12% of drawn shrub above it, even excluding the bad tile. **The shrub/tree boundary is genuinely fuzzy at SRER** — mesquite spans it — so a hard height rule will always misfile some objects. That is a known cost of the §3 definition, not a defect introduced here, and it sets a ceiling on how well any classifier can separate the two.

---

# 16. Run 8 — the bias prediction holds, the mechanism does not

Six shrub polygons on `515000_3526000` reclassified to tree (§15.5). No geometry drawn, moved or deleted; only `class_code` changed on the six whose median CHM was at or above 2.0 m. Selection was by measurement, not row index, and the write went through a staged file with a round-trip check.

**The tile is now well-labelled by every height measure**: Youden J 0.165 → **0.717**, drawn-shrub median 2.85 → **1.29 m**, shrub pixels on CHM ≥ 2 m 40.9% → **5.3%**. Site-wide J rose 0.487 → **0.675** with the optimum still exactly 2.00 m. It went from the worst tile to the second best.

## 16.1 What happened

| | run 5 | run 7 | **run 8** |
|---|---|---|---|
| shrub train px | 1,255 | 1,738 | 1,515 |
| tree train px | 1,959 | 1,994 | 2,217 |
| macro-F1 | **0.855** | 0.833 | 0.828 |
| shrub F1 | **0.658** | 0.642 | **0.606** |
| tree F1 | **0.902** | 0.829 | 0.845 |
| **shrub area bias** | −1.99% | −7.08% | **−1.06%** |
| **tree area bias** | +5.41% | +10.53% | **+5.68%** |
| shrub → tree | **0.140** | 0.226 | **0.226** |
| tree → shrub | **0.054** | 0.112 | 0.117 |

**Both area biases recovered completely.** Shrub returned to −1.06%, better than run 5. Tree returned to +5.68% from +10.53%, matching run 5's +5.41%. The §15 diagnosis that six mislabelled polygons were driving the damage is confirmed on that measure.

## 16.2 But `shrub → tree` did not move at all, and that matters

**0.226 in run 7, 0.226 in run 8.** Identical to three decimals, while tree bias halved.

§13.1 claimed tree's over-prediction *is* shrub leaking into tree, and §14.2 predicted that fixing the shrub/tree labels would reduce both together. **They did not move together.** Bias fell; the confusion that was supposed to cause it did not.

**The explanation is that the bias improvement is largely definitional, not behavioural.** Area bias is computed from the confusion matrix over the labelled sample. Moving six polygons from the shrub row to the tree row changes the row totals directly — true tree rose from 1,994 to 2,217 pixels — so the ratio of predicted to true tree improves whether or not the classifier changed its mind about anything. The classifier is still calling 22.6% of shrub tree.

**§13.1 is therefore still unconfirmed.** It survives as a hypothesis, but run 8 does not support it: the mechanism it named did not respond.

## 16.3 The labelling campaign has been net negative against run 5

| | run 5 | run 8 |
|---|---|---|
| macro-F1 | **0.855** | 0.828 |
| shrub F1 | **0.658** | 0.606 |
| tree F1 | **0.902** | 0.845 |
| `515000_3526000` fold | 0.668 | **0.646** |
| fold sd | 0.077 | 0.083 |

Runs 6, 7 and 8 added labels to `515000_3526000` and every discrimination metric is worse than run 5, including the target tile's own fold, which is now the worst it has ever been. Fold spread widened rather than narrowed.

Run 8 still carries **+260 shrub and +258 tree pixels** from that tile relative to run 5, and those remaining labels are height-consistent — median 1.29 m, only 5.3% above 2 m. **So the harm is not explained by the height rule any more.** Something else about this tile's labels is hurting the model, and height was only part of it.

**Run 5 remains the best state of the ground truth and the reference for everything downstream.**

## 16.4 What this actually established

1. **`H_TREE_MIN` = 2.0 m is validated** and the height-consistency check is worth having as a gate (§15).
2. **Label consistency dominates label count** — 483 extra shrub pixels in run 7 bought nothing because 41% contradicted the definition.
3. **Area bias can improve for definitional reasons.** It must be read alongside a confusion-based measure such as `shrub → tree`, or a relabelling will look like a model improvement.
4. **§13.1 is not confirmed.** Tree over-prediction and shrub-to-tree confusion did not move together when the labels were corrected.

## 16.5 Next

- **Revert to run 5 for any downstream work.** Stage 4_1 should stay on run 5 until this is resolved.
- **Review `515000_3526000`'s labels visually in QGIS against the 10 cm RGB**, not against the CHM. Height is now consistent; appearance may not be. The tile has the site's highest tree-band cover, so it may simply be poor ground on which to draw shrub at all.
- **Consider excluding the tile's shrub labels entirely** as a test: a run with run 5's label set plus only the six tree reclassifications would isolate whether the new shrub polygons help or hurt.
- **One polygon on this tile carries no `class_code`** and is silently ignored by every stage. Worth a check across all tiles.
