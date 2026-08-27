# Stage 3_1 — RF-A per-pixel classification results

Produced by `run_stage3_1_random_forest_ground_truth_classification.py`, which implements **`instructions5.md` §5 Step 1d**. The text refers to the spec step throughout — see `instructions5.md` §0.1 for why the spec's step numbers and the scripts' stage numbers differ.

**Site**: SRER 2022 · **Seed**: 6 + 2022 = 2028 · **Runs**: 1 (baseline), 2 (polygon subsampling), `3_smoke` (smoke test), 3 (10 tiles), **4 (pending — see §11)** · **Variants**: RF-A_A … RF-A_D

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

**Definitions**: every metric on this page — recall, precision, F1, support, macro-F1, folds, feature importance, `prediction_quality`, `margin` — is defined in [`stage3_1_definitions.md`](stage3_1_definitions.md), with what to look for and the trap attached to each.

**Scripts**, all taking `--run N`:

```
python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 2
python run_stage3_2_generate_ground_truth_classification_plots.py    config/srer_2022.json --run 2
python run_stage3_3_create_qgis_results_project.py                         --run 2     # or set RUN in the QGIS console
```

A completed run is **frozen**: `run_stage3_1_random_forest_ground_truth_classification.py` refuses to write into a run directory that already holds a report, and names the next free run instead. `--force` overrides, deliberately.

## Runs

| Run | Tiles | Change | Subsampling | Train pixels (bare/grass/shrub/tree) |
|---|---|---|---|---|
| **1** | 5 | Baseline | off | 8,200 / 4,102 / 844 / 1,190 |
| **2** | 5 | Polygon subsampling | max 100 px/polygon | 2,159 / 2,824 / 844 / 1,190 |
| **`3_smoke`** | **10** | run 3 smoke test — quintile-stratified tiles, advanced shrub review | max 100 px/polygon | 2,671 / 3,218 / **1,531** / 1,530 |
| **3** | **10** | the finished run — gate passed | max 100 px/polygon | **3,601 / 4,717 / 2,087 / 2,158** |
| **4** | **10** | **PENDING** — two part-flown tiles replaced, RGB unflown ground masked, shadow and clusters refit (§10) | max 100 px/polygon | *not yet run* |

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

All 196 shrub labels came from CHM-derived candidates (§1.3), and `CHM` is this model's top feature at 0.193. So its shrub score partly measures *can a model with CHM reproduce a CHM threshold*. The test is to rasterize the labelling rule and measure agreement with the prediction (`stage3_1_definitions.md` §13):

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

# 11. Run 4 — what changes from run 3, recorded before it runs

**Status: not yet executed.** This section is written *before* the run so the changes are on record independently of whatever the scores turn out to be. Fill in results below once it completes.

```
python run_stage3_1_random_forest_ground_truth_classification.py config/srer_2022.json --run 4 --frameworks A B C D
```

> **RUN 4 CHANGES SIX THINGS AT ONCE. It is not a clean comparison against run 3, and no single score movement can be attributed to a single cause.** Three of the six are corrections to defects that were present in run 3, so run 3's numbers are not a clean baseline either — they were computed over partly-unusable ground. Run 4 replaces run 3 as the reference; do not average or interpolate between them.

## 11.1 The six changes

**1. Two part-flown tiles replaced** (`instructions5.md` §2A, `stage4_1_results.md` §2.2).

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

*To be filled once run 4 completes.* Record: per-class F1 for all four variants, fold spread, confusion for `RF-A_C`, feature importance top five, area bias per class, and whether `luma` still ranks first (§9.5, §10 item 6).
