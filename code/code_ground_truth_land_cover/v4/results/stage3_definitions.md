# Stage 3_1 — metric definitions and how to read them

Companion to `stage2_results.md`. Every quantity reported there is defined here, with what a good value looks like **in this project specifically** and the trap attached to each.

> **All Step 1d metrics are DIAGNOSTIC.** They come from cross-validation over training polygons and describe how well the model fits labels the analyst chose. Map accuracy is a different quantity, produced only by the Olofsson area-weighted protocol on an independent probability sample (`instructions5.md` §6). §7 below sets out exactly how the two differ and why one cannot substitute for the other.

---

## 1. The confusion matrix — everything else derives from it

For one class *c*, every prediction falls into one of four boxes:

| | predicted *c* | predicted not *c* |
|---|---|---|
| **truly *c*** | TP — true positive | FN — false negative (**omission**) |
| **truly not *c*** | FP — false positive (**commission**) | TN — true negative |

In the reported matrix, **rows are truth and columns are prediction**. Reading it:

- **Along a row** — where the truth went. Row *shrub*, column *grass* = shrub pixels the model called grass. This is **omission** of shrub.
- **Down a column** — where a prediction came from. Column *shrub*, row *grass* = grass pixels wrongly called shrub. This is **commission** into shrub.
- **The diagonal** is correct predictions.

Reported **row-normalized**, so each row sums to 1 and the diagonal reads directly as recall. Raw counts are printed inside each cell as well, because a rate over small support is easy to over-read — 0.69 over 844 pixels and 0.69 over 30 pixels look identical otherwise.

**What to look for**: not the diagonal, but the largest **off-diagonal** cell. That single number names the model's actual problem. Here it is the grass↔shrub pair, in both directions, in every framework.

---

## 2. Support

**Definition**: the number of true pixels of a class in the validation set — the row total in the confusion matrix.

**Why it comes first**: every other per-class metric is computed over this many pixels, so support sets how much any of them can be trusted. A recall of 0.69 over 844 pixels means something; over 20 pixels it is noise.

**What to look for**: severe imbalance. In this project bare has 8,200 training pixels and shrub 844 — roughly 10:1. That imbalance is the single most important fact about every score on the page, and it is why `class_weight="balanced"` is set and why macro-F1 rather than overall accuracy is the headline.

**The trap**: support in *polygons* and support in *pixels* disagree sharply here. Shrub has the most polygons (125) and the fewest pixels (918), because shrub polygons are ~5 m² and bare polygons ~90 m². The labeling gate counts polygons; the model trains on pixels. Never read one as a proxy for the other.

---

## 3. Recall — did the model find the class?

```
recall = TP / (TP + FN) = TP / (all truly this class)
```

The share of genuinely-*c* pixels the model labelled *c*. Also called **producer's accuracy**, and its complement is **omission error**.

**Range** 0 to 1. **What to look for**: low recall means the class is being *missed*. `RF-A_A` shrub recall 0.454 means over half of all real shrub was called something else — mostly grass.

**The trap**: recall alone is trivially gamed. A model that predicts *shrub* everywhere scores shrub recall 1.0. Recall is only meaningful read beside precision.

---

## 4. Precision — was the model right when it said so?

```
precision = TP / (TP + FP) = TP / (all predicted this class)
```

The share of pixels predicted *c* that really are *c*. Also called **user's accuracy**, and its complement is **commission error**.

**Range** 0 to 1. **What to look for**: low precision means the class is **over-predicted** — the map shows more of it than exists. `RF-A_A` shrub precision 0.371 means nearly two-thirds of everything the map called shrub was not shrub.

**The trap**: precision is also gamed trivially, in the opposite direction. A model that predicts *shrub* for one very obvious pixel and never again scores precision 1.0.

**The pairing that matters**: recall and precision fail in opposite directions, so read them together.

| recall | precision | reading |
|---|---|---|
| low | high | class is missed, but trustworthy when reported |
| high | low | class is over-predicted, unreliable when reported |
| low | low | class is not being learned at all |
| high | high | the class works |

---

## 5. F1 — the single number per class

```
F1 = 2 · (precision · recall) / (precision + recall)
```

The **harmonic** mean of precision and recall, not the arithmetic mean. The harmonic mean is used deliberately: it is dragged down by the *smaller* of the two, so a class cannot buy a good F1 by being excellent on one and poor on the other. Precision 1.0 with recall 0.1 gives F1 = 0.18, not 0.55.

**Range** 0 to 1. **What to look for**: F1 is the right per-class summary for framework selection, and the right thing to track across the `RF-A_*` progression.

**The trap**: F1 ignores true negatives entirely, so it says nothing about how well the model avoids a class. That is the correct behaviour here — with bare at 55% of pixels, any metric counting true negatives would be swamped by it.

---

## 6. Macro-F1 and overall accuracy — and why they disagree

**Overall accuracy** = correct predictions ÷ all predictions. **Every pixel counts once**, so abundant classes dominate.

**Macro-F1** = the unweighted mean of the four per-class F1 values. **Every class counts once**, regardless of size.

They disagree systematically in this project, and the gap is informative rather than a nuisance:

| framework | macro-F1 | overall | gap |
|---|---|---|---|
| `RF-A_A` | 0.788 | 0.913 | 0.125 |
| `RF-A_B` | 0.808 | 0.921 | 0.113 |
| `RF-A_C` | 0.884 | 0.955 | 0.071 |

Overall accuracy is inflated because bare is 55% of pixels and is essentially solved (F1 0.99). It can stay high while shrub — the class the project exists to resolve — fails completely.

**Judge on macro-F1 and on per-class F1. Never report overall accuracy alone** (`instructions5.md` §6.4). Note the gap narrowing as the weak classes improve: the gap is itself a rough read on how unevenly the model performs.

**The trap**: macro-F1 gives a 844-pixel class the same vote as an 8,200-pixel class. That is intended here, but it also means macro-F1 is *noisier* than overall accuracy, and moves a lot when a small class moves a little.

---

## 7. Why none of this is map accuracy

This is the most consequential distinction on the page.

| | Step 1d metrics | Step 6 accuracy (§6) |
|---|---|---|
| Sample | training polygons, chosen by the analyst as clean examples | independent probability sample with known inclusion probabilities |
| Weighting | none — every pixel counts once | **area-weighted** by stratum (`p̂_ij = W_i · n_ij / n_i·`) |
| Estimates | how well the model fits its labels | user's / producer's accuracy and **error-adjusted area**, with 95% intervals |
| Bias | **optimistic** — ambiguous pixels are under-represented by construction | unbiased under the sampling design |
| Reportable as map accuracy | **no** | yes |

Training polygons are drawn where the analyst was *confident*. Mixed, marginal and ambiguous pixels — the ones a map gets wrong — are systematically absent. Every Step 1d number is therefore an upper bound, shrub most of all, where labels came from CHM candidates that are structurally the *detectable* shrubs.

"Recall" and "precision" here are the same formulas as producer's and user's accuracy in §6, but computed on an unweighted convenience sample. **Same arithmetic, different quantity.** Do not carry a number across.

---

## 8. Leave-one-tile-out folds

Train on all train tiles but one, predict the held-out tile, repeat. Reported per fold and pooled across folds.

**Why by tile rather than randomly**: labelled pixels inside one polygon are spatially autocorrelated and nearly identical. A random split would put neighbouring pixels from the same polygon in both train and test, and the model would score its own training data. Holding out a whole tile forces genuine spatial generalization.

**What to look for**: **spread across folds**, not the mean. Tight spread means the estimate is stable. `RF-A_C` runs 0.910 / 0.909 / 0.852 — the third fold is materially weaker, driven by shrub precision collapsing to 0.425, which says shrub is the least stable class even at its improved score.

**The trap**: three folds is a small number to judge spread from, and the tiles are not interchangeable — they differ in composition. A weak fold may mean an unusual tile rather than an unstable model.

---

## 9. Feature importance (Gini / MDI)

Random-forest **mean decrease in impurity**: how much each feature reduced node impurity across all trees, normalized to sum to 1.

**What to look for**: which *groups* rise, not the exact ordering. The informative result in this project is that adding texture pushed three texture measures into the top seven and displaced colour features — that is the evidence texture carries the grass/shrub separation.

**Three traps, all live here:**

1. **MDI is biased toward continuous and high-cardinality features.** All features here are continuous, so the bias is roughly uniform, but it means small differences in rank are not meaningful.
2. **Correlated features split their importance.** `r`, `g`, `b`, `ExG`, `VARI` and `GLI` are all functions of the same three bands, so their individual importances are diluted while a less-correlated feature such as `NDVI` looks relatively stronger. Importance measures *this model's* use of a feature, not the feature's intrinsic worth.
3. **Importance says nothing about direction or correctness.** `luma` ranks first in `RF-A_A` — but luma is also what shadow detection keys on, so it may be separating tree from grass via shadow structure rather than vegetation. High importance is a reason to investigate, not a validation.

---

## 10. `prediction_quality` — normalized Shannon entropy

```
prediction_quality = 1 − H(p) / log(K),   H(p) = −Σ p_c · log p_c,   K = 4
```

Per-pixel, from the class probability vector (`instructions5.md` §3.1).

**Range** 0 to 1. **1.0** = all probability on one class (pure). **0.0** = perfectly even four-way split (0.25 each), the maximum-confusion case.

**Why it exists**: the hard label is `argmax` no matter how weak the winner is. A pixel at 0.33/0.31/0.20/0.16 gets a confident-looking class code. Without this layer, that weakness is discarded.

**What to look for**: the *median per tile*, and its spatial pattern. Low quality should concentrate at class boundaries and in genuinely mixed ground. If it is scattered uniformly, the model is uncertain everywhere and something is wrong.

**Downstream**: this is the `prediction_quality` that ranks end members (Step 4), weights RF-B training (Step 5), and stratifies Step 6 sampling. It is **not** `data_quality`, which describes the PlanetScope observations — the two are never merged (§5.3).

---

## 11. `margin` — top-1 minus top-2 probability

```
margin = p(1) − p(2)
```

The gap between the winning class and the runner-up.

**Range** 0 to 1. **What to look for**: margin and entropy disagree informatively. A pixel split 50/50 between grass and shrub with nothing on bare or tree has *moderate* entropy but **near-zero margin**. Entropy says "somewhat mixed"; margin says "this is a coin flip between two specific classes."

Margin is therefore the layer for finding **which pairs** are confusable, which is exactly the grass/shrub question. Entropy is the layer for how mixed a pixel is overall.

---

## 12. `class_weight="balanced"`

Sets each class's weight inversely proportional to its frequency, so the 844 shrub pixels carry the same total influence as the 8,200 bare pixels.

**Why**: without it the forest minimizes total error, and with bare at 55% of pixels the cheapest way to do that is to under-call shrub.

**What to look for**: balancing raises recall on rare classes and usually *lowers* precision on them — the model becomes readier to guess the rare class. Judge on F1, which prices both.

**The trap**: weighting redistributes attention; it does not create information. 844 pixels is 844 pixels however they are weighted, and the remaining shrub gap is a data problem, not a weighting problem.

---

## 13. Label–feature circularity (leakage) — and how to measure it

The most dangerous failure on this page, because it makes a model look **better** rather than worse.

**Definition**: a feature is circular when it shares an origin with the labels it is being scored against. The model then partly re-derives the labelling rule instead of learning the underlying phenomenon, and every metric computed against those labels is inflated by an unknown amount.

**Where it occurs here**: all 196 shrub labels came from CHM-derived candidates (§1.3) — components of the band `H_GRASS_MAX <= CHM < H_TREE_MIN`, accepted by the analyst. `RF-A_D` and `RF-A_E` receive **CHM as a feature**. So their shrub score partly measures *can a model with CHM reproduce a CHM threshold*, which is close to tautological.

Classes labelled by hand — bare, grass, tree — are **not** circular, because no CHM rule generated them.

### Detecting it: compare the prediction against the labelling rule directly

Do not reason about circularity — measure it. Rasterize the rule that generated the labels, and compute IoU against the model's prediction for that class.

Measured for shrub against the CHM band:

| tile | `RF-A_C` (no CHM) | `RF-A_D` (CHM) |
|---|---|---|
| 511000_3527000 | 0.157 | **0.407** |
| 511000_3528000 | 0.181 | **0.530** |
| 511000_3529000 | 0.172 | **0.508** |
| 515000_3530000 | 0.423 | **0.572** |
| 515000_3531000 | 0.417 | **0.578** |

Agreement with the labelling rule roughly **triples on the train tiles** the moment CHM enters the feature set. `RF-A_D` is not merely using height as one signal among twenty — its shrub map converges on the threshold that produced its own training labels.

### Reading a circular metric

**A circular score is not fake, it is unquantifiable.** The CHM band *is* the reference definition of shrub (`instructions5.md` §3), so reproducing it is reproducing the reference. What cannot be recovered is how much of the score reflects genuine learning versus rule re-derivation — and that share is unknown, not small.

Three rules:

1. **Never compare a circular score like-for-like with a non-circular one.** `RF-A_D` shrub 0.768 and `RF-A_C` shrub 0.658 are not the same kind of number.
2. **Use the non-circular classes as a control.** Bare, grass and tree are hand-labelled, so their change measures what the feature genuinely buys. From `RF-A_C` to `RF-A_D`: tree **+0.031**, grass **+0.011**, bare **+0.002** — against shrub **+0.110**. Height legitimately helps identify trees; the shrub gain is several times larger than any clean class's, which is the signature.
3. **Ask whether the circular feature survives transfer.** CHM is unavailable or offset at the AmeriFlux sites (§1.4, `instructions5.md` R1), so any `RF-A_D` shrub advantage does not travel. A score that cannot transfer is not a reason to select a framework whose purpose is transfer.

### Why the design already anticipates this

`instructions5.md` §4.1 frames `RF-A_D`/`RF-A_E` as the **reference/labeller**, not as transfer candidates, precisely because they carry CHM. Framework selection happens among the transferable variants (`RF-A_A` through `RF-A_C`). The circularity measured above is the empirical justification for that split, not a surprise.

**The generalizable habit**: whenever a label source and a feature share an origin, rasterize the label rule and measure agreement against the prediction. It converts a hand-wave into a number.

---

## 14. Reading order

1. **Support** — how much data is behind each class. Everything else is conditioned on it.
2. **Per-class F1** — which classes work.
3. **Recall vs precision** on any weak class — is it missed, or over-predicted?
4. **Largest off-diagonal cell** — what it is being confused *with*.
5. **Fold spread** — is the estimate stable?
6. **Macro-F1** — the single comparable number across `RF-A_*` variants.
7. **Overall accuracy** — last, and only for context.
8. **Feature importance** — to explain a difference already observed, never to establish one.
9. **Circularity check (§13)** — before accepting any gain on a class whose labels share an origin with a feature.
