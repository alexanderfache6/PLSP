# Retired scripts

Code that was written, run, and superseded. **Kept, not deleted** — each of these produced results that are cited elsewhere in the project, and a reader who finds those results needs to be able to find the code that made them.

Nothing here is on the execution path. `instructions5.md` §0.1 lists what is.

---

## `run_stage1_7_generate_segments.py`

**Retired 2026-08-18.** SLIC segmentation at 1 m plus per-segment spectral, texture, shape and context features. Was stage 1_7.

**Why it was retired: nothing consumed its output.** No script reads `segment_features_*.npz`; `run_stage3_1_random_forest_ground_truth_classification.py` does not define a segments directory at all. It had become a dormant script producing an unread 447 MB.

**The decision behind that**, recorded in `results/stage3_results.md` §1.2: `instructions5.md` §5 Step 1d originally specified RF over SLIC segments, and that turned out to be unworkable here. SLIC segments are a uniform 9 px while the median accepted shrub polygon is 5 px — 0.56 of one segment. At the ≥70% coverage rule, shrub yielded **22 training segments against bare's 738**, a 34:1 imbalance that would have made shrub effectively unpredictable. It was the v3 failure in mirror image (`instructions2.md` §4.5: a starved training set produced 93–97% tree cover). Training moved to per-pixel, which gives shrub 918 samples instead of 22.

**A second reason, specific to this script's metadata.** It wrote a `framework_features` block naming a `D` and an `E`:

- `D` = per-segment means of all 20 bands
- `E` = `D` plus distribution (std, skew, min, max, median), shape, and context features

**Those letters are not the `RF-A_*` framework letters**, and the collision was actively misleading. `RF-A_E` was retired for an unrelated reason — it resolved to a feature set identical to `RF-A_D` (`instructions5.md` §4.1) — whereas this script's `E` genuinely added features. Two different meanings of "E", one of them retired, in the same project.

**Its config key `stage1_7_generate_segments` is retained** in `config/srer_2022.json`, annotated as retired. It records the parameters the existing segment files on disk were built with, which is the only provenance those files have.

**The 447 MB of segment outputs in `stage1_data_and_features/segments/` were left in place** — retiring the script is not a reason to delete data. Delete them deliberately, or keep them if a segment-level comparison is ever revisited.

---

## `run_stage5_1_fit_phenology_fractional_cover_timing_only.py`

**Retired 2026-09-15.** The original stage 5_1: RF-B fitted on 10 TIMING-ONLY features, the 7 date metrics plus DurGU, DurGD and LOS. Was the stage 5 model of record from its first run until the consolidation.

**Why it was retired: timing alone does not predict fractional cover.** At SRER 2022 the joint model scored MAE 0.1845 cross-validated and 0.1925 on held-out tiles, against a constant-mean baseline of 0.1919. That is +3.9 per cent and -0.3 per cent skill: on held-out tiles it was indistinguishable from predicting the site mean. Every cross-validated R squared sat between -0.06 and +0.06. Recorded in `results/stage5_results.md` section 9.

**It is kept because it is the control.** Sections 9 and 11 quote its numbers as the baseline that the greening layers had to beat, and its outputs are still on disk in `stage5_phenology_model_prediction/run5/`.

## `run_stage5_1b_fit_phenology_evi_fractional_cover.py`

**Retired 2026-09-15.** A controlled variant of stage 5_1 that added the three EVI2 greening layers, EVImax, EVIamp and EVIarea, for 13 features, and changed nothing else: same QA mask, same stage 4 targets, same folds, same models, same 7,803,438 pixels.

**Why it was retired: it won, so its feature set became the only one.** Adding greening moved the joint model from -0.3 per cent to +8.4 per cent skill on held-out tiles, turned every test-tile R squared positive, and cut pure-block test MAE from 0.3380 to 0.1375. Recorded in `results/stage5_results.md` section 11.

**Its code now IS stage 5_1.** The consolidated script was copied from this file, so nothing was rewritten from scratch: only the feature set name, the output label and the documentation changed. The first 10 feature columns still match the timing-only script bit for bit, which is what made the comparison in section 11 fair and is asserted again after the consolidation.

**Its outputs are in `stage5_phenology_model_prediction/run5_timing_evi/`** and are left in place. The consolidated 5_1 refuses to write into a run directory whose report names a different feature set, so neither that directory nor `run5` can be overwritten by accident.
