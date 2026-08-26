# Retired scripts

Code that was written, run, and superseded. **Kept, not deleted** — each of these produced results that are cited elsewhere in the project, and a reader who finds those results needs to be able to find the code that made them.

Nothing here is on the execution path. `instructions5.md` §0.1 lists what is.

To bring one back: move it up to `v4/`, restore its stage number in the execution order, and check its config key still matches the current schema.

---

## `run_stage1_7_generate_segments.py`

**Retired 2026-08-18.** SLIC segmentation at 1 m plus per-segment spectral, texture, shape and context features. Was stage 1_7.

**Why it was retired: nothing consumed its output.** No script reads `segment_features_*.npz`; `run_stage3_1_random_forest_ground_truth_classification.py` does not define a segments directory at all. It had become a dormant script producing an unread 447 MB.

**The decision behind that**, recorded in `results/stage3_1_results.md` §1.2: `instructions5.md` §5 Step 1d originally specified RF over SLIC segments, and that turned out to be unworkable here. SLIC segments are a uniform 9 px while the median accepted shrub polygon is 5 px — 0.56 of one segment. At the ≥70% coverage rule, shrub yielded **22 training segments against bare's 738**, a 34:1 imbalance that would have made shrub effectively unpredictable. It was the v3 failure in mirror image (`instructions2.md` §4.5: a starved training set produced 93–97% tree cover). Training moved to per-pixel, which gives shrub 918 samples instead of 22.

**A second reason, specific to this script's metadata.** It wrote a `framework_features` block naming a `D` and an `E`:

- `D` = per-segment means of all 20 bands
- `E` = `D` plus distribution (std, skew, min, max, median), shape, and context features

**Those letters are not the `RF-A_*` framework letters**, and the collision was actively misleading. `RF-A_E` was retired for an unrelated reason — it resolved to a feature set identical to `RF-A_D` (`instructions5.md` §4.1) — whereas this script's `E` genuinely added features. Two different meanings of "E", one of them retired, in the same project.

**Its config key `stage1_7_generate_segments` is retained** in `config/srer_2022.json`, annotated as retired. It records the parameters the existing segment files on disk were built with, which is the only provenance those files have.

**The 447 MB of segment outputs in `stage1_data_and_features/segments/` were left in place** — retiring the script is not a reason to delete data. Delete them deliberately, or keep them if a segment-level comparison is ever revisited.
