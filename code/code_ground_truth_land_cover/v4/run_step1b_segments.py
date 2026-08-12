#!/usr/bin/env python3
"""Step 1b - SLIC segmentation and per-segment features (instructions5.md Step 1b).

Segments each tile at 1 m and summarizes the Step 1a feature stack per segment.

Two feature sets are produced, matching the D / E split in section 4.1:

  D  "the layers"       - the mean of each stack band, one number per band
  E  "full feature set" - D plus within-segment distribution statistics, shape,
                          and context

E is the accuracy ceiling. If E does not beat D by a worthwhile margin, D is the
better framework because it is simpler, so the two must be built from the same
segmentation to be comparable.

SEGMENTATION IS FRAMEWORK-AGNOSTIC AND DELIBERATELY SO. It runs on RGB-derived
bands only - the framework A input set, the lowest common denominator across
A-E and the only thing reproducible from NAIP alone. Segmenting on CHM or NIR
would give A-C a different segmentation from D/E, which would confound the
comparison that section 4.1 exists to make: frameworks must vary by input
features only, never by the partition those features are measured over.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
EPS = 1e-9


def resolve(root, *parts):
    return Path(str(root)).expanduser().joinpath(*parts)


def load_stack(path):
    with rasterio.open(path) as ds:
        names = list(ds.descriptions)
        arr = ds.read().astype("float32")
        profile = ds.profile
    return names, arr, profile


def slic_labels(stack, names, cfg):
    """SLIC on the framework-A bands, with compactness calibrated to hit a target count."""
    from skimage.segmentation import slic

    s1b = cfg["step1b"]
    chans = [names.index(n) for n in s1b["segmentation_bands"]]
    img = np.stack([np.nan_to_num(stack[i], nan=0.0) for i in chans], axis=-1)
    img = (img - img.mean(axis=(0, 1))) / (img.std(axis=(0, 1)) + EPS)

    target = s1b["target_segments"]
    compactness = s1b["compactness"]

    def run(request):
        seg = slic(
            img,
            n_segments=int(request),
            compactness=compactness,
            channel_axis=-1,
            start_label=1,
            enforce_connectivity=True,
        )
        return seg, int(seg.max())

    # Compactness controls segment SHAPE (how far a region may deform to follow
    # an edge), not segment COUNT - measured here, the delivered count varies by
    # under 1% across compactness 0.1-20. n_segments is the count lever, but
    # connectivity enforcement means the request is not the delivery, so one
    # proportional correction is applied.
    seg, n = run(target)
    best = (seg, n, compactness, int(target))
    for _ in range(s1b["calibration_iters"]):
        if n == 0 or abs(n - target) / target <= s1b["count_tolerance"]:
            break
        request = max(1, round(target * target / n))
        seg, n = run(request)
        if abs(n - target) < abs(best[1] - target):
            best = (seg, n, compactness, request)
    return best


def segment_stats(labels, values, n_seg):
    """mean, std and skew per segment, from bincount moments (fast, exact)."""
    flat = labels.ravel()
    v = np.nan_to_num(values.ravel(), nan=0.0).astype(np.float64)
    valid = np.isfinite(values.ravel()).astype(np.float64)
    cnt = np.bincount(flat, weights=valid, minlength=n_seg + 1)[1:]
    cnt_safe = np.maximum(cnt, 1.0)
    s1 = np.bincount(flat, weights=v * valid, minlength=n_seg + 1)[1:]
    s2 = np.bincount(flat, weights=v * v * valid, minlength=n_seg + 1)[1:]
    s3 = np.bincount(flat, weights=v**3 * valid, minlength=n_seg + 1)[1:]
    mean = s1 / cnt_safe
    m2 = np.maximum(s2 / cnt_safe - mean**2, 0.0)
    m3 = s3 / cnt_safe - 3 * mean * (s2 / cnt_safe) + 2 * mean**3
    std = np.sqrt(m2)
    skew = np.where(m2 > EPS, m3 / (m2**1.5 + EPS), 0.0)
    return mean, std, skew, cnt


def build_features(labels, stack, names, cfg, n_seg):
    """Return (feature_dict, d_columns, e_columns)."""
    idx = np.arange(1, n_seg + 1)
    feats = {"segment_id": idx}
    d_cols, e_cols = [], []

    for name, band in zip(names, stack):
        mean, std, skew, cnt = segment_stats(labels, band, n_seg)
        feats[f"{name}_mean"] = mean
        d_cols.append(f"{name}_mean")
        feats[f"{name}_std"] = std
        feats[f"{name}_skew"] = skew
        e_cols += [f"{name}_std", f"{name}_skew"]
        clean = np.nan_to_num(band, nan=0.0)
        feats[f"{name}_min"] = ndi.minimum(clean, labels, idx)
        feats[f"{name}_max"] = ndi.maximum(clean, labels, idx)
        feats[f"{name}_median"] = ndi.median(clean, labels, idx)
        e_cols += [f"{name}_min", f"{name}_max", f"{name}_median"]

    # ---- shape (E only)
    from skimage.measure import regionprops_table

    props = regionprops_table(
        labels,
        properties=("label", "area", "perimeter", "eccentricity", "solidity", "extent"),
    )
    order = np.argsort(props["label"])
    for key in ("area", "perimeter", "eccentricity", "solidity", "extent"):
        col = np.zeros(n_seg, dtype=np.float64)
        col[props["label"][order] - 1] = np.asarray(props[key])[order]
        feats[f"shape_{key}"] = col
        e_cols.append(f"shape_{key}")
    per = feats["shape_perimeter"]
    feats["shape_circularity"] = np.where(
        per > EPS, 4 * np.pi * feats["shape_area"] / (per**2 + EPS), 0.0
    )
    e_cols.append("shape_circularity")

    # ---- context (E only): neighbour contrast, and distance to the nearest tall object
    neighbours = segment_adjacency(labels, n_seg)
    for name in cfg["step1b"]["context_bands"]:
        if name not in names:
            continue
        own = feats[f"{name}_mean"]
        nb = np.array([own[list(s)].mean() if s else own[i] for i, s in enumerate(neighbours)])
        feats[f"ctx_{name}_nbmean"] = nb
        feats[f"ctx_{name}_contrast"] = own - nb
        e_cols += [f"ctx_{name}_nbmean", f"ctx_{name}_contrast"]

    if "CHM" in names:
        chm = np.nan_to_num(stack[names.index("CHM")], nan=0.0)
        tall = chm >= cfg["parameters"]["H_TREE_MIN"]
        dist = ndi.distance_transform_edt(~tall) if tall.any() else np.full(chm.shape, np.inf)
        dist = np.where(np.isfinite(dist), dist, chm.shape[0])
        feats["ctx_dist_to_tree_m"] = ndi.mean(dist, labels, idx)
        e_cols.append("ctx_dist_to_tree_m")

    feats["n_pixels"] = segment_stats(labels, stack[0], n_seg)[3]
    return feats, d_cols, e_cols


def segment_adjacency(labels, n_seg):
    """Neighbour sets from horizontally/vertically touching label pairs."""
    pairs = set()
    for a, b in (
        (labels[:, :-1].ravel(), labels[:, 1:].ravel()),
        (labels[:-1, :].ravel(), labels[1:, :].ravel()),
    ):
        diff = a != b
        for x, y in zip(a[diff], b[diff]):
            pairs.add((x - 1, y - 1) if x < y else (y - 1, x - 1))
    nbrs = [set() for _ in range(n_seg)]
    for i, j in pairs:
        nbrs[i].add(j)
        nbrs[j].add(i)
    return nbrs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--tiles", nargs="+")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    site, year = cfg["site"], cfg["year"]
    feat_dir = resolve(cfg["results_root"], "01_pixel_classification", "features")
    out_dir = resolve(cfg["results_root"], "01_pixel_classification", "segments")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    d_cols = e_cols = None
    for tile in args.tiles or list(cfg["tiles"]):
        src = feat_dir / f"features_{site}_{tile}_{year}.tif"
        if not src.exists():
            sys.exit(f"missing Step 1a features: {src}")
        names, stack, profile = load_stack(src)

        seg, n_seg, compactness, request = slic_labels(stack, names, cfg)
        feats, d_cols, e_cols = build_features(seg, stack, names, cfg, n_seg)

        lab_profile = dict(profile, dtype="int32", count=1, nodata=0, compress="deflate")
        with rasterio.open(out_dir / f"segments_{site}_{tile}_{year}.tif", "w", **lab_profile) as ds:
            ds.write(seg.astype("int32"), 1)

        cols = list(feats)
        table = np.column_stack([np.asarray(feats[c], dtype="float32") for c in cols])
        np.savez_compressed(
            out_dir / f"segment_features_{site}_{tile}_{year}.npz",
            columns=np.array(cols),
            data=table,
        )
        px = feats["n_pixels"]
        print(
            f"[{tile}] segments={n_seg:>6} (requested {request})  compactness={compactness:.2f}  "
            f"px/seg mean={px.mean():.1f} median={np.median(px):.0f}  features={len(cols) - 1}"
        )
        summary.append(
            {
                "tile": tile,
                "n_segments": n_seg,
                "compactness": compactness,
                "n_segments_requested": request,
                "px_per_segment_mean": round(float(px.mean()), 2),
                "px_per_segment_median": float(np.median(px)),
            }
        )

    spec = {
        "site": site,
        "year": year,
        "segmentation_bands": cfg["step1b"]["segmentation_bands"],
        "segmentation_note": (
            "SLIC runs on framework-A (RGB-derived) bands only, so every framework "
            "shares one segmentation and A-E differ by input features alone."
        ),
        "target_segments": cfg["step1b"]["target_segments"],
        "framework_features": {
            "D": d_cols,
            "E": d_cols + e_cols,
            "A": None,
            "B": None,
            "C": None,
        },
        "framework_feature_note": (
            "A/B/C use the same per-band means as D, restricted to their band groups "
            "in the Step 1a band spec. D = per-segment means of all 20 bands. "
            "E = D plus distribution, shape and context features, and serves as the "
            "accuracy ceiling: if E does not beat D materially, D wins on simplicity."
        ),
        "n_features": {"D": len(d_cols), "E": len(d_cols) + len(e_cols)},
        "tiles": summary,
    }
    (out_dir / f"segments_{site}_{year}_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nD features: {len(d_cols)}   E features: {len(d_cols) + len(e_cols)}")
    print(f"wrote {out_dir / f'segments_{site}_{year}_spec.json'}")


if __name__ == "__main__":
    main()
