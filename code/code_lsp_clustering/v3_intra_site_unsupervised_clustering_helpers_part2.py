from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import joblib
import geopandas as gpd
from shapely.geometry import Point

from v3_intra_site_unsupervised_clustering_helpers_part1 import get_ordered_feature_names, pixel_to_xy

"""
Step 3.1 — Extract K-means and GMM cluster centroids (and GMM
covariances) from persisted models.

- K-means cluster centroids (standardized feature space)
- GMM cluster means (standardized feature space)
- GMM per-cluster covariance matrices, normalized to full (k, F, F)
  shape regardless of whether the fitted covariance_type was "full"
  or "diag" (diagonal covariances are expanded into diagonal
  matrices), to support Mahalanobis distance computation in Step 3.2.
"""


# ---------------------------------------------------------------------------
# Report loading (fresh-start helper)
# ---------------------------------------------------------------------------

def load_report(output_dir, config):
    """
    Load the run report JSON from disk.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Results directory containing the report file.
    config : dict
        Config dict; expected key "report" giving the report filename.

    Returns
    -------
    dict
        Parsed report JSON.

    Raises
    ------
    FileNotFoundError
        If the report file does not exist at the expected path.
    """
    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    if not report_path.exists():
        raise FileNotFoundError(
            f"Report file not found: {report_path}. Ensure Steps 1.1 "
            f"through 2.7 have been run for this site/config."
        )
    with open(report_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Centroid and covariance extraction
# ---------------------------------------------------------------------------

def extract_kmeans_centroids(kmeans_model, feature_names):
    """
    Extract K-means cluster centroids as a labeled DataFrame.

    Parameters
    ----------
    kmeans_model : sklearn.cluster.KMeans
        Fitted K-means model loaded from disk (Step 2.7 output).
    feature_names : list[str]
        Ordered list of feature names matching the columns the model
        was fit on.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (k, len(feature_names)), indexed by
        cluster ID (0 to k-1), columns are feature names.
    """
    centroids = kmeans_model.cluster_centers_
    return pd.DataFrame(
        centroids,
        index=[f"cluster_{i}" for i in range(centroids.shape[0])],
        columns=feature_names,
    )


def extract_gmm_centroids(gmm_model, feature_names):
    """
    Extract GMM cluster means as a labeled DataFrame.

    Parameters
    ----------
    gmm_model : sklearn.mixture.GaussianMixture
        Fitted GMM model loaded from disk (Step 2.7 output).
    feature_names : list[str]
        Ordered list of feature names matching the columns the model
        was fit on.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (k, len(feature_names)), indexed by
        cluster ID (0 to k-1), columns are feature names.
    """
    means = gmm_model.means_
    return pd.DataFrame(
        means,
        index=[f"cluster_{i}" for i in range(means.shape[0])],
        columns=feature_names,
    )


def extract_gmm_covariances_full(gmm_model, n_features):
    """
    Extract GMM per-cluster covariance matrices, normalized to full
    (k, F, F) shape regardless of the fitted covariance_type.

    For `covariance_type="full"`, `gmm_model.covariances_` is already
    (k, F, F) and is returned unchanged. For `covariance_type="diag"`,
    `gmm_model.covariances_` is (k, F) (diagonal variances only); each
    row is expanded into an (F, F) diagonal matrix so downstream
    Mahalanobis distance computation can treat both covariance types
    identically.

    Parameters
    ----------
    gmm_model : sklearn.mixture.GaussianMixture
        Fitted GMM model loaded from disk (Step 2.7 output). Must have
        `covariance_type` attribute set to "full" or "diag".
    n_features : int
        Number of features (F), used to validate shape and to build
        diagonal matrices for the "diag" case.

    Returns
    -------
    np.ndarray
        Covariance array of shape (k, F, F).

    Raises
    ------
    ValueError
        If `gmm_model.covariance_type` is neither "full" nor "diag".
    """
    cov_type = gmm_model.covariance_type
    k = gmm_model.means_.shape[0]

    if cov_type == "full":
        return gmm_model.covariances_

    if cov_type == "diag":
        full_covs = np.zeros((k, n_features, n_features), dtype=np.float64)
        for i in range(k):
            full_covs[i] = np.diag(gmm_model.covariances_[i])
        return full_covs

    raise ValueError(
        f"Unsupported covariance_type '{cov_type}' for Mahalanobis "
        f"distance normalization. Expected 'full' or 'diag'."
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_step_3_1(config, output_dir, year):
    """
    Execute Step 3.1: load persisted K-means and GMM models for one
    site/year and extract cluster centroids (and GMM covariances)
    from disk, assuming a fresh-start execution with no in-memory
    state from prior steps.

    Reloads everything needed from disk: the run report (to find
    Step 2.7 model paths and chosen k), the fitted K-means and GMM
    model files (joblib), and reconstructs the feature name ordering
    from config (matching what Step 1.5 used).

    Parameters
    ----------
    config : dict
        Fully-loaded config dict for this site. Expected keys:
        - "report" : report filename
        - "metric_layers", "derived_features" : used to reconstruct
          feature ordering via `get_ordered_feature_names`
    output_dir : str or pathlib.Path
        Results directory for this run (containing report.json and
        all persisted model/scaler files).
    year : str
        Year string identifying which year's models to load.

    Returns
    -------
    dict
        Dictionary with keys:
        - "feature_names" : list[str], the 10 ordered feature names
        - "kmeans_k" : int, chosen K-means cluster count for this year
        - "gmm_k" : int, chosen GMM component count for this year
        - "gmm_covariance_type" : str, "full" or "diag"
        - "kmeans_centroids" : pandas.DataFrame (k_kmeans, F)
        - "gmm_centroids" : pandas.DataFrame (k_gmm, F)
        - "gmm_covariances" : np.ndarray (k_gmm, F, F), normalized to
          full covariance shape regardless of fitted covariance_type

    Raises
    ------
    FileNotFoundError
        If the report file or either model file is missing.
    KeyError
        If `year` is not present in the report's
        "step_2_7_final_models" section.
    """
    print(f'{"="*10} Step 3.1 {"="*10}')

    report = load_report(output_dir, config)

    if "step_2_7_final_models" not in report:
        raise KeyError(
            "Report does not contain 'step_2_7_final_models'. Ensure "
            "Step 2.7 has been run for this site before Step 3.1."
        )
    final_models_report = report["step_2_7_final_models"]
    if year not in final_models_report:
        raise KeyError(
            f"No Step 2.7 model entry found for year '{year}' in the "
            f"run report."
        )
    entry = final_models_report[year]

    kmeans_path = Path(entry["kmeans_model_path"])
    gmm_path = Path(entry["gmm_model_path"])
    if not kmeans_path.exists():
        raise FileNotFoundError(f"K-means model file not found: {kmeans_path}")
    if not gmm_path.exists():
        raise FileNotFoundError(f"GMM model file not found: {gmm_path}")

    kmeans_model = joblib.load(kmeans_path)
    gmm_model = joblib.load(gmm_path)

    feature_names = get_ordered_feature_names(config["metric_layers"], config["derived_features"])
    n_features = len(feature_names)

    kmeans_centroids = extract_kmeans_centroids(kmeans_model, feature_names)
    gmm_centroids = extract_gmm_centroids(gmm_model, feature_names)
    gmm_covariances = extract_gmm_covariances_full(gmm_model, n_features)

    print(f"Step 3.1 complete — year {year}:")
    print(f"K-means: k={entry['kmeans_k']}, centroids shape {kmeans_centroids.shape}")
    print(f"GMM: k={entry['gmm_k']}, covariance_type='{entry['gmm_covariance_type']}', centroids shape {gmm_centroids.shape}, covariances shape {gmm_covariances.shape}")

    return {
        "feature_names": feature_names,
        "kmeans_k": entry["kmeans_k"],
        "gmm_k": entry["gmm_k"],
        "gmm_covariance_type": entry["gmm_covariance_type"],
        "kmeans_centroids": kmeans_centroids,
        "gmm_centroids": gmm_centroids,
        "gmm_covariances": gmm_covariances,
    }


"""
Step 3.2 — Pairwise centroid distance matrices.

Computes within-site pairwise distances between cluster centroids:
- K-means: Euclidean distance in standardized feature space.
- GMM: Mahalanobis distance using the pooled (averaged) covariance
  matrix of each cluster pair, since each GMM component has its own
  covariance rather than a single shared one.

Distance matrices are symmetric (k, k) arrays with zero diagonal,
suitable for direct heatmap visualization.
"""



# ---------------------------------------------------------------------------
# K-means: Euclidean distance
# ---------------------------------------------------------------------------

def compute_kmeans_distance_matrix(kmeans_centroids):
    """
    Compute the pairwise Euclidean distance matrix between K-means
    cluster centroids.

    Parameters
    ----------
    kmeans_centroids : pandas.DataFrame
        DataFrame of shape (k, F), as produced by
        `extract_kmeans_centroids` (Step 3.1), indexed by cluster ID.

    Returns
    -------
    pandas.DataFrame
        Square (k, k) distance matrix, indexed and columned by the
        same cluster IDs as `kmeans_centroids`. Diagonal is zero.
    """
    values = kmeans_centroids.to_numpy(dtype=np.float64)
    distances = squareform(pdist(values, metric="euclidean"))
    return pd.DataFrame(
        distances, index=kmeans_centroids.index, columns=kmeans_centroids.index
    )


# ---------------------------------------------------------------------------
# GMM: Mahalanobis distance (pooled covariance per pair)
# ---------------------------------------------------------------------------

def _mahalanobis_pair(mean_i, mean_j, cov_i, cov_j, regularization=1e-6):
    """
    Compute the Mahalanobis distance between two cluster centroids
    using their pooled (averaged) covariance matrix.

    The pooled covariance is `(cov_i + cov_j) / 2`. A small ridge
    (`regularization`) is added to the diagonal before inversion to
    guard against near-singular covariance matrices (common for
    small or degenerate clusters, especially under "full"
    covariance_type with limited samples).

    Parameters
    ----------
    mean_i, mean_j : np.ndarray
        Centroid vectors of shape (F,) for clusters i and j.
    cov_i, cov_j : np.ndarray
        Covariance matrices of shape (F, F) for clusters i and j.
    regularization : float, optional
        Ridge term added to the pooled covariance diagonal before
        inversion, for numerical stability. Defaults to 1e-6.

    Returns
    -------
    float
        Mahalanobis distance between `mean_i` and `mean_j`.
    """
    pooled_cov = (cov_i + cov_j) / 2.0
    F = pooled_cov.shape[0]
    pooled_cov_reg = pooled_cov + regularization * np.eye(F)

    diff = mean_i - mean_j
    inv_cov = np.linalg.inv(pooled_cov_reg)
    dist_sq = diff @ inv_cov @ diff.T
    return float(np.sqrt(max(dist_sq, 0.0)))


def compute_gmm_mahalanobis_distance_matrix(gmm_centroids, gmm_covariances,
                                              regularization=1e-6):
    """
    Compute the pairwise Mahalanobis distance matrix between GMM
    cluster centroids, using each pair's pooled (averaged) covariance
    matrix.

    Since each GMM component has its own covariance rather than a
    single shared covariance, a symmetric pairwise distance requires
    a choice of which covariance to use for each pair. This function
    uses the average of the two clusters' covariance matrices for
    each pair, which is the standard convention for comparing
    Gaussian components and guarantees a symmetric distance matrix.

    An alternative, more statistically rigorous option is a
    Bhattacharyya-style distance that formally accounts for both mean
    and covariance differences between the two full distributions;
    this is not implemented here but can be substituted if a more
    rigorous divergence measure is needed later.

    Parameters
    ----------
    gmm_centroids : pandas.DataFrame
        DataFrame of shape (k, F), as produced by
        `extract_gmm_centroids` (Step 3.1), indexed by cluster ID.
    gmm_covariances : np.ndarray
        Covariance array of shape (k, F, F), as produced by
        `extract_gmm_covariances_full` (Step 3.1), in the same
        cluster order as `gmm_centroids`.
    regularization : float, optional
        Ridge term added to each pooled covariance before inversion.
        Defaults to 1e-6.

    Returns
    -------
    pandas.DataFrame
        Square (k, k) distance matrix, indexed and columned by the
        same cluster IDs as `gmm_centroids`. Diagonal is zero.

    Raises
    ------
    ValueError
        If `gmm_centroids` row count does not match
        `gmm_covariances` first-axis size.
    """
    k = gmm_centroids.shape[0]
    if gmm_covariances.shape[0] != k:
        raise ValueError(
            f"gmm_centroids has {k} clusters but gmm_covariances has "
            f"{gmm_covariances.shape[0]} — must match."
        )

    means = gmm_centroids.to_numpy(dtype=np.float64)
    distances = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        for j in range(i + 1, k):
            d = _mahalanobis_pair(
                means[i], means[j], gmm_covariances[i], gmm_covariances[j],
                regularization=regularization,
            )
            distances[i, j] = d
            distances[j, i] = d

    return pd.DataFrame(
        distances, index=gmm_centroids.index, columns=gmm_centroids.index
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_distance_matrix(distance_df, output_dir, config, year, method):
    """
    Write a pairwise centroid distance matrix to CSV.

    Parameters
    ----------
    distance_df : pandas.DataFrame
        Square distance matrix from `compute_kmeans_distance_matrix`
        or `compute_gmm_mahalanobis_distance_matrix`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict; expected key
        `config["results"]["step_3_2"]["{method}_distance_matrix_template"]`
        with a "{year}" placeholder.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's template to use.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    template = config["results"]["step_3_2"][f"{method}_distance_matrix_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    distance_df.to_csv(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_step_3_2(config, output_dir, step_3_1_results, update_report_fn, year):
    """
    Execute Step 3.2: compute within-site pairwise centroid distance
    matrices for K-means (Euclidean) and GMM (Mahalanobis, pooled
    covariance per pair).

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected key:
        - `results.step_3_2.kmeans_distance_matrix_template`
        - `results.step_3_2.gmm_distance_matrix_template`
        - `report` : report filename
    output_dir : str or pathlib.Path
        Results directory (containing the report JSON).
    year : str
        Year string.
    step_3_1_results : dict
        Output of `run_step_3_1` for the same site/year, containing
        "kmeans_centroids", "gmm_centroids", "gmm_covariances".
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict
        Dictionary with keys:
        - "kmeans_distance_matrix" : pandas.DataFrame (k, k)
        - "gmm_distance_matrix" : pandas.DataFrame (k, k)
        - "kmeans_distance_csv" : str, output path
        - "gmm_distance_csv" : str, output path
        Also writes both CSVs to disk and updates the run report.
    """
    print(f'{"="*10} Step 3.2 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    kmeans_centroids = step_3_1_results["kmeans_centroids"]
    gmm_centroids = step_3_1_results["gmm_centroids"]
    gmm_covariances = step_3_1_results["gmm_covariances"]

    kmeans_dist = compute_kmeans_distance_matrix(kmeans_centroids)
    gmm_dist = compute_gmm_mahalanobis_distance_matrix(gmm_centroids, gmm_covariances)

    kmeans_csv = write_distance_matrix(kmeans_dist, output_dir, config, year, "kmeans")
    gmm_csv = write_distance_matrix(gmm_dist, output_dir, config, year, "gmm")

    step_report = {
        year: {
            "kmeans_distance_csv": str(kmeans_csv),
            "gmm_distance_csv": str(gmm_csv),
            "kmeans_k": step_3_1_results["kmeans_k"],
            "gmm_k": step_3_1_results["gmm_k"],
            "gmm_covariance_type": step_3_1_results["gmm_covariance_type"],
            "gmm_distance_method": "mahalanobis_pooled_covariance",
        }
    }
    update_report_fn(report_path, "step_3_2_centroid_distances", step_report)

    print(f"Step 3.2 complete — year {year}:")
    print(f"K-means distance matrix ({kmeans_dist.shape}) written to {kmeans_csv}")
    print(f"GMM distance matrix ({gmm_dist.shape}) written to {gmm_csv}")

    return {
        "kmeans_distance_matrix": kmeans_dist,
        "gmm_distance_matrix": gmm_dist,
        "kmeans_distance_csv": str(kmeans_csv),
        "gmm_distance_csv": str(gmm_csv),
    }


"""
Step 3.3 — Distance matrix heatmap visualization.

Renders the K-means (Euclidean) and GMM (Mahalanobis) pairwise
centroid distance matrices from Step 3.2 as annotated heatmaps, for
visual identification of similar (low-distance) vs. distinct
(high-distance) cluster pairs — supporting candidate endmember
selection in later Phase 3 steps.
"""


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distance_heatmap(distance_df, title, out_path, cmap="viridis_r"):
    """
    Render a single annotated heatmap of a pairwise centroid distance
    matrix.

    Parameters
    ----------
    distance_df : pandas.DataFrame
        Square (k, k) distance matrix, as produced by
        `compute_kmeans_distance_matrix` or
        `compute_gmm_mahalanobis_distance_matrix` (Step 3.2).
    title : str
        Plot title.
    out_path : str or pathlib.Path
        Output path for the saved PNG figure.
    cmap : str, optional
        Matplotlib colormap name. Defaults to "viridis_r" (reversed,
        so darker cells indicate smaller distance / higher
        similarity).

    Returns
    -------
    pathlib.Path
        Absolute path to the saved figure.
    """
    k = distance_df.shape[0]
    values = distance_df.to_numpy()

    fig_size = max(6, k * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(values, cmap=cmap, aspect="equal")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(distance_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(distance_df.index)
    ax.set_title(title)

    # Annotate each cell with its distance value.
    # Text color flips to white on dark cells for readability.
    vmax = values.max() if values.max() > 0 else 1.0
    for i in range(k):
        for j in range(k):
            val = values[i, j]
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                color=text_color, fontsize=8,
            )

    fig.colorbar(im, ax=ax, label="Distance", shrink=0.8)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_distance_heatmap(distance_df, output_dir, config, year, method):
    """
    Generate and write a distance-matrix heatmap for one method, using
    the config's Step 3.3 filename template.

    Parameters
    ----------
    distance_df : pandas.DataFrame
        Square distance matrix for this method.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict; expected key
        `config["results"]["step_3_3"]["{method}_heatmap_template"]`
        with a "{year}" placeholder.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's template and title to use.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PNG.

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    template = config["results"]["step_3_3"][f"{method}_heatmap_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename

    distance_label = "Euclidean" if method == "kmeans" else "Mahalanobis"
    method_label = "K-means" if method == "kmeans" else "GMM"
    title = f"{method_label} centroid pairwise {distance_label} distance — {year}"

    return plot_distance_heatmap(distance_df, title, out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_step_3_3(config, output_dir, step_3_2_results, update_report_fn, year):
    """
    Execute Step 3.3: render heatmap visualizations of the K-means and
    GMM pairwise centroid distance matrices from Step 3.2.

    This step produces visual diagnostics only — no automated
    endmember selection happens here. The analyst reviews these
    heatmaps (alongside NAIP overlays, in a later step) to identify
    which cluster pairs are visually/statistically similar (candidate
    duplicates or the same underlying class) versus distinct
    (candidate separate endmembers).

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected keys:
        - `results.step_3_3.kmeans_heatmap_template`
        - `results.step_3_3.gmm_heatmap_template`
        - `report` : report filename
    output_dir : str or pathlib.Path
        Results directory (containing the report JSON).
    year : str
        Year string.
    step_3_2_results : dict
        Output of `run_step_3_2` for the same site/year, containing
        "kmeans_distance_matrix" and "gmm_distance_matrix".
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict
        Dictionary with keys "kmeans_heatmap_png" and
        "gmm_heatmap_png" (absolute paths as strings). Also writes
        both PNGs to disk and updates the run report.
    """
    print(f'{"="*10} Step 3.3 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    kmeans_dist = step_3_2_results["kmeans_distance_matrix"]
    gmm_dist = step_3_2_results["gmm_distance_matrix"]

    kmeans_png = write_distance_heatmap(kmeans_dist, output_dir, config, year, "kmeans")
    gmm_png = write_distance_heatmap(gmm_dist, output_dir, config, year, "gmm")

    step_report = {
        year: {
            "kmeans_heatmap_png": str(kmeans_png),
            "gmm_heatmap_png": str(gmm_png),
        }
    }
    update_report_fn(report_path, "step_3_3_distance_heatmaps", step_report)

    print(f"Step 3.3 complete — year {year}:")
    print(f"K-means heatmap written to {kmeans_png}")
    print(f"GMM heatmap written to {gmm_png}")

    return {
        "kmeans_heatmap_png": str(kmeans_png),
        "gmm_heatmap_png": str(gmm_png),
    }


"""
Step 3.4 — Per-cluster pixel sampling for NAIP inspection.

Loads the full-raster cluster label GeoTIFFs (Step 2.8 output) for
K-means and GMM, identifies "interior" pixels per cluster (pixels
whose 5x5 neighborhood is majority the same cluster label, to avoid
sampling noisy boundary/speckle pixels), draws 20 random interior
pixels per cluster, and converts sampled pixel indices to projected
(x, y) coordinates in the raster's native UTM CRS (no reprojection).

Output is a per-cluster sample table (CSV) intended for manual visual
cross-referencing against NAIP imagery. No automated NAIP thumbnail
extraction is performed at this step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import generic_filter


NODATA_VALUE = -1


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------

def load_cluster_raster(tif_path):
    """
    Load a full-raster cluster-label GeoTIFF (Step 2.8 output).

    Parameters
    ----------
    tif_path : str or pathlib.Path
        Path to the cluster-label GeoTIFF.

    Returns
    -------
    tuple[np.ndarray, rasterio.profiles.Profile]
        - label_raster : ndarray of shape (H, W), dtype int16, with
          cluster labels (0 to k-1) and `NODATA_VALUE` (-1) for
          nodata pixels.
        - profile : rasterio profile of the raster (CRS, transform,
          etc.).

    Raises
    ------
    FileNotFoundError
        If `tif_path` does not exist.
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"Cluster raster not found: {tif_path}")
    with rasterio.open(tif_path) as src:
        label_raster = src.read(1)
        profile = src.profile
    return label_raster, profile


# ---------------------------------------------------------------------------
# Interior-pixel identification (5x5 majority filter)
# ---------------------------------------------------------------------------

def _majority_match_fraction(neighborhood, center_value):
    """
    Compute the fraction of a flattened neighborhood window that
    matches a given center value.

    Used as the callable for `scipy.ndimage.generic_filter`; not
    intended to be called directly outside that context.

    Parameters
    ----------
    neighborhood : np.ndarray
        Flattened array of neighborhood pixel values (as provided by
        `generic_filter`).
    center_value : int or float
        The value to compare against (the center pixel's own value).

    Returns
    -------
    float
        Fraction of neighborhood pixels equal to `center_value`.
    """
    return np.mean(neighborhood == center_value)


def compute_interior_mask(label_raster, window_size=5, majority_threshold=0.5):
    """
    Identify "interior" pixels: pixels whose neighborhood window is
    majority the same cluster label as the pixel itself, excluding
    nodata pixels entirely.

    This avoids sampling from cluster boundaries or isolated
    speckle/noise pixels, which are less representative of a
    cluster's core signature.

    Parameters
    ----------
    label_raster : np.ndarray
        Integer array of shape (H, W), cluster labels with
        `NODATA_VALUE` for nodata pixels.
    window_size : int, optional
        Side length of the square neighborhood window (must be odd).
        Defaults to 5 (5x5 neighborhood).
    majority_threshold : float, optional
        Minimum fraction of neighborhood pixels that must match the
        center pixel's label for it to be considered "interior".
        Defaults to 0.5 (strict majority).

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W). True where the pixel is
        non-nodata AND its neighborhood majority-matches its own
        label.

    Raises
    ------
    ValueError
        If `window_size` is not odd.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}.")

    valid_mask = label_raster != NODATA_VALUE

    # Pad with NODATA_VALUE so edge pixels' neighborhoods don't wrap
    # or falsely match across the raster boundary.
    pad = window_size // 2
    padded = np.pad(
        label_raster, pad_width=pad, mode="constant", constant_values=NODATA_VALUE
    )

    match_fraction = generic_filter(
        padded,
        function=_match_fraction_with_center,
        size=window_size,
        mode="constant",
        cval=NODATA_VALUE,
    )
    # Crop back to original shape (generic_filter preserves padded shape).
    match_fraction = match_fraction[pad:-pad, pad:-pad] if pad > 0 else match_fraction

    interior_mask = valid_mask & (match_fraction >= majority_threshold)
    return interior_mask


def _match_fraction_with_center(neighborhood):
    """
    Callable for `generic_filter`: compute the fraction of a flattened
    neighborhood matching its own center value.

    `generic_filter` passes a flattened 1D array of the neighborhood
    window; the center element's index within that flattened array is
    `len(neighborhood) // 2` for an odd-sized square window.

    Parameters
    ----------
    neighborhood : np.ndarray
        Flattened neighborhood window values.

    Returns
    -------
    float
        Fraction of neighborhood pixels equal to the center pixel's
        value. Returns 0.0 if the center pixel is nodata (so nodata
        centers never pass the majority threshold).
    """
    center_idx = len(neighborhood) // 2
    center_value = neighborhood[center_idx]
    if center_value == NODATA_VALUE:
        return 0.0
    return np.mean(neighborhood == center_value)


# ---------------------------------------------------------------------------
# Per-cluster sampling
# ---------------------------------------------------------------------------

def sample_interior_pixels_per_cluster(label_raster, interior_mask, k,
                                        n_samples_per_cluster, seed):
    """
    Draw a random sample of interior pixel locations for each cluster.

    Parameters
    ----------
    label_raster : np.ndarray
        Integer array of shape (H, W), cluster labels with
        `NODATA_VALUE` for nodata pixels.
    interior_mask : np.ndarray
        Boolean array of shape (H, W), as produced by
        `compute_interior_mask`. True where the pixel qualifies as
        "interior" for sampling.
    k : int
        Number of clusters (cluster IDs assumed to be 0 to k-1).
    n_samples_per_cluster : int
        Number of pixel locations to sample per cluster (fewer are
        returned if a cluster has fewer interior pixels available).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from cluster ID to (rows, cols) arrays of sampled
        pixel indices for that cluster.
    """
    rng = np.random.default_rng(seed)
    samples_by_cluster = {}

    for cluster_id in range(k):
        cluster_interior_mask = interior_mask & (label_raster == cluster_id)
        rows, cols = np.where(cluster_interior_mask)

        n_available = rows.size
        n_draw = min(n_samples_per_cluster, n_available)

        if n_draw == 0:
            samples_by_cluster[cluster_id] = (
                np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            )
            continue

        chosen_idx = rng.choice(n_available, size=n_draw, replace=False)
        samples_by_cluster[cluster_id] = (rows[chosen_idx], cols[chosen_idx])

    return samples_by_cluster


# ---------------------------------------------------------------------------
# Pixel-to-coordinate conversion (reuses Step 1.3's pixel_to_xy)
# ---------------------------------------------------------------------------

def build_cluster_sample_table(samples_by_cluster, transform, method, year):
    """
    Assemble a combined sample table (across all clusters) with
    projected (x, y) coordinates in the raster's native CRS, plus an
    empty `naip_class` column for manual ground-truth annotation in
    QGIS.

    `naip_class` is an integer code with the following mapping,
    intended to be populated manually after visual inspection against
    NAIP imagery:

        0 = bare
        1 = grass
        2 = woody

    Parameters
    ----------
    samples_by_cluster : dict[int, tuple[np.ndarray, np.ndarray]]
        Output of `sample_interior_pixels_per_cluster`.
    transform : affine.Affine
        Rasterio affine transform of the cluster raster.
    method : str
        Method label ("kmeans" or "gmm"), recorded as a column for
        reference.
    year : str
        Year string, recorded as a column for reference.

    Returns
    -------
    pandas.DataFrame
        Columns: cluster_id, row, col, x, y, method, year,
        naip_class. One row per sampled pixel, across all clusters.
        `naip_class` is initialized as a nullable integer (pandas
        "Int64") with all values missing (<NA>), to be manually
        populated in QGIS with 0 (bare), 1 (grass), or 2 (woody).
    """
    rows_all = []
    cols_all = []
    cluster_ids_all = []

    for cluster_id, (rows, cols) in samples_by_cluster.items():
        rows_all.append(rows)
        cols_all.append(cols)
        cluster_ids_all.append(np.full(rows.size, cluster_id, dtype=np.int64))

    rows_all = np.concatenate(rows_all) if rows_all else np.array([], dtype=np.int64)
    cols_all = np.concatenate(cols_all) if cols_all else np.array([], dtype=np.int64)
    cluster_ids_all = (
        np.concatenate(cluster_ids_all) if cluster_ids_all else np.array([], dtype=np.int64)
    )

    x, y = pixel_to_xy(rows_all, cols_all, transform)

    df = pd.DataFrame({
        "cluster_id": cluster_ids_all,
        "row": rows_all,
        "col": cols_all,
        "x": x,
        "y": y,
        "method": method,
        "year": str(year),
    })
    # Nullable integer dtype so the column can hold missing values
    # (<NA>) until manually populated: 0=bare, 1=grass, 2=woody.
    df["naip_class"] = pd.array([pd.NA] * len(df), dtype="Int64")
    return df


# ---------------------------------------------------------------------------
# GeoPackage assembly
# ---------------------------------------------------------------------------

def build_cluster_sample_geodataframe(sample_df, crs):
    """
    Convert a flat per-cluster sample DataFrame (with x, y columns)
    into a GeoDataFrame with Point geometries, ready for GeoPackage
    export.

    Parameters
    ----------
    sample_df : pandas.DataFrame
        Output of `build_cluster_sample_table`, containing at minimum
        "x" and "y" columns in the raster's native projected CRS.
    crs : rasterio.crs.CRS or str or dict
        Coordinate reference system of the source raster (e.g., from
        `profile["crs"]`), used to georeference the output layer.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with the same attribute columns as `sample_df`
        plus a "geometry" column of Point features, in the given CRS.
    """
    geometry = [Point(xy) for xy in zip(sample_df["x"], sample_df["y"])]
    gdf = gpd.GeoDataFrame(sample_df.copy(), geometry=geometry, crs=crs)
    return gdf


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_cluster_sample_table(gdf, output_dir, config, year, method):
    """
    Write the per-cluster NAIP-inspection sample table as a
    GeoPackage point layer, using the config's Step 3.4 filename
    template.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Sample table with Point geometry, from
        `build_cluster_sample_geodataframe`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict; expected key
        `config["results"]["step_3_4"]["{method}_sample_table_template"]`
        with a "{year}" placeholder. The template's file extension
        should be ".gpkg".
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's template to use.

    Returns
    -------
    pathlib.Path
        Absolute path to the written GeoPackage file.

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    template = config["results"]["step_3_4"][f"{method}_sample_table_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename

    gdf.to_file(out_path, driver="GPKG", layer=f"{method}_naip_samples_{year}")
    return out_path


# ---------------------------------------------------------------------------
# Per-method orchestration
# ---------------------------------------------------------------------------

def run_naip_sampling_for_method(config, output_dir, year, method, k,
                                   n_samples_per_cluster, seed,
                                   window_size=5, majority_threshold=0.5):
    """
    Run interior-pixel sampling for NAIP inspection for one clustering
    method (K-means or GMM), loading the full-raster cluster GeoTIFF
    from disk (Step 2.8 output), and writing results as an editable
    GeoPackage point layer.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected key
        `config["results"]["step_2_8"]["{method}_full_raster_tif_template"]`
        with "{year}" and "{k}" placeholders.
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's cluster raster to load.
    k : int
        Number of clusters for this method/year.
    n_samples_per_cluster : int
        Number of interior pixels to sample per cluster.
    seed : int
        Random seed for reproducibility.
    window_size : int, optional
        Neighborhood window size for interior-pixel identification.
        Defaults to 5.
    majority_threshold : float, optional
        Minimum neighborhood match fraction for interior
        classification. Defaults to 0.5.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, pathlib.Path, dict]
        - sample_gdf : combined per-cluster sample GeoDataFrame.
        - gpkg_path : path to the written GeoPackage.
        - stats : dict with per-cluster interior pixel counts and
          samples drawn, for report logging.

    Raises
    ------
    FileNotFoundError
        If the cluster raster file does not exist.
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    tif_template = config["results"]["step_2_8"][f"{method}_full_raster_tif_template"]
    tif_filename = tif_template.replace("{year}", str(year)).replace("{k}", str(k))
    tif_path = Path(output_dir) / tif_filename

    label_raster, profile = load_cluster_raster(tif_path)
    interior_mask = compute_interior_mask(
        label_raster, window_size=window_size, majority_threshold=majority_threshold
    )
    samples_by_cluster = sample_interior_pixels_per_cluster(
        label_raster, interior_mask, k, n_samples_per_cluster, seed
    )

    sample_df = build_cluster_sample_table(
        samples_by_cluster, profile["transform"], method, year
    )
    sample_gdf = build_cluster_sample_geodataframe(sample_df, profile["crs"])
    gpkg_path = write_cluster_sample_table(sample_gdf, output_dir, config, year, method)

    stats = {
        str(cid): {
            "n_interior_available": int(np.sum(
                (interior_mask) & (label_raster == cid)
            )),
            "n_sampled": int(len(rows)),
        }
        for cid, (rows, cols) in samples_by_cluster.items()
    }

    return sample_gdf, gpkg_path, stats

# ---------------------------------------------------------------------------
# Top-level orchestration (both methods)
# ---------------------------------------------------------------------------

def run_step_3_4(config, output_dir, year, step_3_1_results,
                  n_samples_per_cluster=20, seed=99, window_size=5,
                  majority_threshold=0.5, update_report_fn=None):
    """
    Execute Step 3.4: per-cluster interior-pixel sampling for NAIP
    inspection, for both K-means and GMM full-raster cluster maps.

    Loads each method's full-raster cluster GeoTIFF (Step 2.8 output)
    from disk, identifies interior pixels (5x5 neighborhood majority
    match by default), draws a random sample per cluster, and writes
    a combined sample table (with native-CRS projected coordinates)
    per method.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected keys:
        - `results.step_2_8.kmeans_full_raster_tif_template`
        - `results.step_2_8.gmm_full_raster_tif_template`
        - `results.step_3_4.kmeans_sample_table_template`
        - `results.step_3_4.gmm_sample_table_template`
        - `report` : report filename
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.
    step_3_1_results : dict
        Output of `run_step_3_1` for the same site/year, used to read
        "kmeans_k" and "gmm_k" (needed to resolve full-raster GeoTIFF
        filenames and to iterate cluster IDs).
    n_samples_per_cluster : int, optional
        Number of interior pixels to sample per cluster. Defaults to
        20.
    seed : int, optional
        Random seed for reproducibility. Defaults to 99.
    window_size : int, optional
        Neighborhood window size for interior-pixel identification.
        Defaults to 5.
    majority_threshold : float, optional
        Minimum neighborhood match fraction for interior
        classification. Defaults to 0.5.
    update_report_fn : callable, optional
        Function with signature `update_report_fn(report_path,
        section_name, content)`. If provided, appends
        "step_3_4_naip_sampling" to the run report.

    Returns
    -------
    dict
        Dictionary with keys "kmeans_sample_df", "gmm_sample_df",
        "kmeans_csv", "gmm_csv" — sample tables and their output
        paths for both methods.
    """
    print(f'{"="*10} Step 3.4 {"="*10}')

    kmeans_k = step_3_1_results["kmeans_k"]
    gmm_k = step_3_1_results["gmm_k"]

    kmeans_df, kmeans_gpkg, kmeans_stats = run_naip_sampling_for_method(
        config, output_dir, year, "kmeans", kmeans_k,
        n_samples_per_cluster, seed, window_size, majority_threshold,
    )
    gmm_df, gmm_gpkg, gmm_stats = run_naip_sampling_for_method(
        config, output_dir, year, "gmm", gmm_k,
        n_samples_per_cluster, seed, window_size, majority_threshold,
    )

    print(f"Step 3.4 complete — year {year}:")
    print(f"K-means: {len(kmeans_df)} total samples across {kmeans_k} clusters, written to {kmeans_gpkg}")
    print(f"GMM: {len(gmm_df)} total samples across {gmm_k} clusters, written to {gmm_gpkg}")

    if update_report_fn is not None:
        report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
        step_report = {
            year: {
                "kmeans_gpkg": str(kmeans_gpkg),
                "gmm_gpkg": str(gmm_gpkg),
                "n_samples_per_cluster": n_samples_per_cluster,
                "seed": seed,
                "window_size": window_size,
                "majority_threshold": majority_threshold,
                "kmeans_per_cluster_stats": kmeans_stats,
                "gmm_per_cluster_stats": gmm_stats,
            }
        }
        update_report_fn(report_path, "step_3_4_naip_sampling", step_report)

    return {
        "kmeans_sample_df": kmeans_df,
        "gmm_sample_df": gmm_df,
        "kmeans_gpkg": str(kmeans_gpkg),
        "gmm_gpkg": str(gmm_gpkg),
    }



"""
Step 3.5 — Per-cluster purity assessment from NAIP annotations.

Reads back the annotated GeoPackages (Step 3.4 output, with
`naip_class` populated in QGIS: 0=bare, 1=grass, 2=shrub, 3=tree),
computes per-cluster class distribution, majority-class purity
fraction, and Shannon entropy, flags clusters meeting an 80% purity
threshold as candidate pure endmembers, and produces a summary table
and stacked bar chart per clustering method.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_LABELS = {0: "bare", 1: "grass", 2: "shrub", 3: "tree"}
CLASS_COLORS = {0: "#c2b280", 1: "#7cb342", 2: "#8d6e63", 3: "#1b5e20"}
PURITY_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Loading annotated GeoPackages
# ---------------------------------------------------------------------------

def load_annotated_samples(gpkg_path):
    """
    Load an annotated per-cluster sample GeoPackage (Step 3.4 output,
    with `naip_class` populated).

    Parameters
    ----------
    gpkg_path : str or pathlib.Path
        Path to the GeoPackage file.

    Returns
    -------
    pandas.DataFrame
        Attribute table (geometry dropped), containing at minimum
        "cluster_id" and "naip_class" columns.

    Raises
    ------
    FileNotFoundError
        If `gpkg_path` does not exist.
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Annotated GeoPackage not found: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    return pd.DataFrame(gdf.drop(columns="geometry"))


def check_annotation_completeness(df, k):
    """
    Check for missing (null) `naip_class` annotations per cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotated sample table, containing "cluster_id" and
        "naip_class" columns.
    k : int
        Number of clusters expected (cluster IDs 0 to k-1).

    Returns
    -------
    dict[int, dict]
        Mapping from cluster_id to {"n_total": int, "n_annotated":
        int, "n_missing": int}, for every cluster ID 0 to k-1
        (including clusters absent from `df` entirely, reported with
        n_total=0).
    """
    completeness = {}
    for cluster_id in range(k):
        cluster_df = df[df["cluster_id"] == cluster_id]
        n_total = len(cluster_df)
        n_missing = int(cluster_df["naip_class"].isna().sum())
        n_annotated = n_total - n_missing
        completeness[cluster_id] = {
            "n_total": n_total,
            "n_annotated": n_annotated,
            "n_missing": n_missing,
        }
    return completeness


# ---------------------------------------------------------------------------
# Purity and entropy computation
# ---------------------------------------------------------------------------

def compute_shannon_entropy(class_counts):
    """
    Compute the Shannon entropy of a class count distribution, in
    bits (base-2 log).

    Parameters
    ----------
    class_counts : dict[int, int]
        Mapping from class code to count of annotated points in that
        class (zero counts allowed/expected for absent classes).

    Returns
    -------
    float
        Shannon entropy in bits. Zero if all annotated points belong
        to a single class (maximum purity); higher values indicate
        more even mixing across classes. Returns 0.0 if there are no
        annotated points.
    """
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in class_counts.values():
        if count == 0:
            continue
        p = count / total
        entropy -= p * np.log2(p)
    return float(entropy)


def compute_cluster_purity(df, k):
    """
    Compute per-cluster class distribution, majority-class purity
    fraction, and Shannon entropy from annotated sample points.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotated sample table, containing "cluster_id" and
        "naip_class" columns (naip_class values in {0, 1, 2, 3},
        possibly with NaN for unannotated points, which are excluded
        from purity/entropy computation).
    k : int
        Number of clusters (cluster IDs 0 to k-1).

    Returns
    -------
    pandas.DataFrame
        One row per cluster_id (0 to k-1), columns: "cluster_id",
        "n_bare", "n_grass", "n_shrub", "n_tree", "n_annotated",
        "majority_class_code", "majority_class_label",
        "purity_fraction", "shannon_entropy_bits",
        "meets_purity_threshold" (bool, purity_fraction >= 0.80).
    """
    rows = []
    for cluster_id in range(k):
        cluster_df = df[df["cluster_id"] == cluster_id]
        annotated = cluster_df["naip_class"].dropna().astype(int)

        class_counts = {c: int((annotated == c).sum()) for c in CLASS_LABELS}
        n_annotated = int(annotated.size)

        if n_annotated == 0:
            majority_code = None
            majority_label = None
            purity_fraction = 0.0
            entropy = 0.0
        else:
            majority_code = max(class_counts, key=class_counts.get)
            majority_label = CLASS_LABELS[majority_code]
            purity_fraction = class_counts[majority_code] / n_annotated
            entropy = compute_shannon_entropy(class_counts)

        rows.append({
            "cluster_id": cluster_id,
            "n_bare": class_counts[0],
            "n_grass": class_counts[1],
            "n_shrub": class_counts[2],
            "n_tree": class_counts[3],
            "n_annotated": n_annotated,
            "majority_class_code": majority_code,
            "majority_class_label": majority_label,
            "purity_fraction": purity_fraction,
            "shannon_entropy_bits": entropy,
            "meets_purity_threshold": purity_fraction >= PURITY_THRESHOLD,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_purity_stacked_bar(purity_df, method, year, out_path):
    """
    Render a stacked bar chart of per-cluster class composition,
    colored by class, with a purity-fraction line overlaid on a
    secondary y-axis.

    Clusters remain in their original cluster_id order (0 to k-1) on
    the x-axis. Within each individual bar, the four class segments
    are stacked in descending order of count (largest at the bottom,
    smallest at the top), so the majority class is always the
    bottom-most, most visually dominant segment regardless of which
    class it is. A line plot of purity_fraction (majority-class
    fraction) is drawn on a secondary y-axis (0-100%) to make
    per-cluster purity directly readable without needing text
    annotations.

    Parameters
    ----------
    purity_df : pandas.DataFrame
        Output of `compute_cluster_purity`. Must be in cluster_id
        order (0 to k-1); no reordering is performed.
    method : str
        Method label ("kmeans" or "gmm"), used in the plot title.
    year : str
        Year string, used in the plot title.
    out_path : str or pathlib.Path
        Output path for the saved PNG figure.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved figure.
    """
    df = purity_df.sort_values("cluster_id").reset_index(drop=True)
    k = len(df)
    class_cols = {0: "n_bare", 1: "n_grass", 2: "n_shrub", 3: "n_tree"}

    fig, ax = plt.subplots(figsize=(max(8, k * 0.9), 6))
    ax2 = ax.twinx()

    positions = np.arange(k)
    bottoms = np.zeros(k)

    # For each bar (cluster), sort its own four segments by count
    # descending, then stack them bottom-to-top in that order.
    # Track which class occupies which stack level per bar so the
    # legend still reflects true class colors consistently.
    plotted_class_handles = {}

    for i, row in df.iterrows():
        counts_this_bar = [
            (class_code, row[col_name]) for class_code, col_name in class_cols.items()
        ]
        counts_this_bar.sort(key=lambda x: x[1], reverse=True)

        running_bottom = 0.0
        for class_code, count in counts_this_bar:
            if count == 0:
                continue
            bar_container = ax.bar(
                positions[i], count, bottom=running_bottom,
                color=CLASS_COLORS[class_code], edgecolor="white",
                linewidth=0.5, width=0.7,
            )
            running_bottom += count
            if class_code not in plotted_class_handles:
                plotted_class_handles[class_code] = bar_container

    # Purity line on secondary axis.
    ax2.plot(
        positions, df["purity_fraction"] * 100, color="black", marker="o",
        markersize=5, linewidth=1.5, label="Purity (%)", zorder=5,
    )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Purity fraction (%)")

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Annotated point count")
    ax.set_title(f"{method.upper()} cluster class composition — {year}")
    ax.set_xticks(positions)
    ax.set_xticklabels(df["cluster_id"])
    ax.grid(alpha=0.2, axis="y")

    # Build a combined legend: class color patches + purity line.
    legend_handles = [
        plotted_class_handles[c][0] for c in sorted(plotted_class_handles)
    ]
    legend_labels = [CLASS_LABELS[c] for c in sorted(plotted_class_handles)]
    line_handle, line_label = ax2.get_legend_handles_labels()
    ax.legend(
        legend_handles + line_handle, legend_labels + line_label,
        loc="upper right", fontsize=8,
    )

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_purity_table(purity_df, output_dir, config, year, method):
    """
    Write the per-cluster purity summary table to CSV.

    Parameters
    ----------
    purity_df : pandas.DataFrame
        Output of `compute_cluster_purity`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict; expected key
        `config["results"]["step_3_5"]["{method}_purity_table_template"]`
        with a "{year}" placeholder.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's template to use.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    template = config["results"]["step_3_5"][f"{method}_purity_table_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    purity_df.to_csv(out_path, index=False)
    return out_path


def write_purity_plot(purity_df, output_dir, config, year, method):
    """
    Generate and write the per-cluster class composition stacked bar
    chart.

    Parameters
    ----------
    purity_df : pandas.DataFrame
        Output of `compute_cluster_purity`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict; expected key
        `config["results"]["step_3_5"]["{method}_purity_plot_template"]`
        with a "{year}" placeholder.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's template to use.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PNG.

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    template = config["results"]["step_3_5"][f"{method}_purity_plot_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    return plot_purity_stacked_bar(purity_df, method, year, out_path)


# ---------------------------------------------------------------------------
# Per-method orchestration
# ---------------------------------------------------------------------------

def run_purity_assessment_for_method(config, output_dir, year, method, k):
    """
    Run purity assessment for one clustering method (K-means or GMM),
    loading the annotated GeoPackage from disk (Step 3.4 output, now
    with `naip_class` populated).

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected key
        `config["results"]["step_3_4"]["{method}_sample_table_template"]`
        with a "{year}" placeholder.
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's annotated samples to load.
    k : int
        Number of clusters for this method/year.

    Returns
    -------
    tuple[pandas.DataFrame, dict, pathlib.Path, pathlib.Path]
        - purity_df : per-cluster purity/entropy summary.
        - completeness : per-cluster annotation completeness stats.
        - csv_path : path to written purity CSV.
        - plot_path : path to written stacked bar chart PNG.
    """
    gpkg_template = config["results"]["step_3_4"][f"{method}_sample_table_template"]
    gpkg_filename = gpkg_template.replace("{year}", str(year))
    gpkg_path = Path(output_dir) / gpkg_filename

    annotated_df = load_annotated_samples(gpkg_path)
    completeness = check_annotation_completeness(annotated_df, k)
    purity_df = compute_cluster_purity(annotated_df, k)

    csv_path = write_purity_table(purity_df, output_dir, config, year, method)
    plot_path = write_purity_plot(purity_df, output_dir, config, year, method)

    return purity_df, completeness, csv_path, plot_path


# ---------------------------------------------------------------------------
# Top-level orchestration (both methods)
# ---------------------------------------------------------------------------

def run_step_3_5(config, output_dir, year, step_3_1_results, update_report_fn):
    """
    Execute Step 3.5: per-cluster purity assessment from NAIP
    annotations, for both K-means and GMM.

    Reads the annotated GeoPackages (Step 3.4 output, populated
    manually in QGIS), computes per-cluster class distribution,
    majority-class purity fraction, and Shannon entropy, flags
    clusters meeting the 80% purity threshold, and writes summary
    tables and stacked bar charts per method.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected keys:
        - `results.step_3_4.kmeans_sample_table_template`
        - `results.step_3_4.gmm_sample_table_template`
        - `results.step_3_5.kmeans_purity_table_template`
        - `results.step_3_5.gmm_purity_table_template`
        - `results.step_3_5.kmeans_purity_plot_template`
        - `results.step_3_5.gmm_purity_plot_template`
        - `report` : report filename
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.
    step_3_1_results : dict
        Output of `run_step_3_1` for the same site/year, used to read
        "kmeans_k" and "gmm_k".
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict
        Dictionary with keys "kmeans_purity_df", "gmm_purity_df",
        "kmeans_csv", "gmm_csv", "kmeans_plot", "gmm_plot".
    """
    print(f'{"="*10} Step 3.5 {"="*10}')

    kmeans_k = step_3_1_results["kmeans_k"]
    # gmm_k = step_3_1_results["gmm_k"]

    kmeans_purity, kmeans_completeness, kmeans_csv, kmeans_plot = (
        run_purity_assessment_for_method(config, output_dir, year, "kmeans", kmeans_k)
    )
    # gmm_purity, gmm_completeness, gmm_csv, gmm_plot = (
    #     run_purity_assessment_for_method(config, output_dir, year, "gmm", gmm_k)
    # )

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    step_report = {
        year: {
            "purity_threshold": PURITY_THRESHOLD,
            "kmeans": {
                "purity_csv": str(kmeans_csv),
                "purity_plot_png": str(kmeans_plot),
                "completeness": kmeans_completeness,
                "clusters_meeting_threshold": kmeans_purity.loc[
                    kmeans_purity["meets_purity_threshold"], "cluster_id"
                ].tolist(),
                "purity_summary": kmeans_purity.to_dict(orient="records"),
            },
            # "gmm": {
            #     "purity_csv": str(gmm_csv),
            #     "purity_plot_png": str(gmm_plot),
            #     "completeness": gmm_completeness,
            #     "clusters_meeting_threshold": gmm_purity.loc[
            #         gmm_purity["meets_purity_threshold"], "cluster_id"
            #     ].tolist(),
            #     "purity_summary": gmm_purity.to_dict(orient="records"),
            # },
        }
    }
    update_report_fn(report_path, "step_3_5_purity_assessment", step_report)

    print(f"Step 3.5 complete — year {year}:")

    return {
        "kmeans_purity_df": kmeans_purity,
        # "gmm_purity_df": gmm_purity,
        "kmeans_csv": str(kmeans_csv),
        # "gmm_csv": str(gmm_csv),
        "kmeans_plot": str(kmeans_plot),
        # "gmm_plot": str(gmm_plot),
    }


"""
Step 3.6b — Reclassified cluster raster (GeoTIFF + PNG).

Applies the cluster-to-class mapping (Step 3.6) to the full-raster
cluster-label GeoTIFF (Step 2.8 output), producing a reclassified
5-category raster: bare (0), grass (1), shrub (2), tree (3), mixed
(4), with nodata (-1) preserved. Writes both a GeoTIFF and a
categorical PNG using the mapped class colors, with mixed clusters
rendered in orange.
"""

from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


RECLASS_NODATA_VALUE = -1

MIXED_CLASS_LABEL = "mixed"
MIXED_CLASS_COLOR = "#FB6C00"

CONFIDENCE_THRESHOLDS = {
    "high": 0.80,
    "moderate": 0.50,
}

RECLASS_CODE_MAP = {
    "bare": 0,
    "grass": 1,
    "shrub": 2,
    "tree": 3,
    "mixed": 4,
}
RECLASS_LABELS = {v: k for k, v in RECLASS_CODE_MAP.items()}
RECLASS_COLORS = {
    0: CLASS_COLORS[0],   # bare
    1: CLASS_COLORS[1],   # grass
    2: CLASS_COLORS[2],   # shrub
    3: CLASS_COLORS[3],   # tree
    4: MIXED_CLASS_COLOR, # mixed, orange
}



def assign_confidence_tier(purity_fraction):
    """
    Assign a confidence tier to a purity fraction using the 0.80 /
    0.50 cutoffs.

    Parameters
    ----------
    purity_fraction : float
        Majority-class fraction for a cluster, in [0, 1].

    Returns
    -------
    str
        "high" if purity_fraction >= 0.80, "moderate" if >= 0.50 and
        < 0.80, otherwise "mixed".
    """
    if purity_fraction >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if purity_fraction >= CONFIDENCE_THRESHOLDS["moderate"]:
        return "moderate"
    return "mixed"


def build_cluster_to_class_mapping(purity_df):
    """
    Build a cluster-to-class mapping table from per-cluster purity
    results, applying the 0.80 / 0.50 confidence cutoffs.

    Clusters with purity >= 0.50 are assigned their majority class
    (bare, grass, shrub, or tree) with a "high" or "moderate"
    confidence tier. Clusters with purity < 0.50 are assigned the
    "mixed" class instead of their nominal majority class, since a
    majority below 50% does not reliably describe the cluster's
    ground composition; these are tagged with confidence_tier
    "mixed" and mapped_class_code None.

    Parameters
    ----------
    purity_df : pandas.DataFrame
        Output of `compute_cluster_purity`, containing "cluster_id",
        "majority_class_code", "majority_class_label",
        "purity_fraction".

    Returns
    -------
    pandas.DataFrame
        One row per cluster, columns:
        - "cluster_id" : int
        - "mapped_class_code" : int or None (None if mixed)
        - "mapped_class_label" : str, one of "bare", "grass",
          "shrub", "tree", or "mixed"
        - "purity_fraction" : float
        - "confidence_tier" : str, "high", "moderate", or "mixed"
        - "mapped_color" : str, hex color for the mapped class
          (CLASS_COLORS lookup for real classes, MIXED_CLASS_COLOR
          for mixed)
    """
    rows = []
    for _, row in purity_df.iterrows():
        purity = row["purity_fraction"]
        tier = assign_confidence_tier(purity)

        if tier == "mixed":
            mapped_class_code = None
            mapped_class_label = MIXED_CLASS_LABEL
            mapped_color = MIXED_CLASS_COLOR
        else:
            mapped_class_code = row["majority_class_code"]
            mapped_class_label = row["majority_class_label"]
            mapped_color = CLASS_COLORS[mapped_class_code]

        rows.append({
            "cluster_id": row["cluster_id"],
            "mapped_class_code": mapped_class_code,
            "mapped_class_label": mapped_class_label,
            "purity_fraction": purity,
            "confidence_tier": tier,
            "mapped_color": mapped_color,
        })

    return pd.DataFrame(rows)




# ---------------------------------------------------------------------------
# Remapping
# ---------------------------------------------------------------------------

def build_cluster_to_reclass_lookup(mapping_df, k):
    """
    Build a lookup array mapping original cluster_id (0 to k-1) to the
    new reclassified code (0=bare, 1=grass, 2=shrub, 3=tree, 4=mixed).

    Parameters
    ----------
    mapping_df : pandas.DataFrame
        Output of `build_cluster_to_class_mapping`, containing
        "cluster_id" and "mapped_class_label".
    k : int
        Number of original clusters (cluster IDs 0 to k-1). Every
        cluster ID in this range must be present in `mapping_df`.

    Returns
    -------
    np.ndarray
        Integer array of shape (k,), dtype int16, where index i gives
        the reclassified code for original cluster_id i.

    Raises
    ------
    ValueError
        If any cluster_id in 0..k-1 is missing from `mapping_df`.
    """
    lookup = np.full(k, RECLASS_NODATA_VALUE, dtype=np.int16)
    mapping_by_id = mapping_df.set_index("cluster_id")["mapped_class_label"]

    for cluster_id in range(k):
        if cluster_id not in mapping_by_id.index:
            raise ValueError(
                f"cluster_id {cluster_id} missing from mapping_df; "
                f"expected all IDs 0..{k-1}."
            )
        label = mapping_by_id.loc[cluster_id]
        lookup[cluster_id] = RECLASS_CODE_MAP[label]

    return lookup


def apply_reclassification(label_raster, lookup):
    """
    Apply a cluster-id-to-reclass-code lookup to a full-raster cluster
    label array, preserving nodata.

    Parameters
    ----------
    label_raster : np.ndarray
        Integer array of shape (H, W), original cluster labels (0 to
        k-1) with `NODATA_VALUE` (-1) for nodata pixels (as produced
        by Step 2.8).
    lookup : np.ndarray
        Lookup array of shape (k,), from
        `build_cluster_to_reclass_lookup`, mapping original cluster_id
        to new reclass code.

    Returns
    -------
    np.ndarray
        Integer array of shape (H, W), dtype int16, with reclassified
        codes (0=bare, 1=grass, 2=shrub, 3=tree, 4=mixed) and
        `RECLASS_NODATA_VALUE` (-1) preserved at original nodata
        pixels.
    """
    reclassified = np.full(label_raster.shape, RECLASS_NODATA_VALUE, dtype=np.int16)
    valid_mask = label_raster != NODATA_VALUE
    reclassified[valid_mask] = lookup[label_raster[valid_mask]]
    return reclassified


# ---------------------------------------------------------------------------
# GeoTIFF writing
# ---------------------------------------------------------------------------

def write_reclass_geotiff(reclass_raster, reference_profile, out_path):
    """
    Write a reclassified cluster raster to GeoTIFF, using the CRS,
    transform, and shape from a reference raster profile.

    Parameters
    ----------
    reclass_raster : np.ndarray
        Integer array of shape (H, W), dtype int16, with reclassified
        codes and `RECLASS_NODATA_VALUE` for nodata pixels.
    reference_profile : rasterio.profiles.Profile
        Profile of the reference raster (e.g., from the original
        Step 2.8 cluster GeoTIFF), providing CRS, transform, width,
        and height.
    out_path : str or pathlib.Path
        Output GeoTIFF path.

    Returns
    -------
    pathlib.Path
        Absolute path to the written GeoTIFF.
    """
    profile = reference_profile.copy()
    profile.update(
        dtype="int16",
        count=1,
        nodata=RECLASS_NODATA_VALUE,
        compress="lzw",
    )
    out_path = Path(out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(reclass_raster, 1)
    return out_path


# ---------------------------------------------------------------------------
# PNG visualization
# ---------------------------------------------------------------------------

def plot_reclass_map(reclass_raster, method_name, year, out_path):
    """
    Render a categorical PNG visualization of a reclassified cluster
    raster: bare, grass, shrub, tree colored per `CLASS_COLORS`, mixed
    rendered in orange, and nodata pixels shown in black.

    Parameters
    ----------
    reclass_raster : np.ndarray
        Integer array of shape (H, W), dtype int16, with reclassified
        codes (0=bare, 1=grass, 2=shrub, 3=tree, 4=mixed) and
        `RECLASS_NODATA_VALUE` (-1) for nodata pixels.
    method_name : str
        Method name for the plot title, e.g., "K-means" or "GMM".
    year : str
        Year string for the plot title.
    out_path : str or pathlib.Path
        Output PNG path.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PNG.
    """
    n_classes = len(RECLASS_LABELS)
    ordered_colors = ["black"] + [RECLASS_COLORS[i] for i in range(n_classes)]
    cmap = ListedColormap(ordered_colors)

    bounds = [RECLASS_NODATA_VALUE - 0.5] + [i - 0.5 for i in range(n_classes + 1)]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(reclass_raster, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"{method_name} reclassified cluster map — {year}")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_handles = [
        Patch(facecolor=RECLASS_COLORS[i], label=RECLASS_LABELS[i].capitalize())
        for i in range(n_classes)
    ]
    legend_handles.append(Patch(facecolor="black", label="No data / QA-failed"))
    ax.legend(
        handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9,
    )

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

def resolve_reclass_filenames(config, year, method):
    """
    Resolve output TIF and PNG filenames for the reclassified cluster
    raster for a given method and year.

    Parameters
    ----------
    config : dict
        Config dict; expected keys under
        `config["results"]["step_3_6"]`: "{method}_reclass_tif_template",
        "{method}_reclass_png_template", each with a "{year}"
        placeholder.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's templates to use.

    Returns
    -------
    tuple[str, str]
        (tif_filename, png_filename), with placeholders substituted.

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    tif_template = config["results"]["step_3_6"][f"{method}_reclass_tif_template"]
    png_template = config["results"]["step_3_6"][f"{method}_reclass_png_template"]

    tif_filename = tif_template.replace("{year}", str(year))
    png_filename = png_template.replace("{year}", str(year))
    return tif_filename, png_filename


# ---------------------------------------------------------------------------
# Per-method orchestration
# ---------------------------------------------------------------------------

def run_reclassification_for_method(config, output_dir, year, method, k,
                                     mapping_df):
    """
    Run reclassification for one clustering method (K-means or GMM),
    loading the original full-raster cluster GeoTIFF from disk (Step
    2.8 output), applying the cluster-to-class mapping, and writing a
    reclassified GeoTIFF and PNG.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected key
        `config["results"]["step_2_8"]["{method}_full_raster_tif_template"]`
        with "{year}" and "{k}" placeholders.
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.
    method : {"kmeans", "gmm"}
        Which method's cluster raster to reclassify.
    k : int
        Number of clusters for this method/year.
    mapping_df : pandas.DataFrame
        Output of `build_cluster_to_class_mapping` for this
        method/year.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        (tif_path, png_path) for the written reclassified outputs.
    """
    tif_template = config["results"]["step_2_8"][f"{method}_full_raster_tif_template"]
    tif_filename = tif_template.replace("{year}", str(year)).replace("{k}", str(k))
    original_tif_path = Path(output_dir) / tif_filename

    label_raster, profile = load_cluster_raster(original_tif_path)

    lookup = build_cluster_to_reclass_lookup(mapping_df, k)
    reclass_raster = apply_reclassification(label_raster, lookup)

    tif_out_name, png_out_name = resolve_reclass_filenames(config, year, method)
    tif_path = write_reclass_geotiff(
        reclass_raster, profile, Path(output_dir) / tif_out_name
    )
    method_label = "GMM" if method == "gmm" else "K-means"
    png_path = plot_reclass_map(
        reclass_raster, method_label, year, Path(output_dir) / png_out_name
    )

    return tif_path, png_path


# ---------------------------------------------------------------------------
# Top-level orchestration (both methods)
# ---------------------------------------------------------------------------

def run_step_3_6(config, output_dir, year, step_3_1_results,
                  kmeans_mapping_df, gmm_mapping_df, update_report_fn):
    """
    Execute Step 3.6 (reclassification stage): apply the
    cluster-to-class mapping to produce reclassified GeoTIFF and PNG
    outputs for both K-means and GMM cluster rasters.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict. Expected keys:
        - `results.step_2_8.kmeans_full_raster_tif_template`
        - `results.step_2_8.gmm_full_raster_tif_template`
        - `results.step_3_6.kmeans_reclass_tif_template`
        - `results.step_3_6.kmeans_reclass_png_template`
        - `results.step_3_6.gmm_reclass_tif_template`
        - `results.step_3_6.gmm_reclass_png_template`
        - `report` : report filename
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.
    step_3_1_results : dict
        Output of `run_step_3_1`, used to read "kmeans_k" and
        "gmm_k".
    kmeans_mapping_df : pandas.DataFrame
        Output of `build_cluster_to_class_mapping` for K-means.
    gmm_mapping_df : pandas.DataFrame
        Output of `build_cluster_to_class_mapping` for GMM.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict
        Dictionary with keys "kmeans_tif", "kmeans_png", "gmm_tif",
        "gmm_png" (absolute paths as strings).
    """
    print(f'{"="*10} Step 3.6 — reclassification {"="*10}')

    kmeans_k = step_3_1_results["kmeans_k"]
    # gmm_k = step_3_1_results["gmm_k"]

    kmeans_tif, kmeans_png = run_reclassification_for_method(config, output_dir, year, "kmeans", kmeans_k, kmeans_mapping_df)
    # gmm_tif, gmm_png = run_reclassification_for_method(config, output_dir, year, "gmm", gmm_k, gmm_mapping_df)

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    step_report = {
        year: {
            "kmeans_reclass_tif": str(kmeans_tif),
            "kmeans_reclass_png": str(kmeans_png),
            # "gmm_reclass_tif": str(gmm_tif),
            # "gmm_reclass_png": str(gmm_png),
            "reclass_code_map": RECLASS_CODE_MAP,
        }
    }
    update_report_fn(report_path, "step_3_6_reclassification", step_report)

    print(f"Step 3.6 complete — year {year}:")
    print(f"K-means reclass: {kmeans_tif}, {kmeans_png}")
    # print(f"GMM reclass: {gmm_tif}, {gmm_png}")

    return {
        "kmeans_tif": str(kmeans_tif),
        "kmeans_png": str(kmeans_png),
        # "gmm_tif": str(gmm_tif),
        # "gmm_png": str(gmm_png),
    }