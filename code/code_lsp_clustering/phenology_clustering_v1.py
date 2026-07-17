
import json
import numpy as np
import rasterio
from pathlib import Path
import pickle
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for HPC/headless runs
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
from hdbscan.validity import validity_index


def build_config_path(config_filename):
    cwd = Path(os.getcwd())
    config_path = cwd / "configs" / config_filename
    return config_path


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def resolve_paths(config):
    root = Path(config["data_path"])
    for layer in config["metric_layers"]:
        p = Path(layer["path"])
        layer["path"] = str(p) if p.is_absolute() else str(root / p)
    for layer in config["qa_layers"]:
        p = Path(layer["path"])
        layer["path"] = str(p) if p.is_absolute() else str(root / p)
    return config


def build_output_dir(config):
    script_path = Path(config["script_path"])
    results_dir = script_path / "results" / config["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_metric_stack(metric_layers):
    bands = []
    names = []
    profile = None
    for layer in metric_layers:
        with rasterio.open(layer["path"]) as src:
            bands.append(src.read(1).astype(np.float32))
            names.append(layer["name"])
            if profile is None:
                profile = src.profile
    stack = np.stack(bands, axis=-1)
    return stack, names, profile


def build_qa_mask(qa_layers, shape, logic="AND"):
    masks = []
    for layer in qa_layers:
        with rasterio.open(layer["path"]) as src:
            qa = src.read(1)
        low, high = layer["valid_range"]
        masks.append((qa >= low) & (qa <= high))

    if logic == "AND":
        mask = np.logical_and.reduce(masks)
    elif logic == "OR":
        mask = np.logical_or.reduce(masks)
    else:
        raise ValueError(f"Unsupported qa_logic: {logic}")
    return mask


def stratified_grid_sample(mask, cell_size_px, samples_per_cell, random_seed):
    # 1 pixel sampled in each cell_size_px x cell_size_px block
    rng = np.random.default_rng(random_seed)
    n_rows, n_cols = mask.shape
    sampled_rows, sampled_cols = [], []

    for r0 in range(0, n_rows, cell_size_px):
        for c0 in range(0, n_cols, cell_size_px):
            r1 = min(r0 + cell_size_px, n_rows)
            c1 = min(c0 + cell_size_px, n_cols)
            cell_mask = mask[r0:r1, c0:c1]
            valid_r, valid_c = np.where(cell_mask)
            if len(valid_r) == 0:
                continue
            n_take = min(samples_per_cell, len(valid_r))
            idx = rng.choice(len(valid_r), size=n_take, replace=False)
            sampled_rows.extend(valid_r[idx] + r0) # shift to global coordinates
            sampled_cols.extend(valid_c[idx] + c0)

    return np.array(sampled_rows), np.array(sampled_cols)


def preprocess(X, feature_names, scaling="standard"):
    X = X.copy()
 
    if scaling == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "none":
        scaler = None
        X_scaled = X
    else:
        raise ValueError(f"Unsupported scaling: {scaling}")

    return X_scaled, scaler, feature_names


def run_pca_diagnostic(X_scaled, feature_names, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    corr_matrix = np.corrcoef(X_scaled, rowvar=False)
    return {
        "pca_model": pca,
        "loadings": pca.components_.T,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "feature_correlation_matrix": corr_matrix,
        "feature_names": feature_names
    }


def plot_pca_diagnostics(pca_results, out_dir, filename):
    explained_var = pca_results["explained_variance_ratio"]
    loadings = pca_results["loadings"]
    feature_names = pca_results["feature_names"]

    cumulative_var = np.cumsum(explained_var)
    n_components = len(explained_var)

    weighted_loadings = np.abs(loadings) * explained_var[np.newaxis, :]
    feature_importance = weighted_loadings.sum(axis=1)

    order = np.argsort(feature_importance)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_importance = feature_importance[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    x = np.arange(1, n_components + 1)
    ax1.plot(x, explained_var, marker="o", label="Individual", color="tab:blue")
    ax1.plot(x, cumulative_var, marker="s", label="Cumulative", color="tab:orange")
    ax1.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Scree Plot")
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.barh(sorted_names, sorted_importance, color="tab:green")
    ax2.set_xlabel("Variance-Weighted Absolute Loading (summed across PCs)")
    ax2.set_title("Feature Importance (PCA-derived)")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PCA diagnostic plot to {out_path}")
    return out_path


def evaluate_kmeans_gmm(X, k_range, method="kmeans", covariance_type="full"):
    results = []
    for k in range(k_range[0], k_range[1] + 1):
        print(f"{method} {k}")
        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(X)
            score_metric = model.inertia_
        elif method == "gmm":
            model = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
            labels = model.fit_predict(X)
            score_metric = model.bic(X)
        else:
            raise ValueError(method)

        sil = silhouette_score(X, labels, sample_size=min(10000, len(X)), random_state=42)
        dbi = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

        results.append({
            "method": method, "k": k, "inertia_or_bic": score_metric,
            "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": ch,
            "model": model, "labels": labels
        })
    return results


def evaluate_hdbscan(X, min_cluster_sizes, min_samples=None):
    results = []
    for mcs in min_cluster_sizes:
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples)
        labels = model.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = float(np.mean(labels == -1))
        try:
            dbcv = validity_index(X.astype(np.float64), labels)
        except Exception:
            dbcv = np.nan
        results.append({
            "method": "hdbscan", "min_cluster_size": mcs, "n_clusters": n_clusters,
            "noise_fraction": noise_frac, "dbcv": dbcv,
            "model": model, "labels": labels
        })
    return results


def plot_kmeans_validity(csv_path, out_dir, filename):
    df = pd.read_csv(csv_path)
    df = df.sort_values("k")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(df["k"], df["inertia_or_bic"], marker="o", color="tab:blue")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Plot (Inertia vs. k)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["k"], df["silhouette"], marker="o", color="tab:green")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score (higher=better)")
    ax.set_title("Silhouette vs. k")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["k"], df["davies_bouldin"], marker="o", color="tab:red")
    ax.set_xlabel("k")
    ax.set_ylabel("Davies-Bouldin Index (lower=better)")
    ax.set_title("Davies-Bouldin vs. k")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["k"], df["calinski_harabasz"], marker="o", color="tab:purple")
    ax.set_xlabel("k")
    ax.set_ylabel("Calinski-Harabasz Score (higher=better)")
    ax.set_title("Calinski-Harabasz vs. k")
    ax.grid(alpha=0.3)

    fig.suptitle("K-Means Cluster Validity Diagnostics", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved k-means validity diagnostics to {out_path}")
    return out_path


def plot_kmeans_normalized_consensus(csv_path, out_dir, filename):
    df = pd.read_csv(csv_path).sort_values("k")

    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    elbow_norm = 1 - minmax(df['inertia_or_bic'])
    sil_norm = minmax(df["silhouette"])
    dbi_norm = 1 - minmax(df["davies_bouldin"])
    ch_norm = minmax(df["calinski_harabasz"])

    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(df["k"], elbow_norm, marker="o", label="Elbow (norm., inverted)")
    ax.plot(df["k"], sil_norm, marker="o", label="Silhouette (norm.)")
    ax.plot(df["k"], dbi_norm, marker="o", label="Davies-Bouldin (norm., inverted)")
    ax.plot(df["k"], ch_norm, marker="o", label="Calinski-Harabasz (norm.)")
    ax.set_xlabel("k")
    ax.set_ylabel("Normalized score (higher = better, all metrics)")
    ax.set_title("Normalized Validity Metrics — Consensus k")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved consensus plot to {out_path}")
    return out_path


def plot_gmm_validity(csv_path, out_dir, filename):
    df = pd.read_csv(csv_path)
    df = df.sort_values("k")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: BIC ---
    ax = axes[0, 0]
    ax.plot(df["k"], df["inertia_or_bic"], marker="o", color="tab:blue")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.set_title("BIC vs. k")
    ax.grid(alpha=0.3)

    # --- Panel 2: Silhouette ---
    ax = axes[0, 1]
    ax.plot(df["k"], df["silhouette"], marker="o", color="tab:green")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette vs. k")
    ax.grid(alpha=0.3)

    # --- Panel 3: Davies-Bouldin ---
    ax = axes[1, 0]
    ax.plot(df["k"], df["davies_bouldin"], marker="o", color="tab:red")
    ax.set_xlabel("k")
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_title("Davies-Bouldin vs. k")
    ax.grid(alpha=0.3)

    # --- Panel 4: Calinski-Harabasz ---
    ax = axes[1, 1]
    ax.plot(df["k"], df["calinski_harabasz"], marker="o", color="tab:purple")
    ax.set_xlabel("k")
    ax.set_ylabel("Calinski-Harabasz Score (higher=better)")
    ax.set_title("Calinski-Harabasz vs. k")
    ax.grid(alpha=0.3)

    fig.suptitle("GMM Cluster Validity Diagnostics", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved GMM validity diagnostics to {out_path}")
    return out_path


def plot_gmm_normalized_consensus(csv_path, out_dir, filename):
    df = pd.read_csv(csv_path).sort_values("k")

    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    bic_norm = 1 - minmax(df["inertia_or_bic"])  # invert: lower BIC = better
    sil_norm = minmax(df["silhouette"])
    dbi_norm = 1 - minmax(df["davies_bouldin"])
    ch_norm = minmax(df["calinski_harabasz"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["k"], bic_norm, marker="o", label="BIC (norm., inverted)")
    ax.plot(df["k"], sil_norm, marker="o", label="Silhouette (norm.)")
    ax.plot(df["k"], dbi_norm, marker="o", label="Davies-Bouldin (norm., inverted)")
    ax.plot(df["k"], ch_norm, marker="o", label="Calinski-Harabasz (norm.)")
    ax.set_xlabel("k")
    ax.set_ylabel("Normalized score (higher = better, all metrics)")
    ax.set_title("Normalized Validity Metrics — Consensus k (GMM)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved GMM consensus plot to {out_path}")
    return out_path


def plot_hdbscan_validity(csv_path, out_dir, filename):
    df = pd.read_csv(csv_path)
    df = df.sort_values("min_cluster_size")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: number of clusters found vs. min_cluster_size ---
    ax = axes[0]
    ax.plot(df["min_cluster_size"], df["n_clusters"], marker="o", color="tab:blue")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("Number of Clusters Found")
    ax.set_title("Cluster Count vs. min_cluster_size")
    ax.grid(alpha=0.3)

    # --- Panel 2: noise fraction vs. min_cluster_size ---
    ax = axes[1]
    ax.plot(df["min_cluster_size"], df["noise_fraction"], marker="o", color="tab:orange")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("Noise Fraction (unassigned pixels)")
    ax.set_title("Noise Fraction vs. min_cluster_size")
    ax.grid(alpha=0.3)

    # --- Panel 3: DBCV (relative validity) vs. min_cluster_size ---
    ax = axes[2]
    ax.plot(df["min_cluster_size"], df["dbcv"], marker="o", color="tab:green")
    if df["dbcv"].notna().any():
        best_mcs = df.loc[df["dbcv"].idxmax(), "min_cluster_size"]
        ax.axvline(best_mcs, color="gray", linestyle="--", linewidth=1, label=f"best mcs={best_mcs}")
        ax.legend()
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("DBCV (higher=better)")
    ax.set_title("DBCV vs. min_cluster_size")
    ax.grid(alpha=0.3)

    fig.suptitle("HDBSCAN Cluster Validity Diagnostics", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved HDBSCAN validity diagnostics to {out_path}")
    if df["dbcv"].notna().any():
        print(f"DBCV-best min_cluster_size: {best_mcs}")
    return out_path


def step_header(step):
    print(f"\n[{step}] {'='*40}")


def main(config_filename):
    STEP = "step00"
    step_header(STEP)
    config_path = build_config_path(config_filename)
    print(f"{config_path=}")
    config = load_config(config_path)
    config = resolve_paths(config)
    out_dir = build_output_dir(config)
    print(f"{out_dir=}")
    config_results = config["results"]

    # 1. load metric stack
    STEP = "step01"
    step_header(STEP)
    metrics_stack, feature_names, profile = load_metric_stack(config["metric_layers"])
    print(f"{metrics_stack.shape=}")
    print(f"{feature_names=}")
    # print(f"{profile=}")

    # 2. QA mask (strict AND across layers)
    STEP = "step02"
    step_header(STEP)
    qa_mask = build_qa_mask(config["qa_layers"], metrics_stack.shape[:2], logic=config["qa_logic"])
    nodata_mask = np.any(np.isnan(metrics_stack), axis=-1)
    qa_mask &= ~nodata_mask # NOTE and qa mask and not NAN data mask
    print(f"{qa_mask.shape=}")

    # 3. stratified grid sample
    STEP = "step03"
    step_header(STEP)
    config_sampling = config["sampling"]
    sampled_rows, sampled_cols = stratified_grid_sample(qa_mask, config_sampling["cell_size_px"], config_sampling["samples_per_cell"], config_sampling["random_seed"])
    X_sampled_metrics_raw = metrics_stack[sampled_rows, sampled_cols, :]
    print(f"Sampled {X_sampled_metrics_raw.shape[0]} pixels from {qa_mask.sum()} valid pixels ({qa_mask.sum() / qa_mask.size:.1%} of tile is valid).")

    if config["output"]["save_sample_table"]:
        df = pd.DataFrame(X_sampled_metrics_raw, columns=feature_names)
        df["row"] = sampled_rows
        df["col"] = sampled_cols
        file_name = out_dir / config_results[STEP]["stratified_grid_samples"]
        df.to_csv(file_name, index=False)
        print(f"saved: {file_name}")

    # 4. preprocessing
    STEP = "step04"
    step_header(STEP)
    config_preprocessing = config["preprocessing"]
    X_scaled, scaler, feature_names_out = preprocess(X_sampled_metrics_raw, feature_names, scaling=config_preprocessing["scaling"])
    if config["output"]["save_scaler"] and scaler is not None:
        file_name = out_dir / config_results[STEP]["scaler"]
        with open(file_name, "wb") as f:
            pickle.dump(scaler, f)
        print(f"saved: {file_name}")

    # 5. PCA diagnostic
    STEP = "step05"
    step_header(STEP)
    pca_results = None
    if config["pca"]["run"]:
        pca_results = run_pca_diagnostic(X_scaled, feature_names_out, config["pca"]["n_components"])

        corr_path = out_dir / config_results[STEP]["feature_correlation_matrix"]
        np.savetxt(corr_path, pca_results["feature_correlation_matrix"], delimiter=",", header=",".join(feature_names_out), comments="")
        print(f"saved: {corr_path}")

        plot_pca_diagnostics(pca_results, out_dir, config_results[STEP]["pca_feature_importance_plot"])
        print("Explained variance ratio by PC:", pca_results["explained_variance_ratio"])

    # 6. clustering sweeps
    STEP = "step06"
    step_header(STEP)
    config_clustering = config["clustering"]
    # all_results = {}
    # if "kmeans" in config_clustering["methods"]:
    #     all_results["kmeans"] = evaluate_kmeans_gmm(X_scaled, config_clustering["kmeans"]["k_range"], method="kmeans")
    # if "gmm" in config_clustering["methods"]:
    #     all_results["gmm"] = evaluate_kmeans_gmm(X_scaled, config_clustering["gmm"]["k_range"], method="gmm", covariance_type=config_clustering["gmm"]["covariance_type"])
    # if "hdbscan" in config_clustering["methods"]:
    #     all_results["hdbscan"] = evaluate_hdbscan(X_scaled, config_clustering["hdbscan"]["min_cluster_sizes"], config_clustering["hdbscan"]["min_samples"])

    # 7. save summary tables
    STEP = "step07"
    step_header(STEP)
    # for method, results in all_results.items():
    #     summary = [{k: v for k, v in r.items() if k not in ("model", "labels")} for r in results]
    #     summary_key = f"{method}_validity_summary"
    #     summary_path = out_dir / config_results[STEP][summary_key]
    #     pd.DataFrame(summary).to_csv(summary_path, index=False)
    #     print(f"saved: {summary_path}")

    if "kmeans" in config_clustering["methods"]:
        summary_path = out_dir / config_results[STEP]["kmeans_validity_summary"]
        plot_kmeans_validity(summary_path, out_dir, config_results[STEP]["kmeans_validity_plot"])
        plot_kmeans_normalized_consensus(summary_path, out_dir, config_results[STEP]["kmeans_consensus_plot"])

    if "gmm" in config_clustering["methods"]:
        summary_path = out_dir / config_results[STEP]["gmm_validity_summary"]
        plot_gmm_validity(summary_path, out_dir, config_results[STEP]["gmm_validity_plot"])
        plot_gmm_normalized_consensus(summary_path, out_dir, config_results[STEP]["gmm_consensus_plot"])

    if "hdbscan" in config_clustering["methods"]:
        summary_path = out_dir / config_results[STEP]["hdbscan_validity_summary"]
        plot_hdbscan_validity(summary_path, out_dir, config_results[STEP]["hdbscan_validity_plot"])

    # # 8. pickle full results for later full-raster prediction
    # STEP = "step08"
    # step_header(STEP)
    # if config["output"]["save_model"]:
    #     file_name = out_dir / config_results[STEP]["clustering_results"]
    #     with open(file_name, "wb") as f:
    #         pickle.dump(all_results, f)
    #     print(f"saved: {file_name}")

    print('done')

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise ValueError("Usage: python phenology_clustering.py <config_filename.json>")
    main(sys.argv[1])