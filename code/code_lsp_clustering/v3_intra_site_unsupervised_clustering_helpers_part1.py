import json
import os
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rasterio
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import joblib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch



# =============================================================================
# Step 1.1 — Load Moon et al. PlanetScope LSP timing metrics
# =============================================================================
"""
Step 1.1 — Load Moon et al. PlanetScope LSP timing metrics for target
sites and years.

Loads the 7 raw timing metric GeoTIFFs per site per year into stacked
arrays with associated raster profiles. Metric layer paths and QA layer
paths follow the naming convention:

    {order}_{year}_{metric}.tif

where `order` is a leading integer, `{year}` is the 4-digit year, and
`metric` is the metric short name (e.g., OGI, 50PCGI, OGMx).

Fill value (config['qa']['fill_value']) in metric layers is converted
to NaN at load time.

QA layer loading is handled in Step 1.2.
"""

# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

def build_config_path(config_filename, config_dir="configs"):
    """
    Build absolute path to a config file located under a config directory
    within the current working directory.

    Parameters
    ----------
    config_filename : str
        Filename of the config JSON
    config_dir : str, optional
        Subdirectory (relative to current working directory) where config
        files live. Defaults to "configs".

    Returns
    -------
    pathlib.Path
        Absolute path to the config file.
    """
    cwd = Path(os.getcwd())
    return cwd / config_dir / config_filename


def load_config(config_path):
    """
    Load a JSON config file from disk.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the JSON config file.

    Returns
    -------
    dict
        Parsed config as a Python dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def compute_run_name(config):
    """
    Compute the run name as the concatenation of site_id, method, and
    run_number using underscores.

    Parameters
    ----------
    config : dict
        Config dict containing keys `site_id`, `method`, `run_number`.

    Returns
    -------
    str
        Run name of the form "{site_id}_{method}_{run_number}".
    """
    site_id = config["site_id"]
    method = config["method"]
    run_number = config["run_number"]
    return f"{site_id}_{method}_{run_number}"


def build_output_dir(config):
    """
    Create and return the output results directory for the current run.

    The directory is located under `{output_path}/{run_name}` where
    `run_name` is derived from site_id, method, and run_number.

    Parameters
    ----------
    config : dict
        Config dict containing keys `output_path`, `site_id`, `method`,
        `run_number`.

    Returns
    -------
    pathlib.Path
        Absolute path to the output results directory (created if
        missing).
    """
    output_path = Path(config["output_path"])
    run_name = compute_run_name(config)
    results_dir = output_path / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ---------------------------------------------------------------------------
# Path parsing and resolution
# ---------------------------------------------------------------------------

def parse_layer_filename(filename):
    """
    Parse a Moon et al. LSP layer filename into (order, year, metric_name).

    The filename convention is `{order}_{year}_{metric}.tif`, parsed by
    splitting on underscores. The metric name may itself contain digits
    (e.g., "50PCGI").

    Parameters
    ----------
    filename : str
        Filename to parse. May be a full path or bare filename; only the
        basename is used.

    Returns
    -------
    tuple[int, str, str]
        Tuple of (order, year, metric_name) where:
        - order is the leading integer (int),
        - year is the 4-digit year (str),
        - metric_name is the metric short name with ".tif" stripped (str).

    Raises
    ------
    ValueError
        If the filename does not match the expected 3-part convention.
    """
    basename = Path(filename).name
    parts = basename.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Filename '{basename}' does not match expected format "
            f"'{{order}}_{{year}}_{{metric}}.tif' (got {len(parts)} parts)."
        )
    order_str, year_str, metric_with_ext = parts
    try:
        order = int(order_str)
    except ValueError:
        raise ValueError(
            f"Leading token '{order_str}' in '{basename}' is not an integer."
        )
    if not (year_str.isdigit() and len(year_str) == 4):
        raise ValueError(
            f"Year token '{year_str}' in '{basename}' is not a 4-digit year."
        )
    metric_name = metric_with_ext.replace(".tif", "")
    return order, year_str, metric_name


def resolve_layer_paths_for_year(layer_entries, config, year):
    """
    Expand {site_name} and {year} placeholders and prefix data root for
    a list of layer entries, for one specific year.

    Assumes data directory layout `{data_path}/{site_name}/{year}/{filename}`
    (with {site_name} and {year} substituted from config's `data_path`
    template).

    Parameters
    ----------
    layer_entries : list[dict]
        List of layer entries from config, each having a "path" key with
        a filename that may contain the string "{year}" as year placeholder.
    config : dict
        Full config dict, containing "data_path" (a template that may
        include "{site_name}" and "{year}" placeholders) and "site_name".
    year : str
        4-digit year string used to substitute "{year}" placeholders.

    Returns
    -------
    list[dict]
        New list of layer entries with "path" set to the absolute resolved
        path. Original entries are not modified.
    """
    root = Path(config['data_path'])
    root = Path(str(root).replace("{site_name}", config['site_name']))
    root = Path(str(root).replace("{year}", str(year)))

    resolved = []
    for entry in layer_entries:
        new_entry = dict(entry)
        raw_path = Path(new_entry["path"])
        resolved_path = str(root / raw_path).replace("{year}", str(year))
        new_entry["path"] = resolved_path
        resolved.append(new_entry)
    return resolved


# ---------------------------------------------------------------------------
# Metric stack loading (single year)
# ---------------------------------------------------------------------------

def _assert_profiles_match(profile_a, profile_b, layer_a_path, layer_b_path):
    """
    Assert that two rasterio profiles have matching shape, CRS, and
    transform.

    Parameters
    ----------
    profile_a, profile_b : dict-like
        Rasterio profile dictionaries.
    layer_a_path, layer_b_path : str
        File paths corresponding to the two profiles, used in error
        messages.

    Raises
    ------
    ValueError
        If width, height, CRS, or transform differ between the profiles.
    """
    checks = [
        ("width", profile_a["width"], profile_b["width"]),
        ("height", profile_a["height"], profile_b["height"]),
        ("crs", profile_a["crs"], profile_b["crs"]),
        ("transform", profile_a["transform"], profile_b["transform"]),
    ]
    mismatches = [name for name, a, b in checks if a != b]
    if mismatches:
        raise ValueError(
            f"Profile mismatch between '{layer_a_path}' and '{layer_b_path}': "
            f"{mismatches} differ."
        )


def load_metric_stack(metric_layers, config):
    """
    Load a set of Moon et al. LSP timing-metric GeoTIFFs for one site and
    one year into a stacked float32 array.

    Layers are sorted by their `order` integer (parsed from filename)
    before stacking to guarantee a consistent feature-column order across
    calls. All layers must share the same width, height, CRS, and
    transform; a mismatch raises ValueError. Values equal to
    `config['qa']['fill_value']` are converted to NaN.

    Parameters
    ----------
    metric_layers : list[dict]
        List of layer entries, each containing at minimum a resolved
        absolute "path" to a GeoTIFF.
    config : dict
        Full config dict; used to read `config['qa']['fill_value']`.

    Returns
    -------
    tuple[np.ndarray, list[str], rasterio.profiles.Profile]
        - stack : ndarray of shape (H, W, F), dtype float32, with
          fill values replaced by NaN. F is the number of layers,
          ordered by the parsed `order` integer.
        - metric_names : list of str of length F, matching the stacking
          order; names are taken directly from parsed filenames.
        - profile : the rasterio Profile of the first layer (all layers
          share the same profile).

    Raises
    ------
    FileNotFoundError
        If any layer path does not exist on disk.
    ValueError
        If layers disagree on width, height, CRS, or transform, or if a
        filename does not match the expected convention.
    """
    parsed = []
    for entry in metric_layers:
        path = entry["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Metric layer not found: {path}")
        order, year_str, metric_name = parse_layer_filename(path)
        parsed.append((order, year_str, metric_name, path))
    parsed.sort(key=lambda item: item[0])

    bands = []
    metric_names = []
    profile = None
    reference_path = None

    for order, year_str, metric_name, path in parsed:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            layer_profile = src.profile

        if profile is None:
            profile = layer_profile
            reference_path = path
        else:
            _assert_profiles_match(profile, layer_profile, reference_path, path)

        band[band == config['qa']['fill_value']] = np.nan
        bands.append(band)
        metric_names.append(metric_name)

    stack = np.stack(bands, axis=-1)
    return stack, metric_names, profile


# ---------------------------------------------------------------------------
# Metric stack loading (multi-year)
# ---------------------------------------------------------------------------

def _assert_cross_year_profiles_match(profiles_by_year):
    """
    Assert that raster profiles are consistent across years.

    All profiles must share the same width, height, CRS, and transform.

    Parameters
    ----------
    profiles_by_year : dict[str, rasterio.profiles.Profile]
        Mapping from year string to raster profile of that year's stack.

    Raises
    ------
    ValueError
        If any two years disagree on width, height, CRS, or transform.
    """
    years = list(profiles_by_year.keys())
    if len(years) < 2:
        return
    reference_year = years[0]
    reference_profile = profiles_by_year[reference_year]
    for year in years[1:]:
        _assert_profiles_match(
            reference_profile,
            profiles_by_year[year],
            layer_a_path=f"year={reference_year}",
            layer_b_path=f"year={year}",
        )


def load_metric_stack_multi(config):
    """
    Load Moon et al. LSP timing-metric stacks for one site across
    multiple years, keyed by year.

    Wraps `load_metric_stack` in a per-year loop, resolving layer paths
    for each year from the config's `data_path`, `years`, and
    `metric_layers` fields. Verifies that raster profiles are consistent
    across years and raises if not.

    Parameters
    ----------
    config : dict
        Config dict containing at minimum:
        - "data_path" : str, root data directory template
        - "years"     : list[str], target years to load
        - "metric_layers" : list[dict], each with a "path" template that
          may contain "{year}" placeholder

    Returns
    -------
    dict[str, tuple[np.ndarray, list[str], rasterio.profiles.Profile]]
        Mapping from year (str) to (stack, metric_names, profile) tuple
        as produced by `load_metric_stack`.

    Raises
    ------
    FileNotFoundError
        If any metric layer file is missing for any requested year.
    ValueError
        If layers within a year disagree on profile, if years disagree
        on profile, or if a filename does not match the expected
        convention.
    """
    years = [str(y) for y in config["years"]]
    metric_layers_template = config["metric_layers"]

    stacks_by_year = {}
    profiles_by_year = {}
    for year in years:
        resolved_layers = resolve_layer_paths_for_year(
            metric_layers_template, config, year
        )
        stack, metric_names, profile = load_metric_stack(
            resolved_layers, config
        )
        stacks_by_year[year] = (stack, metric_names, profile)
        profiles_by_year[year] = profile

    _assert_cross_year_profiles_match(profiles_by_year)
    return stacks_by_year


# ---------------------------------------------------------------------------
# Report file utilities
# ---------------------------------------------------------------------------

def initialize_report(config, output_dir):
    """
    Initialize (or reuse) the run report JSON file for this pipeline
    execution.

    If the report file already exists at the target location, it is
    loaded and returned unmodified so downstream steps can append. If
    it does not exist, a new report is created with run metadata (site
    name, method, run number, run name, timestamps) and no step
    entries.

    Parameters
    ----------
    config : dict
        Config dict containing at minimum "site_name", "method",
        "run_number", "report" (filename).
    output_dir : str or pathlib.Path
        Directory where the report file lives (typically the results
        directory returned by `build_output_dir`).

    Returns
    -------
    pathlib.Path
        Absolute path to the report file. The file is guaranteed to
        exist after this call.
    """

    # NOTE TODO report per year, doesn't work for multi year
    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    if report_path.exists():
        return report_path

    now_iso = datetime.now(timezone.utc).isoformat()
    initial = {
        "run_metadata": {
            "site_name": config["site_name"],
            "method": config["method"],
            "run_number": config["run_number"],
            "run_name": compute_run_name(config),
            "created_at": now_iso,
            "updated_at": now_iso,
        },
    }
    with open(report_path, "w") as f:
        json.dump(initial, f, indent=2)
    return report_path


def update_report(report_path, section_name, content):
    """
    Read-modify-write update to the run report JSON file.

    Adds or overwrites a top-level section keyed by `section_name` and
    updates the `updated_at` timestamp under `run_metadata`.

    Parameters
    ----------
    report_path : str or pathlib.Path
        Path to the report file. Must exist (call `initialize_report`
        first).
    section_name : str
        Top-level key under which `content` is stored. Overwrites any
        existing section with the same name.
    content : dict
        JSON-serializable content for this section.

    Returns
    -------
    dict
        The full report dict after the update, for optional inspection.

    Raises
    ------
    FileNotFoundError
        If `report_path` does not exist.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report file not found: {report_path}. "
            f"Call initialize_report first."
        )
    with open(report_path, "r") as f:
        report = json.load(f)
    report[section_name] = content
    report["run_metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def run_step_1_1(config, output_dir):
    """
    Execute Step 1.1: load PlanetScope LSP timing-metric stacks for
    all configured years.

    Initializes the run report if not already present, then loads the
    7 raw timing-metric GeoTIFFs per configured year (via
    `load_metric_stack_multi`) into stacked arrays, validating shape,
    CRS, and transform consistency both within each year and across
    years.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory (as returned by `build_output_dir`).

    Returns
    -------
    dict[str, tuple[np.ndarray, list[str], rasterio.profiles.Profile]]
        Per-year metric stacks keyed by year string.

    Raises
    ------
    FileNotFoundError
        If any metric layer file is missing for any requested year.
    ValueError
        If metric layers within a year, or across years, disagree on
        shape, CRS, or transform, or if a filename does not match the
        expected convention.
    """
    print(f'{"="*10} Step 1.1 {"="*10}')
    initialize_report(config, output_dir)
    stacks_by_year = load_metric_stack_multi(config)

    for year, (stack, metric_names, profile) in stacks_by_year.items():
        print(f"{year}: shape={stack.shape}, features={metric_names}")

    return stacks_by_year


# =============================================================================
# Step 1.2 — Build per-year QA masks
# =============================================================================
"""
Step 1.2 — Build per-year QA masks for Moon et al. PlanetScope LSP
data.

QA masks are boolean arrays identifying pixels that are valid for
downstream clustering analysis. Each mask is constructed from one or
more QA layers, combined via the config's `qa.logic` field ("AND" or
"OR"). Pixels with fill value `qa.fill_value` in any QA layer are
treated as invalid.

Retention statistics for each year are computed and added to the run
report JSON.
"""

# ---------------------------------------------------------------------------
# QA layer loading
# ---------------------------------------------------------------------------

def load_qa_layers(qa_layers, config):
    """
    Load QA GeoTIFF layers for one site and one year into a dict of
    float32 arrays keyed by QA metric name.

    Fill values (config['qa']['fill_value']) are converted to NaN so
    downstream mask logic can treat missing data uniformly via NaN
    checks.

    Parameters
    ----------
    qa_layers : list[dict]
        List of resolved QA layer entries, each with "path" and
        "valid_range".
    config : dict
        Full config dict; used to read `config['qa']['fill_value']`.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from QA metric name (parsed from filename) to the
        corresponding float32 array with fill values converted to NaN.

    Raises
    ------
    FileNotFoundError
        If any QA layer file is missing.
    ValueError
        If any filename does not match the expected convention.
    """
    qa_arrays = {}
    for entry in qa_layers:
        path = entry["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"QA layer not found: {path}")
        _, _, qa_name = parse_layer_filename(path)
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        arr[arr == config['qa']['fill_value']] = np.nan
        qa_arrays[qa_name] = arr
    return qa_arrays


# ---------------------------------------------------------------------------
# QA mask construction
# ---------------------------------------------------------------------------

def build_qa_mask(qa_arrays, qa_layers, logic="AND"):
    """
    Construct a boolean QA mask from loaded QA arrays and their
    configured valid ranges.

    A pixel passes an individual QA layer if the value is non-NaN and
    within the layer's `valid_range` (inclusive on both bounds). Layer
    results are combined via `logic`:

    - "AND" : pixel must pass every layer to be valid
    - "OR"  : pixel need pass only one layer

    Parameters
    ----------
    qa_arrays : dict[str, np.ndarray]
        Mapping from QA metric name to float32 array (as produced by
        `load_qa_layers`).
    qa_layers : list[dict]
        List of resolved QA layer entries with "path" and
        "valid_range".
    logic : {"AND", "OR"}
        Combination logic. Defaults to "AND".

    Returns
    -------
    np.ndarray
        Boolean array of shape (H, W). True where the pixel passes the
        combined QA criteria.

    Raises
    ------
    ValueError
        If `logic` is not "AND" or "OR", or if a QA layer's parsed
        name is missing from `qa_arrays`.
    """
    per_layer_masks = []
    for entry in qa_layers:
        _, _, qa_name = parse_layer_filename(entry["path"])
        if qa_name not in qa_arrays:
            raise ValueError(
                f"QA layer '{qa_name}' not found in loaded qa_arrays. "
                f"Available: {list(qa_arrays.keys())}"
            )
        arr = qa_arrays[qa_name]
        low, high = entry["valid_range"]
        layer_mask = (~np.isnan(arr)) & (arr >= low) & (arr <= high)
        per_layer_masks.append(layer_mask)

    if logic == "AND":
        return np.logical_and.reduce(per_layer_masks)
    if logic == "OR":
        return np.logical_or.reduce(per_layer_masks)
    raise ValueError(f"Unsupported qa_logic: '{logic}'. Expected 'AND' or 'OR'.")


# ---------------------------------------------------------------------------
# Retention reporting
# ---------------------------------------------------------------------------

def report_qa_retention(mask, year):
    """
    Compute QA retention statistics for a single-year mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean QA mask of shape (H, W).
    year : str
        4-digit year string associated with the mask.

    Returns
    -------
    dict
        Retention statistics for this year.
    """
    n_total = int(mask.size)
    n_valid = int(mask.sum())
    n_invalid = n_total - n_valid
    return {
        "year": str(year),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "valid_fraction": f"{n_valid / n_total if n_total > 0 else 0.0:.3f}",
        "invalid_fraction": f"{n_invalid / n_total if n_total > 0 else 0.0:.3f}",
    }


# ---------------------------------------------------------------------------
# Single-year and multi-year QA mask wrappers
# ---------------------------------------------------------------------------

def build_qa_mask_for_year(config, year):
    """
    Build a QA mask for one site and one year.

    Resolves QA layer paths from config's `data_path`, `qa.layers`, and
    `qa.logic` fields, loads the layers, and constructs the boolean
    mask.

    Parameters
    ----------
    config : dict
        Config dict containing "data_path", "qa.layers", "qa.logic".
    year : str
        4-digit year string.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W). True where the pixel passes QA.

    Raises
    ------
    FileNotFoundError
        If any QA layer file is missing for this year.
    ValueError
        If QA logic is invalid or a QA layer name mismatches.
    """
    resolved_qa_layers = resolve_layer_paths_for_year(config['qa']['layers'], config, year)
    qa_arrays = load_qa_layers(resolved_qa_layers, config)
    mask = build_qa_mask(
        qa_arrays, resolved_qa_layers, logic=config['qa']['logic']
    )
    return mask


def build_qa_masks_multi(config):
    """
    Build per-year QA masks for one site across multiple years.

    Each year has its own QA layers and its own resulting mask; masks
    are not intersected or unioned across years. Returned as a dict
    keyed by year string.

    Parameters
    ----------
    config : dict
        Config dict containing "years", "data_path", "qa.layers",
        "qa.logic".

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from year (str) to boolean mask of shape (H, W).

    Raises
    ------
    FileNotFoundError
        If any QA layer file is missing for any requested year.
    ValueError
        If QA logic is invalid or QA layer names mismatch within a
        year.
    """
    years = [str(y) for y in config["years"]]
    masks_by_year = {}
    for year in years:
        masks_by_year[year] = build_qa_mask_for_year(config, year)
        print(f"{year}: shape={masks_by_year[year].shape}")
    return masks_by_year


# ---------------------------------------------------------------------------
# Top-level orchestration for Step 1.2
# ---------------------------------------------------------------------------

def run_step_1_2(config, output_dir):
    """
    Execute Step 1.2: build per-year QA masks and update the run
    report with retention statistics.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory (as returned by `build_output_dir`).

    Returns
    -------
    dict[str, np.ndarray]
        Per-year QA masks keyed by year string. Also updates the
        report on disk.
    """
    print(f'{"="*10} Step 1.2 {"="*10}')

    report_path = initialize_report(config, output_dir)
    masks_by_year = build_qa_masks_multi(config)

    retention_by_year = {
        year: report_qa_retention(mask, year)
        for year, mask in masks_by_year.items()
    }
    update_report(report_path, "step_1_2_qa_retention", retention_by_year)
    return masks_by_year


# =============================================================================
# Step 1.3 — Stratified grid sampling
# =============================================================================
"""
Step 1.3 — Stratified grid sampling of QA-passing pixels.

For each year, divides the raster into square cells of
`cell_size_px × cell_size_px` and draws `samples_per_cell` random
QA-passing pixels from each cell. Cells with zero QA-passing pixels
are skipped.

Random seed is derived per year as `random_seed_base + int(year)` to
guarantee independent (but reproducible) samples across years.

Outputs one CSV per year with sample metadata and metric values.
"""

# ---------------------------------------------------------------------------
# Sampling core
# ---------------------------------------------------------------------------

def _cell_bounds(height, width, cell_size):
    """
    Yield (row_start, row_end, col_start, col_end) bounds for each
    grid cell tiling a raster of the given height and width.

    Parameters
    ----------
    height : int
        Raster height in pixels.
    width : int
        Raster width in pixels.
    cell_size : int
        Cell size in pixels (square cells).

    Yields
    ------
    tuple[int, int, int, int]
        (row_start, row_end, col_start, col_end) with end indices
        exclusive.
    """
    for row_start in range(0, height, cell_size):
        row_end = min(row_start + cell_size, height)
        for col_start in range(0, width, cell_size):
            col_end = min(col_start + cell_size, width)
            yield row_start, row_end, col_start, col_end


def stratified_grid_sample(mask, cell_size, samples_per_cell, seed):
    """
    Draw stratified grid samples of pixel indices from a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array of shape (H, W). True where sampling is
        permitted (i.e., QA-passing pixels).
    cell_size : int
        Cell side length in pixels.
    samples_per_cell : int
        Number of samples to draw per cell (capped by the number of
        True pixels in that cell).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        - rows : ndarray of shape (N,), int64, sampled row indices
        - cols : ndarray of shape (N,), int64, sampled col indices
        - stats : dict with cell coverage counts and total samples
          drawn.
    """
    rng = np.random.default_rng(seed)
    height, width = mask.shape

    rows_out = []
    cols_out = []
    n_cells_total = 0
    n_cells_sampled = 0
    n_cells_empty = 0

    for row_start, row_end, col_start, col_end in _cell_bounds(
        height, width, cell_size
    ):
        n_cells_total += 1
        cell = mask[row_start:row_end, col_start:col_end]
        true_local_idx = np.flatnonzero(cell)
        if true_local_idx.size == 0:
            n_cells_empty += 1
            continue

        n_draw = min(samples_per_cell, true_local_idx.size)
        chosen_local = rng.choice(true_local_idx, size=n_draw, replace=False)

        cell_width = col_end - col_start
        local_rows, local_cols = np.divmod(chosen_local, cell_width)
        rows_out.append(local_rows + row_start)
        cols_out.append(local_cols + col_start)
        n_cells_sampled += 1

    if rows_out:
        rows = np.concatenate(rows_out).astype(np.int64)
        cols = np.concatenate(cols_out).astype(np.int64)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)

    stats = {
        "n_cells_total": int(n_cells_total),
        "n_cells_sampled": int(n_cells_sampled),
        "n_cells_empty": int(n_cells_empty),
        "n_samples": int(rows.size),
    }
    return rows, cols, stats


# ---------------------------------------------------------------------------
# Pixel-to-coordinate conversion
# ---------------------------------------------------------------------------

def pixel_to_xy(rows, cols, transform):
    """
    Convert (row, col) pixel indices to projected (x, y) coordinates
    using a rasterio affine transform.

    Parameters
    ----------
    rows : np.ndarray
        Row indices, shape (N,).
    cols : np.ndarray
        Column indices, shape (N,).
    transform : affine.Affine
        Rasterio affine transform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, y) coordinate arrays of shape (N,), dtype float64.
    """
    xs, ys = rasterio.transform.xy(
        transform, rows.tolist(), cols.tolist(), offset="center"
    )
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sample table assembly
# ---------------------------------------------------------------------------

def _read_qa_values_at_pixels(qa_layer_entries, rows, cols, fill_value):
    """
    Read QA layer values at a set of pixel indices, returning a dict
    keyed by QA metric name.

    Parameters
    ----------
    qa_layer_entries : list[dict]
        Resolved QA layer entries (each with absolute "path").
    rows, cols : np.ndarray
        Pixel indices at which to sample values, shape (N,).
    fill_value : int or float
        Sentinel converted to NaN.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from QA metric name to sampled values, shape (N,).
    """
    values_by_name = {}
    for entry in qa_layer_entries:
        _, _, qa_name = parse_layer_filename(entry["path"])
        with rasterio.open(entry["path"]) as src:
            arr = src.read(1).astype(np.float32)
        arr[arr == fill_value] = np.nan
        values_by_name[qa_name] = arr[rows, cols]
    return values_by_name


def build_sample_table(
    metric_stack, metric_names, rows, cols, transform, year, qa_values_by_name
):
    """
    Assemble a per-year sample table (pandas DataFrame) from sampled
    pixels.

    Parameters
    ----------
    metric_stack : np.ndarray
        Metric stack of shape (H, W, F).
    metric_names : list[str]
        Feature names matching the last axis of `metric_stack`.
    rows, cols : np.ndarray
        Sampled pixel indices, shape (N,).
    transform : affine.Affine
        Rasterio affine transform of the raster grid.
    year : str
        4-digit year string, written to every row.
    qa_values_by_name : dict[str, np.ndarray]
        QA values at sampled pixels, keyed by QA name.

    Returns
    -------
    pandas.DataFrame
        Sample table.
    """
    x, y = pixel_to_xy(rows, cols, transform)
    metric_values = metric_stack[rows, cols, :]

    data = {
        "row": rows,
        "col": cols,
        "x": x,
        "y": y,
        "year": [str(year)] * rows.size,
    }
    for i, name in enumerate(metric_names):
        data[name] = metric_values[:, i]
    for qa_name, qa_vals in qa_values_by_name.items():
        data[qa_name] = qa_vals

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Per-year sampling
# ---------------------------------------------------------------------------

def sample_year(config, year, mask, metric_stack, metric_names, profile):
    """
    Draw stratified grid samples for one year and assemble the sample
    table.

    Parameters
    ----------
    config : dict
        Config dict containing "sampling", "qa", "data_path".
    year : str
        4-digit year string.
    mask : np.ndarray
        Boolean QA mask for this year, shape (H, W).
    metric_stack : np.ndarray
        Metric stack for this year, shape (H, W, F).
    metric_names : list[str]
        Feature names matching the last axis of `metric_stack`.
    profile : rasterio.profiles.Profile
        Raster profile for this year (used to extract transform).

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        (sample_table, sampling_stats).

    Raises
    ------
    ValueError
        If `sampling.method` is not "stratified_grid".
    """
    sampling_cfg = config["sampling"]
    method = sampling_cfg.get("method", "stratified_grid")
    if method != "stratified_grid":
        raise ValueError(
            f"Unsupported sampling method: '{method}'. Only "
            f"'stratified_grid' is currently implemented."
        )

    cell_size = int(sampling_cfg["cell_size_px"])
    samples_per_cell = int(sampling_cfg["samples_per_cell"])
    base_seed = int(sampling_cfg["random_seed_base"])
    year_seed = base_seed + int(year)

    rows, cols, stats = stratified_grid_sample(
        mask, cell_size=cell_size, samples_per_cell=samples_per_cell,
        seed=year_seed,
    )
    stats["seed"] = year_seed

    # FIX C/D: use nested qa config schema, pass full config to resolver
    resolved_qa_layers = resolve_layer_paths_for_year(
        config["qa"]["layers"], config, year
    )
    qa_values_by_name = _read_qa_values_at_pixels(
        resolved_qa_layers, rows, cols, fill_value=config["qa"]["fill_value"]
    )

    df = build_sample_table(
        metric_stack=metric_stack,
        metric_names=metric_names,
        rows=rows,
        cols=cols,
        transform=profile["transform"],
        year=year,
        qa_values_by_name=qa_values_by_name,
    )
    return df, stats


def write_sample_table(df, output_dir, config, year):
    """
    Write a per-year sample table to CSV using the results filename
    template.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample table to write.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string used to substitute "{year}" in the filename
        template.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_1_3"]["sampled_pixels_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_step_1_3(config, output_dir, masks_by_year, stacks_by_year):
    """
    Execute Step 1.3: stratified grid sampling per year, sample table
    assembly and CSV output, and report update.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    masks_by_year : dict[str, np.ndarray]
        Per-year QA masks.
    stacks_by_year : dict[str, tuple[np.ndarray, list[str], rasterio.profiles.Profile]]
        Per-year metric stacks.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from year (str) to sample DataFrame.
    """
    print(f'{"="*10} Step 1.3 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    sample_tables = {}
    sampling_report = {}

    for year in [str(y) for y in config["years"]]:
        mask = masks_by_year[year]
        stack, metric_names, profile = stacks_by_year[year]

        df, stats = sample_year(
            config=config,
            year=year,
            mask=mask,
            metric_stack=stack,
            metric_names=metric_names,
            profile=profile,
        )
        out_path = write_sample_table(df, output_dir, config, year)

        sample_tables[year] = df
        sampling_report[year] = {
            **stats,
            "output_csv": str(out_path),
            "cell_coverage_fraction": (
                stats["n_cells_sampled"] / stats["n_cells_total"]
                if stats["n_cells_total"] > 0 else 0.0
            ),
        }

    update_report(report_path, "step_1_3_sampling", sampling_report)
    return sample_tables


# =============================================================================
# Step 1.4 — Derived duration features
# =============================================================================

def write_combined_features_table(df, output_dir, config, year):
    """
    Write the combined feature table (raw + derived) to CSV using the
    config's Step 1.4 filename template.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined feature table to write.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_1_4"]["combined_features_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    df.to_csv(out_path, index=False)
    return out_path


SUPPORTED_OPERATIONS = {"-"}


def compute_derived_feature(df, metric1, metric2, operation, allowed_names):
    """
    Compute a single derived feature from two operand columns and an
    operation code.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing `metric1` and `metric2` as columns.
    metric1 : str
        Name of the left-hand-side operand column.
    metric2 : str
        Name of the right-hand-side operand column.
    operation : str
        Operation code. Only "-" is currently supported.
    allowed_names : set[str] or list[str]
        Set of column names permitted as operands.

    Returns
    -------
    pandas.Series
        Result of applying `operation` to `df[metric1]` and
        `df[metric2]`.

    Raises
    ------
    ValueError
        If `operation` is not supported, or if `metric1` or `metric2`
        is not in `allowed_names`.
    """
    allowed_names_set = set(allowed_names)
    for name in (metric1, metric2):
        if name not in allowed_names_set:
            raise ValueError(
                f"Operand '{name}' not in allowed operand set. "
                f"Allowed: {sorted(allowed_names_set)}."
            )
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation '{operation}'. "
            f"Supported: {sorted(SUPPORTED_OPERATIONS)}."
        )
    if operation == "-":
        return df[metric1] - df[metric2]
    raise ValueError(f"Operation dispatch fell through for '{operation}'.")


def add_derived_features(df, derived_features_config, metric_names):
    """
    Add derived feature columns to a copy of the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input sample table.
    derived_features_config : list[dict]
        List of derived-feature specs, each with "name", "metric1",
        "metric2", "operation".
    metric_names : list[str]
        Names of raw metric columns available as operands.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        - out_df : DataFrame copy with derived columns appended.
        - derived_names : list of derived column names in config
          order.
    """
    out_df = df.copy()
    derived_names = []
    for entry in derived_features_config:
        name = entry["name"]
        metric1 = entry["metric1"]
        metric2 = entry["metric2"]
        operation = entry["operation"]
        out_df[name] = compute_derived_feature(
            out_df, metric1, metric2, operation, metric_names
        )
        derived_names.append(name)
    return out_df, derived_names


def compute_feature_stats(series):
    """
    Compute basic descriptive statistics for a numeric series,
    ignoring NaN values.

    Parameters
    ----------
    series : pandas.Series or np.ndarray
        Numeric values to summarize.

    Returns
    -------
    dict
        Dictionary with keys "min", "max", "mean", "sd", "n_valid",
        "n_nan".
    """
    arr = np.asarray(series, dtype=np.float64)
    n_total = arr.size
    n_nan = int(np.isnan(arr).sum())
    n_valid = n_total - n_nan
    if n_valid == 0:
        return {
            "min": None, "max": None, "mean": None, "sd": None,
            "n_valid": 0, "n_nan": n_nan,
        }
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "sd": float(np.nanstd(arr, ddof=1)) if n_valid > 1 else 0.0,
        "n_valid": int(n_valid),
        "n_nan": n_nan,
    }


def run_step_1_4(config, output_dir, sample_tables_by_year, metric_names, update_report_fn):
    """
    Execute Step 1.4: compute derived duration features per year and
    write combined feature CSVs.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    sample_tables_by_year : dict[str, pandas.DataFrame]
        Per-year sample tables from Step 1.3.
    metric_names : list[str]
        Names of raw metric columns in the sample tables.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Per-year combined feature tables (raw + derived).
    """
    print(f'{"="*10} Step 1.4 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    derived_features_config = config["derived_features"]

    combined_tables = {}
    step_report = {}

    for year, df in sample_tables_by_year.items():
        combined_df, derived_names = add_derived_features(
            df, derived_features_config, metric_names
        )
        out_path = write_combined_features_table(
            combined_df, output_dir, config, year
        )

        derived_stats = {
            name: compute_feature_stats(combined_df[name])
            for name in derived_names
        }
        step_report[year] = {
            "derived_feature_names": derived_names,
            "derived_feature_stats": derived_stats,
            "output_csv": str(out_path),
            "n_rows": int(len(combined_df)),
        }
        combined_tables[year] = combined_df

    update_report_fn(report_path, "step_1_4_derived_features", step_report)
    return combined_tables


# =============================================================================
# Step 1.5 — Standardize (z-score) timing features per year
# =============================================================================
"""
Step 1.5 — Standardize (z-score) timing features per year.

Fits an independent sklearn StandardScaler per year on the 10 timing
features (7 raw metrics in order-parsed order + 3 derived duration
features in config order). Scaled values overwrite the raw feature
columns in the output table; non-feature columns (row, col, x, y,
year, NumCycles, QA) are preserved unchanged.

A defensive NaN check runs before fitting: any NaN present in the
feature columns raises ValueError, since the pipeline trusts the QA
mask (Step 1.2) as the sole validity filter.

Fitted scalers are serialized with joblib for reproducibility.
"""

# ---------------------------------------------------------------------------
# Feature ordering
# ---------------------------------------------------------------------------

def get_ordered_feature_names(metric_layers_config, derived_features_config):
    """
    Determine the deterministic 10-feature ordering used for scaling.

    Parameters
    ----------
    metric_layers_config : list[dict]
        Resolved or unresolved metric layer entries.
    derived_features_config : list[dict]
        Derived feature specs, each with a "name" key.

    Returns
    -------
    list[str]
        Ordered list of feature names: raw metric names followed by
        derived feature names.

    Raises
    ------
    ValueError
        If a metric layer filename does not match the expected
        convention.
    """
    parsed = []
    for entry in metric_layers_config:
        filename = Path(entry["path"]).name
        parts = filename.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Filename '{filename}' does not match expected format "
                f"'{{order}}_{{year}}_{{metric}}.tif'."
            )
        order = int(parts[0])
        metric_name = parts[2].replace(".tif", "")
        parsed.append((order, metric_name))
    parsed.sort(key=lambda item: item[0])
    raw_names = [name for _, name in parsed]

    derived_names = [entry["name"] for entry in derived_features_config]

    return raw_names + derived_names


# ---------------------------------------------------------------------------
# NaN validation
# ---------------------------------------------------------------------------

def assert_no_nan_in_features(df, feature_names, year):
    """
    Raise ValueError if any NaN is present in the specified feature
    columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined feature table (from Step 1.4) for one year.
    feature_names : list[str]
        Names of the feature columns to check.
    year : str
        Year string, used only for the error message.

    Raises
    ------
    ValueError
        If any NaN is found in any of `feature_names`.
    """
    nan_counts = {
        name: int(df[name].isna().sum())
        for name in feature_names
        if df[name].isna().any()
    }
    if nan_counts:
        raise ValueError(
            f"NaN detected in feature columns for year {year}, which "
            f"violates the trust-QA design assumption. "
            f"NaN counts per column: {nan_counts}."
        )


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_and_transform_features(df, feature_names):
    """
    Fit a StandardScaler on the specified feature columns and return
    the scaled values along with the fitted scaler.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names to scale.

    Returns
    -------
    tuple[np.ndarray, sklearn.preprocessing.StandardScaler]
        - scaled : ndarray of shape (N, len(feature_names)).
        - scaler : the fitted StandardScaler instance.
    """
    X = df[feature_names].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled, scaler


def build_scaled_table(df, feature_names, scaled_values):
    """
    Build a new DataFrame with feature columns overwritten by their
    scaled values, preserving all non-feature columns unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Original combined feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names being replaced.
    scaled_values : np.ndarray
        Scaled feature matrix of shape (N, len(feature_names)).

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with feature columns replaced by scaled values.
    """
    out_df = df.copy()
    for i, name in enumerate(feature_names):
        out_df[name] = scaled_values[:, i]
    return out_df


def summarize_scaler_params(scaler, feature_names):
    """
    Extract fitted StandardScaler parameters per feature for reporting.

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler instance.
    feature_names : list[str]
        Feature names in the same order as the scaler's mean_ and
        scale_ attributes.

    Returns
    -------
    dict[str, dict]
        Mapping from feature name to fitted mean/sd.
    """
    return {
        name: {
            "fitted_mean": float(scaler.mean_[i]),
            "fitted_sd": float(scaler.scale_[i]),
        }
        for i, name in enumerate(feature_names)
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_scaled_features_table(df, output_dir, config, year):
    """
    Write the scaled feature table to CSV using the config's Step 1.5
    filename template.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table to write.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_1_5"]["scaled_features_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    df.to_csv(out_path, index=False)
    return out_path


def write_scaler(scaler, output_dir, config, year):
    """
    Serialize the fitted StandardScaler to disk via joblib.

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler instance.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written joblib file.
    """
    template = config["results"]["step_1_5"]["scaler_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    joblib.dump(scaler, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_step_1_5(config, output_dir, combined_tables_by_year, update_report_fn):
    """
    Execute Step 1.5: standardize timing features per year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    combined_tables_by_year : dict[str, pandas.DataFrame]
        Per-year combined feature tables from Step 1.4.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Per-year scaled feature tables.

    Raises
    ------
    ValueError
        If NaN is detected in feature columns for any year.
    """
    print(f'{"="*10} Step 1.5 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    feature_names = get_ordered_feature_names(
        config["metric_layers"], config["derived_features"]
    )

    scaled_tables = {}
    step_report = {}

    for year, df in combined_tables_by_year.items():
        assert_no_nan_in_features(df, feature_names, year)

        scaled_values, scaler = fit_and_transform_features(df, feature_names)
        scaled_df = build_scaled_table(df, feature_names, scaled_values)

        csv_path = write_scaled_features_table(scaled_df, output_dir, config, year)
        scaler_path = write_scaler(scaler, output_dir, config, year)

        post_scaling_stats = {
            name: compute_feature_stats(scaled_df[name])
            for name in feature_names
        }
        fitted_params = summarize_scaler_params(scaler, feature_names)

        step_report[year] = {
            "feature_names": feature_names,
            "fitted_scaler_params": fitted_params,
            "post_scaling_stats": post_scaling_stats,
            "output_csv": str(csv_path),
            "output_scaler": str(scaler_path),
            "n_samples": int(len(df)),
        }
        scaled_tables[year] = scaled_df

    update_report_fn(report_path, "step_1_5_scaling", step_report)
    return scaled_tables


# =============================================================================
# Step 2.1 — PCA and correlation diagnostics
# =============================================================================
"""
Step 2.1 — PCA and correlation diagnostics on standardized timing
features. Diagnostic only — PCA output is not used to transform
features for downstream clustering.
"""

def compute_feature_correlation_matrix(df, feature_names):
    """
    Compute the Pearson correlation matrix among the specified feature
    columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names.

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix.
    """
    return df[feature_names].corr(method="pearson")


def write_correlation_matrix(corr_df, output_dir, config, year):
    """
    Write the feature correlation matrix to CSV.

    Parameters
    ----------
    corr_df : pandas.DataFrame
        Correlation matrix.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_2_1"]["correlation_matrix_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    corr_df.to_csv(out_path)
    return out_path


def fit_pca(df, feature_names, n_components=None):
    """
    Fit PCA on the standardized feature matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names.
    n_components : int or None, optional
        Number of components to keep.

    Returns
    -------
    tuple[sklearn.decomposition.PCA, np.ndarray]
        - pca : fitted PCA instance.
        - transformed : projected samples in PCA space.
    """
    X = df[feature_names].to_numpy(dtype=np.float64)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X)
    return pca, transformed


def summarize_pca_variance(pca):
    """
    Build a per-component summary of explained variance and cumulative
    explained variance.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA instance.

    Returns
    -------
    pandas.DataFrame
        Per-component variance summary.
    """
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    return pd.DataFrame({
        "component": np.arange(1, len(ratios) + 1),
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance": cumulative,
    })


def get_pca_loadings(pca, feature_names):
    """
    Build a DataFrame of PCA component loadings.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA instance.
    feature_names : list[str]
        Feature names, in the fit order.

    Returns
    -------
    pandas.DataFrame
        Loadings DataFrame, indexed by feature name, columns PC1..PCn.
    """
    n_components = pca.components_.shape[0]
    col_names = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(
        pca.components_.T, index=feature_names, columns=col_names
    )


def top_loading_features(loadings_df, n_components_to_report=5, top_n=5):
    """
    Identify the top-contributing features (by absolute loading) for
    each of the first N principal components.

    Parameters
    ----------
    loadings_df : pandas.DataFrame
        Loadings DataFrame.
    n_components_to_report : int, optional
        Number of leading components to summarize.
    top_n : int, optional
        Number of top features to report per component.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from component name to top loadings.
    """
    result = {}
    n_available = min(n_components_to_report, loadings_df.shape[1])
    for i in range(n_available):
        pc_name = loadings_df.columns[i]
        col = loadings_df[pc_name]
        ranked = col.reindex(col.abs().sort_values(ascending=False).index)
        top = ranked.head(top_n)
        result[pc_name] = [
            {"feature": feat, "loading": float(val)}
            for feat, val in top.items()
        ]
    return result


def plot_pca_diagnostics(loadings_df, variance_df, out_path):
    """
    Generate a two-panel PCA diagnostic figure: explained variance
    (scree plot with cumulative line) and a loadings heatmap.

    Parameters
    ----------
    loadings_df : pandas.DataFrame
        Loadings DataFrame.
    variance_df : pandas.DataFrame
        Explained variance summary.
    out_path : str or pathlib.Path
        Output path for the saved PNG figure.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved figure.
    """
    n_show = min(10, loadings_df.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax0 = axes[0]
    ax0.bar(
        variance_df["component"], variance_df["explained_variance_ratio"],
        color="steelblue", label="Explained variance ratio",
    )
    ax0.plot(
        variance_df["component"], variance_df["cumulative_explained_variance"],
        color="darkorange", marker="o", label="Cumulative explained variance",
    )
    ax0.set_xlabel("Principal component")
    ax0.set_ylabel("Explained variance ratio")
    ax0.set_title("PCA explained variance")
    ax0.legend(loc="center right")
    ax0.set_xticks(variance_df["component"])

    ax1 = axes[1]
    subset = loadings_df.iloc[:, :n_show]
    im = ax1.imshow(subset.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(n_show))
    ax1.set_xticklabels(subset.columns)
    ax1.set_yticks(range(len(subset.index)))
    ax1.set_yticklabels(subset.index)
    ax1.set_title("PCA loadings (top components)")
    fig.colorbar(im, ax=ax1, label="Loading")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_pca_variance_table(variance_df, output_dir, config, year):
    """
    Write the PCA explained variance summary to CSV.

    Parameters
    ----------
    variance_df : pandas.DataFrame
        Output of `summarize_pca_variance`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_2_1"]["pca_explained_variance_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    variance_df.to_csv(out_path, index=False)
    return out_path


def run_step_2_1(config, output_dir, scaled_tables_by_year, feature_names, update_report_fn):
    """
    Execute Step 2.1: correlation matrix and PCA diagnostics per year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    scaled_tables_by_year : dict[str, pandas.DataFrame]
        Per-year scaled feature tables from Step 1.5.
    feature_names : list[str]
        Ordered list of feature column names.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, dict]
        Per-year PCA diagnostic results.
    """
    print(f'{"="*10} Step 2.1 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    n_components = config["models"]["pca"]["n_components"]

    step_report = {}
    results_by_year = {}

    for year, df in scaled_tables_by_year.items():
        corr_df = compute_feature_correlation_matrix(df, feature_names)
        corr_path = write_correlation_matrix(corr_df, output_dir, config, year)

        pca, _ = fit_pca(df, feature_names, n_components=n_components)
        variance_df = summarize_pca_variance(pca)
        loadings_df = get_pca_loadings(pca, feature_names)
        top_loadings = top_loading_features(loadings_df)

        variance_path = write_pca_variance_table(variance_df, output_dir, config, year)

        plot_template = config["results"]["step_2_1"]["pca_plot_template"]
        plot_filename = plot_template.replace("{year}", str(year))
        plot_path = plot_pca_diagnostics(loadings_df, variance_df, Path(output_dir) / plot_filename)

        step_report[year] = {
            "n_components_fit": int(pca.n_components_),
            "explained_variance_ratio": variance_df["explained_variance_ratio"].tolist(),
            "cumulative_explained_variance": variance_df["cumulative_explained_variance"].tolist(),
            "top_loading_features": top_loadings,
            "correlation_matrix_csv": str(corr_path),
            "pca_variance_csv": str(variance_path),
            "pca_plot_png": str(plot_path),
        }
        results_by_year[year] = {
            "pca": pca,
            "loadings": loadings_df,
            "variance_df": variance_df,
            "correlation_matrix": corr_df,
        }

    update_report_fn(report_path, "step_2_1_pca", step_report)
    return results_by_year


# =============================================================================
# Step 2.2 — K-means diagnostic sweep
# =============================================================================
"""
Step 2.2 — K-means diagnostic sweep across a range of k values.
Diagnostic only; no models persisted here.
"""

def derive_kmeans_seed(base_seed, year, k):
    """
    Derive a deterministic random seed for a specific K-means fit.

    Parameters
    ----------
    base_seed : int
        Base seed from config.
    year : str or int
        Year associated with this fit.
    k : int
        Number of clusters for this fit.

    Returns
    -------
    int
        Derived seed.
    """
    return int(base_seed) + int(year) + int(k)


def fit_kmeans_single(X, k, n_init, seed):
    """
    Fit a single K-means model and compute diagnostic metrics.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix of shape (N, F).
    k : int
        Number of clusters.
    n_init : int
        Number of centroid initializations.
    seed : int
        Random seed for this fit.

    Returns
    -------
    dict
        Fit results including labels, inertia, silhouette,
        cluster_sizes, n_iter.
    """
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels) if k >= 2 and k < X.shape[0] else np.nan

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "k": int(k),
        "seed": int(seed),
        "labels": labels,
        "inertia": float(model.inertia_),
        "silhouette": float(sil) if not np.isnan(sil) else None,
        "cluster_sizes": cluster_sizes,
        "n_iter": int(model.n_iter_),
    }


def run_kmeans_sweep(df, feature_names, k_range, n_init, base_seed, year):
    """
    Run K-means across a range of k values for one year.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names.
    k_range : list[int]
        [min_k, max_k] inclusive.
    n_init : int
        Number of centroid initializations per fit.
    base_seed : int
        Base seed for deriving per-(year, k) seeds.
    year : str
        Year string.

    Returns
    -------
    list[dict]
        List of per-k result dictionaries, ascending k order.
    """
    X = df[feature_names].to_numpy(dtype=np.float64)
    k_min, k_max = k_range
    results = []
    for k in tqdm(range(k_min, k_max + 1)):
        seed = derive_kmeans_seed(base_seed, year, k)
        result = fit_kmeans_single(X, k, n_init, seed)
        results.append(result)
    return results


def summarize_kmeans_sweep(sweep_results, year):
    """
    Build a per-k summary DataFrame from a K-means sweep.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of `run_kmeans_sweep`.
    year : str
        Year string.

    Returns
    -------
    pandas.DataFrame
        Per-k summary table.
    """
    rows = []
    for r in sweep_results:
        sizes = list(r["cluster_sizes"].values())
        rows.append({
            "year": str(year),
            "k": r["k"],
            "seed": r["seed"],
            "inertia": r["inertia"],
            "silhouette": r["silhouette"],
            "n_iter": r["n_iter"],
            "min_cluster_size": min(sizes) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "n_clusters_realized": len(sizes),
        })
    return pd.DataFrame(rows)


def write_kmeans_sweep_summary(summary_df, output_dir, config, year):
    """
    Write the K-means sweep summary table to CSV.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of `summarize_kmeans_sweep`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_2_2"]["kmeans_sweep_summary_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    summary_df.to_csv(out_path, index=False)
    return out_path


def write_kmeans_labels(sweep_results, output_dir, config, year, df):
    """
    Write per-pixel cluster labels for every k in the sweep to a wide
    CSV.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of `run_kmeans_sweep`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.
    df : pandas.DataFrame
        Original scaled feature table for this year.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    id_cols = ["row", "col", "x", "y", "year"]
    out_df = df[id_cols].copy()
    for r in sweep_results:
        out_df[f"kmeans_k{r['k']}"] = r["labels"]

    template = config["results"]["step_2_2"]["kmeans_labels_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    out_df.to_csv(out_path, index=False)
    return out_path


def run_step_2_2(config, output_dir, scaled_tables_by_year, feature_names,
                  update_report_fn):
    """
    Execute Step 2.2: K-means diagnostic sweep across k for every year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    scaled_tables_by_year : dict[str, pandas.DataFrame]
        Per-year scaled feature tables from Step 1.5.
    feature_names : list[str]
        Ordered list of feature column names.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    tuple[dict[str, list[dict]], dict[str, pandas.DataFrame]]
        - sweep_results_by_year : per-year raw sweep results.
        - summary_by_year : per-year summary DataFrame.
    """
    print(f'{"="*10} Step 2.2 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    kmeans_cfg = config["models"]["kmeans"]
    k_range = kmeans_cfg["k_range"]
    n_init = kmeans_cfg["n_init"]
    base_seed = kmeans_cfg["base_seed"]

    sweep_results_by_year = {}
    summary_by_year = {}
    step_report = {}

    for year, df in scaled_tables_by_year.items():
        sweep_results = run_kmeans_sweep(
            df, feature_names, k_range, n_init, base_seed, year
        )
        summary_df = summarize_kmeans_sweep(sweep_results, year)

        summary_path = write_kmeans_sweep_summary(summary_df, output_dir, config, year)
        labels_path = write_kmeans_labels(sweep_results, output_dir, config, year, df)

        step_report[year] = {
            "k_range": k_range,
            "n_init": n_init,
            "sweep_summary": summary_df.to_dict(orient="records"),
            "sweep_summary_csv": str(summary_path),
            "labels_csv": str(labels_path),
        }
        sweep_results_by_year[year] = sweep_results
        summary_by_year[year] = summary_df

    update_report_fn(report_path, "step_2_2_kmeans_sweep", step_report)

    return sweep_results_by_year, summary_by_year


# =============================================================================
# Step 2.3 — K-means validity diagnostic plots
# =============================================================================
"""
Step 2.3 — K-means validity diagnostic plots. Manual review only.
"""

def build_ordered_cluster_size_matrix(sweep_results):
    """
    Build a matrix of cluster sizes per k, ordered largest to smallest
    within each k.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of `run_kmeans_sweep`.

    Returns
    -------
    tuple[list[int], np.ndarray]
        - k_values : list of k values.
        - size_matrix : ndarray of shape (K, max_k).
    """
    k_values = [r["k"] for r in sweep_results]
    max_k = max(k_values)

    size_matrix = np.zeros((len(sweep_results), max_k), dtype=np.int64)
    for i, r in enumerate(sweep_results):
        sizes_sorted = sorted(r["cluster_sizes"].values(), reverse=True)
        size_matrix[i, :len(sizes_sorted)] = sizes_sorted

    return k_values, size_matrix


def plot_kmeans_validity(summary_df, sweep_results, year, out_path):
    """
    Generate a three-panel K-means validity diagnostic figure.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Per-k summary table for one year.
    sweep_results : list[dict]
        Full per-k sweep results for the same year.
    year : str
        Year string.
    out_path : str or pathlib.Path
        Output path for the saved PNG figure.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax0 = axes[0]
    ax0.plot(summary_df["k"], summary_df["inertia"], marker="o", color="steelblue")
    ax0.set_xlabel("k (number of clusters)")
    ax0.set_ylabel("Inertia (within-cluster sum of squares)")
    ax0.set_title(f"K-means elbow plot — {year}")
    ax0.set_xticks(summary_df["k"])
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    ax1.plot(summary_df["k"], summary_df["silhouette"], marker="o", color="darkorange")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Mean silhouette score")
    ax1.set_title(f"K-means silhouette scores — {year}")
    ax1.set_xticks(summary_df["k"])
    ax1.grid(alpha=0.3)

    ax2 = axes[2]
    k_values, size_matrix = build_ordered_cluster_size_matrix(sweep_results)
    max_k = size_matrix.shape[1]

    cmap = plt.get_cmap("viridis", max_k)
    bottoms = np.zeros(len(k_values))
    for slot in range(max_k):
        heights = size_matrix[:, slot]
        ax2.bar(
            k_values, heights, bottom=bottoms,
            color=cmap(slot), edgecolor="white", linewidth=0.3,
            width=0.8,
        )
        bottoms += heights

    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("Cluster size (pixel count)")
    ax2.set_title(f"Cluster size distribution (largest→smallest) — {year}")
    ax2.set_xticks(k_values)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_kmeans_validity_plot(summary_df, sweep_results, output_dir, config, year):
    """
    Generate and write the K-means validity diagnostic plot.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Per-k summary table for this year.
    sweep_results : list[dict]
        Full per-k sweep results for this year.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PNG.
    """
    template = config["results"]["step_2_3"]["kmeans_validity_plot_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    return plot_kmeans_validity(summary_df, sweep_results, year, out_path)


def run_step_2_3(config, output_dir, kmeans_sweep_results_by_year,
                  kmeans_summary_by_year, update_report_fn):
    """
    Execute Step 2.3: generate K-means validity diagnostic plots per
    year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    kmeans_sweep_results_by_year : dict[str, list[dict]]
        Per-year full sweep results from Step 2.2.
    kmeans_summary_by_year : dict[str, pandas.DataFrame]
        Per-year summary tables from Step 2.2.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from year to validity plot path.

    Raises
    ------
    KeyError
        If a year present in one dict is missing from the other.
    """
    print(f'{"="*10} Step 2.3 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    plot_paths_by_year = {}
    step_report = {}

    for year, sweep_results in kmeans_sweep_results_by_year.items():
        if year not in kmeans_summary_by_year:
            raise KeyError(
                f"Year '{year}' present in kmeans_sweep_results_by_year "
                f"but missing from kmeans_summary_by_year."
            )
        summary_df = kmeans_summary_by_year[year]

        plot_path = write_kmeans_validity_plot(summary_df, sweep_results, output_dir, config, year)
        plot_paths_by_year[year] = plot_path

        step_report[year] = {
            "validity_plot_png": str(plot_path),
        }

    update_report_fn(report_path, "step_2_3_kmeans_validity", step_report)

    print(f"Step 2.3 complete — K-means validity plots written for years: "
          f"{list(plot_paths_by_year.keys())}")
    for year, path in plot_paths_by_year.items():
        print(f"{year}: {path}")

    return plot_paths_by_year


# =============================================================================
# Step 2.4 — GMM diagnostic sweep
# =============================================================================
"""
Step 2.4 — GMM diagnostic sweep across a range of k values and two
covariance types. Diagnostic only; no models persisted here.
"""

def derive_gmm_seed(base_seed, year, k, covariance_type, config):
    """
    Derive a deterministic random seed for a specific GMM fit.

    Parameters
    ----------
    base_seed : int
        Base seed from config (`models.gmm.base_seed`).
    year : str or int
        Year associated with this fit.
    k : int
        Number of mixture components for this fit.
    covariance_type : str
        Covariance type ("full" or "diag").
    config : dict
        Full config dict; used to read
        `config['models']['gmm']['covariance_offset_seed']`.

    Returns
    -------
    int
        Derived seed.
    """
    offset = config['models']['gmm']['covariance_offset_seed'][covariance_type]
    return int(base_seed) + int(year) + int(k) + offset


def fit_gmm_single(X, k, covariance_type, seed):
    """
    Fit a single GMM model and compute diagnostic metrics.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix of shape (N, F).
    k : int
        Number of mixture components.
    covariance_type : str
        Covariance type ("full" or "diag").
    seed : int
        Random seed for this fit.

    Returns
    -------
    dict
        Fit results including labels, posteriors, bic, aic,
        log_likelihood, silhouette, converged, n_iter, cluster_sizes.
    """
    model = GaussianMixture(
        n_components=k, covariance_type=covariance_type, random_state=seed,
    )
    model.fit(X)

    posteriors = model.predict_proba(X)
    labels = np.argmax(posteriors, axis=1)

    sil = silhouette_score(X, labels) if k >= 2 and k < X.shape[0] else np.nan

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "k": int(k),
        "covariance_type": covariance_type,
        "seed": int(seed),
        "labels": labels,
        "posteriors": posteriors,
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "log_likelihood": float(model.score(X)),
        "silhouette": float(sil) if not np.isnan(sil) else None,
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "cluster_sizes": cluster_sizes,
    }


def run_gmm_sweep(df, feature_names, k_range, covariance_types, base_seed, year, config):
    """
    Run GMM across a range of k values and a set of covariance types
    for one year.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names.
    k_range : list[int]
        [min_k, max_k] inclusive.
    covariance_types : list[str]
        Covariance types to test.
    base_seed : int
        Base seed for deriving per-(year, k, covariance_type) seeds.
    year : str
        Year string.
    config : dict
        Full config dict, passed through to `derive_gmm_seed`.

    Returns
    -------
    list[dict]
        List of per-(covariance_type, k) result dictionaries.
    """
    X = df[feature_names].to_numpy(dtype=np.float64)
    k_min, k_max = k_range
    results = []
    for covariance_type in covariance_types:
        for k in tqdm(range(k_min, k_max + 1)):
            seed = derive_gmm_seed(base_seed, year, k, covariance_type, config)
            result = fit_gmm_single(X, k, covariance_type, seed)
            results.append(result)
    return results


def summarize_gmm_sweep(sweep_results, year):
    """
    Build a per-(k, covariance_type) summary DataFrame from a GMM
    sweep.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of `run_gmm_sweep`.
    year : str
        Year string.

    Returns
    -------
    pandas.DataFrame
        Per-(k, covariance_type) summary table.
    """
    rows = []
    for r in sweep_results:
        sizes = list(r["cluster_sizes"].values())
        rows.append({
            "year": str(year),
            "k": r["k"],
            "covariance_type": r["covariance_type"],
            "seed": r["seed"],
            "bic": r["bic"],
            "aic": r["aic"],
            "log_likelihood": r["log_likelihood"],
            "silhouette": r["silhouette"],
            "converged": r["converged"],
            "n_iter": r["n_iter"],
            "min_cluster_size": min(sizes) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "n_clusters_realized": len(sizes),
        })
    return pd.DataFrame(rows)


def write_gmm_sweep_summary(summary_df, output_dir, config, year):
    """
    Write the GMM sweep summary table to CSV.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of `summarize_gmm_sweep`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    template = config["results"]["step_2_4"]["gmm_sweep_summary_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    summary_df.to_csv(out_path, index=False)
    return out_path


def write_gmm_labels(sweep_results, output_dir, config, year, df):
    """
    Write per-pixel hard cluster labels for every (k, covariance_type)
    combination to a wide CSV.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of `run_gmm_sweep`.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.
    df : pandas.DataFrame
        Original scaled feature table for this year.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    id_cols = ["row", "col", "x", "y", "year"]
    out_df = df[id_cols].copy()
    for r in sweep_results:
        col_name = f"gmm_{r['covariance_type']}_k{r['k']}"
        out_df[col_name] = r["labels"]

    template = config["results"]["step_2_4"]["gmm_labels_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    out_df.to_csv(out_path, index=False)
    return out_path


def run_step_2_4(config, output_dir, scaled_tables_by_year, feature_names,
                  update_report_fn):
    """
    Execute Step 2.4: GMM diagnostic sweep across k and covariance
    type for every year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    scaled_tables_by_year : dict[str, pandas.DataFrame]
        Per-year scaled feature tables from Step 1.5.
    feature_names : list[str]
        Ordered list of feature column names.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    tuple[dict[str, list[dict]], dict[str, pandas.DataFrame]]
        - sweep_results_by_year : per-year raw sweep results.
        - summary_by_year : per-year summary DataFrame.
    """
    print(f'{"="*10} Step 2.4 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    gmm_cfg = config["models"]["gmm"]
    k_range = gmm_cfg["k_range"]
    covariance_types = gmm_cfg["covariance_types"]
    base_seed = gmm_cfg["base_seed"]

    sweep_results_by_year = {}
    summary_by_year = {}
    step_report = {}

    for year, df in scaled_tables_by_year.items():
        print(f'  {year}: running GMM sweep')
        sweep_results = run_gmm_sweep(
            df, feature_names, k_range, covariance_types, base_seed, year, config
        )
        summary_df = summarize_gmm_sweep(sweep_results, year)

        summary_path = write_gmm_sweep_summary(summary_df, output_dir, config, year)
        labels_path = write_gmm_labels(sweep_results, output_dir, config, year, df)

        n_not_converged = int((~summary_df["converged"]).sum())

        step_report[year] = {
            "k_range": k_range,
            "covariance_types": covariance_types,
            "n_fits_not_converged": n_not_converged,
            "sweep_summary": summary_df.to_dict(orient="records"),
            "sweep_summary_csv": str(summary_path),
            "labels_csv": str(labels_path),
        }
        sweep_results_by_year[year] = sweep_results
        summary_by_year[year] = summary_df

    update_report_fn(report_path, "step_2_4_gmm_sweep", step_report)

    print(f"Step 2.4 complete — GMM sweep run for years: "
          f"{list(sweep_results_by_year.keys())}")

    return sweep_results_by_year, summary_by_year


# =============================================================================
# Step 2.5 — GMM validity diagnostic plots
# =============================================================================
"""
Step 2.5 — GMM validity diagnostic plots. Manual review only — k and
covariance-type selection (Step 2.6) is performed by the analyst.
"""

COVARIANCE_TYPE_COLORS = {"full": "steelblue", "diag": "firebrick"}


def build_ordered_cluster_size_matrix_gmm(sweep_results, covariance_type):
    """
    Build a matrix of cluster sizes per k for one covariance type,
    ordered largest to smallest within each k.

    Parameters
    ----------
    sweep_results : list[dict]
        Output of `run_gmm_sweep`.
    covariance_type : str
        Covariance type to filter on ("full" or "diag").

    Returns
    -------
    tuple[list[int], np.ndarray]
        - k_values : list of k values for this covariance type.
        - size_matrix : ndarray of shape (K, max_k).
    """
    filtered = [r for r in sweep_results if r["covariance_type"] == covariance_type]
    filtered.sort(key=lambda r: r["k"])
    k_values = [r["k"] for r in filtered]
    max_k = max(k_values)

    size_matrix = np.zeros((len(filtered), max_k), dtype=np.int64)
    for i, r in enumerate(filtered):
        sizes_sorted = sorted(r["cluster_sizes"].values(), reverse=True)
        size_matrix[i, :len(sizes_sorted)] = sizes_sorted

    return k_values, size_matrix


def _plot_cluster_size_panel(ax, sweep_results, covariance_type, year):
    """
    Draw a stacked cluster-size bar chart (largest to smallest) onto
    a given axis, for one covariance type.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    sweep_results : list[dict]
        Full GMM sweep results for the year.
    covariance_type : str
        Covariance type to filter and plot.
    year : str
        Year string, used in the panel title.

    Returns
    -------
    None
    """
    k_values, size_matrix = build_ordered_cluster_size_matrix_gmm(
        sweep_results, covariance_type
    )
    max_k = size_matrix.shape[1]
    cmap = plt.get_cmap("viridis", max_k)

    bottoms = np.zeros(len(k_values))
    for slot in range(max_k):
        heights = size_matrix[:, slot]
        ax.bar(
            k_values, heights, bottom=bottoms,
            color=cmap(slot), edgecolor="white", linewidth=0.3,
            width=0.8,
        )
        bottoms += heights

    ax.set_xlabel("k (number of components)")
    ax.set_ylabel("Cluster size (pixel count)")
    ax.set_title(f"Cluster sizes — covariance='{covariance_type}' — {year}")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.3, axis="y")


def plot_gmm_validity(summary_df, sweep_results, year, out_path):
    """
    Generate a four-panel GMM validity diagnostic figure.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Per-(k, covariance_type) summary table for one year.
    sweep_results : list[dict]
        Full sweep results for the same year.
    year : str
        Year string.
    out_path : str or pathlib.Path
        Output path for the saved PNG figure.

    Returns
    -------
    pathlib.Path
        Absolute path to the saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    covariance_types = sorted(summary_df["covariance_type"].unique())

    ax_ic = axes[0, 0]
    for cov_type in covariance_types:
        sub = summary_df[summary_df["covariance_type"] == cov_type].sort_values("k")
        color = COVARIANCE_TYPE_COLORS.get(cov_type, None)
        ax_ic.plot(sub["k"], sub["bic"], marker="o", linestyle="-",
                   color=color, label=f"BIC ({cov_type})")
        ax_ic.plot(sub["k"], sub["aic"], marker="s", linestyle="--",
                   color=color, label=f"AIC ({cov_type})", alpha=0.7)
    ax_ic.set_xlabel("k (number of components)")
    ax_ic.set_ylabel("Information criterion value")
    ax_ic.set_title(f"GMM BIC / AIC — {year}")
    ax_ic.legend(fontsize=8)
    ax_ic.grid(alpha=0.3)

    ax_sil = axes[0, 1]
    for cov_type in covariance_types:
        sub = summary_df[summary_df["covariance_type"] == cov_type].sort_values("k")
        color = COVARIANCE_TYPE_COLORS.get(cov_type, None)
        ax_sil.plot(sub["k"], sub["silhouette"], marker="o", linestyle="-",
                    color=color, label=cov_type)
    ax_sil.set_xlabel("k (number of components)")
    ax_sil.set_ylabel("Mean silhouette score (hard labels)")
    ax_sil.set_title(f"GMM silhouette scores — {year}")
    ax_sil.legend(fontsize=8)
    ax_sil.grid(alpha=0.3)

    for ax, cov_type in zip([axes[1, 0], axes[1, 1]], covariance_types[:2]):
        _plot_cluster_size_panel(ax, sweep_results, cov_type, year)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_gmm_validity_plot(summary_df, sweep_results, output_dir, config, year):
    """
    Generate and write the GMM validity diagnostic plot for one year.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Per-(k, covariance_type) summary table for this year.
    sweep_results : list[dict]
        Full sweep results for this year.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PNG.
    """
    template = config["results"]["step_2_5"]["gmm_validity_plot_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    return plot_gmm_validity(summary_df, sweep_results, year, out_path)


def run_step_2_5(config, output_dir, gmm_sweep_results_by_year,
                  gmm_summary_by_year, update_report_fn):
    """
    Execute Step 2.5: generate GMM validity diagnostic plots per year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    gmm_sweep_results_by_year : dict[str, list[dict]]
        Per-year full sweep results from Step 2.4.
    gmm_summary_by_year : dict[str, pandas.DataFrame]
        Per-year summary tables from Step 2.4.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from year to validity plot path.

    Raises
    ------
    KeyError
        If a year present in one dict is missing from the other.
    """
    print(f'{"="*10} Step 2.5 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))

    plot_paths_by_year = {}
    step_report = {}

    for year, sweep_results in gmm_sweep_results_by_year.items():
        if year not in gmm_summary_by_year:
            raise KeyError(
                f"Year '{year}' present in gmm_sweep_results_by_year "
                f"but missing from gmm_summary_by_year."
            )
        summary_df = gmm_summary_by_year[year]

        plot_path = write_gmm_validity_plot(summary_df, sweep_results, output_dir, config, year)
        plot_paths_by_year[year] = plot_path


        step_report[year] = {
            "validity_plot_png": str(plot_path),
        }

    update_report_fn(report_path, "step_2_5_gmm_validity", step_report)

    print(f"Step 2.5 complete — GMM validity plots written for years: "
          f"{list(plot_paths_by_year.keys())}")
    for year, path in plot_paths_by_year.items():
        print(f"  {year}: {path}")

    return plot_paths_by_year


# =============================================================================
# Step 2.6 — Manual cluster count (and covariance type) selection
# =============================================================================
"""
Step 2.6 — Manual cluster count (and covariance type) selection.
No clustering or computation happens in this step.
"""

def print_years(config):
    """
    Print the list of years configured for this run.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.

    Returns
    -------
    list[str]
        The list of years as strings, in config order.
    """
    years = [str(y) for y in config["years"]]
    print(f"Years configured for this run ({len(years)} total):")
    for i, year in enumerate(years):
        print(f"  [{i}] {year}")
    return years


def validate_list_length(values, years, list_name):
    """
    Validate that a provided list has exactly one entry per year.

    Parameters
    ----------
    values : list
        List of values provided by the analyst.
    years : list[str]
        List of years from config.
    list_name : str
        Name of the list, used in error messages.

    Raises
    ------
    ValueError
        If `len(values) != len(years)`.
    """
    if len(values) != len(years):
        raise ValueError(
            f"'{list_name}' has {len(values)} entries but there are "
            f"{len(years)} configured years {years}. Provide exactly "
            f"one entry per year, in the same order."
        )


def validate_kmeans_k_values(kmeans_k_values, years, k_range):
    """
    Validate K-means k selections: correct length and each value
    within the swept range.

    Parameters
    ----------
    kmeans_k_values : list[int]
        One k per year, positionally matched to `years`.
    years : list[str]
        List of years from config.
    k_range : list[int]
        [min_k, max_k] swept range.

    Raises
    ------
    ValueError
        If the list length is wrong, or any k is outside `k_range`.
    """
    validate_list_length(kmeans_k_values, years, "kmeans_k_values")
    k_min, k_max = k_range
    for year, k in zip(years, kmeans_k_values):
        if not (k_min <= int(k) <= k_max):
            raise ValueError(
                f"K-means k={k} for year '{year}' is outside the "
                f"swept range [{k_min}, {k_max}]."
            )


def validate_gmm_k_values(gmm_k_values, years, k_range):
    """
    Validate GMM k selections: correct length and each value within
    the swept range.

    Parameters
    ----------
    gmm_k_values : list[int]
        One k per year, positionally matched to `years`.
    years : list[str]
        List of years from config.
    k_range : list[int]
        [min_k, max_k] swept range.

    Raises
    ------
    ValueError
        If the list length is wrong, or any k is outside `k_range`.
    """
    validate_list_length(gmm_k_values, years, "gmm_k_values")
    k_min, k_max = k_range
    for year, k in zip(years, gmm_k_values):
        if not (k_min <= int(k) <= k_max):
            raise ValueError(
                f"GMM k={k} for year '{year}' is outside the swept "
                f"range [{k_min}, {k_max}]."
            )


def validate_gmm_covariance_types(gmm_covariance_types, years, valid_types):
    """
    Validate GMM covariance type selections: correct length and each
    value among the previously swept covariance types.

    Parameters
    ----------
    gmm_covariance_types : list[str]
        One covariance type per year, positionally matched to `years`.
    years : list[str]
        List of years from config.
    valid_types : list[str]
        Covariance types swept in Step 2.4.

    Raises
    ------
    ValueError
        If the list length is wrong, or any covariance type is not
        among `valid_types`.
    """
    validate_list_length(gmm_covariance_types, years, "gmm_covariance_types")
    valid_set = set(valid_types)
    for year, cov_type in zip(years, gmm_covariance_types):
        if cov_type not in valid_set:
            raise ValueError(
                f"GMM covariance_type='{cov_type}' for year '{year}' "
                f"is not among the swept types {sorted(valid_set)}."
            )


def populate_selected_clusters(config, years, kmeans_k_values, gmm_k_values,
                                gmm_covariance_types):
    """
    Build the `selected_clusters` config block from the three
    positionally-matched input lists and insert it into the config
    dict.

    Parameters
    ----------
    config : dict
        Config dict to update in place.
    years : list[str]
        List of years from config.
    kmeans_k_values : list[int]
        One K-means k per year.
    gmm_k_values : list[int]
        One GMM k per year.
    gmm_covariance_types : list[str]
        One GMM covariance type per year.

    Returns
    -------
    dict
        The updated config dict.
    """
    config["selected_clusters"] = {
        "kmeans": {
            year: int(k) for year, k in zip(years, kmeans_k_values)
        },
        "gmm": {
            year: {"k": int(k), "covariance_type": cov_type}
            for year, k, cov_type in zip(years, gmm_k_values, gmm_covariance_types)
        },
    }
    return config


def write_config(config, config_path):
    """
    Write the (updated) config dict back to its JSON file on disk.

    Parameters
    ----------
    config : dict
        Config dict to serialize.
    config_path : str or pathlib.Path
        Path to the config JSON file to overwrite.

    Returns
    -------
    pathlib.Path
        Absolute path to the written config file.
    """
    config_path = Path(config_path)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def run_step_2_6(config, config_path, kmeans_k_values, gmm_k_values,
                  gmm_covariance_types):
    """
    Execute Step 2.6: validate and record manually chosen k values
    (and GMM covariance types), one per year, into config.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    config_path : str or pathlib.Path
        Path to the config JSON file, to be overwritten.
    kmeans_k_values : list[int]
        Analyst-chosen K-means k per year, matching `config["years"]`
        order.
    gmm_k_values : list[int]
        Analyst-chosen GMM k per year.
    gmm_covariance_types : list[str]
        Analyst-chosen GMM covariance type per year.

    Returns
    -------
    dict
        The updated config dict, including `selected_clusters`.

    Raises
    ------
    ValueError
        If any input list's length does not match the number of
        years, if any k is outside its swept range, or if any
        covariance type was not among the swept types.
    """
    years = print_years(config)

    validate_kmeans_k_values(
        kmeans_k_values, years, config["models"]["kmeans"]["k_range"]
    )
    validate_gmm_k_values(
        gmm_k_values, years, config["models"]["gmm"]["k_range"]
    )
    validate_gmm_covariance_types(
        gmm_covariance_types, years, config["models"]["gmm"]["covariance_types"]
    )

    config = populate_selected_clusters(
        config, years, kmeans_k_values, gmm_k_values, gmm_covariance_types
    )
    write_config(config, config_path)

    print("Selected clusters recorded:")
    print(json.dumps(config["selected_clusters"], indent=2))

    return config


# ---------------------------------------------------------------------------
# Selection validators consumed by Step 2.7 (previously missing)
# ---------------------------------------------------------------------------

def validate_kmeans_selection(config):
    """
    Validate that a K-means cluster count has been manually selected
    for every configured year (via Step 2.6), and that each selection
    falls within the previously swept k range.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict, expected to contain
        `config["selected_clusters"]["kmeans"]` (populated by
        `run_step_2_6`).

    Returns
    -------
    dict[str, int]
        Mapping from year (str) to validated chosen k.

    Raises
    ------
    KeyError
        If `selected_clusters.kmeans` is missing entirely, or missing
        an entry for any configured year.
    ValueError
        If a selected k falls outside the swept `k_range`.
    """
    years = [str(y) for y in config["years"]]
    k_min, k_max = config["models"]["kmeans"]["k_range"]

    if "selected_clusters" not in config or "kmeans" not in config["selected_clusters"]:
        raise KeyError(
            "config['selected_clusters']['kmeans'] is missing. "
            "Run Step 2.6 to record a chosen k per year before "
            "running Step 2.7."
        )

    selections = config["selected_clusters"]["kmeans"]
    validated = {}
    for year in years:
        if year not in selections:
            raise KeyError(
                f"No K-means k selection found for year '{year}' in "
                f"config['selected_clusters']['kmeans']."
            )
        k = int(selections[year])
        if not (k_min <= k <= k_max):
            raise ValueError(
                f"Selected K-means k={k} for year '{year}' is outside "
                f"the swept range [{k_min}, {k_max}]."
            )
        validated[year] = k
    return validated


def validate_gmm_selection(config):
    """
    Validate that a GMM cluster count and covariance type have been
    manually selected for every configured year (via Step 2.6), and
    that each selection falls within the previously swept k range and
    covariance type list.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict, expected to contain
        `config["selected_clusters"]["gmm"]` (populated by
        `run_step_2_6`).

    Returns
    -------
    dict[str, dict]
        Mapping from year (str) to {"k": int, "covariance_type": str}.

    Raises
    ------
    KeyError
        If `selected_clusters.gmm` is missing entirely, missing an
        entry for any configured year, or missing "k" /
        "covariance_type" within a year's entry.
    ValueError
        If a selected k falls outside the swept `k_range`, or a
        selected covariance_type was not among the swept
        `covariance_types`.
    """
    years = [str(y) for y in config["years"]]
    k_min, k_max = config["models"]["gmm"]["k_range"]
    valid_cov_types = set(config["models"]["gmm"]["covariance_types"])

    if "selected_clusters" not in config or "gmm" not in config["selected_clusters"]:
        raise KeyError(
            "config['selected_clusters']['gmm'] is missing. Run "
            "Step 2.6 to record a chosen (k, covariance_type) per "
            "year before running Step 2.7."
        )

    selections = config["selected_clusters"]["gmm"]
    validated = {}
    for year in years:
        if year not in selections:
            raise KeyError(
                f"No GMM selection found for year '{year}' in "
                f"config['selected_clusters']['gmm']."
            )
        entry = selections[year]
        if "k" not in entry or "covariance_type" not in entry:
            raise KeyError(
                f"GMM selection for year '{year}' must contain both "
                f"'k' and 'covariance_type'. Got: {entry}."
            )
        k = int(entry["k"])
        covariance_type = entry["covariance_type"]
        if not (k_min <= k <= k_max):
            raise ValueError(
                f"Selected GMM k={k} for year '{year}' is outside the "
                f"swept range [{k_min}, {k_max}]."
            )
        if covariance_type not in valid_cov_types:
            raise ValueError(
                f"Selected GMM covariance_type='{covariance_type}' for "
                f"year '{year}' was not among the swept covariance "
                f"types {sorted(valid_cov_types)}."
            )
        validated[year] = {"k": k, "covariance_type": covariance_type}
    return validated


# =============================================================================
# Step 2.7 — Refit and persist analyst-chosen K-means and GMM models
# =============================================================================
"""
Step 2.7 — Refit and persist analyst-chosen K-means and GMM models.
This is the only step in Phase 2 where models are saved.
"""

def fit_final_kmeans(df, feature_names, k, n_init, base_seed, year):
    """
    Refit K-means at the analyst-chosen k for one year.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names.
    k : int
        Analyst-chosen number of clusters.
    n_init : int
        Number of centroid initializations.
    base_seed : int
        Base seed from config.
    year : str
        Year string.

    Returns
    -------
    tuple[sklearn.cluster.KMeans, np.ndarray]
        - model : fitted KMeans instance.
        - labels : ndarray of hard cluster assignments.
    """
    X = df[feature_names].to_numpy(dtype=np.float64)
    seed = derive_kmeans_seed(base_seed, year, k)
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(X)
    return model, labels


def fit_final_gmm(df, feature_names, k, covariance_type, base_seed, year, config):
    """
    Refit GMM at the analyst-chosen k and covariance type for one
    year.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled feature table for one year.
    feature_names : list[str]
        Ordered list of feature column names.
    k : int
        Analyst-chosen number of mixture components.
    covariance_type : str
        Analyst-chosen covariance type ("full" or "diag").
    base_seed : int
        Base seed from config.
    year : str
        Year string.
    config : dict
        Full config dict, passed through to `derive_gmm_seed` (FIX B:
        previously omitted, causing a TypeError).

    Returns
    -------
    tuple[sklearn.mixture.GaussianMixture, np.ndarray, np.ndarray]
        - model : fitted GaussianMixture instance.
        - labels : ndarray of hard cluster assignments.
        - posteriors : ndarray of soft cluster membership
          probabilities.
    """
    X = df[feature_names].to_numpy(dtype=np.float64)
    seed = derive_gmm_seed(base_seed, year, k, covariance_type, config)
    model = GaussianMixture(
        n_components=k, covariance_type=covariance_type, random_state=seed,
    )
    model.fit(X)
    posteriors = model.predict_proba(X)
    labels = np.argmax(posteriors, axis=1)
    return model, labels, posteriors


def write_final_kmeans_model(model, k, output_dir, config, year):
    """
    Serialize the final K-means model to disk via joblib.

    Parameters
    ----------
    model : sklearn.cluster.KMeans
        Fitted model.
    k : int
        Chosen number of clusters, substituted into the filename.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written joblib file.
    """
    template = config["results"]["step_2_7"]["kmeans_model_template"]
    filename = template.replace("{year}", str(year)).replace("{k}", str(k))
    out_path = Path(output_dir) / filename
    joblib.dump(model, out_path)
    return out_path


def write_final_gmm_model(model, k, output_dir, config, year):
    """
    Serialize the final GMM model to disk via joblib.

    Parameters
    ----------
    model : sklearn.mixture.GaussianMixture
        Fitted model.
    k : int
        Chosen number of mixture components, substituted into the
        filename.
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written joblib file.
    """
    template = config["results"]["step_2_7"]["gmm_model_template"]
    filename = template.replace("{year}", str(year)).replace("{k}", str(k))
    out_path = Path(output_dir) / filename
    joblib.dump(model, out_path)
    return out_path


def write_final_labels_table(df, kmeans_labels, gmm_labels, gmm_posteriors,
                              output_dir, config, year):
    """
    Write a combined per-pixel table of final K-means labels, GMM
    hard labels, and GMM posterior probabilities to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Original scaled feature table for this year.
    kmeans_labels : np.ndarray
        Final K-means hard labels, shape (N,).
    gmm_labels : np.ndarray
        Final GMM hard labels, shape (N,).
    gmm_posteriors : np.ndarray
        Final GMM posterior probabilities, shape (N, k_gmm).
    output_dir : str or pathlib.Path
        Results directory.
    config : dict
        Config dict.
    year : str
        Year string.

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV.
    """
    id_cols = ["row", "col", "x", "y", "year"]
    out_df = df[id_cols].copy()
    out_df["kmeans_label"] = kmeans_labels
    out_df["gmm_label"] = gmm_labels
    for j in range(gmm_posteriors.shape[1]):
        out_df[f"gmm_posterior_{j}"] = gmm_posteriors[:, j]

    template = config["results"]["step_2_7"]["final_labels_template"]
    filename = template.replace("{year}", str(year))
    out_path = Path(output_dir) / filename
    out_df.to_csv(out_path, index=False)
    return out_path


def run_step_2_7(config, output_dir, scaled_tables_by_year, feature_names,
                  kmeans_selection, gmm_selection, update_report_fn):
    """
    Execute Step 2.7: refit and persist the analyst-chosen K-means and
    GMM models per year.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    scaled_tables_by_year : dict[str, pandas.DataFrame]
        Per-year scaled feature tables from Step 1.5.
    feature_names : list[str]
        Ordered list of feature column names.
    kmeans_selection : dict[str, int]
        Validated K-means selections (from
        `validate_kmeans_selection`).
    gmm_selection : dict[str, dict]
        Validated GMM selections (from `validate_gmm_selection`).
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, dict]
        Per-year dictionary with model paths, labels path, and chosen
        k / covariance type. Also writes files to disk and updates
        the run report.
    """
    print(f'{"="*10} Step 2.7 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    kmeans_n_init = config["models"]["kmeans"]["n_init"]
    kmeans_base_seed = config["models"]["kmeans"]["base_seed"]
    gmm_base_seed = config["models"]["gmm"]["base_seed"]

    results_by_year = {}
    step_report = {}

    for year, df in scaled_tables_by_year.items():
        k_kmeans = kmeans_selection[year]
        kmeans_model, kmeans_labels = fit_final_kmeans(
            df, feature_names, k_kmeans, kmeans_n_init, kmeans_base_seed, year
        )
        kmeans_model_path = write_final_kmeans_model(
            kmeans_model, k_kmeans, output_dir, config, year
        )

        gmm_entry = gmm_selection[year]
        k_gmm = gmm_entry["k"]
        cov_type = gmm_entry["covariance_type"]
        gmm_model, gmm_labels, gmm_posteriors = fit_final_gmm(
            df, feature_names, k_gmm, cov_type, gmm_base_seed, year, config
        )
        gmm_model_path = write_final_gmm_model(
            gmm_model, k_gmm, output_dir, config, year
        )

        labels_csv_path = write_final_labels_table(
            df, kmeans_labels, gmm_labels, gmm_posteriors,
            output_dir, config, year,
        )

        results_by_year[year] = {
            "kmeans_model_path": str(kmeans_model_path),
            "gmm_model_path": str(gmm_model_path),
            "labels_csv_path": str(labels_csv_path),
            "kmeans_k": k_kmeans,
            "gmm_k": k_gmm,
            "gmm_covariance_type": cov_type,
        }
        step_report[year] = results_by_year[year]

    update_report_fn(report_path, "step_2_7_final_models", step_report)
    return results_by_year


# =============================================================================
# Step 2.8 — Full-raster cluster prediction and visualization
# =============================================================================
"""
Step 2.8 — Full-raster cluster prediction and visualization.

Applies the analyst-chosen, already-fitted K-means and GMM models
(persisted to disk in Step 2.7) to every QA-passing pixel in the full
raster, producing a wall-to-wall cluster assignment map per year per
method. Models and scalers are loaded from their joblib files on disk.
"""

NODATA_VALUE = -1


def load_scaler_for_year(config, output_dir, year):
    """
    Load the fitted StandardScaler for one year from its joblib file
    on disk (as written in Step 1.5).

    Parameters
    ----------
    config : dict
        Config dict.
    output_dir : str or pathlib.Path
        Results directory.
    year : str
        Year string.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        The loaded, fitted scaler.

    Raises
    ------
    FileNotFoundError
        If the scaler file does not exist at the resolved path.
    """
    template = config["results"]["step_1_5"]["scaler_template"]
    filename = template.replace("{year}", str(year))
    scaler_path = Path(output_dir) / filename
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    return joblib.load(scaler_path)


def load_final_models_for_year(config, output_dir, year, final_models_report):
    """
    Load the fitted K-means and GMM models for one year from their
    joblib files on disk (as written in Step 2.7).

    Parameters
    ----------
    config : dict
        Config dict (unused directly here; paths are taken from
        `final_models_report`).
    output_dir : str or pathlib.Path
        Results directory (unused directly).
    year : str
        Year string, used to key into `final_models_report`.
    final_models_report : dict[str, dict]
        The `step_2_7_final_models` report section (or equivalent
        dict), keyed by year.

    Returns
    -------
    dict
        Dictionary with keys "kmeans_model", "gmm_model", "kmeans_k",
        "gmm_k".

    Raises
    ------
    KeyError
        If `year` is not present in `final_models_report`.
    FileNotFoundError
        If either model file does not exist at its recorded path.
    """
    if year not in final_models_report:
        raise KeyError(
            f"No Step 2.7 model entry found for year '{year}' in "
            f"final_models_report."
        )
    entry = final_models_report[year]

    kmeans_path = Path(entry["kmeans_model_path"])
    gmm_path = Path(entry["gmm_model_path"])
    if not kmeans_path.exists():
        raise FileNotFoundError(f"K-means model file not found: {kmeans_path}")
    if not gmm_path.exists():
        raise FileNotFoundError(f"GMM model file not found: {gmm_path}")

    return {
        "kmeans_model": joblib.load(kmeans_path),
        "gmm_model": joblib.load(gmm_path),
        "kmeans_k": entry["kmeans_k"],
        "gmm_k": entry["gmm_k"],
    }


def assemble_full_feature_array(metric_stack, metric_names, derived_features_config,
                                 feature_names):
    """
    Build the full-raster (H, W, F) feature array in the same feature
    order used for scaling and clustering, including derived duration
    features computed pixel-wise across the whole raster.

    Parameters
    ----------
    metric_stack : np.ndarray
        Raw metric stack of shape (H, W, M).
    metric_names : list[str]
        Names corresponding to the last axis of `metric_stack`.
    derived_features_config : list[dict]
        Derived feature specs from config.
    feature_names : list[str]
        Full ordered list of feature names.

    Returns
    -------
    np.ndarray
        Feature array of shape (H, W, len(feature_names)).

    Raises
    ------
    ValueError
        If any derived feature's operation is unsupported, or if a
        required metric name is missing from `metric_names`.
    """
    name_to_idx = {name: i for i, name in enumerate(metric_names)}
    derived_by_name = {d["name"]: d for d in derived_features_config}

    bands = []
    for name in feature_names:
        if name in name_to_idx:
            bands.append(metric_stack[:, :, name_to_idx[name]])
        elif name in derived_by_name:
            spec = derived_by_name[name]
            if spec["operation"] != "-":
                raise ValueError(
                    f"Unsupported operation '{spec['operation']}' for "
                    f"derived feature '{name}'."
                )
            m1 = spec["metric1"]
            m2 = spec["metric2"]
            if m1 not in name_to_idx or m2 not in name_to_idx:
                raise ValueError(
                    f"Derived feature '{name}' references unknown "
                    f"metric(s): '{m1}', '{m2}'."
                )
            bands.append(
                metric_stack[:, :, name_to_idx[m1]] - metric_stack[:, :, name_to_idx[m2]]
            )
        else:
            raise ValueError(
                f"Feature '{name}' is neither a raw metric nor a "
                f"configured derived feature."
            )

    return np.stack(bands, axis=-1).astype(np.float64)


def full_raster_predict(feature_array, mask, scaler, model):
    """
    Apply a fitted scaler and clustering model to every QA-passing
    pixel in a full-raster feature array.

    Parameters
    ----------
    feature_array : np.ndarray
        Feature array of shape (H, W, F).
    mask : np.ndarray
        Boolean QA mask of shape (H, W).
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for this year.
    model : sklearn.cluster.KMeans or sklearn.mixture.GaussianMixture
        Fitted clustering model.

    Returns
    -------
    np.ndarray
        Integer array of shape (H, W), dtype int16, with cluster
        labels at QA-passing pixels and `NODATA_VALUE` elsewhere.
    """
    H, W, F = feature_array.shape
    label_raster = np.full((H, W), NODATA_VALUE, dtype=np.int16)

    rows, cols = np.where(mask)
    if rows.size == 0:
        return label_raster

    X = feature_array[rows, cols, :].astype(np.float64)
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)

    label_raster[rows, cols] = labels.astype(np.int16)
    return label_raster


def write_cluster_geotiff(label_raster, reference_profile, out_path):
    """
    Write a cluster-label raster to GeoTIFF, using the CRS, transform,
    and shape from a reference raster profile.

    Parameters
    ----------
    label_raster : np.ndarray
        Integer array of shape (H, W), dtype int16.
    reference_profile : rasterio.profiles.Profile
        Profile of the reference raster.
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
        nodata=NODATA_VALUE,
        compress="lzw",
    )
    out_path = Path(out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(label_raster, 1)
    return out_path


def plot_cluster_map(label_raster, k, method_name, year, out_path):
    """
    Render a categorical PNG visualization of a cluster-label raster
    using the tab20 colormap.

    Parameters
    ----------
    label_raster : np.ndarray
        Integer array of shape (H, W), dtype int16.
    k : int
        Number of clusters.
    method_name : str
        Method name for the plot title.
    year : str
        Year string for the plot title.
    out_path : str or pathlib.Path
        Output PNG path.

    Returns
    -------
    pathlib.Path
        Absolute path to the written PNG.
    """
    base_cmap = plt.get_cmap("tab20", 20)
    cluster_colors = [base_cmap(i % 20) for i in range(k)]
    full_colors = ["black"] + cluster_colors
    cmap = ListedColormap(full_colors)

    bounds = [NODATA_VALUE - 0.5] + [i - 0.5 for i in range(k + 1)]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(label_raster, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"{method_name} full-raster cluster assignment — {year} (k={k})")
    ax.set_xticks([])
    ax.set_yticks([])

    legend_handles = [
        Patch(facecolor=cluster_colors[i], label=f"Cluster {i}")
        for i in range(k)
    ]
    legend_handles.append(Patch(facecolor="black", label="No data / QA-failed"))
    ax.legend(
        handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, ncol=1 if k <= 12 else 2,
    )

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def resolve_full_raster_filenames(config, year, k, method):
    """
    Resolve output TIF and PNG filenames for a given method, year, and
    k.

    Parameters
    ----------
    config : dict
        Config dict.
    year : str
        Year string.
    k : int
        Chosen number of clusters for this method/year.
    method : {"kmeans", "gmm"}
        Which method's templates to use.

    Returns
    -------
    tuple[str, str]
        (tif_filename, png_filename).

    Raises
    ------
    ValueError
        If `method` is not "kmeans" or "gmm".
    """
    if method not in ("kmeans", "gmm"):
        raise ValueError(f"Unsupported method '{method}'. Expected 'kmeans' or 'gmm'.")

    tif_template = config["results"]["step_2_8"][f"{method}_full_raster_tif_template"]
    png_template = config["results"]["step_2_8"][f"{method}_full_raster_png_template"]

    tif_filename = tif_template.replace("{year}", str(year)).replace("{k}", str(k))
    png_filename = png_template.replace("{year}", str(year)).replace("{k}", str(k))
    return tif_filename, png_filename


def run_step_2_8(config, output_dir, stacks_by_year, masks_by_year,
                  final_models_report, feature_names, update_report_fn):
    """
    Execute Step 2.8: full-raster cluster prediction and visualization
    for K-means and GMM, per year, loading scalers and models from
    disk.

    Parameters
    ----------
    config : dict
        Fully-loaded config dict.
    output_dir : str or pathlib.Path
        Results directory.
    stacks_by_year : dict[str, tuple[np.ndarray, list[str], rasterio.profiles.Profile]]
        Per-year raw metric stacks from Step 1.1.
    masks_by_year : dict[str, np.ndarray]
        Per-year QA masks from Step 1.2.
    final_models_report : dict[str, dict]
        The `step_2_7_final_models` report section, keyed by year.
    feature_names : list[str]
        Ordered list of feature column names.
    update_report_fn : callable
        Function with signature `update_report_fn(report_path,
        section_name, content)`.

    Returns
    -------
    dict[str, dict]
        Per-year dictionary with output paths and chosen k for each
        method.

    Raises
    ------
    KeyError
        If a year is missing from `final_models_report`.
    FileNotFoundError
        If a scaler or model file does not exist at its expected
        path.
    ValueError
        Propagated from `assemble_full_feature_array`.
    """
    print(f'{"="*10} Step 2.8 {"="*10}')

    report_path = Path(output_dir) / config["report"].replace("{year}", str(config['years'][0]))
    derived_features_config = config["derived_features"]

    step_report = {}
    outputs_by_year = {}

    for year, (metric_stack, metric_names, _stack_profile) in stacks_by_year.items():
        print(f'{year=}')
        mask = masks_by_year[year]

        scaler = load_scaler_for_year(config, output_dir, year)
        loaded_models = load_final_models_for_year(
            config, output_dir, year, final_models_report
        )

        resolved_qa_layers = resolve_layer_paths_for_year(
            config["qa"]["layers"], config, year
        )
        ref_qa_path = resolved_qa_layers[0]["path"]
        with rasterio.open(ref_qa_path) as ref_src:
            reference_profile = ref_src.profile

        feature_array = assemble_full_feature_array(
            metric_stack, metric_names, derived_features_config, feature_names
        )

        year_outputs = {}
        for method in ("kmeans", "gmm"):
            print(f'{method=}')

            model = loaded_models[f"{method}_model"]
            k = loaded_models[f"{method}_k"]

            print('full_raster_predict')
            label_raster = full_raster_predict(feature_array, mask, scaler, model)

            print('write')
            tif_filename, png_filename = resolve_full_raster_filenames(
                config, year, k, method
            )
            tif_path = write_cluster_geotiff(
                label_raster, reference_profile, Path(output_dir) / tif_filename
            )
            png_path = plot_cluster_map(
                label_raster, k, "GMM" if method == "gmm" else "K-means",
                year, Path(output_dir) / png_filename,
            )

            year_outputs[f"{method}_tif"] = str(tif_path)
            year_outputs[f"{method}_png"] = str(png_path)
            year_outputs[f"{method}_k"] = k

        outputs_by_year[year] = year_outputs
        step_report[year] = year_outputs

    update_report_fn(report_path, "step_2_8_full_raster_clustering", step_report)
    return outputs_by_year