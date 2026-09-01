# NOTE run `run_get_site_tif_counts.sh` then run this

#!/usr/bin/env python3
"""Combine per-site NEW/ARCHIVE tif count CSVs into two site-wide DataFrames.

Reads the per-site monthly and yearly CSVs produced by
`get_site_tif_counts.sh` (run for both "new" and "archive" modes via
`run_get_site_tif_counts.sh`) from the current working directory,
combines them into two DataFrames spanning the full global time range,
writes them to CSV, then deletes the per-site input files.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

METADATA_CSV = Path("/projectnb/modislc/users/fache/src/PLSP/code/selected_sites_info/data/01_selected_sites_raw_handgenerated_2.csv")
INPUT_DIR = Path.cwd()
OUTPUT_DIR = Path.cwd()

MONTHLY_OUTPUT_NAME = "2A_all_sites_monthly_tif_counts.csv"
YEARLY_OUTPUT_NAME  = "2B_all_sites_yearly_tif_counts.csv"
SUMMARY_PNG_NAME    = "2C_all_sites_summary.png"

MODES = ("ARCHIVE", "NEW")  # column order per site: archive, new, total
GRANULARITIES = ("monthly", "yearly")

def discover_sites(metadata_csv: Path, input_dir: Path) -> list[tuple[str, str]]:
    """Discover sites with per-site CSVs on disk and map them to their site_id.

    Iterates rows of the metadata CSV in order. For each row, checks whether
    any of the four per-site CSVs (monthly/yearly x NEW/ARCHIVE) exist in
    `input_dir` under `site_name`. If not, checks `site_name_2`. The names
    are treated as exclusive: only one will match. Sites with no matching
    files are skipped with a warning.

    Args:
        metadata_csv: Path to the metadata CSV containing at minimum the
            columns `site_id`, `site_name`, and `site_name_2`.
        input_dir: Directory to look in for the per-site CSVs.

    Returns:
        Ordered list of `(file_stem, site_id)` tuples, in the order the
        rows appear in the metadata CSV, restricted to sites that have at
        least one per-site CSV present. `file_stem` is whichever of
        `site_name` or `site_name_2` matched files on disk.
    """
    meta = pd.read_csv(metadata_csv)

    discovered: list[tuple[str, str]] = []
    for _, row in meta.iterrows():
        site_id = str(row["plsp_raw_id"])
        # site_name = row.get("site_name")
        # site_name_2 = row.get("site_name_2")

        candidates = []
        # if isinstance(site_name, str) and site_name.strip():
        #     candidates.append(site_name.strip())
        # if isinstance(site_name_2, str) and site_name_2.strip():
        #     candidates.append(site_name_2.strip())
        
        candidates.append(site_id.strip())

        matched_stem: str | None = None
        for stem in candidates:
            if _any_site_file_exists(stem, input_dir):
                matched_stem = stem
                break

        if matched_stem is None:
            print(f"Warning: no per-site CSVs found for site_id={site_id} "
                  f"(tried {candidates}). Skipping.")
            continue

        discovered.append((matched_stem, site_id))
    
    print(f"{discovered=}")

    return discovered


def _any_site_file_exists(file_stem: str, input_dir: Path) -> bool:
    """Return True if any of the four per-site CSVs exist for `file_stem`.

    Args:
        file_stem: The filename prefix (either `site_name` or `site_name_2`).
        input_dir: Directory to look in.

    Returns:
        True if at least one of the four expected CSVs exists.
    """
    for granularity in GRANULARITIES:
        for mode in MODES:
            path = _site_file_path(file_stem, granularity, mode, input_dir)
            if path.exists():
                return True
    return False


def _site_file_path(
    file_stem: str, granularity: str, mode: str, input_dir: Path
) -> Path:
    """Build the expected path for one per-site CSV.

    Args:
        file_stem: Filename prefix.
        granularity: Either "monthly" or "yearly".
        mode: Either "NEW" or "ARCHIVE".
        input_dir: Directory the file lives in.

    Returns:
        The `Path` to the per-site CSV (may or may not exist on disk).
    """
    return input_dir / f"{file_stem}_{granularity}_tif_counts_{mode}.csv"


def read_site_counts(
    file_stem: str, granularity: str, mode: str, input_dir: Path
) -> pd.DataFrame:
    """Read a single per-site count CSV, tolerating missing/empty files.

    Args:
        file_stem: Filename prefix (whichever of `site_name` / `site_name_2`
            was resolved for this site).
        granularity: Either "monthly" or "yearly".
        mode: Either "NEW" or "ARCHIVE".
        input_dir: Directory containing the CSV.

    Returns:
        DataFrame with columns `["year", "month", "count"]` for monthly or
        `["year", "count"]` for yearly. If the file is missing or has no
        data rows, an empty DataFrame with the correct columns is returned
        and a warning is printed.
    """
    path = _site_file_path(file_stem, granularity, mode, input_dir)
    expected_cols = ["year", "month", "count"] if granularity == "monthly" else ["year", "count"]

    if not path.exists():
        print(f"Warning: missing file {path.name}; treating as all zeros.")
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(path)
    if df.empty:
        print(f"Warning: file {path.name} has no data rows; treating as all zeros.")
        return pd.DataFrame(columns=expected_cols)

    return df[expected_cols]


def build_monthly_df(
    discovered: list[tuple[str, str]], input_dir: Path
) -> pd.DataFrame:
    """Build the combined monthly DataFrame across all discovered sites.

    Rows span every month from the earliest `(year, month)` seen across
    all sites and both NEW/ARCHIVE files, to the latest such
    `(year, month)`, with no gaps. Missing months for a given site are
    filled with 0. Per-site columns are `<site_id>_archive`,
    `<site_id>_new`, `<site_id>_total` (where `_total = archive + new`).
    Site column order matches `discovered`.

    Args:
        discovered: Ordered `(file_stem, site_id)` pairs from `discover_sites`.
        input_dir: Directory containing the per-site CSVs.

    Returns:
        DataFrame with columns `year, month, <site_id>_archive,
        <site_id>_new, <site_id>_total, ...`.
    """
    site_frames: dict[str, dict[str, pd.DataFrame]] = {}
    all_year_months: set[tuple[int, int]] = set()

    for file_stem, site_id in discovered:
        site_frames[site_id] = {}
        for mode in MODES:
            df = read_site_counts(file_stem, "monthly", mode, input_dir)
            if not df.empty:
                df = df.astype({"year": int, "month": int, "count": int})
                all_year_months.update(zip(df["year"], df["month"]))
            site_frames[site_id][mode] = df

    if not all_year_months:
        print("Warning: no monthly data found for any site.")
        cols = ["year", "month"] + [
            f"{sid}_{suf}" for _, sid in discovered for suf in ("archive", "new", "total")
        ]
        return pd.DataFrame(columns=cols)

    index_df = _full_monthly_index(all_year_months)

    out = index_df.copy()
    for _, site_id in discovered:
        archive_df = site_frames[site_id]["ARCHIVE"]
        new_df = site_frames[site_id]["NEW"]

        archive_series = _align_monthly(archive_df, index_df)
        new_series = _align_monthly(new_df, index_df)

        out[f"{site_id}_archive"] = archive_series
        out[f"{site_id}_new"] = new_series
        out[f"{site_id}_total"] = archive_series + new_series

    return out


def build_yearly_df(
    discovered: list[tuple[str, str]], input_dir: Path
) -> pd.DataFrame:
    """Build the combined yearly DataFrame across all discovered sites.

    Rows span every year from the earliest to the latest year seen across
    all sites and both NEW/ARCHIVE files, with no gaps. Missing years for
    a given site are filled with 0. Per-site columns are
    `<site_id>_archive`, `<site_id>_new`, `<site_id>_total`. Site column
    order matches `discovered`.

    Args:
        discovered: Ordered `(file_stem, site_id)` pairs from `discover_sites`.
        input_dir: Directory containing the per-site CSVs.

    Returns:
        DataFrame with columns `year, <site_id>_archive, <site_id>_new,
        <site_id>_total, ...`.
    """
    site_frames: dict[str, dict[str, pd.DataFrame]] = {}
    all_years: set[int] = set()

    for file_stem, site_id in discovered:
        site_frames[site_id] = {}
        for mode in MODES:
            df = read_site_counts(file_stem, "yearly", mode, input_dir)
            if not df.empty:
                df = df.astype({"year": int, "count": int})
                all_years.update(df["year"].tolist())
            site_frames[site_id][mode] = df

    if not all_years:
        print("Warning: no yearly data found for any site.")
        cols = ["year"] + [
            f"{sid}_{suf}" for _, sid in discovered for suf in ("archive", "new", "total")
        ]
        return pd.DataFrame(columns=cols)

    index_df = pd.DataFrame(
        {"year": list(range(min(all_years), max(all_years) + 1))}
    )

    out = index_df.copy()
    for _, site_id in discovered:
        archive_series = _align_yearly(site_frames[site_id]["ARCHIVE"], index_df)
        new_series = _align_yearly(site_frames[site_id]["NEW"], index_df)

        out[f"{site_id}_archive"] = archive_series
        out[f"{site_id}_new"] = new_series
        out[f"{site_id}_total"] = archive_series + new_series

    return out


def _full_monthly_index(year_months: set[tuple[int, int]]) -> pd.DataFrame:
    """Build a gap-free monthly index DataFrame spanning min to max.

    Args:
        year_months: Set of `(year, month)` tuples that were observed.

    Returns:
        DataFrame with columns `year, month`, sorted ascending, containing
        every month from the earliest to the latest observed.
    """
    start = min(year_months)
    end = max(year_months)

    rows = []
    y, m = start
    end_y, end_m = end
    while (y, m) <= (end_y, end_m):
        rows.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return pd.DataFrame(rows, columns=["year", "month"])


def _align_monthly(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.Series:
    """Align a per-site monthly frame to the full index, filling gaps with 0.

    Args:
        df: Per-site DataFrame with columns `year`, `month`, `count`. May
            be empty.
        index_df: DataFrame with the full `year`, `month` index.

    Returns:
        Integer Series of counts aligned to `index_df`, with missing rows
        set to 0.
    """
    if df.empty:
        return pd.Series([0] * len(index_df), index=index_df.index, dtype=int)

    merged = index_df.merge(df, on=["year", "month"], how="left")
    return merged["count"].fillna(0).astype(int)


def _align_yearly(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.Series:
    """Align a per-site yearly frame to the full index, filling gaps with 0.

    Args:
        df: Per-site DataFrame with columns `year`, `count`. May be empty.
        index_df: DataFrame with the full `year` index.

    Returns:
        Integer Series of counts aligned to `index_df`, with missing rows
        set to 0.
    """
    if df.empty:
        return pd.Series([0] * len(index_df), index=index_df.index, dtype=int)

    merged = index_df.merge(df, on=["year"], how="left")
    return merged["count"].fillna(0).astype(int)


def save_combined_dfs(
    df_monthly: pd.DataFrame,
    df_yearly: pd.DataFrame,
    monthly_path: Path,
    yearly_path: Path
) -> tuple[Path, Path]:
    """Write the two combined DataFrames to CSV.

    Args:
        df_monthly: Combined monthly DataFrame.
        df_yearly: Combined yearly DataFrame.
        monthly_path
        yearly_path

    Returns:
        Tuple `(monthly_path, yearly_path)` of the written files.
    """

    df_monthly.to_csv(monthly_path, index=False)
    df_yearly.to_csv(yearly_path, index=False)

    print(f"Wrote {monthly_path}")
    print(f"Wrote {yearly_path}")

    return monthly_path, yearly_path


def cleanup_per_site_files(
    discovered: list[tuple[str, str]], input_dir: Path
) -> None:
    """Delete the four per-site CSVs for every discovered site.

    Only files that actually exist are removed; missing ones are ignored
    silently (they were already treated as zeros upstream).

    Args:
        discovered: Ordered `(file_stem, site_id)` pairs from `discover_sites`.
        input_dir: Directory the per-site CSVs live in.
    """
    removed = 0
    path_dne = 0
    for file_stem, _site_id in discovered:
        for granularity in GRANULARITIES:
            for mode in MODES:
                path = _site_file_path(file_stem, granularity, mode, input_dir)
                if path.exists():
                    path.unlink() # delete file
                    removed += 1
                else:
                    print(f'dne: {path}')
                    path_dne += 1
    print(f"Removed {removed} per-site CSV file(s).")
    print(f"Paths that do not exist {path_dne}")

def plot_summary(
    df_monthly: pd.DataFrame,
    df_yearly: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Render and save the two-panel summary figure as PNG.

    Top subplot: overlaid multi-line time series of monthly totals
    (archive + new) per site, with one line per site colored from the
    `Set3` qualitative colormap.

    Bottom subplot: grouped bar chart per year with one bar per site
    per year. Each bar is internally stacked with the archive portion
    (alpha=0.5) below the new portion (alpha=1.0), reusing the same
    per-site `Set3` colors as the top subplot. The bottom legend
    contains two proxy handles indicating the archive/new shading
    convention; site identity is conveyed by the top subplot's legend.

    Args:
        df_monthly: Combined monthly DataFrame from `build_monthly_df`.
            Must contain `year`, `month`, and `<site_id>_total` columns.
        df_yearly: Combined yearly DataFrame from `build_yearly_df`.
            Must contain `year`, `<site_id>_archive`, and
            `<site_id>_new` columns.
        output_dir: Directory to save the PNG into.

    Returns:
        Path to the saved PNG file.

    Raises:
        ValueError: If the number of sites exceeds the number of
            discrete colors available in the `Set3` colormap (12).
    """

    # Discover site_ids from column names, preserving DataFrame order
    site_ids = [
        col[: -len("_total")]
        for col in df_monthly.columns
        if col.endswith("_total")
    ]
    n_sites = len(site_ids)

    # Discrete Set3 swatches, one per site
    set3_colors = plt.get_cmap("Set3").colors
    if n_sites > len(set3_colors):
        raise ValueError(
            f"Too many sites ({n_sites}) for Set3 colormap "
            f"({len(set3_colors)} colors available)."
        )
    site_colors = dict(zip(site_ids, set3_colors[:n_sites]))

    fig, (ax_line, ax_bar) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top subplot: overlaid monthly line chart ---
    dates = pd.to_datetime(df_monthly[["year", "month"]].assign(day=1))
    for site_id in site_ids:
        ax_line.plot(
            dates,
            df_monthly[f"{site_id}_total"],
            label=site_id,
            color=site_colors[site_id],
            linewidth=1.5,
        )
    ax_line.set_xlabel("month")
    ax_line.set_ylabel("count")
    ax_line.set_title("monthly tif counts per site")
    ax_line.xaxis.set_major_locator(mdates.YearLocator())
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_line.xaxis.set_minor_locator(mdates.MonthLocator())
    ax_line.set_xlim(left=dates.min().replace(month=1, day=1)) # NOTE start year for all data
    ax_line.legend(
        loc="upper left",
        # bbox_to_anchor=(1.01, 1.0),
        fontsize=10,
        title="Site ID"
    )
    ax_line.grid(True, alpha=0.3)

    # --- Bottom subplot: grouped bars per year, stacked archive/new ---
    years = df_yearly["year"].to_numpy()
    group_width = 0.8
    bar_width = group_width / n_sites

    for i, site_id in enumerate(site_ids):
        offset = (i - (n_sites - 1) / 2) * bar_width
        x = years + offset

        archive_vals = df_yearly[f"{site_id}_archive"].to_numpy()
        new_vals = df_yearly[f"{site_id}_new"].to_numpy()

        ax_bar.bar(
            x, archive_vals,
            width=bar_width,
            color=site_colors[site_id],
            alpha=0.5,
            edgecolor="none",
        )
        ax_bar.bar(
            x, new_vals,
            width=bar_width,
            bottom=archive_vals,
            color=site_colors[site_id],
            alpha=1.0,
            edgecolor="none",
        )

    ax_bar.set_xlabel("year")
    ax_bar.set_ylabel("count")
    ax_bar.set_title("yearly tif counts per site")
    ax_bar.set_xticks(years)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # Two proxy handles for the archive/new shading convention
    proxy_archive = Patch(facecolor="gray", alpha=0.5, label="archive")
    proxy_new = Patch(facecolor="gray", alpha=1.0, label="new")
    ax_bar.legend(handles=[proxy_archive, proxy_new], loc="upper left", fontsize=10)

    fig.tight_layout()

    out_path = output_dir / SUMMARY_PNG_NAME
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {out_path}")
    return out_path

def main() -> None:
    """Run the full pipeline: discover, combine, save, cleanup."""
    print(f"Reading metadata from: {METADATA_CSV}")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    monthly_path = OUTPUT_DIR / MONTHLY_OUTPUT_NAME
    yearly_path = OUTPUT_DIR / YEARLY_OUTPUT_NAME

    if not monthly_path.exists() or not yearly_path.exists():
        print('running')

        discovered = discover_sites(METADATA_CSV, INPUT_DIR)
        if not discovered:
            print("No sites with per-site CSVs found. Nothing to do.")
            return

        print(f"Discovered {len(discovered)} site(s) with data:")
        for file_stem, site_id in discovered:
            print(f"{site_id}  <-  {file_stem}")
        
        df_monthly = build_monthly_df(discovered, INPUT_DIR)
        df_yearly = build_yearly_df(discovered, INPUT_DIR)

        save_combined_dfs(df_monthly, df_yearly, monthly_path, yearly_path)
        print(discovered)

        cleanup_per_site_files(discovered, INPUT_DIR)
    else:
        print('monthly and yearly files were already created, delete if a new run needs to happen')
    
    # create visualization
    df_monthly = pd.read_csv(monthly_path)
    df_yearly = pd.read_csv(yearly_path)
    print('plotting')
    plot_summary(df_monthly, df_yearly, OUTPUT_DIR)

    print("done")


if __name__ == "__main__":
    main()