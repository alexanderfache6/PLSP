'''
"Good Quality" Pixel check on `NumCycles` and `QA`
'''

import csv
import glob
import os
import sys

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns∏
from aquarel import load_theme
import matplotlib as mpl


def get_nc_files(sites, product_dirs):
    dirs = []
    for site in sites:
        for product_dir in product_dirs:
            dirs.append(Path('/projectnb/modislc/users/fache/data/planet/') / product_dir / site)

    nc_files = []
    for d in dirs:
        if os.path.isdir(d):
            nc_files.extend(glob.glob(os.path.join(d, '*.nc'), recursive=True))
    nc_files = sorted(set(nc_files))

    return nc_files


def load(ds, name):
    if name not in ds.variables:
        raise KeyError(f'variable "{name}" not found (available: {list(ds.variables)})')

    arr = ds.variables[name][:]
    if np.ma.isMaskedArray(arr): # NetCDF marks missing pixels as masked, fill missing values
        arr = arr.filled(-1) # -1 fails NumCycles>=1 and QA==1
    return np.asarray(arr)


def analyze_pixel_quality(path, QA):
    with Dataset(path, 'r') as ds:
        num_cycles_layer = load(ds, 'NumCycles')
        qa_layer = load(ds, 'QA')

        return np.count_nonzero((num_cycles_layer >= 1) & (qa_layer >= 1) & (qa_layer <= QA)), int(num_cycles_layer.size)


def run_all_analyze_pixel_quality(nc_files, QA):
    results = []
    for path in tqdm(nc_files):
        good_pixels, total_pixels = analyze_pixel_quality(path, QA)
        percent_good_pixels = 100.0 * good_pixels / total_pixels
        results.append({
            'file': path,
            'site': Path(path).parent.name,
            'year': Path(path).stem.split('_')[-1],
            'good_pixels': good_pixels,
            'total_pixels': total_pixels,
            'percent_good_pixels': f'{percent_good_pixels:.3f}%'
        })
    
    return results


def save_results_to_csv(results, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'site', 'year', 'good_pixels', 'total_pixels', 'percent_good_pixels'])
        writer.writeheader()
        writer.writerows(results)
        print(f'saved: {csv_file}')


def save_results_to_png(csv_file, png_file, QA):
    df = pd.read_csv(csv_file)
    df['year'] = pd.to_numeric(df['year'])
    df['percent_good_pixels'] = pd.to_numeric(df['percent_good_pixels'].astype(str).str.rstrip('%'))
    df = df.sort_values('year')
    df['year'] = df['year'].astype(int).astype(str)

    theme = load_theme('arctic_dark')
    theme.apply()
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x='site',
        y='percent_good_pixels',
        hue='year',
        palette='Blues',
        ax=ax
    )

    ax.set_xlabel('Site')
    ax.set_ylabel('Good Pixels (%)')
    ax.set_ylim(bottom=0, top=100)
    ax.set_title(f"Good Pixels % with NumCycles >= 1 and QA >= 1 and QA <= {QA}")
    ax.legend(title='Year', loc='upper right', frameon=True)

    theme.apply_transforms()
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)

    fig.tight_layout()
    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f'saved: {png_file}')


def main(QA_level):
    sites = ['ARM_Southern_Great_Plains_site', 'Mountainair_Pinyon-Juniper_Woodland', 'NEON_Konza_Prairie_Biological_Station', 'Santa_Rita_Grassland', 'Santa_Rita_Mesquite', 'Sevilleta_shrubland', 'Walnut_Gulch_Kendall_Grasslands', 'Walnut_Gulch_Lucky_Hills_Shrub', 'Willard_Juniper_Savannah']
    print(f'{len(sites)=}')
    product_dirs = ['PLSP_production_nc', 'PLSP_stage_nc']
    print(f'{len(product_dirs)=}')

    nc_files = get_nc_files(sites, product_dirs)
    print(f'{len(nc_files)=}')
    results = run_all_analyze_pixel_quality(nc_files, QA_level)
    print(f'{len(results)=}')

    csv_file = f'/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_analysis/check_site_pixel_quality_results_QA{QA_level}.csv'
    png_file = f'/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_analysis/check_site_pixel_quality_results_QA{QA_level}.png'
    save_results_to_csv(results, csv_file)
    save_results_to_png(csv_file, png_file, QA_level)

    print('done')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python run_check_site_pixel_quality.py <QA_level>")
    main(int(sys.argv[1]))