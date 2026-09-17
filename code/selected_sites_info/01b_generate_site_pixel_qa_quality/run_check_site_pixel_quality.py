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
import seaborn as sns
from aquarel import load_theme
import matplotlib as mpl


SITES_DIR = Path(__file__).resolve().parent.parent
SITES_CSV = SITES_DIR / 'data' / '01_selected_sites_raw_handgenerated_2.csv'

OUT_DIR = Path('/projectnb/modislc/users/fache/src/PLSP/code/selected_sites_info/01b_generate_site_pixel_qa_quality')
OUT_FILE_PRE = 'check_site_pixel_quality_results_QA'


def get_sites(csv_file):
    '''Site directory names come from the plsp_raw_id column, domain from the domain column
    of the hand-generated site list. Returns a list of (site, domain) tuples.'''
    with open(csv_file, newline='') as f:
        sites = [(row['plsp_raw_id'].strip(), row['neon_domain'].strip()) for row in csv.DictReader(f)]
    sites.sort(key=lambda s: (int(s[1].lstrip('Dd')), s[0])) # dort by domain ascending then site name ascending
    return sites


def get_nc_files(sites, product_dirs):
    site_names = [site for site, domain in sites]

    dirs = []
    for site in site_names:
        for product_dir in product_dirs:
            dirs.append(Path('/projectnb/modislc/users/fache/data/planet/') / product_dir / site)

    nc_files = []
    for d in dirs:
        if os.path.isdir(d):
            nc_files.extend(glob.glob(os.path.join(d, '*.nc'), recursive=True))
    nc_files = sorted(set(nc_files))

    found = {Path(p).parent.name for p in nc_files}
    for site in site_names:
        if site not in found:
            print(f'WARNING: no .nc files for "{site}" in any of {product_dirs}', file=sys.stderr)

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

        return np.count_nonzero((num_cycles_layer == 1) & (qa_layer >= 1) & (qa_layer <= QA)), int(num_cycles_layer.size)


def run_all_analyze_pixel_quality(nc_files, QA, site_domain_map):
    results = []
    for path in tqdm(nc_files):
        good_pixels, total_pixels = analyze_pixel_quality(path, QA)
        percent_good_pixels = 100.0 * good_pixels / total_pixels
        site = Path(path).parent.name
        results.append({
            'file': path,
            'site': site,
            'neon_domain': site_domain_map.get(site, ''),
            'year': Path(path).stem.split('_')[-1],
            'good_pixels': good_pixels,
            'total_pixels': total_pixels,
            'percent_good_pixels': f'{percent_good_pixels:.3f}%'
        })

    return results


def save_results_to_csv(results, csv_file):
    # domain is intentionally excluded from the saved CSV
    fieldnames = ['file', 'site', 'year', 'good_pixels', 'total_pixels', 'percent_good_pixels']
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
        print(f'saved: {csv_file}')

def save_results_to_png(csv_file, png_file, QA, site_domain_map):
    df = pd.read_csv(csv_file)
    df['year'] = pd.to_numeric(df['year'])
    df['percent_good_pixels'] = pd.to_numeric(df['percent_good_pixels'].astype(str).str.rstrip('%'))
 
    # domain isn't in the CSV, so re-attach it here just for labeling
    df['domain'] = df['site'].map(site_domain_map).fillna('')
    df['domain_num'] = df['domain'].str.lstrip('Dd').replace('', np.nan).astype(float)
    df['site_label'] = df.apply(lambda r: f"{r['site']} ({r['domain']})" if r['domain'] else r['site'], axis=1)
 
    df = df.sort_values(by=['domain_num', 'site', 'year'], ascending=True)
    df['year'] = df['year'].astype(int).astype(str)
 
    # explicit x-axis order: domain (numeric), then site name
    x_order = df[['domain_num', 'site', 'site_label']].drop_duplicates().sort_values(by=['domain_num', 'site'])['site_label'].tolist()
 
    theme = load_theme('arctic_dark')
    theme.apply()
    mpl.rcParams['font.family'] = 'DejaVu Sans'
 
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x='site_label',
        y='percent_good_pixels',
        hue='year',
        order=x_order,
        palette='Blues',
        ax=ax
    )
 
    ax.set_xlabel('Site')
    ax.set_ylabel('Good Pixels (%)')
    ax.set_ylim(bottom=0, top=100)
    ax.set_title(f"Good Pixels (%) for NumCycles = 1 and QA = [{', '.join(str(x) for x in range(1, QA + 1, 1))}]")
    ax.legend(title='Year', loc='upper left', frameon=True)
 
    theme.apply_transforms()
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
 
    fig.tight_layout()
    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f'saved: {png_file}')


def main(QA_level):
    sites = get_sites(SITES_CSV)
    print(f'{SITES_CSV=}')
    print(f'{len(sites)=}')
    print(f'{sites=}')
    site_domain_map = dict(sites)

    product_dirs = ['PLSP_production_nc', 'PLSP_stage_nc']
    print(f'{len(product_dirs)=}')

    nc_files = get_nc_files(sites, product_dirs)
    print(f'{len(nc_files)=}')
    results = run_all_analyze_pixel_quality(nc_files, QA_level, site_domain_map)
    print(f'{len(results)=}')

    csv_file = OUT_DIR / f'{OUT_FILE_PRE}{QA_level}.csv'
    png_file = OUT_DIR / f'{OUT_FILE_PRE}{QA_level}.png'
    save_results_to_csv(results, csv_file)
    save_results_to_png(csv_file, png_file, QA_level, site_domain_map)

    print('done')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python run_check_site_pixel_quality.py <QA_level>")
    main(int(sys.argv[1]))