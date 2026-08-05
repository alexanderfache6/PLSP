from pathlib import Path

TILE_IDS = [
    '515000_3530000',
    '515000_3531000',
    '511000_3528000',
]
YEAR = '2022'
SITE_ID = 'SRER'
SITE_NAME = 'Santa_Rita_Experimental_Range_NEON'

SAVE_DIR = Path('/projectnb/modislc/users/fache/results/results_ground_truth_land_cover')
OUTPUT_DIR = SAVE_DIR / SITE_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_DATA_PATH = Path(f'/projectnb/modislc/users/fache/data/NEON/{SITE_NAME}')


def build_paths(tile_id):
    rgb = SITE_DATA_PATH / 'NEON_images-camera-ortho-mosaic' / f'NEON.D14.{SITE_ID}.DP3.30010.001.2022-08.basic' / f'{YEAR}_{SITE_ID}_5_{tile_id}_image.tif'
    vi_dir = SITE_DATA_PATH / 'NEON_indices-veg-spectrometer-bidir-mosaic' / f'NEON.D14.{SITE_ID}.DP3.30026.002.2022-08.basic' / f'NEON_D14_{SITE_ID}_DP3_{tile_id}_bidirectional_VegIndices'
    ndvi = vi_dir / f'NEON_D14_{SITE_ID}_DP3_{tile_id}_bidirectional_NDVI.tif'
    savi = vi_dir / f'NEON_D14_{SITE_ID}_DP3_{tile_id}_bidirectional_SAVI.tif'
    chm = SITE_DATA_PATH / 'NEON_struct-ecosystem' / f'NEON.D14.{SITE_ID}.DP3.30015.001.2022-08.basic' / f'NEON_D14_{SITE_ID}_DP3_{tile_id}_CHM.tif'
    return rgb, ndvi, savi, chm