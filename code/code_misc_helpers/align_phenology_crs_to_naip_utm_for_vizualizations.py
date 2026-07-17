import subprocess
from pathlib import Path

# USAGE
SITE = 'Willard_Juniper_Savannah'
YEAR = '2022'

data_folder = Path("/projectnb/modislc/users/fache/data/planet/phenologyGeotiff") / SITE / YEAR
crs_folder = Path("/projectnb/modislc/users/fache/data/planet/phenologyGeotiff") / SITE / f"{YEAR}_crs"
srs = "EPSG:32612"
ullr = ["595230", "3516717", "605232", "3506715"]  # xmin ymax xmax ymin

# TODO read one mosaic to get correct srs, ullr per site

print(f'{"-"*10} running for {SITE} {YEAR} {"-"*10}')

if not crs_folder.exists():
    crs_folder.mkdir()
    print(f'created: {crs_folder}')
else:
    print(f'exists: {crs_folder}')

for src_file in data_folder.glob("*.tif"):
    out_file = crs_folder / f'{src_file.stem}_crs.tif'

    cmd = [
        "gdal_translate",
        "-a_srs", srs,
        "-a_ullr", *ullr,
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=3",
        "-co", "TILED=YES",
        str(src_file), str(out_file)
    ]

    print(f"{src_file.name} -> {out_file.name}")
    result = subprocess.run(cmd, cwd=data_folder, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}")

print(f'{"-"*10} done {"-"*10}')
