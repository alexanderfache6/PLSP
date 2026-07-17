import subprocess
from pathlib import Path

# --- CONFIG ---
folder = Path("/projectnb/modislc/users/fache/data/planet/phenologyGeotiff/Walnut_Gulch_Kendall_Grasslands/2020")
srs = "EPSG:32612"
ullr = ["595230", "3516717", "605232", "3506715"]  # xmin ymax xmax ymin
# ---------------

for src in folder.glob("*.tif"):
    # skip files that are themselves already-processed outputs
    if src.stem.endswith("_crs"):
        continue

    out = src.with_name(src.stem + "_crs.tif")

    if out.exists():
        print(f"Skipping {src.name} (already has {out.name})")
        continue

    cmd = [
        "gdal_translate",
        "-a_srs", srs,
        "-a_ullr", *ullr,
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=3",
        "-co", "TILED=YES",
        str(src), str(out)
    ]

    print(f"Processing {src.name} -> {out.name}")
    result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    else:
        print(f"  Done.")

print("All files processed.")