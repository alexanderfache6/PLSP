#!/usr/bin/env bash
##
# run_get_site_tif_counts.sh
#
# Runs get_site_tif_counts.sh for each site in SITES, for both
# "new" and "archive" modes, launching each in the background
# without waiting for any to finish.

set -uo pipefail

SCRIPT="./0_get_site_tif_count.sh"

# --- List of site names is defined here ---
SITES=(
    "ARM_Southern_Great_Plains_site-_Lamont"
    "Mountainair_Pinyon-Juniper_Woodland"
    "Konza_Prairie_Biological_Station_NEON"
    "Santa_Rita_Grassland"
    "Santa_Rita_Mesquite"
    "Sevilleta_shrubland"
    "Walnut_Gulch_Kendall_Grasslands"
    "Walnut_Gulch_Lucky_Hills_Shrub"
    "Willard_Juniper_Savannah"
)

MODES=("new" "archive")

for site in "${SITES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "Running: $SCRIPT $site $mode"
        "$SCRIPT" "$site" "$mode" &
        # echo "  -> PID $!"
    done
done

echo "All jobs launched."