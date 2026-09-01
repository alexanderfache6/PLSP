#!/usr/bin/env bash
##
# run_get_site_tif_counts.sh
#
# Runs get_site_tif_counts.sh for each site in SITES, for both
# "new" and "archive" modes, launching each in the background
# without waiting for any to finish.

set -uo pipefail

SCRIPT="./0_helper_get_site_tif_count.sh"

# --- List of site names is defined here ---
SITES=(
    # "ARM_Southern_Great_Plains_site-_Lamont"
    # "Mountainair_Pinyon-Juniper_Woodland"
    # "Konza_Prairie_Biological_Station_NEON"
    # "Santa_Rita_Grassland"
    # "Santa_Rita_Mesquite"
    # "Sevilleta_shrubland"
    # "Walnut_Gulch_Lucky_Hills_Shrub"

    # NOTE list taken from code/selected_sites_info/data/01_selected_sites_raw_handgenerated_2.csv
    "Santa_Rita_Experimental_Range_NEON"
    "Walnut_Gulch_Kendall_Grasslands"
    "Onaqui_NEON"
    "Reynolds_Creek_Wyoming_big_sagebrush"
    "Moab_NEON"
    "Willard_Juniper_Savannah"
    "San_Joaquin_Experimental_Range_NEON"
    "Tonzi_Ranch"
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