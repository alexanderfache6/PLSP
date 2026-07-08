#!/usr/bin/env bash
##
# run_all_get_site_tif_counts.sh
#
# Runs get_site_tif_counts.sh for each site in SITES, for both
# "new" and "archive" modes, launching each in the background
# without waiting for any to finish.

set -uo pipefail

SCRIPT="./get_site_tif_counts.sh"

# --- List of site names is defined here ---
SITES=(
    "Walnut_Gulch_Kendall_Grasslands"
    "Willard_Juniper_Savannah"
)

MODES=("new" "archive")

for site in "${SITES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "Running: $SCRIPT $site $mode"
        "$SCRIPT" "$site" "$mode" &
        echo "  -> PID $!"
    done
done

echo "All jobs launched."