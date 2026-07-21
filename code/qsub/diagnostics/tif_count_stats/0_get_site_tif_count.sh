#!/usr/bin/env bash
##
# Usage: ./0_get_site_tif_count.sh <site_name> <mode>
#   <mode> must be either "new" or "archive"
#
# Given a site name, looks in the appropriate base directory (recursively,
# including all subdirectories) for .tif files named like:
#   20210221_172235_0e0f_3B_AnalyticMS_DN_udm_clip.tif
# and aggregates counts of files per year/month, sorted ascending,
# filling in any missing months (between the earliest and latest
# found) with a count of 0. Writes result as CSV: year,month,count
# to a file named <site_name>_monthly_tif_count_<MODE>.csv

set -euo pipefail

SITE_NAME="${1:-}"
MODE="${2:-}"

if [[ -z "$SITE_NAME" || -z "$MODE" ]]; then
    echo "Usage: $0 <site_name> <new|archive>" >&2
    exit 1
fi

# Normalize mode to lowercase for comparison
MODE_LC="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"

case "$MODE_LC" in
    new)
        BASE_DIR="/projectnb/modislc/users/fache/data/planet/raw/${SITE_NAME}/data"
        MONTHLY_OUTPUT_FILE="${SITE_NAME}_monthly_tif_counts_NEW.csv"
        YEARLY_OUTPUT_FILE="${SITE_NAME}_yearly_tif_counts_NEW.csv"
        ;;
    archive)
        BASE_DIR="/projectnb/planet/PLSP/raw/${SITE_NAME}/data"
        MONTHLY_OUTPUT_FILE="${SITE_NAME}_monthly_tif_counts_ARCHIVE.csv"
        YEARLY_OUTPUT_FILE="${SITE_NAME}_yearly_tif_counts_ARCHIVE.csv"
        ;;
    *)
        echo "Error: invalid mode '$MODE'. Must be 'new' or 'archive'." >&2
        exit 1
        ;;
esac

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Directory does not exist: $BASE_DIR" >&2
    exit 1
fi

# Temp file to hold extracted YYYY MM pairs, one per line (space-separated)
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

# Walk each subdirectory under BASE_DIR, announce it, then find .tif files
# directly within it (maxdepth 1 since we're doing the recursion ourselves).
while IFS= read -r -d '' dir; do
    # echo "\nchecking directory: $dir"

    find "$dir" -maxdepth 1 -type f -iname "*.tif" -print0 | \
    while IFS= read -r -d '' file; do
        fname="$(basename "$file")"

        if [[ "$fname" =~ ^([0-9]{4})([0-9]{2})([0-9]{2}) ]]; then
            year="${BASH_REMATCH[1]}"
            month="${BASH_REMATCH[2]}"
            echo "${year} ${month}" >> "$TMP_FILE"
        else
            echo "Warning: skipping file with unexpected name format: $fname" >&2
        fi
    done
done < <(find "$BASE_DIR" -type d -print0)

if [[ ! -s "$TMP_FILE" ]]; then
    echo "No matching .tif files found under $BASE_DIR" >&2
    echo "year,month,count" > "$MONTHLY_OUTPUT_FILE"
    echo "year,count" > "$YEARLY_OUTPUT_FILE"
    exit 0
fi

# Aggregate counts into an associative array keyed "YYYY MM"
declare -A counts
while read -r year month; do
    key="${year} ${month}"
    counts["$key"]=$(( ${counts["$key"]:-0} + 1 ))
done < <(sort "$TMP_FILE")

# Determine earliest and latest year/month
sorted_keys=$(printf '%s\n' "${!counts[@]}" | sort -k1,1n -k2,2n)
start_year=$(echo "$sorted_keys" | head -n1 | awk '{print $1}')
start_month=$(echo "$sorted_keys" | head -n1 | awk '{print $2}')
end_year=$(echo "$sorted_keys" | tail -n1 | awk '{print $1}')
end_month=$(echo "$sorted_keys" | tail -n1 | awk '{print $2}')

# Strip any leading zeros for arithmetic (avoid octal interpretation)
start_month=$((10#$start_month))
end_month=$((10#$end_month))

# Associative array to hold yearly totals
declare -A yearly_counts

{
    echo "year,month,count"

    y="$start_year"
    m="$start_month"

    while (( y < end_year || ( y == end_year && m <= end_month ) )); do
        key_padded_month=$(printf "%02d" "$m")
        key="${y} ${key_padded_month}"
        count="${counts[$key]:-0}"
        printf "%s,%02d,%d\n" "$y" "$m" "$count"

        yearly_counts["$y"]=$(( ${yearly_counts["$y"]:-0} + count ))

        m=$((m + 1))
        if (( m > 12 )); then
            m=1
            y=$((y + 1))
        fi
    done
} > "$MONTHLY_OUTPUT_FILE"

{
    echo "year,count"
    for yr in $(printf '%s\n' "${!yearly_counts[@]}" | sort -n); do
        printf "%s,%d\n" "$yr" "${yearly_counts[$yr]}"
    done
} > "$YEARLY_OUTPUT_FILE"

echo "Monthly counts written to: $MONTHLY_OUTPUT_FILE"
echo "Yearly counts written to:  $YEARLY_OUTPUT_FILE"