#!/bin/bash

# goes down one level and summarize number of files and total size of directories

# USAGE
# run_get_dir_size_stats.sh /projectnb/modislc/users/fache/data/planet
# run_get_dir_size_stats.sh /projectnb/modislc/users/fache/data/planet/raw/
# run_get_dir_size_stats.sh /projectnb/modislc/users/fache/data/planet/raw/Willard_Juniper_Savannah/data

human_size() {
    local bytes="$1"
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec --suffix=B "$bytes"
    else
        echo "${bytes}B"
    fi
}

process_dir() {
    local parent="$1"

    if [[ ! -d "$parent" ]]; then
        echo "SKIP (not a directory): $parent" >&2
        return
    fi

    # Iterate over immediate subdirectories only
    find "$parent" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r sub; do
        # Count files (regular files) recursively within this subdir
        file_count=$(find "$sub" -type f | wc -l)

        # Total size in bytes of all files within this subdir
        total_bytes=$(find "$sub" -type f -exec stat --format='%s' {} + 2>/dev/null \
                        | awk '{sum+=$1} END {print sum+0}')

        printf "%-60s files=%-8s size=%s (%s)\n" \
            "$sub" "$file_count" "$(human_size "$total_bytes")" "${total_bytes}B"
    done
}

if [[ $# -gt 0 ]]; then
    dirs=("$@")
else
    mapfile -t dirs
fi

for d in "${dirs[@]}"; do
    echo "=== Parent: $d ==="
    process_dir "$d"
    echo
done