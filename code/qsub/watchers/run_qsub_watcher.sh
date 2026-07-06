#!/bin/bash
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# run_qsub_watcher.sh
# Watch a directory for new files and count them every N seconds
#
# Usage:
#   ./run_qsub_watcher.sh <relative_dir_pattern> [seconds_interval] [--log]
#
# Examples:
#   ./run_qsub_watcher.sh phenology/Walnut_Gulch_Kendall_Grasslands/chunk_phenology_*.rda
#   ./run_qsub_watcher.sh phenology/Walnut_Gulch_Kendall_Grasslands/chunk_phenology_*.rda 30
#   ./run_qsub_watcher.sh phenology/Walnut_Gulch_Kendall_Grasslands/chunk_phenology_*.rda 30 --log
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

ROOT_DIR="/projectnb/modislc/users/fache/data/planet"
DIR_PATTERN=$1
SECONDS_INTERVAL=${2:-60}
MODE=$3  # optional --log flag

if [ -z "$DIR_PATTERN" ]; then
  echo "[error] no directory pattern specified"
  echo "usage: ./run_qsub_watcher.sh <relative_dir_pattern> [seconds_interval] [--log]"
  echo "example: ./run_qsub_watcher.sh phenology/Walnut_Gulch_Kendall_Grasslands/chunk_phenology_*.rda 30 --log"
  exit 1
fi

FULL_PATTERN="${ROOT_DIR}/${DIR_PATTERN}"

echo "[watching] $FULL_PATTERN"
echo "[interval] every ${SECONDS_INTERVAL}s"

if [ "$MODE" == "--log" ]; then
  echo "[mode] log"
  echo "----"
  while true; do
    count=$(ls ${FULL_PATTERN} 2>/dev/null | wc -l)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - files: $count"
    sleep $SECONDS_INTERVAL
  done
else
  echo "[mode] watch"
  echo "----"
  watch -n $SECONDS_INTERVAL "echo \$(date '+%Y-%m-%d %H:%M:%S') — files: \$(ls ${FULL_PATTERN} 2>/dev/null | wc -l)"
fi