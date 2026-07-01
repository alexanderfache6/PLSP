#!/bin/bash
#$ -V
# export current environment variables into job
#$ -pe omp 10
# request multiple cores
#$ -l h_rt=1:00:00
# hard time limit
#$ -o /projectnb/modislc/users/fache/logs/planet/
# output log
#$ -e /projectnb/modislc/users/fache/logs/planet/
# error log

# #$ -m ae
# #$ fache@bu.edu

siteNum=$1
Rfile=/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/01_quality_mask_and_mosaic.R

echo "Submitting $Rfile with site number $siteNum"

module load R

R --vanilla --args $siteNum < $Rfile

# run in any directory with:
# qsub qsub_01.sh <siteNum>