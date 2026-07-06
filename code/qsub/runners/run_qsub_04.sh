#!/bin/bash
#$ -V
# export current environment variables into job
#$ -pe omp 4
# request multiple cores
#$ -l h_rt=24:00:00
# hard time limit
#$ -o /projectnb/modislc/users/fache/logs/planet/
# output log
#$ -e /projectnb/modislc/users/fache/logs/planet/
# error log

#$ -l mem_per_core=16G
# allocated memory per worker, assigned to SGE NSLOTS

siteNumber=$1
Rfile=/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/04_generate_geotiff_product_layers.R

echo "Submitting $Rfile with site number $siteNumber"

module load R

R --vanilla --args $siteNumber < $Rfile


# USAGE
# run with
# qsub run_qsub_04.sh <siteNumber>
