#!/bin/bash
#$ -V
# export current environment variables into job
#$ -pe omp 8
# request multiple cores
#$ -l h_rt=12:00:00
# hard time limit
#$ -o /projectnb/modislc/users/fache/logs/planet/
# output log
#$ -e /projectnb/modislc/users/fache/logs/planet/
# error log
#$ -l mem_per_core=16G
# allocated memory per worker, assigned to SGE NSLOTS
#$ -m ea
# send an email when the job ends or is aborted

siteNumber=$1
Rfile=/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/01_quality_mask_and_mosaic.R

echo "Submitting $Rfile with site number $siteNumber"

module load R

R --vanilla --args $siteNumber < $Rfile


# USAGE
# run with
# qsub run_qsub_01.sh <siteNumber>


# NOTE for 1 site for 1 year (ex 274 dates) takes about 30min of run time (excluding queue wait time)