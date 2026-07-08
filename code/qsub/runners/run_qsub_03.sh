#!/bin/bash
#$ -V
# export current environment variables into job
#$ -pe omp 28
# request multiple cores
#$ -l h_rt=24:00:00
# hard time limit
#$ -o /projectnb/modislc/users/fache/logs/planet/
# output log
#$ -e /projectnb/modislc/users/fache/logs/planet/
# error log
#$ -l mem_per_core=4G
# allocated memory per worker, assigned to SGE NSLOTS
#$ -m ea
# send an email when the job ends or is aborted

siteNumber=$1
Rfile=/projectnb/modislc/users/fache/src/PLSP/code/code_lsp_refactored/03_run_per_pixel_LSP.R

echo "Submitting $Rfile with site number $siteNumber"

module load R

R --vanilla --args $siteNumber < $Rfile


# USAGE
# run with
# qsub run_qsub_03.sh <siteNumber>


# NOTE each chunk takes ~30min per core at omp 28 4GB for 3 years of LSP for 200 chunks, ~8 batches so ~4 hours