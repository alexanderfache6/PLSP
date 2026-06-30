#!/bin/bash
echo Submitting $1
R --vanilla < ~/01_img_process.R $1

# run by
# qsub -V -pe omp 28 -l h_rt=12:00:00 qsub_01_image_process.sh <numSite>