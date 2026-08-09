qrsh -pe omp 4 -P modislc


qgpus
gpus -v

qrsh -l gpus=1 -l gpu_c=6.0 -pe omp 4

hostname

nvidia-smi