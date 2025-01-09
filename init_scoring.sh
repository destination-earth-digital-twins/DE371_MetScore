#!/bin/bash -l
#SBATCH -J DE371_stylegan
#SBATCH -A p200177
#SBATCH -q default 
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --time=0-48:00:00

export OMP_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.1
export NVHPC_CUDA_HOME=/usr/local/cuda-12.1
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard
export APPTAINER_BINDPATH="/project/home/p200177/DE_371:/project/home/p200177/DE_371/,/project/scratch/p200177/DE_371:/project/scratch/p200177/DE_371/"
module load Apptainer/1.2.4-GCCcore-12.3.0


apptainer exec --nv /project/home/p200177/DE_371/resources/apptainer_container/container_for_scoring.sif python3 main.py --config '/project/home/p200177/DE_371/experiments_WP1/temporal_diff_samples/scores/config.yml'
