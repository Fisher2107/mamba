#!/bin/bash

# Grid Engine options
#$ -N job  # Name of the job
#$ -wd /exports/eddie/scratch/s2517783/mamba # Run the job from the scratch directory
#$ -l h_rt=1:00:00  # Request a runtime
#$ -q gpu          # Submit the job to the gpu queue
#$ -pe gpu-a100 1  # Request NNODE A100 GPUs
#$ -l h_vmem=80G    # Request memory per core
#$ -l rl9=true    # rocky linux update

# Load the module system
. /etc/profile.d/modules.sh
. /exports/applications/support/set_qlogin_environment.sh

# Load the CUDA module
module load cuda/12.1.1

# Activate the conda environment for CUDA
source /exports/eddie/scratch/s2517783/miniconda3/bin/activate base
cd /exports/eddie/scratch/s2517783/mamba
conda activate tsp

python tsp_gpustats.py
