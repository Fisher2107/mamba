#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=00:03:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=tc064-s2517783

# Load the required modules
module load nvidia/nvhpc/24.5
pwd
source ../miniconda3/bin/activate
conda activate base


python pt_reader.py --checkpoint 'checkpoints/big/64_G_city20_20-07_16-04.pt'
