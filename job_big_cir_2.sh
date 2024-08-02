#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=tc064-s2517783

# Load the required modules
module load nvidia/nvhpc/24.5

pwd
source ../miniconda3/bin/activate
conda activate base

# Set the TRANSFORMERS_CACHE environment variable to a writable directory
export TRANSFORMERS_CACHE="${PWD}/.cache/huggingface/hub"
mkdir -p "${TRANSFORMERS_CACHE}"

# Set the Triton cache directory to a writable location
export TRITON_CACHE_DIR="${PWD}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

python tsp.py --save_loc 'checkpoints/big_cirrus/64_G_city50' --nb_layers 3  --nb_epochs 6000  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 220 --last_layer 'pointer' --reverse True --action 'next_city'  --project_name 'big_cirrus' --recycle_data 10 --wandb
