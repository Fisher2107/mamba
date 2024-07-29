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

python tsp.py --save_loc 'checkpoints/big_cirrus/64_G_city20' --nb_layers 3  --nb_epochs 6000  --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 600 --last_layer 'pointer' --reverse True --action 'next_city'  --project_name 'big_cirrus' 
