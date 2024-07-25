#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]

# Load the required modules
module load nvidia/nvhpc/24.5

srun ./cuda_test.x

python tsp.py --save_loc 'checkpoints/gpu/mamba2_75_ng' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
