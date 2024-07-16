#!/bin/bash
#$ -N job_gpu_1
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=4:00:00
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=80G
#$ -l rl9=true

. /etc/profile.d/modules.sh
. /exports/applications/support/set_qlogin_environment.sh

module load cuda/12.1.1

source /exports/eddie/scratch/s2517783/miniconda3/bin/activate base
cd /exports/eddie/scratch/s2517783/mamba
conda activate tsp

python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz50' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz30' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 30 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz40' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 40 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz75' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 75 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz125' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 125 --reverse True
