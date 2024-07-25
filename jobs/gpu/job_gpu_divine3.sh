#!/bin/bash
#$ -N job_gpu_divine2
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

python tsp.py --save_loc 'checkpoints/gpu/mamba2_75_ng' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
