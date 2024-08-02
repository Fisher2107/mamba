#!/bin/bash
#$ -N job_share_big_2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=48:00:00
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

python tsp.py --save_loc 'checkpoints/big/share_100_big_fix' --nb_epochs 6000 --nb_batch_per_epoch 40 --city_count 100  --mamba2 True --bsz 220 --reverse True  --checkpoint 'checkpoints/big/share_50_general_27-07_13-29.pt' --action 'next_city'
