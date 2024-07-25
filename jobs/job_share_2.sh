#!/bin/bash
#$ -N job_share_2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=24:00:00
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

python tsp.py --save_loc 'checkpoints/big/share_20_general'  --nb_epochs 1000 --nb_batch_per_epoch 10 --city_count 20 --mamba2 True  --bsz 600 --reverse True  --checkpoint 'checkpoints/big/64_G_city10_general_20-07_01-48.pt' --action 'next_city'
