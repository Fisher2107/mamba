#!/bin/bash
#$ -N job_final_1
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

python tsp.py --save_loc 'checkpoints/big/BIG_city100_2'  --nb_epochs 7000 --mamba2 True  --city_count 100 --nb_batch_per_epoch 40 --bsz 220 --reverse True  --project_name 'Mamba_big' --checkpoint 'checkpoints/big/share_100_big_fix_30-07_05-40.pt' --action 'next_city' 
