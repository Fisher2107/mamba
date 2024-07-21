#!/bin/bash
#$ -N job_big_21
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

python tsp.py --save_loc 'checkpoints/big/16_G_city20_general' --nb_layers 3  --nb_epochs 4000 --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 2000 --last_layer 'pointer' --reverse True --d_model 16 --city_range '10,30'
