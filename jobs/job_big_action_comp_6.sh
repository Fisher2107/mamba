#!/bin/bash
#$ -N job_big_action_comp_6
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

python tsp.py --save_loc 'checkpoints/action/mamba2_tour3' --nb_layers 3  --nb_epochs 1000 --nb_batch_per_epoch 40 --city_count 50 --recycle_data 5 --mamba2 True --bsz 220 --reverse True --project_name 'Action2'
