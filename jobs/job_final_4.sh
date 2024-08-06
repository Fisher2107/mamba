#!/bin/bash
#$ -N job_final_4
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

python tsp.py --save_loc 'checkpoints/abalation/city20_128_R'  --nb_epochs 10000  --mamba2 True --city_count 20 --nb_batch_per_epoch 10  --bsz 600 --reverse_start True  --project_name 'Abalation' --d_model 128 --nb_layers 1 
