#!/bin/bash
#$ -N job_final_3_100
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

python tsp.py --save_loc 'checkpoints/abalation/city100_128'  --nb_epochs 30000  --mamba2 True --city_count 100 --nb_batch_per_epoch 100  --bsz 100  --project_name 'Abalation' --d_model 128 --nb_layers 1 --checkpoint 'checkpoints/abalation/city50_128.pt'
