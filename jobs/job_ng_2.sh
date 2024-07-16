#!/bin/bash
#$ -N job_ng_2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=12:00:00
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

python tsp.py --save_loc 'checkpoints/mamba_ng/city50/lay3_bsz150_bpe10' --nb_layers 3  --nb_epochs 100  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 150 --last_layer 'pointer' --reverse True --mlp_cls 'identity'
