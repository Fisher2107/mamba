#!/bin/bash
#$ -N job_city50_2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=23:00:00
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=80G
#$ -l rl9=true

. /etc/profile.d/modules.sh
. /exports/applications/support/set_qlogin_environment.sh

module load cuda/12.1.1

source /exports/eddie/scratch/s2517783/miniconda3/bin/activate base
cd /exports/eddie/scratch/s2517783/mamba
conda activate tspp

source ~/.bashrc

python tsp.py --save_loc 'checkpoints/city50/mamba2' --nb_layers 3  --nb_epochs 2000 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --recycle_data 10
