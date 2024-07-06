#!/bin/bash
#$ -N fourier_16_v2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=5:00:00
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=40G
#$ -l rl9=true

. /etc/profile.d/modules.sh
. /exports/applications/support/set_qlogin_environment.sh

module load cuda/12.1.1

source /exports/eddie/scratch/s2517783/miniconda3/bin/activate base
cd /exports/eddie/scratch/s2517783/mamba
conda activate tspp

python tsp.py --save_loc 'checkpoints/embed2/fourier1_10city_4l' --nb_layers 4 --nb_epochs 2000 --city_count 10 --fourier_scale 1
