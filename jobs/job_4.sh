#!/bin/bash
#$ -N job_4
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=5:00:00
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

python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reversestart_128_point' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
