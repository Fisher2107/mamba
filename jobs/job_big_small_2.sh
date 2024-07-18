#!/bin/bash
#$ -N job_big_small_2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=1:00:00
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

python tsp.py --save_loc 'checkpoints/big/64_NG_city10'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 50 --nb_batch_per_epoch 10 --bsz 1200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --mlp_cls 'identity'
