#!/bin/bash
#$ -N job_gpu_1
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
conda activate tsp

python tsp.py --save_loc 'checkpoints/gpu/lay/mamba2_50_lay1' --nb_layers 1  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 150 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/lay/mamba2_50_lay2' --nb_layers 2  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 150 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/lay/mamba2_50_lay3' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 150 --reverse True

python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_5' --nb_layers 3  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_10' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_20' --nb_layers 3  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_50' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_100' --nb_layers 3  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True

python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_5_d16' --nb_layers 3  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_10_d16' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_20_d16' --nb_layers 3  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_50_d16' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_100_d16' --nb_layers 3  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16

python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz10' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 10 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz20' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz100' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 100 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz150' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 150 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/bsz/mamba2_50_bsz200' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 200 --reverse True
