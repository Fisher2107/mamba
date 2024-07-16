#!/bin/bash
#$ -N job_gpu_2
#$ -wd /exports/eddie/scratch/s2517783/mamba
#$ -l h_rt=4:00:00
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

python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_75_d16' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16 --last_layer 'identity'

python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_5_lay1' --nb_layers 1  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --last_layer 'identity'
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_10_lay1' --nb_layers 1  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --last_layer 'identity'
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_20_lay1' --nb_layers 1  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --last_layer 'identity'
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_50_lay1' --nb_layers 1  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --last_layer 'identity'
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_100_lay1' --nb_layers 1  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --last_layer 'identity'
python tsp.py --save_loc 'checkpoints/gpu/city_count/mamba2_75_lay1' --nb_layers 1  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --last_layer 'identity'



python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_75' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_75_d16' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16

python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_5_lay1' --nb_layers 1  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_10_lay1' --nb_layers 1  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_20_lay1' --nb_layers 1  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_50_lay1' --nb_layers 1  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_100_lay1' --nb_layers 1  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point/mamba2_75_lay1' --nb_layers 1  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True




python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_75' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_75_d16' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True --d_model 16

python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_5_lay1' --nb_layers 1  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_10_lay1' --nb_layers 1  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_20_lay1' --nb_layers 1  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_50_lay1' --nb_layers 1  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_100_lay1' --nb_layers 1  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
python tsp.py --save_loc 'checkpoints/gpu/city_count_point_ng/mamba2_75_lay1' --nb_layers 1  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --mlp_cls 'identitiy' --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 50 --reverse True
