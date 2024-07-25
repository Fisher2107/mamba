#!/bin/bash
#!/bin/bash
#$ -N job_gpu_divine2
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

python tsp.py --save_loc 'checkpoints/gpu/tour/mamba2_75_d16' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --bsz 100 --reverse True --d_model 16


python tsp.py --save_loc 'checkpoints/gpu/mamba2_5' --nb_layers 3  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_10' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_20' --nb_layers 3  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_50' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_75' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_100' --nb_layers 3  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_120' --nb_layers 3  --nb_epochs 5 --city_count 120 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'

python tsp.py --save_loc 'checkpoints/gpu/mamba2_5_ng' --nb_layers 3  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_10_ng' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_20_ng' --nb_layers 3  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_50_ng' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_75_ng' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_100_ng' --nb_layers 3  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_120_ng' --nb_layers 3  --nb_epochs 5 --city_count 120 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --mlp_cls 'identity'

python tsp.py --save_loc 'checkpoints/gpu/mamba2_5_d16' --nb_layers 3  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/mamba2_10_d16' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/mamba2_20_d16' --nb_layers 3  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/mamba2_50_d16' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/mamba2_75_d16' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/mamba2_100_d16' --nb_layers 3  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16
python tsp.py --save_loc 'checkpoints/gpu/mamba2_120_d16' --nb_layers 3  --nb_epochs 5 --city_count 120 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --d_model 16

python tsp.py --save_loc 'checkpoints/gpu/mamba2_5_lay1' --nb_layers 1  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_10_lay1' --nb_layers 1  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_20_lay1' --nb_layers 1  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_50_lay1' --nb_layers 1  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_75_lay1' --nb_layers 1  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_100_lay1' --nb_layers 1  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_120_lay1' --nb_layers 1  --nb_epochs 5 --city_count 120 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city'

python tsp.py --save_loc 'checkpoints/gpu/mamba2_5_general' --nb_layers 3  --nb_epochs 5 --city_count 5 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '4,10'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_10_general' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '5,15'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_20_general' --nb_layers 3  --nb_epochs 5 --city_count 20 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '15,25'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_50_general' --nb_layers 3  --nb_epochs 5 --city_count 50 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '45,55'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_75_general' --nb_layers 3  --nb_epochs 5 --city_count 75 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '70,80'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_100_general' --nb_layers 3  --nb_epochs 5 --city_count 100 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '95,105'
python tsp.py --save_loc 'checkpoints/gpu/mamba2_120_general' --nb_layers 3  --nb_epochs 5 --city_count 120 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 0 --wandb --bsz 100 --reverse True --action 'next_city' --city_range '115,125'