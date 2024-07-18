#!/bin/bash

#NG 64
python tsp.py --save_loc 'checkpoints/big/64_NG_city10_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 10 --nb_batch_per_epoch 10 --bsz 1200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --city_range '5,15'
python tsp.py --save_loc 'checkpoints/big/64_NG_city10'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 50 --nb_batch_per_epoch 10 --bsz 1200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/big/64_NG_city20_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 800 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --city_range '10,30'
python tsp.py --save_loc 'checkpoints/big/64_NG_city20'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 800 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/big/64_NG_city50'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 500 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity'
python tsp.py --save_loc 'checkpoints/big/64_NG_city100'        --nb_layers 3  --nb_epochs 4 --mamba2 True --city_count 100 --nb_batch_per_epoch 40 --bsz 200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity'
#NG 16
python tsp.py --save_loc 'checkpoints/big/16_NG_city10_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 10 --nb_batch_per_epoch 40 --bsz 2000 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --d_model 16 --city_range '5,15'
python tsp.py --save_loc 'checkpoints/big/16_NG_city10'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 10 --nb_batch_per_epoch 40 --bsz 2000 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --d_model 16
python tsp.py --save_loc 'checkpoints/big/16_NG_city20_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 20 --nb_batch_per_epoch 40 --bsz 1600 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --d_model 16 --city_range '10,30'
python tsp.py --save_loc 'checkpoints/big/16_NG_city20'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 20 --nb_batch_per_epoch 40 --bsz 1600 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --d_model 16
python tsp.py --save_loc 'checkpoints/big/16_NG_city50'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 1000 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --d_model 16
python tsp.py --save_loc 'checkpoints/big/16_NG_city100'        --nb_layers 3  --nb_epochs 4 --mamba2 True --city_count 100 --nb_batch_per_epoch 40 --bsz 400 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --mlp_cls 'identity' --d_model 16
#G 64
python tsp.py --save_loc 'checkpoints/big/64_G_city10_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 10 --nb_batch_per_epoch 40 --bsz 600 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --city_range '5,15'
python tsp.py --save_loc 'checkpoints/big/64_G_city10'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 10 --nb_batch_per_epoch 40 --bsz 600 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb
python tsp.py --save_loc 'checkpoints/big/64_G_city20_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 20 --nb_batch_per_epoch 40 --bsz 400 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --city_range '10,30'
python tsp.py --save_loc 'checkpoints/big/64_G_city20'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 20 --nb_batch_per_epoch 40 --bsz 400 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb
python tsp.py --save_loc 'checkpoints/big/64_G_city50'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 220 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb
python tsp.py --save_loc 'checkpoints/big/64_G_city100'        --nb_layers 3  --nb_epochs 4 --mamba2 True --city_count 100 --nb_batch_per_epoch 40 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb
#G 16
python tsp.py --save_loc 'checkpoints/big/16_G_city10_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 10 --nb_batch_per_epoch 40 --bsz 1200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --d_model 16 --city_range '5,15'
python tsp.py --save_loc 'checkpoints/big/16_G_city10'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 10 --nb_batch_per_epoch 40 --bsz 1200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --d_model 16
python tsp.py --save_loc 'checkpoints/big/16_G_city20_general' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 20 --nb_batch_per_epoch 40 --bsz 800 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --d_model 16 --city_range '10,30'
python tsp.py --save_loc 'checkpoints/big/16_G_city20'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 20 --nb_batch_per_epoch 40 --bsz 800 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --d_model 16
python tsp.py --save_loc 'checkpoints/big/16_G_city50'         --nb_layers 3  --nb_epochs 4  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 500 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --d_model 16
python tsp.py --save_loc 'checkpoints/big/16_G_city100'        --nb_layers 3  --nb_epochs 4 --mamba2 True --city_count 100 --nb_batch_per_epoch 40 --bsz 200 --last_layer 'pointer' --reverse True --pynvml True --gpu_id $CUDA_VISIBLE_DEVICES --wandb --d_model 16

#Pointer Generalisation
#python tsp.py --save_loc 'checkpoints/pointer_generalisation/normal_pointer' --nb_layers 3  --nb_epochs 2000 --city_count 10 --last_layer 'pointer' #Test to see if anything changed
#python tsp.py --save_loc 'checkpoints/pointer_generalisation/5_10' --nb_layers 3  --nb_epochs 2000 --city_count 10 --last_layer 'pointer' --city_range '5,10'
#python tsp.py --save_loc 'checkpoints/pointer_generalisation/5_20' --nb_layers 3  --nb_epochs 2000 --city_count 10 --last_layer 'pointer' --city_range '5,20'

#Mamba2 50 city
#python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_100' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 100 --last_layer 'pointer' --reverse True
#python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_150' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 150 --last_layer 'pointer' --reverse True
#python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_recy_100' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --recycle_data 20 --bsz 100 --last_layer 'pointer' --reverse True
#python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_recy_150' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --recycle_data 20 --bsz 150 --last_layer 'pointer' --reverse True
#python tsp.py --save_loc 'checkpoints/city50/mamba2_profiler_150' --nb_layers 3  --nb_epochs 5 --mamba2 True --city_count 50 --nb_batch_per_epoch 1 --bsz 150 --last_layer 'pointer' --reverse True --profiler True
#python tsp.py --save_loc 'checkpoints/city50/mamba2_profiler_100' --nb_layers 3  --nb_epochs 5 --mamba2 True --city_count 50 --nb_batch_per_epoch 1 --bsz 100 --last_layer 'pointer' --reverse True --profiler True

#python tsp.py --save_loc 'checkpoints/gpu/mamba2_10_recy5' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 5 --mamba2 True --pynvml True --gpu_id 1 --wandb

####
