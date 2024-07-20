#!/bin/bash

#action = next_city
python tsp.py --save_loc 'checkpoints/action/city10_next_city' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 10 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id 0 --wandb --action 'next_city'
python tsp.py --save_loc 'checkpoints/action/city20_next_city' --nb_layers 3  --nb_epochs 10  --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id 0 --wandb --action 'next_city'
python tsp.py --save_loc 'checkpoints/action/city50_next_city' --nb_layers 3  --nb_epochs 10  --mamba2 True --city_count 50 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id 0 --wandb --action 'next_city'
python tsp.py --save_loc 'checkpoints/action/city100_next_city' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 100 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True  --pynvml True --gpu_id 0 --wandb --action 'next_city'

#action = tour
python tsp.py --save_loc 'checkpoints/action/city10' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 10 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id 0 --wandb
python tsp.py --save_loc 'checkpoints/action/city20' --nb_layers 3  --nb_epochs 10  --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id 0 --wandb
python tsp.py --save_loc 'checkpoints/action/city50' --nb_layers 3  --nb_epochs 10  --mamba2 True --city_count 50 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True --pynvml True --gpu_id 0 --wandb 
python tsp.py --save_loc 'checkpoints/action/city100' --nb_layers 3  --nb_epochs 10 --mamba2 True --city_count 100 --nb_batch_per_epoch 10 --bsz 60 --last_layer 'pointer' --reverse True  --pynvml True --gpu_id 0 --wandb



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
