#!/bin/bash
#FINAL EDDIE RUNS
python tsp.py --save_loc 'checkpoints/big/BIG_city100_2'  --nb_epochs 7000 --mamba2 True  --city_count 100 --nb_batch_per_epoch 40 --bsz 220 --reverse True  --project_name 'Mamba_big' --checkpoint 'checkpoints/big/share_100_big_fix_30-07_05-40.pt' --action 'next_city' 
python tsp.py --save_loc 'checkpoints/big/BIG_city20_2'  --nb_epochs 12000  --mamba2 True --city_count 20 --nb_batch_per_epoch 10  --bsz 600 --reverse True  --project_name 'Mamba_big' --checkpoint 'checkpoints/big_cirrus/64_G_city20.pt'
python tsp.py --save_loc 'checkpoints/abalation/city20'  --nb_epochs 10000  --mamba2 True --city_count 20 --nb_batch_per_epoch 40  --bsz 220 --reverse True  --project_name 'Abalation' --d_model 128 --nb_layers 1
python tsp.py --save_loc 'checkpoints/abalation/city20'  --nb_epochs 10000  --mamba2 True --city_count 20 --nb_batch_per_epoch 40  --bsz 220 --reverse True  --project_name 'Abalation' --d_model 256 --nb_layers 1
python tsp.py --save_loc 'checkpoints/abalation/city20'  --nb_epochs 10000  --mamba2 True --city_count 20 --nb_batch_per_epoch 40  --bsz 220 --reverse True  --project_name 'Abalation' --d_model 128


#CIRRUS REDO
#python tsp.py --save_loc 'checkpoints/big_cirrus/64_G_city20' --nb_layers 3  --nb_epochs 6000  --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 600 --last_layer 'pointer' --reverse True --action 'next_city'  --project_name 'big_cirrus' 
#python tsp.py --save_loc 'checkpoints/big_cirrus/64_G_city50' --nb_layers 3  --nb_epochs 6000  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 220 --last_layer 'pointer' --reverse True --action 'next_city'  --project_name 'big_cirrus' --recycle_data 10
#python tsp.py --save_loc 'checkpoints/big_cirrus/64_G_city100' --nb_layers 3  --nb_epochs 6000 --mamba2 True --city_count 100 --nb_batch_per_epoch 40 --bsz 220 --last_layer 'pointer' --reverse True --action 'next_city'  --project_name 'big_cirrus' --recycle_data 10


#ABALATION

#SHARE BIG
#python tsp.py --save_loc 'checkpoints/big/share_50_big' --nb_epochs 5500 --nb_batch_per_epoch 40 --city_count 50  --mamba2 True --bsz 220 --reverse True  --checkpoint 'checkpoints/big/share_50_general_27-07_13-29.pt'
#python tsp.py --save_loc 'checkpoints/big/share_100_big' --nb_epochs 6000 --nb_batch_per_epoch 40 --city_count 100  --mamba2 True --bsz 220 --reverse True  --checkpoint 'checkpoints/big/share_50_general_27-07_13-29.pt' --action 'next_city'

#action = next_city
#python tsp.py --save_loc 'checkpoints/action/city50_big_nondet' --nb_layers 3  --nb_epochs 30  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 150 --last_layer 'pointer' --reverse True --project_name 'Mamba_action' --non_det True

#python tsp.py --save_loc 'checkpoints/action/city50_next_city_big_nondet' --nb_layers 3  --nb_epochs 30  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 150 --last_layer 'pointer' --reverse True --project_name 'Mamba_action' --action 'next_city' --non_det True
#python tsp.py --save_loc 'checkpoints/action/city50_next_city_big' --nb_layers 3  --nb_epochs 30  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 150 --last_layer 'pointer' --reverse True --project_name 'Mamba_action' --action 'next_city'

#python tsp.py --save_loc 'checkpoints/action/city50_batch' --nb_layers 3  --nb_epochs 3  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 500 --last_layer 'pointer' --reverse True --wandb --pynvml True --gpu_id 0 --action 'next_city'
#python tsp.py --save_loc 'checkpoints/action/city50_batch_tour' --nb_layers 3  --nb_epochs 3  --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 60 --last_layer 'pointer' --reverse True --wandb --pynvml True --gpu_id 0
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
