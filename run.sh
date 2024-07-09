#!/bin/bash

#Mamba1 v Mamba2
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 2000 --city_count 10
#python tsp.py --save_loc 'checkpoints/mamba2/mamba2' --nb_layers 3  --nb_epochs 2000 --mamba2 True --city_count 10
#python tsp.py --save_loc 'checkpoints/mamba1_vs_2/profiler/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --profiler True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 5
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 10
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 10 --nb_batch_per_epoch 20 

#python tsp.py --save_loc 'checkpoints/mamba1_vs_2/profiler/mamba2' --nb_layers 3  --nb_epochs 5 --city_count 10 --mamba2 True --profiler True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 5 --mamba2 True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 10 --mamba2 True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 10 --nb_batch_per_epoch 20 --mamba2 True


#Mamba2 50 city
python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_100' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 100 --last_layer 'pointer' --reverse True
python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_150' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --bsz 150 --last_layer 'pointer' --reverse True
python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_recy_100' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --recycle_data 20 --bsz 100 --last_layer 'pointer' --reverse True
python tsp.py --save_loc 'checkpoints/city50/mamba2_rev_recy_150' --nb_layers 3  --nb_epochs 100 --mamba2 True --city_count 50 --nb_batch_per_epoch 40 --recycle_data 20 --bsz 150 --last_layer 'pointer' --reverse True
python tsp.py --save_loc 'checkpoints/city50/mamba2_profiler_150' --nb_layers 3  --nb_epochs 5 --mamba2 True --city_count 50 --nb_batch_per_epoch 1 --bsz 150 --last_layer 'pointer' --reverse True --profiler True
python tsp.py --save_loc 'checkpoints/city50/mamba2_profiler_100' --nb_layers 3  --nb_epochs 5 --mamba2 True --city_count 50 --nb_batch_per_epoch 1 --bsz 100 --last_layer 'pointer' --reverse True --profiler True

####
