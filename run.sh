#!/bin/bash

#Mamba1 v Mamba2
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 2000 --city_count 10
#python tsp.py --save_loc 'checkpoints/mamba2/mamba2' --nb_layers 3  --nb_epochs 2000 --mamba2 True --city_count 10

#Pointer Network
#python tsp.py --save_loc 'checkpoints/mamba2/pointer64' --mamba2 True --nb_layers 3  --nb_epochs 2000 --city_count 10 --pointer True 
#python tsp.py --save_loc 'checkpoints/mamba2/pointer32' --mamba2 True --nb_layers 3  --nb_epochs 2000 --city_count 10 --d_model 32 --pointer True
#python tsp.py --save_loc 'checkpoints/mamba2/pointer16' --mamba2 True --nb_layers 3  --nb_epochs 2000 --city_count 10 --d_model 16 --pointer True
#python tsp.py --save_loc 'checkpoints/mamba2/standard64' --mamba2 True --nb_layers 3  --nb_epochs 2000 --city_count 10
#python tsp.py --save_loc 'checkpoints/mamba2/standard32' --mamba2 True --nb_layers 3  --nb_epochs 2000 --city_count 10 --d_model 32
#python tsp.py --save_loc 'checkpoints/mamba2/standard16' --mamba2 True --nb_layers 3  --nb_epochs 2000 --city_count 10 --d_model 16

#Reverse Layers
#python tsp.py --save_loc 'checkpoints/mamba2_reverse/mabma2' --nb_layers 3  --nb_epochs 2000 --city_count 10 --mamba2 True #Use from above
python tsp.py --save_loc 'checkpoints/mamba2_reverse/mamba2_reverse_point' --nb_layers 3  --nb_epochs 2000 --mamba2 True --city_count 10 --reverse True --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/mamba2_reverse/mamba2_reverse2_point' --nb_layers 3  --nb_epochs 2000 --mamba2 True --city_count 10 --reverse_start True --last_layer 'pointer'


####
