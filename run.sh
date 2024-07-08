#!/bin/bash

#Mamba1 v Mamba2
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 2000 --city_count 10
#python tsp.py --save_loc 'checkpoints/mamba2/mamba2' --nb_layers 3  --nb_epochs 2000 --mamba2 True --city_count 10
python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --profiler True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 5
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 10
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 10 --nb_batch_per_epoch 20 

python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --mamba2 True --profiler True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 5 --mamba2 True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 10 --city_count 10 --recycle_data 10 --mamba2 True
#python tsp.py --save_loc 'checkpoints/mamba2/mamba1' --nb_layers 3  --nb_epochs 5 --city_count 10 --recycle_data 10 --nb_batch_per_epoch 20 --mamba2 True

####
