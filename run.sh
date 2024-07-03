#!/bin/bash

#Reverse10 Experiments
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reversestart_128' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10 --d_model 128
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reverse_128' --nb_layers 3  --reverse True --nb_epochs 2000 --city_count 10 --d_model 128
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_128' --nb_layers 3 --nb_epochs 2000 --city_count 10 --d_model 128

python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reversestart_128_point' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reverse_128_point' --nb_layers 3  --reverse True --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_128_point' --nb_layers 3 --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
####
