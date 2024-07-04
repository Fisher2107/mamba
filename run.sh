#!/bin/bash

#Reverse5 Experiments
python tsp.py --save_loc 'checkpoints/reverse/fixed_reversestart' --nb_layers 3  --reverse_start True --nb_epochs 2000
python tsp.py --save_loc 'checkpoints/reverse/fixed_reverse' --nb_layers 3  --reverse True --nb_epochs 2000
python tsp.py --save_loc 'checkpoints/reverse/fixed' --nb_layers 3 --nb_epochs 2000

#Reverse10 Experiments
python tsp.py --save_loc 'checkpoints/reverse10/fixed_reversestart' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10
python tsp.py --save_loc 'checkpoints/reverse10/fixed_reverse' --nb_layers 3  --reverse True --nb_epochs 2000 --city_count 10
python tsp.py --save_loc 'checkpoints/reverse10/fixed' --nb_layers 3 --nb_epochs 2000 --city_count 10


python tsp.py --save_loc 'checkpoints/reverse10/fixed_reversestart_128' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10 --d_model 128
python tsp.py --save_loc 'checkpoints/reverse10/fixed_reverse_128' --nb_layers 3  --reverse True --nb_epochs 2000 --city_count 10 --d_model 128
python tsp.py --save_loc 'checkpoints/reverse10/fixed_128' --nb_layers 3 --nb_epochs 2000 --city_count 10 --d_model 128

python tsp.py --save_loc 'checkpoints/reverse10/fixed_reversestart_128_point' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/reverse10/fixed_reverse_128_point' --nb_layers 3  --reverse True --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/reverse10/fixed_128_point' --nb_layers 3 --nb_epochs 2000 --city_count 10 --d_model 128 --last_layer 'pointer'
####
