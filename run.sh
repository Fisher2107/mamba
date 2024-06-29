#!/bin/bash

#Start Token Experiments
#python tsp.py --start -1 --save_loc 'checkpoints/start2/startneg1' --test_folder_name 'start_min1'
#python tsp.py --start -0.1 --save_loc 'checkpoints/start2/startnegpoint1' --test_folder_name 'start_min01'
#python tsp.py --start 1.5 --save_loc 'checkpoints/start2/start1andhalf' --test_folder_name 'start_1p5'
#python tsp.py --start 2 --save_loc 'checkpoints/start2/start2'
#python tsp.py --start 2.5 --save_loc 'checkpoints/start2/start2andhalf' --test_folder_name 'start_2p5'
#python tsp.py --start 3 --save_loc 'checkpoints/start2/start3'
#python tsp.py --start 5 --save_loc 'checkpoints/start2/start5'
#python tsp.py --start 100 --save_loc 'checkpoints/start2/start100'

#Pointer Experiments
python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_3l_64' --nb_layers 3 --mamba2 True  --reverse True --d_model 64 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_3l_32' --nb_layers 3 --mamba2 True  --reverse True --d_model 32 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_3l_64' --nb_layers 4 --mamba2 True  --reverse True --d_model 64 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_3l_32' --nb_layers 4 --mamba2 True  --reverse True --d_model 32 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
