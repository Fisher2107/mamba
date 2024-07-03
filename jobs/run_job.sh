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

#Reverse Experiments
#python tsp.py --save_loc 'checkpoints/reverse/mamba2_reversestart_3l' --nb_layers 3  --reverse_start True --nb_epochs 1000
#python tsp.py --save_loc 'checkpoints/reverse/mamba2_reversestart_4l' --nb_layers 4  --reverse_start True --nb_epochs 1000
#python tsp.py --save_loc 'checkpoints/reverse/mamba2_reversestart_5l' --nb_layers 5  --reverse_start True --nb_epochs 1000

#Pointer Experiments
#python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_3l_64' --nb_layers 3 --mamba2 True  --reverse True --d_model 64 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
#python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_3l_32' --nb_layers 3 --mamba2 True  --reverse True --d_model 32 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
#python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_4l_64' --nb_layers 4 --mamba2 True  --reverse True --d_model 64 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'
#python tsp.py --save_loc 'checkpoints/pointer_dim/mamba2_4l_32' --nb_layers 4 --mamba2 True  --reverse True --d_model 32 --city_count 10 --nb_epochs 2000 --last_layer 'pointer'

#Reverse10 Experiments
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reversestart' --nb_layers 3  --reverse_start True --nb_epochs 2000 --city_count 10
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reverse' --nb_layers 3  --reverse True --nb_epochs 2000 --city_count 10
python tsp.py --save_loc 'checkpoints/reverse10/mamba2_reverse' --nb_layers 3 --nb_epochs 2000 --city_count 10