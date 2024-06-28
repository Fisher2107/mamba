#!/bin/bash

#python tsp.py --start 5 --save_loc 'checkpoints/start/Linear_mlp_5start'
#python tsp.py --start 'rand' --save_loc 'checkpoints/start/Linear_mlp_randstart'
#python tsp.py --save_loc 'checkpoints/reverse/Linear_reverse3' --nb_epochs 1000  --reverse True --nb_layers 3
#python tsp.py --save_loc 'checkpoints/mamba2/mamba2_3lay' --nb_epochs 1000 --nb_layers 3 --mamba2 True --reverse True --d_model 128 --city_count 10
#python tsp.py --save_loc 'checkpoints/pointer/mamba2_3l_band' --nb_layers 3 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 1000 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/pointer/mamba2_3l_soft' --nb_layers 3 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 1000 --last_layer 'dot_pointer'
#python tsp.py --save_loc 'checkpoints/pointer/mamba2_3l_ident' --nb_layers 3 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 1000
#python tsp.py --save_loc 'checkpoints/pointer/mamba2_1l_ident' --nb_layers 1 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 1000
#python tsp.py --save_loc 'checkpoints/pointer/mamba2_1l_band' --nb_layers 1 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 1000 --last_layer 'pointer'
python tsp.py --save_loc 'checkpoints/pointer/mamba2_1l_soft' --nb_layers 1 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 1000 --last_layer 'dot_pointer'

