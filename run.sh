#!/bin/bash

#python tsp.py --start 5 --save_loc 'checkpoints/start/Linear_mlp_5start'
#python tsp.py --start 'rand' --save_loc 'checkpoints/start/Linear_mlp_randstart'
#python tsp.py --save_loc 'checkpoints/reverse/Linear_reverse3' --nb_epochs 1000  --reverse True --nb_layers 3
#python tsp.py --save_loc 'checkpoints/mamba2/mamba2_3lay' --nb_epochs 1000 --nb_layers 3 --mamba2 True --reverse True --d_model 128 --city_count 10
python tsp.py --save_loc 'checkpoints/mamba2/mamba2_4lay' --nb_epochs 1000 --nb_layers 4 --mamba2 True  --reverse True --d_model 128 --city_count 10 --nb_epochs 6000 --checkpoint 'checkpoints/mamba2/mamba2_4lay_22-06_17-16.pt'
#python tsp.py --save_loc 'checkpoints/mamba2/mamba_3lay' --nb_epochs 1000 --nb_layers 3 --reverse True --d_model 128 --city_count 10
#python tsp.py --save_loc 'checkpoints/mamba2/mamba_4lay' --nb_epochs 1000 --nb_layers 4 --reverse True  --d_model 128 --city_count 10 --nb_epochs 2000 --checkpoint 'checkpoints/mamba2/mamba_4lay_22-06_23-42.pt'

