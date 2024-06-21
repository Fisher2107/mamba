#!/bin/bash

#python tsp.py --start 5 --save_loc 'checkpoints/start/Linear_mlp_5start'
#python tsp.py --start 'rand' --save_loc 'checkpoints/start/Linear_mlp_randstart'
python tsp.py --start 2 --checkpoint 'checkpoints/start/Linear_mlp_2start_again_21-06_11-56.pt' --save_loc 'checkpoints/start/Linear_mlp_2start_again' --nb_epochs 2000

