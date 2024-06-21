#!/bin/bash

#python tsp.py --start 5 --save_loc 'checkpoints/start/Linear_mlp_5start'
#python tsp.py --start 'rand' --save_loc 'checkpoints/start/Linear_mlp_randstart'
python tsp.py --save_loc 'checkpoints/reverse/Linear_reverse3' --nb_epochs 1000  --reverse True --nb_layers 3
python tsp.py --save_loc 'checkpoints/reverse/Linear_3' --nb_epochs 1000  --reverse False --nb_layers 3
python tsp.py --save_loc 'checkpoints/reverse/Linear_reverse4' --nb_epochs 1000  --reverse True --nb_layers 4
python tsp.py --save_loc 'checkpoints/reverse/Linear_reverse5' --nb_epochs 1000  --reverse True --nb_layers 5
python tsp.py --save_loc 'checkpoints/reverse/Linear_5' --nb_epochs 1000  --reverse False --nb_layers 5
