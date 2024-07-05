#!/bin/bash

#Fourier Experiments
python tsp.py --save_loc 'checkpoints/embed2/fourier1_5city_3l' --nb_layers 3  --nb_epochs 2000 --fourier_scale 1
python tsp.py --save_loc 'checkpoints/embed2/fourier2_5city_3l' --nb_layers 3  --nb_epochs 2000 --fourier_scale 2
python tsp.py --save_loc 'checkpoints/embed2/fourier5_5city_3l' --nb_layers 3  --nb_epochs 2000 --fourier_scale 5
python tsp.py --save_loc 'checkpoints/embed2/fourier10_5city_3l' --nb_layers 3  --nb_epochs 2000 --fourier_scale 10
python tsp.py --save_loc 'checkpoints/embed2/linear_5city_3l' --nb_layers 3 --nb_epochs 2000

python tsp.py --save_loc 'checkpoints/embed2/fourier1_5city_4l' --nb_layers 4 --nb_epochs 2000 --fourier_scale 1
python tsp.py --save_loc 'checkpoints/embed2/fourier2_5city_4l' --nb_layers 4 --nb_epochs 2000 --fourier_scale 2
python tsp.py --save_loc 'checkpoints/embed2/fourier5_5city_4l' --nb_layers 4 --nb_epochs 2000 --fourier_scale 5
python tsp.py --save_loc 'checkpoints/embed2/fourier10_5city_4l' --nb_layers 4 --nb_epochs 2000 --fourier_scale 10
python tsp.py --save_loc 'checkpoints/embed2/linear_5city_4l' --nb_layers 4 --nb_epochs 2000

python tsp.py --save_loc 'checkpoints/embed2/fourier1_10city_3l' --nb_layers 3 --nb_epochs 2000 --city_count 10 --fourier_scale 1
python tsp.py --save_loc 'checkpoints/embed2/fourier2_10city_3l' --nb_layers 3 --nb_epochs 2000 --city_count 10 --fourier_scale 2
python tsp.py --save_loc 'checkpoints/embed2/fourier5_10city_3l' --nb_layers 3 --nb_epochs 2000 --city_count 10 --fourier_scale 5
python tsp.py --save_loc 'checkpoints/embed2/fourier10_10city_3l' --nb_layers 3 --nb_epochs 2000 --city_count 10 --fourier_scale 10
python tsp.py --save_loc 'checkpoints/embed2/linear_10city_3l' --nb_layers 3 --nb_epochs 2000 --city_count 10

python tsp.py --save_loc 'checkpoints/embed2/fourier1_10city_4l' --nb_layers 4 --nb_epochs 2000 --city_count 10 --fourier_scale 1
python tsp.py --save_loc 'checkpoints/embed2/fourier2_10city_4l' --nb_layers 4 --nb_epochs 2000 --city_count 10 --fourier_scale 2
python tsp.py --save_loc 'checkpoints/embed2/fourier5_10city_4l' --nb_layers 4 --nb_epochs 2000 --city_count 10 --fourier_scale 5
python tsp.py --save_loc 'checkpoints/embed2/fourier10_10city_4l' --nb_layers 4 --nb_epochs 2000 --city_count 10 --fourier_scale 10
python tsp.py --save_loc 'checkpoints/embed2/linear_10city_4l' --nb_layers 4 --nb_epochs 2000 --city_count 10
####
