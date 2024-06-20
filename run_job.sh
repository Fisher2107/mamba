#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit:
$ -l h_rt=02:00:00
#
# Set working directory to the directory where the job is submitted from:
$ -cwd
#
# Request one GPU in the gpu queue:
$ -q gpu 
$ -pe gpu-a100 1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
$ -l h_vmem=24G

# Initialise the environment modules and load CUDA version 11.0.2
source /exports/applications/support/set_qlogin_environment.sh
module load cuda/11.8

# Run the executable
python tsp.py --start = 5 --save_loc = 'checkpoints/embed/Linear_mlp_5start'
python tsp.py --start = 'rand' --save_loc = 'checkpoints/embed/Linear_mlp_randstart'