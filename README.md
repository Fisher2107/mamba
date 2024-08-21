# Mamba for TSP

This is a fork of of the Mamba github repo https://github.com/state-spaces/mamba

## Installation of Mamba (taken from original mamba repos README)

- [Option] `pip install causal-conv1d>=1.4.0`: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
- `pip install mamba-ssm`: the core Mamba package.

It can also be built from source with `pip install .` from this repository.

If `pip` complains about PyTorch versions, try passing `--no-build-isolation` to `pip`.

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

## Usage Train
To train all mamba TSP models we called the tsp.py file whilst parsing arguments. A typcial example looks like:
python tsp.py --save_loc 'checkpoints/example' --nb_layers 3  --nb_epochs 6000  --mamba2 True --city_count 20 --nb_batch_per_epoch 10 --bsz 600 --last_layer 'pointer' --reverse True

The tsp.py file used many imports however one notatble one is model.py. This contains many helper functions and classes used to train the model such as compute_tour_length, MambaFull and seq_2_seq_generate_tour.

Also when --pynvml True was called we would call functions from the gpu_stats.py file in the main dir. This is what I used to keep track of scalability metrics used in Section Scalability. 

## Usage Inference + checkpoints folder
All trained models create checkpoints saved in the checkpoint folder. When running the mamba model we load the checkpoints into our mamba model, allowing it us to call the model on different datasets.

## Concorde and Transformer
During my dissertation I also used the concorde and transformer model. Their respective repositories can be found here:
https://github.com/jvkersch/pyconcorde
https://github.com/xbresson/TSP_Transformer

## Data Folder
The data folder contains many sub folders. The subfolders named begninng with start contain my test dataset I used to evaluate model performance whilst I was training. They contain 2000 instances of TSP instances and where generated using generate_data.py. Th generlaisation subfolder containes the custom TSP dataset I used in Section 5.2.2 Visualisation. Transformer data contains a 10k tsp instance of city count 50 and 100. This was the test set the transformer model used and which I also used in my analysis section. It also contains labels representing the optimal tour length of each instance. 

## Evals Folder 
This folder contains most of the python files I used to generate plots. It also contains figs which is where I saved the pdf figures generated from the python files. 

## Benchmarks Folder
These contain transformer chekpoints and pyconorder repositories used to call the concorde and transformer model. They have been emptied due to space restrictions however their contents can be found in there repositories. benchmark_solvers.py contains code that computes the greedy tour length used in my figures. It also contains an brute force exact solver that can only be used in low city counts (<=10). The transformer_model.py file contains the code used to call the transformer model used during analysis.

## IPYNB files
These files stored in the main dir are what I used to generate visualistions of the transformer and mamba models. They were also used to generate the boxplots, histograms and heatmaps used in analysis. 

## Jobs Folder
Most of my experiments where conducted on the University of Edinburghs Eddie Cluster using a A100 GPU. Some smaller experiments where conducted on a L4 GPU provided by Lightning AI. We used Eddie to conduct experiments using Jobs or Interactive Sessions. When using jobs we create bash used to request a gpu from Eddie and run experiments. The jobs folder contains my saved jobs. 

## Other folders
These are mostly folders carried over from https://github.com/state-spaces/mamba. For example the mamba_ssm folder is what contains the main Mamba class. 


