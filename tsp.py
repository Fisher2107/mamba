import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from model import MambaFull, generate_data, seq2seq_generate_tour, compute_tour_length, Bandau_Pointer, Dot_Pointer
from datetime import datetime
import argparse
import wandb

#login to wandb
wandb.login()
#parser
parser = argparse.ArgumentParser(description='Train Mamba model')
parser.add_argument('--bsz', type=int, default=600, help='Batch size')
parser.add_argument('--d_model', type=int, default=64, help='Model dimension')#ensure that this is a multiple of 2 if fourier_scale is not None
parser.add_argument('--coord_dim', type=int, default=2, help='Coordinate dimension')
parser.add_argument('--nb_layers', type=int, default=4, help='Number of layers in the model')
parser.add_argument('--mlp_cls', type=str, default='gatedmlp', help='Type of mlp to use')#set as 'identity' or 'gatedmlp'
parser.add_argument('--city_count', type=int, default=5, help='Number of cities')
parser.add_argument('--fourier_scale', type=float, default=None, help='Fourier scale')#If set as None a standard Linear map is used else a gaussian fourier feature mapping is used
parser.add_argument('--start', type=float, default=2, help='Start token')

parser.add_argument('--nb_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--nb_batch_per_epoch', type=int, default=10, help='Number of batches per epoch')

parser.add_argument('--test_size', type=int, default=2000, help='Size of test data')
parser.add_argument('--save_loc', type=str, default='checkpoints/not_named', help='Location to save model')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
parser.add_argument('--recycle_data', type=int, default=0, help='Recycle data')
parser.add_argument('--model_name', type=str, default='Full', help='Model name')
parser.add_argument('--mamba2', type=bool, default=False, help='choose if mamba2 is used')
parser.add_argument('--reverse', type=bool, default=False, help='Reverse even model layers')
parser.add_argument('--reverse_start', type=bool, default=False, help='Set to True if you want to reverse the input')
parser.add_argument('--last_layer', type=str, default='identity', help='Select whether the last layer is identity or pointer')
parser.add_argument('--test_folder_name', type=str, default=None, help='Name of folder where test data is stored')

# Define model parameters and hyperparameters
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

args=DotDict() 

# Update the DotDict instance with the parsed arguments
parsed_args = parser.parse_args()
for key, value in vars(parsed_args).items():
    setattr(args, key, value)

if args.test_folder_name is None and (args.start).is_integer():
    args.test_data_loc=f'data/start_{int(args.start)}/{args.test_size}_{args.city_count}_{args.coord_dim}.pt'
else:
    args.test_data_loc=f'data/{args.test_folder_name}/{args.test_size}_{args.city_count}_{args.coord_dim}.pt'

#Load checkpoint
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
else:
    checkpoint = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.fourier_scale is None:
    args.B = None
else:
    if checkpoint:
        args.B = checkpoint['args'].B
    else:
        args.B = torch.randn(args.d_model // 2, 2).to(device) * args.fourier_scale



args['x_flipped']=False
if args.reverse_start and not args.reverse:
    args['x_flipped']=True
elif args.reverse_start and args.reverse:
    if args.nb_layers%2!=0:
        args['x_flipped']=True
elif args.reverse and not args.reverse_start:
    if args.nb_layers%2==0:
        args['x_flipped']=True
run = wandb.init(
    # Set the project where this run will be logged
    project="Mamba",
    # Track hyperparameters and run metadata
    config=args,
)

#load train and baseline model, where baseline is used to reduce variance in loss function as per the REINFORCE algorithm. 
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls,args.B, args.reverse,args.reverse_start,args.mamba2,args.last_layer).to(device)
model_baseline = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls,args.B, args.reverse,args.reverse_start,args.mamba2,args.last_layer).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model_train.parameters(), lr=1e-4)


if checkpoint:
    if 'model_baseline_state_dict' in checkpoint.keys():
        model_train.load_state_dict(checkpoint['model_train_state_dict'])
        model_baseline.load_state_dict(checkpoint['model_baseline_state_dict'])
    else:
        model_train.load_state_dict(checkpoint['model_state_dict'])
        model_baseline.load_state_dict(checkpoint['model_state_dict']) 

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    tot_time_ckpt = checkpoint['time_tot']
    start_epoch = checkpoint['epoch']
    mean_tour_length_list = checkpoint['mean_tour_length_list']
    mean_tour_length_best = min([i.item() for i in checkpoint['mean_tour_length_list']])
    if 'time_to_reach_best' in checkpoint.keys():    
        best_time = checkpoint['time_to_reach_best']
    else:
        best_time = 0
else:
    tot_time_ckpt, start_epoch = 0,0
    mean_tour_length_list = [] 
    mean_tour_length_best = float('inf') 
    best_time = 0
    model_baseline.load_state_dict(model_train.state_dict())

model_baseline.eval()
#for name, param in model_train.named_parameters():
#    print(f"Parameter: {name}, Size: {param.size()}")
total_params = sum(p.numel() for p in model_train.parameters())
print(f"Total number of parameters: {total_params}")


test_data = torch.load(args.test_data_loc).to(device)
test_data_batches = torch.split(test_data, args.bsz)

print(args)

start_training_time = time.time()
now = datetime.now()
date_time = now.strftime("%d-%m_%H-%M")

# Training loop
for epoch in tqdm(range(start_epoch,args.nb_epochs)):
    model_train.train()
    i= 0 # Tracks the number of steps before we generate new data
    start = time.time()
    L_train_train_total = 0
    L_baseline_train_total = 0
    for step in range(args.nb_batch_per_epoch):

        if i == 0:
            #Inputs will have size (bsz, seq_len, coord_dim)
            inputs = generate_data(device, args.bsz, args.city_count, args.coord_dim,start=args.start)
            i=args.recycle_data
        else: i-=1

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours_train, sumLogProbOfActions = seq2seq_generate_tour(device,model_train,inputs,deterministic=False)
        tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,deterministic=False)
        #get the length of the tours
        with torch.no_grad():
            L_train = compute_tour_length(inputs, tours_train)
            L_baseline = compute_tour_length(inputs, tours_baseline)
            L_train_train_total += L_train.sum()
            L_baseline_train_total += L_baseline.sum()
        #print(f"L_train requires_grad: {L_train.requires_grad}")

        # backprop     
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt

    ###################
    # Evaluate train model and baseline
    ###################
    model_train.eval()
    L_train_total = 0
    L_baseline_total = 0
    
    # Compute tour for model and baseline for test data, making it sure its split to not overload the gpu
    for test_data_batch in test_data_batches:
        tour_train, _ = seq2seq_generate_tour(device, model_train, test_data_batch, deterministic=True)
        tour_baseline, _ = seq2seq_generate_tour(device, model_baseline, test_data_batch, deterministic=True)

        # Get the lengths of the tours and add to the accumulators
        L_train_total += compute_tour_length(test_data_batch, tour_train).sum()
        L_baseline_total += compute_tour_length(test_data_batch, tour_baseline).sum()

    # Compute the average tour lengths
    L_train = L_train_total / args.test_size
    L_baseline = L_baseline_total / args.test_size

    #print(f'Epoch {epoch}, test tour length train: {L_train}, test tour length baseline: {L_baseline}, time one epoch: {time_one_epoch}, time tot: {time_tot}')
    wandb.log({
        "test_tour length train": float(L_train),
        "test_tour length baseline": float(L_baseline),
        "time one epoch": float(time_one_epoch),
        "time tot": float(time_tot),
        "train_tour length train": float(L_train_train_total),
        "train_length baseline": float(L_baseline_train_total)
    })

    mean_tour_length_list.append(L_train)
    # evaluate train model and baseline and update if train model is better
    if L_train < L_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )
        best_time = time_tot

    # Save checkpoint every 10,000 epochs
    if L_train < mean_tour_length_best or epoch % 10 == 0:
        if L_train < mean_tour_length_best:
            mean_tour_length_best = L_train

        # Append to filename
        filename = f"file_{date_time}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_train_state_dict': model_train.state_dict(),
            'model_baseline_state_dict': model_baseline.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean_tour_length_list': mean_tour_length_list,
            'args': args,
            'time_tot': time_tot,
            'time_to_reach_best': best_time,
        }
        torch.save(checkpoint, f'{args.save_loc}_{date_time}.pt' )
