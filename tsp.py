import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from model import MambaFull, generate_data, seq2seq_generate_tour, compute_tour_length
from datetime import datetime
import argparse


#parser
parser = argparse.ArgumentParser(description='Train Mamba model')
parser.add_argument('--bsz', type=int, default=600, help='Batch size')
parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
parser.add_argument('--coord_dim', type=int, default=2, help='Coordinate dimension')
parser.add_argument('--nb_layers', type=int, default=4, help='Number of layers in the model')
parser.add_argument('--mlp_cls', type=str, default='gatedmlp', help='Type of mlp to use')
parser.add_argument('--city_count', type=int, default=5, help='Number of cities')
parser.add_argument('--fourier_scale', type=float, default=None, help='Fourier scale')
parser.add_argument('--nb_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--test_size', type=int, default=2000, help='Size of test data')
parser.add_argument('--nb_batch_per_epoch', type=int, default=10, help='Number of batches per epoch')
parser.add_argument('--save_loc', type=str, default='checkpoints/embed/Linear_mlp_4lay', help='Location to save model')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
parser.add_argument('--start', type=int, default=2, help='Start token')

# Define model parameters and hyperparameters
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

args=DotDict() 

#Args for the model
args.bsz=600
args.d_model = 64 #ensure that this is a multiple of 2
args.coord_dim = 2
args.nb_layers = 4
args.mlp_cls = 'gatedmlp' #set as 'identity' or 'gatedmlp'
args.city_count = 5
args.deterministic = False #used for sampling from the model
args.fourier_scale = None #If set as None a standard Linear map is used else a gaussian fourier feature mapping is used
args.start = 2 #start token
#args.polar = True #TODO

#Args for the training
args.nb_epochs=500
args.test_size=2000
args.nb_batch_per_epoch=10
args.save_loc = 'checkpoints/embed/Linear_mlp_4lay'
args.test_data_loc=f'data/start_{args.start}/test_rand_{args.test_size}_{args.city_count}_{args.coord_dim}.pt'
#0 => data will not be recycled and each step new data is generated, however this will make the gpu spend most of the time loading data. Recommeded val is 100
args.recycle_data=0

# Update the DotDict instance with the parsed arguments
parsed_args = parser.parse_args()
for key, value in vars(parsed_args).items():
    setattr(args, key, value)

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

#model which will be train and baseline as in the REINFORCE algorithm. 
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls, B = args.B).to(device)
model_baseline = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls, B = args.B).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model_train.parameters(), lr=1e-4)


if checkpoint:
    model_train.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    tot_time_ckpt = checkpoint['time_tot']
    start_epoch = checkpoint['epoch']
    mean_tour_length_list = checkpoint['mean_tour_length_list']
    mean_tour_length_best = min([i.item() for i in checkpoint['mean_tour_length_list']])
    print(mean_tour_length_best,mean_tour_length_list[-1])
else:
    tot_time_ckpt, start_epoch = 0,0
    mean_tour_length_list = [] 
    mean_tour_length_best = float('inf') 

model_baseline.load_state_dict(model_train.state_dict())
model_baseline.eval()
for name, param in model_train.named_parameters():
    print(f"Parameter: {name}, Size: {param.size()}")
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
    for step in range(args.nb_batch_per_epoch):

        if i == 0:
            #Inputs will have size (bsz, seq_len, coord_dim)
            inputs = generate_data(device, args.bsz, args.city_count, args.coord_dim,start=args.start)
            i=args.recycle_data
        else: i-=1

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours_train, sumLogProbOfActions = seq2seq_generate_tour(device,model_train,inputs,args.deterministic)
        tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,args.deterministic)
        #get the length of the tours
        with torch.no_grad():
            L_train = compute_tour_length(inputs, tours_train)
            L_baseline = compute_tour_length(inputs, tours_baseline)
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

    print(f'Epoch {epoch}, test tour length train: {L_train}, test tour length baseline: {L_baseline}, time one epoch: {time_one_epoch}, time tot: {time_tot}')

    mean_tour_length_list.append(L_train)
    # evaluate train model and baseline and update if train model is better
    if L_train < L_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )

    # Save checkpoint every 10,000 epochs
    if L_train < mean_tour_length_best or epoch % 10 == 0:
        mean_tour_length_best = L_train

        # Append to filename
        filename = f"file_{date_time}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_train.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean_tour_length_list': mean_tour_length_list,
            'args': args,
            'time_tot': time_tot
        }
        torch.save(checkpoint, f'{args.save_loc}_{date_time}.pt' )