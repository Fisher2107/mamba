import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from model import MambaFull, generate_data, seq2seq_generate_tour, compute_tour_length

# Define model parameters and hyperparameters
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

args=DotDict()
#Args for the model

args.bsz=50
args.d_model = 64
args.coord_dim = 2
args.nb_layers = 2
args.mlp_cls = nn.Identity #nn.Linear
args.city_count = 50
args.sequence_length = args.city_count + 1
args.deterministic = False #used for sampling from the model

#Args for the training
args.nb_epochs=100
args.nb_batch_per_epoch=100 
args.nb_batch_eval=10
#0 => data will not be recycled and each step new data is generated, however this will make the gpu spend most of the time loading data. Recommeded val is 100
args.recycle_data=25

tot_time_ckpt = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#model which will be trained on
model_train = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
#model which will be used as our baseline in the REINFORCE algorithm. This model will not be trained and only used for evaluation
model_baseline = MambaFull(args.d_model, args.city_count, args.nb_layers, args.coord_dim, args.mlp_cls).to(device)
model_baseline.load_state_dict(model_train.state_dict())
model_baseline.eval()

# Define a loss function
loss_fn = nn.CrossEntropyLoss()

# Define an optimizer
optimizer = Adam(model_train.parameters(), lr=1e-4)


mean_tour_length_train_list = [] # List to store loss values
mean_tour_length_train_best = float('inf') # Variable to store the best loss value
best_loss = float('inf')

start_training_time = time.time()

# Training loop
for epoch in tqdm(range(args.nb_epochs)):
    model_train.train()
    i= 0 # Tracks the number of steps before we generate new data
    start = time.time()
    for step in tqdm(range(args.nb_batch_per_epoch)):

        if i == 0:
            #Inputs will have size (bsz, seq_len, coord_dim)
            inputs = generate_data(device, args.bsz, args.city_count, args.coord_dim)
            i=args.recycle_data
        else: i-=1

        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours_train, sumLogProbOfActions = seq2seq_generate_tour(args,device,model_train,inputs)
        tours_baseline, _ = seq2seq_generate_tour(args,device,model_baseline,inputs)
        #get the length of the tours
        L_train = compute_tour_length(inputs, tours_train)
        L_baseline = compute_tour_length(inputs, tours_baseline)

        # backprop     
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt

    ###################
    # Evaluate train model and baseline on 10k random TSP instances
    ###################
    model_train.eval()
    mean_tour_length_train = 0
    mean_tour_length_baseline = 0
    for step in range(0,args.nb_batch_eval):

        # generate a batch of random tsp instances 
        inputs = generate_data(device, args.bsz, args.city_count, args.coord_dim)  

        # compute tour for model and baseline
        with torch.no_grad():
            tour_train, _ = seq2seq_generate_tour(args,device,model_train,inputs)
            tour_baseline, _ = seq2seq_generate_tour(args,device,model_baseline,inputs)
            
        # get the lengths of the tours
        L_train = compute_tour_length(inputs, tour_train)
        L_baseline = compute_tour_length(inputs, tour_baseline)

        # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
        mean_tour_length_train += L_train.mean().item()
        mean_tour_length_baseline += L_baseline.mean().item()

    mean_tour_length_train =  mean_tour_length_train/ args.nb_batch_eval
    mean_tour_length_baseline =  mean_tour_length_baseline/ args.nb_batch_eval
    print(f'Epoch {epoch}, mean tour length train: {mean_tour_length_train}, mean tour length baseline: {mean_tour_length_baseline}, time one epoch: {time_one_epoch}, time tot: {time_tot}')

    mean_tour_length_train_list.append(mean_tour_length_train)
    # evaluate train model and baseline and update if train model is better
    if mean_tour_length_train < mean_tour_length_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )

    # Save checkpoint every 10,000 epochs
    if mean_tour_length_train > mean_tour_length_train_best:
        mean_tour_length_train_best = mean_tour_length_train
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_train.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean_tour_length_list': mean_tour_length_train_list,
        }
        torch.save(checkpoint, 'best_checkpoint.pt')