from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.modules.mlp import GatedMLP
import torch
from torch.distributions.categorical import Categorical
from functools import partial
import torch.nn as nn
import numpy as np

def generate_data(device, batch_size, city_count, coord_dim=2 , start = 2):
    
    #The value of start will signify the start of the decoding phase
    if start == 'rand':
        return torch.rand(batch_size, city_count+1, coord_dim).to(device)
    if start == 0:
        epsilon = 0.02
        random_data = epsilon + (1 - epsilon) * torch.rand(batch_size, city_count, coord_dim).to(device)
    else:
        random_data = torch.rand(batch_size, city_count, coord_dim).to(device)
    
    start_data = torch.full((batch_size, 1, coord_dim), start).to(device)
    return torch.cat((random_data, start_data), dim=1)

def compute_tour_length(x, tour,remove_start_token=True,get_tour_only=True): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, city_count+1, 2) batch of tsp tour instances
             tour of size (bsz, city_count) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    if remove_start_token:
        x = x[:,:-1,:]
    bsz = x.shape[0]
    arange_vec = torch.arange(bsz, device=x.device).unsqueeze(-1)
    tour = tour.to(x.device)

    # Get the cities in the order of the tour
    ordered_cities = x[arange_vec, tour, :] # size(ordered_cities)=(bsz, city_count, 2)

    # Compute the differences between each pair of consecutive cities
    diffs = ordered_cities[:, 1:, :] - ordered_cities[:, :-1, :] # size(diffs)=(bsz, city_count-1, 2)

    # Compute the distance between each pair of consecutive cities
    distances = torch.sqrt(torch.sum(diffs**2, dim=2)) # size(distances)=(bsz, city_count-1)

    # Add the distance from the last city to the first
    distances = torch.cat([distances, torch.norm(ordered_cities[:, 0, :] - ordered_cities[:, -1, :], dim=1).unsqueeze(-1)], dim=1)

    # Sum the distances to get the total length of each tour
    L = torch.sum(distances, dim=1)

    if get_tour_only:
        return L
    #this will both return the tour and the length
    else:
        return L, distances

# Fourier feature mapping
class input_mapping(nn.Module):
    def __init__(self, B, d_model, coord_dim=2,device='cuda'):
        super().__init__()
        self.B = B
        if B is None:
            self.embedding = nn.Linear(coord_dim, d_model)
        else:
            self.embedding = nn.Linear(d_model, d_model)

    def forward(self, x):
        if self.B is None:
            return self.embedding(x)
        else:
            x = 2. * np.pi * x @ self.B.T
            x= torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            return self.embedding(x)


class MambaFull(nn.Module):
    def __init__(self,
    d_model,
    city_count,
    nb_layers,
    coord_dim=2,
    mlp_cls='identity',
    B=None,
    reverse=False,
    reverse_start=False,
    mamba2=False,
    last_layer='identity'):
        super().__init__()
        self.d_model=d_model
        self.city_count=city_count
        self.nb_layers=nb_layers
        if mlp_cls == 'gatedmlp':
            self.mlp_cls = GatedMLP
        elif mlp_cls == 'identity':
            self.mlp_cls = nn.Identity
        else:
            raise ValueError('mlp_cls must be either "gatedmlp" or "identity"')
        self.norm_f = nn.LayerNorm(d_model)

        self.embedding = input_mapping(B,d_model,coord_dim=coord_dim)
        if d_model%16 != 0: raise ValueError('d_model must be a multiple of 16')
        if mamba2:
            self.mixer_cls = partial(Mamba2,d_model,headdim=d_model//4)
        else:
            self.mixer_cls = partial(Mamba,d_model)

        self.layers = nn.ModuleList([
                    Block(dim= d_model,
                        mixer_cls= self.mixer_cls,
                        mlp_cls= self.mlp_cls,
                        fused_add_norm=True,
                        residual_in_fp32=True,
                        )   for _ in range(nb_layers)])

        self.pointer = False
        if last_layer == 'identity':
            self.last_layer = nn.Identity()
        elif last_layer == 'pointer':
            self.last_layer = Bandau_Pointer(d_model,nb_layers,reverse,reverse_start)
            self.pointer = True
        elif last_layer == 'dot_pointer':
            self.last_layer = Dot_Pointer(d_model, city_count)
        else:
            raise ValueError("Last layer must be either ('identity', 'pointer', 'dot_pointer')")

        if last_layer=='identity':#Set output_head to False if we use a last layer that is a pointer network
            self.output_head = nn.Linear(d_model,city_count, bias=False)
        else:
            self.output_head = nn.Identity()
        self.reverse = reverse
        self.reverse_start = reverse_start

    #only include city_count if you want to override the default city_count and you are use last_layer='pointer'
    def forward(self,x,city_count=None):
        x = self.embedding(x)
        if self.reverse_start:
            x = torch.flip(x,[1])

        residual=None
        for i,layer in enumerate(self.layers):
            if self.reverse == True and i>0:
                x , residual = layer(torch.flip(x,[1]),torch.flip(residual,[1]))
            else:
                x , residual = layer(x,residual)

        # Set prenorm=False here since we don't need the residual
        x = layer_norm_fn(
            x,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
            is_rms_norm=True
        )

        if self.pointer:
            x = self.last_layer(x,city_count)
        else:
            x = self.last_layer(x)

        logits = self.output_head(x)
        #mask vistited cities
        return logits

def seq2seq_generate_tour(device,model,inputs,lastlayer,deterministic=False,sum_logactions=True):
    # Mask is used to prevent the model from choosing the same city twice
    bsz = inputs.shape[0]
    city_count = inputs.shape[1] - 1
    mask = torch.ones(bsz, city_count).to(device)
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    tours = []
    # list that will contain Float tensors of shape (bsz,) that gives the log probs of the choices made at time t
    LogProbOfActions = []
    #Construct tour recursively
    for i in range(city_count):
        if lastlayer=='pointer':
            outputs = model(inputs,city_count)[:,-1,:] #outputs of shape (bsz,city_count)
        else:
            outputs = model(inputs)[:,-1,:]
        #print(outputs.shape)
        outputs = outputs.masked_fill_(mask == 0, -float('inf'))
        #print(outputs[0])
        outputs = nn.Softmax(dim=1)(outputs)
        #print(outputs[0])
        if deterministic:
            next_city = torch.argmax(outputs, dim=1) #next_city of shape (bsz,)
        else:
            next_city = Categorical(outputs).sample()
        #print(next_city[0])
        tours.append(next_city)
        LogProbOfActions.append(torch.log(outputs[torch.arange(bsz), next_city]) )
        #Try: next_city = next_city.detach()
        mask[torch.arange(bsz), next_city] = 0
        inputs = torch.cat((inputs, inputs[torch.arange(bsz), next_city, :].unsqueeze(1)), dim=1)
    tours = torch.stack(tours, dim=1).to(device)
    if sum_logactions:
        sumLogProbOfActions = torch.stack(LogProbOfActions, dim=1).sum(dim=1).to(device)
        return tours, sumLogProbOfActions
    else:
        return tours, LogProbOfActions
    
def train_step(model_train, model_baseline, inputs, optimizer, device,L_train_train_total,L_baseline_train_total,gpu_logger,action,non_det):
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    lastlayer = 'identity'
    if model_train.pointer:
        lastlayer = 'pointer'
    reverse_start = model_train.reverse_start
    reverse = model_train.reverse
    
    if action == 'tour':
        if gpu_logger: gpu_logger.log_event('generating tours of train model')
        tours_train, sumLogProbOfActions = seq2seq_generate_tour(device,model_train,inputs,lastlayer=lastlayer,deterministic=False)
        if gpu_logger: gpu_logger.log_event('generating tours of baseline model')
        tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,lastlayer=lastlayer,deterministic=not non_det)

        
        #get the length of the tours
        with torch.no_grad():
            if gpu_logger: gpu_logger.log_event('computing tour length of train model')
            L_train = compute_tour_length(inputs, tours_train)
            if gpu_logger: gpu_logger.log_event('computing tour length of baseline model')
            L_baseline = compute_tour_length(inputs, tours_baseline)
            L_train_train_total += L_train.sum()
            L_baseline_train_total += L_baseline.sum()
        #print(f"L_train requires_grad: {L_train.requires_grad}")

        if gpu_logger: gpu_logger.log_event('computing loss and backprop')
        # backprop     
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    elif action == 'next_city':
        with torch.no_grad():
            tours_train, _ = seq2seq_generate_tour(device,model_train,inputs,lastlayer=lastlayer,deterministic=False)
            tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,lastlayer=lastlayer,deterministic=not non_det)
        
            #get the length of the tours
            if gpu_logger: gpu_logger.log_event('computing tour length of train model')
            L_train = compute_tour_length(inputs, tours_train)
            if gpu_logger: gpu_logger.log_event('computing tour length of baseline model')
            L_baseline = compute_tour_length(inputs, tours_baseline)
            L_train_train_total += L_train.sum()
            L_baseline_train_total += L_baseline.sum()
        
        #Go through tour and backprop
        bsz = inputs.shape[0]
        city_count = inputs.shape[1] - 1
        mask = torch.ones(bsz, city_count).to(device)
        #Construct tour recursively
        optimizer.zero_grad()
        for i in range(city_count):
            if lastlayer=='pointer':
                outputs = model_train(inputs,city_count)[:,-1,:]
            else:
                outputs = model_train(inputs)[:,-1,:]
            outputs = outputs.masked_fill_(mask == 0, -float('inf'))
            outputs = nn.Softmax(dim=1)(outputs)

            next_city = tours_train[:,i]
            LogProbOfAction = torch.log(outputs[torch.arange(bsz), next_city])
            mask[torch.arange(bsz), next_city] = 0
            inputs = torch.cat((inputs, inputs[torch.arange(bsz), next_city, :].unsqueeze(1)), dim=1)
            loss = torch.mean( (L_train - L_baseline)* LogProbOfAction )
            loss.backward()
        optimizer.step()
    
    elif 'importance_sampling' in action:
        reuse_tours = int(action.split('_')[-1])
        if reverse or reverse_start:
            raise ValueError('importance_sampling only works with reverse=False and reverse_start=False')
        
        with torch.no_grad():
            tours_train, sumLogProbOfActions = seq2seq_generate_tour(device,model_train,inputs,lastlayer=lastlayer,deterministic=False)
            tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,lastlayer=lastlayer,deterministic=not non_det)
        
            #get the length of the tours
            if gpu_logger: gpu_logger.log_event('computing tour length of train model')
            L_train = compute_tour_length(inputs, tours_train)
            if gpu_logger: gpu_logger.log_event('computing tour length of baseline model')
            L_baseline = compute_tour_length(inputs, tours_baseline)
            L_train_train_total += L_train.sum()
            L_baseline_train_total += L_baseline.sum()
        

        bsz = inputs.shape[0]
        city_count = inputs.shape[1] - 1
        mask = torch.ones(bsz, city_count).to(device)

        #inputs size (bsz, city_count+1, 2), tours_train size (bsz, city_count)
        for i in range(tours_train.shape[1]-1):
            inputs = torch.cat((inputs, inputs[torch.arange(inputs.shape[0]), tours_train[:,i], :].unsqueeze(1)), dim=1)
        print(inputs.shape) #should be inputs size (bsz, 2*city_count, 2)
        

        for i in range(reuse_tours):
            #Go through tour with teacher forcing
            optimizer.zero_grad()
            if lastlayer=='pointer':
                outputs = model_train(inputs,city_count)[:,-city_count,:]
            else:
                outputs = model_train(inputs)[:,-city_count,:]
            print(outputs.shape)#should be (bsz, city_count,city_count)
            
            for i in range(city_count,0,-1):
                outputs[:,-i,:] = outputs[:,-i,:].masked_fill_(mask == 0, -float('inf'))
                mask[torch.arange(bsz), tours_train[:,-i]] = 0
            
            outputs = nn.Softmax(dim=2)(outputs)
            print(outputs.shape)#should be (bsz, city_count,city_count)
            '''LogProbOfActions = torch.zeros(bsz,city_count).to(device)
            for i in range(bsz):
                for j in range(city_count):
                    LogProbOfActions[i,j] = torch.log(outputs[i,j,tours_train[i,j]])'''
            LogProbOfActions = torch.log(outputs[torch.arange(bsz).unsqueeze(1), torch.arange(city_count).unsqueeze(0), tours_train]).to(device)
            sumLogProbOfActionsnew = LogProbOfActions.sum(dim=1)
            loss = torch.mean( (L_train - L_baseline)* torch.exp(sumLogProbOfActionsnew-sumLogProbOfActions) )
            loss.backward()
            optimizer.step()
             
    else:
        raise ValueError('action must be either "tour" or "next_city"')

    return L_train_train_total, L_baseline_train_total

class Bandau_Pointer(nn.Module):
    def __init__(self, d_model,nb_layers,reverse,reverse_start):
        super().__init__()
        self.d_model=d_model
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, 1, bias=False)
        
        self.x_flipped=False
        if reverse_start and not reverse:
            self.x_flipped=True
        elif reverse_start and reverse:
            if nb_layers%2!=0:
                self.x_flipped=True
        elif reverse and not reverse_start:
            if nb_layers%2==0:
                self.x_flipped=True
            
    def forward(self,x,city_count):
        if self.x_flipped:
            x = torch.flip(x,[1])
        key = self.W1(x[:,:city_count,:])#(bsz,city_count,d_model)
        query = self.W2(x[:,-1,:].unsqueeze(1))#(bsz,1,d_model)
        energy = self.V(torch.tanh(key + query)).squeeze(-1)
        return energy.unsqueeze(1) #returns a tensor of size (bsz,1,city_count)

class Dot_Pointer(nn.Module):
    def __init__(self, d_model,city_count):
        super().__init__()
        self.d_model=d_model
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, city_count, bias=False)
    def forward(self,x):
        key = self.W1(x[:,:self.city_count,:]).reshape(-1,self.d_model,self.city_count)
        query = self.W2(x[:,-1,:].unsqueeze(1))#(bsz,1,d_model)
        value = self.V(x[:,:self.city_count,:])#(bsz,city_count,city_count)
        print(key.shape,query.shape)
        attention_mat = (nn.Softmax(dim=-1)((query@key)/(self.d_model**0.5)))#(bsz,1,city_count)
        energy = attention_mat@value
        return energy.unsqueeze(1) #returns a tensor of size (bsz,1,city_count)