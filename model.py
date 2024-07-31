from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import InferenceParams
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

# Fourier feature mapping or linear mapping
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
                        mixer_cls= partial(self.mixer_cls, layer_idx=i),
                        mlp_cls= self.mlp_cls,
                        fused_add_norm=True,
                        residual_in_fp32=True,
                        )   for i in range(nb_layers)])

        self.pointer = False
        if last_layer == 'identity':
            self.last_layer = nn.Identity()
        elif last_layer == 'pointer':
            self.last_layer = Bandau_Pointer(d_model,nb_layers,reverse,reverse_start)
            self.pointer = True
        else:
            raise ValueError("Last layer must be either ('identity', 'pointer')")

        if last_layer=='identity':#Set output_head to False if we use a last layer that is a pointer network
            self.output_head = nn.Linear(d_model,city_count, bias=False)
        else:
            self.output_head = nn.Identity()
        self.reverse = reverse
        self.reverse_start = reverse_start

    #only include city_count if you want to override the default city_count and you are use last_layer='pointer'
    def forward(self,x,city_count=None,inference_params=None):
        x = self.embedding(x)
        if self.reverse_start:
            x = torch.flip(x,[1])

        residual=None
        for i,layer in enumerate(self.layers):
            if self.reverse == True and i>0:
                x , residual = layer(torch.flip(x,[1]),torch.flip(residual,[1]))
            else:
                if inference_params is None:
                    x , residual = layer(x,residual)
                else:
                    x , residual = layer(x,residual,inference_params=inference_params)

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
            x = self.last_layer(x,city_count,inference_params)
        else:
            x = self.last_layer(x)

        logits = self.output_head(x)
        #mask vistited cities
        return logits

def seq2seq_generate_tour(device, model, inputs, lastlayer, deterministic=False, sum_logactions=True, use_inf_params=True): #set mamba_input to True TODO
    bsz, seq_len, hidden_dim = inputs.shape
    city_count = seq_len - 1
    mask = torch.ones(bsz, city_count).to(device)
    tours = []
    LogProbOfActions = []

    if use_inf_params:
        inference_params = InferenceParams(max_seqlen=seq_len * 2, max_batch_size=bsz)
        # Allocate cache for each Mamba2 layer in MambaFull
        for i, layer in enumerate(model.layers):
            conv_state, ssm_state = layer.mixer.allocate_inference_cache(bsz, seq_len * 2)
            inference_params.key_value_memory_dict[i] = (conv_state, ssm_state)
    else:
        inference_params = None

    # Process the initial input
    logits = model(inputs, city_count, inference_params=inference_params)

    for i in range(city_count):
        last_output = logits[:, -1, :] # shape (bsz,city_count)
        if use_inf_params:
            inference_params.seqlen_offset += 1

        last_output = last_output.masked_fill_(mask == 0, -float('inf'))
        probs = nn.Softmax(dim=1)(last_output)

        if deterministic:
            next_city = torch.argmax(probs, dim=1) #next_city of shape (bsz,)
        else:
            next_city = Categorical(probs).sample()
        #print(next_city[0])
        tours.append(next_city)
        LogProbOfActions.append(torch.log(probs[torch.arange(bsz), next_city]) )
        #Try: next_city = next_city.detach()
        mask[torch.arange(bsz), next_city] = 0

        if use_inf_params:
            next_input = inputs[torch.arange(bsz), next_city, :].unsqueeze(1)
            # Process single token through MambaFull
            logits = model(next_input, city_count, inference_params=inference_params)
            inference_params.seqlen_offset += 1
        else:
            inputs = torch.cat((inputs, inputs[torch.arange(bsz), next_city, :].unsqueeze(1)), dim=1)
            logits = model(inputs,city_count)

    tours = torch.stack(tours, dim=1).to(device)
    if sum_logactions:
        sumLogProbOfActions = torch.stack(LogProbOfActions, dim=1).sum(dim=1).to(device)
        return tours, sumLogProbOfActions
    else:
        return tours, LogProbOfActions
    
def train_step(model_train, model_baseline, inputs, optimizer, device,L_train_train_total,L_baseline_train_total,gpu_logger,action,non_det,use_inf_params=False):
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    lastlayer = 'identity'
    if model_train.pointer:
        lastlayer = 'pointer'
    
    if action == 'tour':
        if gpu_logger: gpu_logger.log_event('generating tours of train model')
        tours_train, sumLogProbOfActions = seq2seq_generate_tour(device,model_train,inputs,lastlayer=lastlayer,deterministic=False,use_inf_params=use_inf_params)
        if gpu_logger: gpu_logger.log_event('generating tours of baseline model')
        tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,lastlayer=lastlayer,deterministic=not non_det,use_inf_params=use_inf_params)

        
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
            tours_train, _ = seq2seq_generate_tour(device,model_train,inputs,lastlayer=lastlayer,deterministic=False,use_inf_params=use_inf_params)
            tours_baseline, _ = seq2seq_generate_tour(device,model_baseline,inputs,lastlayer=lastlayer,deterministic=not non_det,use_inf_params=use_inf_params)
        
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
    
    ##SS
    elif action == 'next_city2':
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
        self.key = None
        self.x_flipped=False
        if reverse_start and not reverse:
            self.x_flipped=True
        elif reverse_start and reverse:
            if nb_layers%2!=0:
                self.x_flipped=True
        elif reverse and not reverse_start:
            if nb_layers%2==0:
                self.x_flipped=True
            
    def forward(self,x,city_count,inference_params=None):
        if inference_params == None or x.shape[1] == city_count+1:
            if self.x_flipped:
                x = torch.flip(x,[1])
            key = self.W1(x[:,:city_count,:])#(bsz,city_count,d_model)
            if inference_params is not None:
                self.key = key
            query = self.W2(x[:,-1,:].unsqueeze(1))#(bsz,1,d_model)
            energy = self.V(torch.tanh(key + query)).squeeze(-1)
            return energy.unsqueeze(1) #returns a tensor of size (bsz,1,city_count)
        
        elif inference_params is not None:
            key = self.key
            query = self.W2(x[:,-1,:].unsqueeze(1))
            energy = self.V(torch.tanh(key + query)).squeeze(-1)
            return energy.unsqueeze(1)

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