from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.modules.mlp import GatedMLP
import torch
from torch.distributions.categorical import Categorical
from functools import partial
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

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

def compute_tour_length(x, tour,remove_start_token=True): 
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

    return L

def plot_tsp(x_coord, x_path, plot_concorde=False, plot_dist_pair=False):
    """
    Helper function to plot TSP tours.
    """

    # pytorch detach
    x_coord = x_coord.detach().cpu()
    x_path = x_path.detach().cpu()
    
    # compute TSP lengths
    length_tsp = compute_tour_length(x_coord, x_path)
    x_coord = x_coord[:,:-1,:]

    # preparation  
    x_coord = np.array(x_coord)
    x_path = np.array(x_path)
    nb_nodes = x_coord.shape[1]
    G = nx.from_numpy_array(np.zeros((nb_nodes,nb_nodes)))
    colors = ['g'] + ['b'] * (nb_nodes - 1)  # Green for 0th node, blue for others
    batch_size = x_coord.shape[0]
    max_nb_plots = 3**2 # max number of TSP plots, x^2 for x rows and x cols 
    nb_plots = batch_size if batch_size<max_nb_plots else max_nb_plots 
    nb_rows = nb_cols = int(nb_plots**0.5)
    if plot_concorde: nb_cols *= 2 # double nb of cols if concorde is plotted 
    f = plt.figure(figsize=(10, 5)) if plot_concorde else plt.figure(figsize=(15, 15)) # figure size  
    
    # gap
    running_time = 0
    gap = 0
    L_concorde = []
    
    # loop over TSPs
    for i in range(nb_plots):
        x_coord_i = x_coord[i]
        pos_i = dict(zip(range(len(x_coord_i)), x_coord_i.tolist()))
        if plot_dist_pair: # Compute pairwise distances matrix for better visualization
            dist_pair_i = squareform(pdist(x_coord_i, metric='euclidean')) 
            G = nx.from_numpy_array(dist_pair_i)
        x_path_i = x_path[i] 
        length_tsp_i = length_tsp[i]
        nodes_pair_tsp_i = []
        for r in range(nb_nodes-1): # compute consecutive nodes in the solution
            nodes_pair_tsp_i.append((x_path_i[r], x_path_i[r+1]))
        nodes_pair_tsp_i.append((x_path_i[nb_nodes-1], x_path_i[0]))
        if plot_concorde: # run concorde solver
            start = time.time()
            graph =  pd.DataFrame({'lat' : x_coord_i[:,0]}); graph['lon'] =  x_coord_i[:,1]
            solver = TSPSolver.from_data( graph.lat, graph.lon, norm="GEO" )  
            solution = solver.solve().tour
            running_time += time.time()-start
            nodes_pair_concorde_i = []
            for r in range(nb_nodes-1):
                nodes_pair_concorde_i.append((solution[r], solution[r+1]))
            nodes_pair_concorde_i.append((solution[nb_nodes-1], solution[0]))
            length_concorde = compute_tour_length(torch.tensor(x_coord_i).unsqueeze(0),torch.tensor(solution).long().unsqueeze(0))
            gap += length_tsp_i/length_concorde - 1.0
            L_concorde.append(length_concorde)
        if plot_concorde:
            subf = f.add_subplot(nb_rows,nb_cols,2*i+1)
            nx.draw_networkx_nodes(G, pos_i, node_color=colors, node_size=20)
            nx.draw_networkx_edges(G, pos_i, edgelist=nodes_pair_tsp_i, alpha=1, width=1, edge_color='r')
            if plot_dist_pair:
                nx.draw_networkx_edges(G, pos_i, alpha=0.3, width=0.5)
            subf.set_title('Length w/ NNetwork : ' + str(length_tsp_i.item())[:5])
            subf = f.add_subplot(nb_rows,nb_cols,2*i+2)
            nx.draw_networkx_nodes(G, pos_i, node_color=colors, node_size=20)
            nx.draw_networkx_edges(G, pos_i, edgelist=nodes_pair_concorde_i, alpha=1, width=1, edge_color='b') #, style='dashed'
            if plot_dist_pair:
                nx.draw_networkx_edges(G, pos_i, alpha=0.3, width=0.5)
            subf.set_title('Length w/ Concorde : ' + str(length_concorde.item())[:5])
        else:
            subf = f.add_subplot(nb_rows,nb_cols,i+1)
            nx.draw_networkx_nodes(G, pos_i, node_color=colors, node_size=20)
            nx.draw_networkx_edges(G, pos_i, edgelist=nodes_pair_tsp_i, alpha=1, width=1, edge_color='r')
            if plot_dist_pair:
                nx.draw_networkx_edges(G, pos_i, alpha=0.3, width=0.5)
            subf.set_title('Length w/ NNetwork : ' + str(length_tsp_i.item())[:5])
       

    
    # gap
    if plot_concorde:
        L_concorde = torch.stack(L_concorde).squeeze()
        print('L_concorde',L_concorde)
        print('Concorde time: {:.3f}sec'.format(running_time))  
        print('gap:',(gap/nb_plots).item())

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
    def __init__(self,d_model,city_count,nb_layers,coord_dim=2,mlp_cls='identity',norm_f=nn.LayerNorm,B=None):
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
        self.norm_f = norm_f(d_model)

        self.embedding = input_mapping(B,d_model,coord_dim=coord_dim)
        self.layers = nn.ModuleList([
                    Block(dim= d_model,
                        mixer_cls= partial(Mamba,d_model),
                        mlp_cls= self.mlp_cls,
                        fused_add_norm=True,
                        residual_in_fp32=True,
                        )   for _ in range(nb_layers)])
        
        self.output_head = nn.Linear(d_model,city_count, bias=False)

    def forward(self,x):
        x = self.embedding(x)

        residual=None
        for layer in self.layers:
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
        logits = self.output_head(x)
        #mask vistited cities
        return logits

def seq2seq_generate_tour(device,model,inputs,deterministic=False):
    # Mask is used to prevent the model from choosing the same city twice
    bsz = inputs.shape[0]
    city_count = inputs.shape[1] - 1
    mask = torch.ones(bsz, city_count).to(device)
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    tours = []
    # list that will contain Float tensors of shape (bsz,) that gives the log probs of the choices made at time t
    sumLogProbOfActions = []
    #Construct tour recursively
    for i in range(city_count):
        #print(i)
        outputs = model(inputs)[:,-1,:]
        #print(outputs[0])
        outputs = outputs.masked_fill_(mask == 0, -float('inf'))
        #print(outputs[0])
        outputs = nn.Softmax(dim=1)(outputs)
        #print(outputs[0])
        if deterministic:
            next_city = torch.argmax(outputs, dim=1)
        else:
            next_city = Categorical(outputs).sample()
        #print(next_city[0])
        tours.append(next_city)
        sumLogProbOfActions.append(torch.log(outputs[torch.arange(bsz), next_city]) )
        mask[torch.arange(bsz), next_city] = 0
        inputs = torch.cat((inputs, inputs[torch.arange(bsz), next_city, :].unsqueeze(1)), dim=1)
    tours = torch.stack(tours, dim=1).to(device)
    sumLogProbOfActions = torch.stack(sumLogProbOfActions, dim=1).sum(dim=1).to(device)
    return tours, sumLogProbOfActions