from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
import torch
from torch.distributions.categorical import Categorical
from functools import partial
import torch.nn as nn

def generate_data(device,batch_size,city_count,coord_dim=2):
    return torch.rand(batch_size,city_count+1,2).to(device)

def compute_tour_length(x, tour): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, city_count+1, 2) batch of tsp tour instances
             tour of size (bsz, city_count) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
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

class MambaFull(nn.Module):
    def __init__(self,d_model,city_count,nb_layers,coord_dim=2,mlp_cls=nn.Identity,norm_f=nn.LayerNorm):
        super().__init__()
        self.d_model=d_model
        self.city_count=city_count
        self.nb_layers=nb_layers
        self.mlp = mlp_cls(d_model)
        self.norm_f = norm_f(d_model)

        self.embedding = nn.Linear(coord_dim, d_model)
        self.layers = nn.ModuleList([
                    Block(dim= d_model,
                        mixer_cls= partial(Mamba,d_model),
                        mlp_cls= mlp_cls,
                        fused_add_norm=True,
                        residual_in_fp32=True,
                        )   for _ in range(nb_layers)])
        
        self.output_head = nn.Linear(d_model,city_count, bias=False)

    def forward(self,x):
        x = self.embedding(x)

        residual=None
        for layer in self.layers:
            x , residual = layer(x,residual)

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

def seq2seq_generate_tour(args,device,model,inputs):
    # Mask is used to prevent the model from choosing the same city twice
    mask = torch.ones(args.bsz, args.city_count).to(device)
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    tours = []
    # list that will contain Float tensors of shape (bsz,) that gives the log probs of the choices made at time t
    sumLogProbOfActions = []
    #Construct tour recursively
    for i in range(args.city_count):
        #print(i)
        outputs = model(inputs)[:,-1,:]
        #print(outputs[0])
        outputs = outputs.masked_fill_(mask == 0, -float('inf'))
        #print(outputs[0])
        outputs = nn.Softmax(dim=1)(outputs)
        #print(outputs[0])
        if args.deterministic:
            next_city = torch.argmax(outputs, dim=1)
        else:
            next_city = Categorical(outputs).sample()
        #print(next_city[0])
        tours.append(next_city)
        sumLogProbOfActions.append(torch.log(outputs[torch.arange(args.bsz), next_city]) )
        mask[torch.arange(args.bsz), next_city] = 0
        inputs = torch.cat((inputs, inputs[torch.arange(args.bsz), next_city, :].unsqueeze(1)), dim=1)
        args.sequence_length += 1
    tours = torch.stack(tours, dim=1).to(device)
    sumLogProbOfActions = torch.stack(sumLogProbOfActions, dim=1).sum(dim=1).to(device)
    return tours, sumLogProbOfActions