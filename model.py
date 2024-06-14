from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
import torch
from functools import partial
import torch.nn as nn

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

def generate_data(device,batch_size,city_count,coord_dim=2):
    return torch.rand(batch_size,city_count+1,2).to(device)