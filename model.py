from mamba_ssm.modules.mamba_simple import Mamba
import torch
import torch.nn as nn

class MambaFull(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.d_model=args['d_model']
        self.vocab_size=args['vocab_size']

        self.embedding = nn.Embedding(args['vocab_size'], args['d_model'])
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args['n_layer'])])
        self.final_norm = RMSNorm(args['d_model'])
        self.output_head = nn.Linear(args['d_model'], args['vocab_size'], bias=False)

    def forward(self,x):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mixer = Mamba(args['d_model'])
        self.norm = RMSNorm(args['d_model'])
        

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x

        return output

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output