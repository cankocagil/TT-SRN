import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from siren.init import siren_uniform_

def sine_init(x):
    siren_uniform_(x, mode='fan_in', c=6)


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SinLayerClass(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dropout = 0.2):
        super().__init__()
        internal_state_dim = int(hidden_dim//2)
        

        self.net = nn.Sequential(
            Siren(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, internal_state_dim),
            nn.GELU(),
            nn.Linear(internal_state_dim, num_heads)
        )
    def forward(self, x):
        return self.net(x)




class SinLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dropout = 0.2):
        super().__init__()
        internal_state_dim = int(hidden_dim//2)
        internal_state_dim2 = int(internal_state_dim//2)

        self.net = nn.Sequential(
            Siren(dim, hidden_dim),
            nn.Dropout(dropout),
            Siren(hidden_dim, internal_state_dim),
            nn.Dropout(dropout),
            nn.Linear(internal_state_dim, internal_state_dim2),
            nn.GELU(),
            nn.Linear(internal_state_dim2, num_heads)
        )
    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dropout = 0.):
        super().__init__()
        internal_state_dim = int(hidden_dim//2)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, internal_state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(internal_state_dim, num_heads),
        )
    def forward(self, x):
        return self.net(x)   