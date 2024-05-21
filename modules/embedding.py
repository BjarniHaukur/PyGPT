import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, input_dim:int, embed_dim:int):
        super(Embedding, self).__init__()
        self.input_dim, self.embed_dim = input_dim, embed_dim
        
        self.weight = nn.Parameter(torch.randn(input_dim, embed_dim))
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.weight[x]