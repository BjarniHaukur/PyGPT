import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm(nn.Module):
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1):
        super(BatchNorm, self).__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        x = (x-mean) / (var+self.eps).sqrt()
        return x*self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1):
        super(LayerNorm, self).__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.training:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        x = (x-mean) / (var+self.eps).sqrt()
        return x*self.weight + self.bias
    