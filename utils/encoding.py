import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    pe = SinusoidalPositionalEncoding(d_model=200, max_len=1000)
    x = torch.randn(1, 200, 200)
    plt.imshow((x - pe(x))[0].detach().numpy()) # extract only the positional encoding
    plt.title("Sinusoidal Positional Encoding")
    plt.show()
    
    