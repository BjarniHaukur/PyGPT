from .base import PyGenerator
from modules import embedding, dgru
from utils.sample import nucleus_sample, sample_with_temp
from utils.tokenizer import BOS_ID, EOS_ID, PAD_ID

import torch
import torch.nn as nn

class PyGRU(PyGenerator):
    def __init__(self, vocab_size:int, hidden_size:int, num_layers:int, **kwargs):
        super(PyGRU, self).__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        
        # self.embed = embedding.PyEmbedding(vocab_size, hidden_size)
        # self.rnn = dgru.PyGRU(hidden_size, hidden_size, num_layers)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.rnn(x, h) # i.e. 100% teacher forcing
        x = self.linear(x)
        return x, h
    
    @torch.no_grad()
    def generate(
            self,
            batch_size:int,
            max_len:int=1000,
            temperature:float=1.0,
            nucleus_threshold:float=None,
            starting_tokens:torch.Tensor=None
        ) -> list[list[int]]:
        """Generates a batch of tokens, returning a list of potentially variable length sequences. If both nucleus_threshold and temperature are supplied, nucleus is used"""
        self.eval()
        device = next(self.parameters()).device
        
        # Prepare the batch of starting tokens
        xt = torch.tensor([BOS_ID] * batch_size, device=device).unsqueeze(1)
        ht = None
        if starting_tokens is not None:
            xt = torch.cat([xt, starting_tokens], dim=1)
    
        tokens = [] if starting_tokens is None else starting_tokens.T.tolist()
        for _ in range(max_len - starting_tokens.shape[1] if starting_tokens is not None else max_len):
            xt, ht = self.forward(xt, ht)
            if nucleus_threshold:
                xt = nucleus_sample(xt[:, -1, :], nucleus_threshold)
            else:
                xt = sample_with_temp(xt[:,-1,:], temperature=temperature)
            tokens.append(xt.tolist())
            xt = xt.unsqueeze(1)
        
        self.train()
        return torch.tensor(tokens).T.tolist()






