from .base import PyGenerator

from modules import embedding, lstm
from utils.sample import nucleus_sample
from utils.tokenizer import BOS_ID, EOS_ID

import torch
import torch.nn as nn

class PyLSTM(PyGenerator):
    def __init__(self, vocab_size:int, hidden_size:int, num_layers:int, **kwargs):
        super(PyLSTM, self).__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        
        self.embed = embedding.PyEmbedding(vocab_size, hidden_size)
        self.rnn = lstm.PyLSTM(hidden_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h=None, c=None):
        x = self.embed(x)
        x, (h, c) = self.rnn(x, h, c) # i.e. 100% teacher forcing
        x = self.linear(x)
        return x, (h, c)
    
    @torch.no_grad()
    def generate(
            self,
            batch_size:int,
            max_len:int=1000,
            nucleus_threshold:float=0.9,
            starting_tokens:torch.Tensor=None
        ) -> list[list[int]]:
        """ Generates a batch of tokens, returning a list of potentially variable length sequences """
        self.eval()
        device = next(self.parameters()).device
        
        # Prepare the batch of starting tokens
        xt = torch.tensor([BOS_ID] * batch_size, device=device).unsqueeze(1)
        if starting_tokens is not None:
            xt = torch.cat([xt, starting_tokens], dim=1)
    
        ht = torch.randn(1, batch_size, self.hidden_size, device=device)
        ct = torch.randn(1, batch_size, self.hidden_size, device=device)

        tokens = []
        for _ in range(max_len):
            xt, (ht, ct) = self.forward(xt, ht, ct)
            xt = nucleus_sample(xt[:, -1, :], nucleus_threshold)
            
            tokens.append(xt.tolist())
            xt = xt.unsqueeze(1)
        
        self.train()
        tokens = torch.tensor(tokens).T.tolist() # (max_len, batch_size) -> (batch_size, max_len)
        return [seq[:seq.index(EOS_ID) + 1] if EOS_ID in seq else seq for seq in tokens] # truncate at EOS token
