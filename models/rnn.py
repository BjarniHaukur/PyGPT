import torch
import torch.nn as nn
from modules import rnn, embedding

from utils.sample import nucleus_sample
from utils.tokenizer import BOS_ID, EOS_ID, PAD_ID


class PyRNN(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, num_layers:int, **kwargs):
        super(PyRNN, self).__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        
        self.embed = embedding.PyEmbedding(vocab_size, hidden_size)
        self.rnn = rnn.PyRNN(hidden_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.rnn(x) # i.e. 100% teacher forcing
        x = self.linear(x)
        return x, h
    
    @torch.no_grad()
    def generate(
            self,
            batch_size:int,
            max_len:int=1000,
            nucleus_threshold:float = None,
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

        tokens = []
        for _ in range(max_len):
            xt = self.embed(xt)
            xt, ht = self.rnn(xt, ht)
            xt = self.linear(xt)
            xt = nucleus_sample(xt[:, -1, :], nucleus_threshold)
            
            tokens.append(xt.tolist())
            xt = xt.unsqueeze(1)
        
        self.train()
        return [x[:x.index(EOS_ID) + 1] for x in tokens]


