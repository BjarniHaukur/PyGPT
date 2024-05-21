import torch
import torch.nn as nn

from utils.sample import nucleus_sample_with_temp
from utils.tokenizer import BOS_ID, EOS_ID

class PyLSTM(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, n_layers:int, **kwargs):
        super(PyLSTM, self).__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.rnn(x) # i.e. 100% teacher forcing
        x = self.linear(x)
        return x, h
    
    @torch.no_grad()
    def generate(self, max_len=1000, starting_tokens:list[int]=None,  nucleus_threshold:float=None, temperature:float=None)->str:
        """ Generates a sequence of tokens, added to starting_tokens """
        self.eval()
        device = next(self.parameters()).device
        xt = torch.tensor([[BOS_ID] + (starting_tokens or [])], device=device)
        ht = torch.randn(1, 1, self.hidden_size, device=device)
        ct = torch.randn(1, 1, self.hidden_size, device=device)

        sample_kwargs = {}
        if nucleus_threshold is not None:
            sample_kwargs['nucleus_threshold'] = nucleus_threshold
        if temperature is not None:
            sample_kwargs['temperature'] = temperature
        
        tokens = starting_tokens or []
        for _ in range(max_len):
            xt = self.embed(xt)
            xt, (ht, ct) = self.rnn(xt, (ht, ct))
            xt = self.linear(xt)
            xt = nucleus_sample_with_temp(xt[:,-1,:], **sample_kwargs)
            token = xt.item()
            if token == EOS_ID:
                break
            tokens.append(token)

        self.train()

        return tokens
