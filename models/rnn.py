import torch
import torch.nn as nn
from modules import rnn, embedding

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
    

    # @torch.no_grad()
    # def generate(model, max_len=1000, starting_tokens:list[int]=None)->str:
    #     model.eval()
    #     device = next(model.parameters()).device
    #     xt = torch.tensor([[BOS_ID] + (starting_tokens or [])], device=device)
    #     ht = torch.randn(1, 1, model.hidden_size, device=device)
        
    #     tokens = starting_tokens or []
    #     for _ in range(max_len):
    #         xt = model.embed(xt)
    #         xt, ht = model.rnn(xt, ht)
    #         xt = model.linear(xt)
    #         xt = nucleus_sample(xt[:,-1,:], nucleus_threshold=0.9)
    #         # xt = sample_with_temp(xt[:,-1,:], temperature=1.0)
    #         token = xt.item()
    #         if token == EOS_ID:
    #             break
    #         tokens.append(token)

    #     model.train()
    #     return ds.tokenizer.detokenize(tokens)
