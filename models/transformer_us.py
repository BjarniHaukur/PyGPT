from .base import PyGenerator

from modules import transformer_decoder, embedding
from utils.tokenizer import BOS_ID, EOS_ID
from utils.sample import nucleus_sample, sample_with_temp

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PyTransformer(PyGenerator):
    def __init__(self,
            vocab_size:int,
            d_model:int,
            d_feedforward:int,
            num_attn_heads:int,
            num_decoder_layers:int,
            context_window_size:int,
            dropout:float = 0.1,
            **kwargs
        ):
        super(PyTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.num_attn_heads = num_attn_heads
        self.num_decoder_layers = num_decoder_layers
        self.context_window_size = context_window_size
        self.dropout = dropout

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = embedding.PositionalEncoding(d_model, dropout=dropout)
        # self.decoder = transformer_decoder.TransformerDecoder(vocab_size, d_model, d_feedforward, num_attn_heads, num_decoder_layers, dropout)
        self.decoder = nn.TransformerDecoder()

    def forward(self, x, hc=None):
        x = self.pos_embed(self.embed(x) * math.sqrt(self.d_model))
        x = self.decoder(x)
        return x, hc

    @torch.no_grad()
    def generate(
            self,
            batch_size:int,
            max_len:int=1000,
            temperature:float=None,
            nucleus_threshold:float=None,
            starting_tokens:torch.Tensor=None
        ) -> list[list[int]]:
        """Generates a batch of tokens, returning a list of potentially variable length sequences. If both nucleus_threshold and temperature are supplied, sampling with temperature is used"""
        self.eval()
        device = next(self.parameters()).device

        # Prepare the batch of starting tokens
        xt = torch.tensor([BOS_ID] * batch_size, device=device).unsqueeze(1)
        hc = None
        if starting_tokens is not None:
            xt = torch.cat([xt, starting_tokens], dim=1)

        tokens = [] if starting_tokens is None else starting_tokens.T.tolist()
        for _ in range(max_len - starting_tokens.shape[1] if starting_tokens is not None else max_len):
            xt, hc = self.forward(xt, hc)
            if temperature:
                xt = sample_with_temp(xt[:,-1,:], temperature=temperature)
            else:
                xt = nucleus_sample(xt[:, -1, :], nucleus_threshold)
            tokens.append(xt.tolist())
            xt = xt.unsqueeze(1)

        self.train()
        return torch.tensor(tokens).T.tolist()
    