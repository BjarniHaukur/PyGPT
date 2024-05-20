import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTransformer(nn.Module):
    def __init__(self,
            vocab_size:int = 376,
            d_model:int = 512,
            d_feedforward:int = 512,
            num_attn_heads:int = 8,
            num_decoder_layers:int = 4,
            context_window_size:int = 512,
            dropout:float = 0.1,
            **kwargs
        ):
        super(PyTransformer, self).__init__()
        ...