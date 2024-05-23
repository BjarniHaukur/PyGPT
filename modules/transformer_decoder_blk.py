import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .attention import PyMultiheadAttention


class AddNorm(nn.Module):
    def __init__(self, num_hiddens, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(num_hiddens)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_hiddens, num_hiddens):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, num_hiddens)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
class PyTransformerDecoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
        super(PyTransformerDecoderBlock, self).__init__()
        self.attention = PyMultiheadAttention(num_hiddens, num_heads, dropout,batch_first=True)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        
    def forward(self, x):
        y = self.addnorm1(x, self.attention(x, x, x))
        return self.addnorm2(y, self.ffn(y))
    
    # Testing the implementation
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    num_hiddens = 32
    ffn_num_hiddens = 64
    num_heads = 2
    dropout = 0
    num_layers = 1

    # Dummy input and states
    X = torch.rand(batch_size, seq_len, num_hiddens)

    enc_outputs = torch.zeros(batch_size, seq_len, num_hiddens)
    # enc_valid_lens = torch.arange(1, seq_len + 1, device=X.device).repeat(batch_size, 1)
    # enc_mask = (enc_valid_lens.unsqueeze(1) >= torch.arange(seq_len, device=enc_valid_lens.device)).bool()
    
    # Initialize custom PyTransformerDecoderBlock
    custom_decoder_block = PyTransformerDecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout)
    custom_output = custom_decoder_block(X)

    # Initialize PyTorch's TransformerDecoderLayer
    pytorch_decoder_layer = nn.TransformerDecoderLayer(d_model=num_hiddens, nhead=num_heads, dim_feedforward=ffn_num_hiddens, dropout=dropout,batch_first=True)
    
    # Copy the weights from the custom model's MultiHead layer to the PyTorch model
    in_proj_weight = torch.stack([custom_decoder_block.attention.W_q.weight, custom_decoder_block.attention.W_k.weight, custom_decoder_block.attention.W_v.weight])
    # in_proj_bias = torch.stack([custom_decoder_block.attention.W_q.bias, custom_decoder_block.attention.W_k.bias, custom_decoder_block.attention.W_v.bias])
    pytorch_decoder_layer.self_attn.in_proj_weight = nn.Parameter(in_proj_weight)
    # pytorch_decoder_layer.self_attn.in_proj_bias = nn.Parameter(in_proj_bias)
    pytorch_decoder_layer.self_attn.out_proj.weight = nn.Parameter(custom_decoder_block.attention.W_o.weight)
    # pytorch_decoder_layer.self_attn.out_proj.bias = nn.Parameter(custom_decoder_block.attention.W_o.bias)

    # Copy the weights from the custom model's AddNorm layer to the PyTorch model
    pytorch_decoder_layer.norm1.weight = nn.Parameter(custom_decoder_block.addnorm1.ln.weight)
    pytorch_decoder_layer.norm1.bias = nn.Parameter(custom_decoder_block.addnorm1.ln.bias)

    pytorch_decoder_layer.norm3.weight = nn.Parameter(custom_decoder_block.addnorm2.ln.weight)
    pytorch_decoder_layer.norm3.bias = nn.Parameter(custom_decoder_block.addnorm2.ln.bias)

    # Copy the weights from the custom model's PositionWiseFFN layer to the PyTorch model
    pytorch_decoder_layer.linear1.weight = nn.Parameter(custom_decoder_block.ffn.dense1.weight)
    pytorch_decoder_layer.linear1.bias = nn.Parameter(custom_decoder_block.ffn.dense1.bias)
    pytorch_decoder_layer.linear2.weight = nn.Parameter(custom_decoder_block.ffn.dense2.weight)
    pytorch_decoder_layer.linear2.bias = nn.Parameter(custom_decoder_block.ffn.dense2.bias)
    
    batch_size, num_steps, _ = X.shape
    tgt_msk = ~torch.tril(torch.ones((num_steps, num_steps), device=X.device)).bool()
    pytorch_output = pytorch_decoder_layer(X, enc_outputs, tgt_mask=tgt_msk, memory_mask=None, tgt_is_causal=False)


    # Check individual blocks
    my_Y = custom_decoder_block.addnorm1(X, custom_decoder_block.attention(X,X,X,tgt_msk))
    torch_Y = pytorch_decoder_layer.norm1(X+pytorch_decoder_layer._sa_block(X, tgt_msk, None, False))
    np.testing.assert_allclose(my_Y.detach().numpy(), torch_Y.detach().numpy())
    my_out = pytorch_decoder_layer.norm3(torch_Y + pytorch_decoder_layer._ff_block(torch_Y))
    torch_out = custom_decoder_block.addnorm2(my_Y, custom_decoder_block.ffn(my_Y))
    np.testing.assert_allclose(my_out.detach().numpy(), torch_out.detach().numpy())

    # Compare the outputs
    np.testing.assert_allclose(custom_output.detach().numpy(), pytorch_output.detach().numpy(), rtol=1e-4)
    print("The custom PyTransformerDecoderBlock output matches with PyTorch's TransformerDecoderLayer output.")

