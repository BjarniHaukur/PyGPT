import torch
import torch.nn as nn
from embedding import PyEmbedding
from transformer_decoder_blk import TransformerDecoderBlock
import math


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = PyEmbedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout=dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self):
        return [None] * len(self.blks)

    def forward(self, X, state):
        # X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[i] = blk.attention1.attention.attention.attention_weights
            
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

# Testing the implementation
if __name__ == "__main__":
    import numpy as np

    vocab_size = 10
    batch_size = 4
    seq_len = 5
    num_hiddens = 32
    ffn_num_hiddens = 64
    num_heads = 2
    dropout = 0
    num_layers = 1

    # Dummy input and states
    X = torch.randint(0, vocab_size, (batch_size, seq_len))

    enc_outputs = torch.zeros(batch_size, seq_len, num_hiddens)
    # enc_valid_lens = torch.arange(1, seq_len + 1, device=X.device).repeat(batch_size, 1)
    # enc_mask = (enc_valid_lens.unsqueeze(1) >= torch.arange(seq_len, device=enc_valid_lens.device)).bool()
    state = [None] * num_layers

    # Initialize custom TransformerDecoderBlock
    custom_decoder = TransformerDecoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)

    # Initialize PyTorch's TransformerDecoderLayer
    pytorch_decoder_layer = nn.TransformerDecoderLayer(d_model=num_hiddens, nhead=num_heads, dim_feedforward=ffn_num_hiddens, dropout=dropout,batch_first=True)
    pytorch_decoder = nn.TransformerDecoder(pytorch_decoder_layer, num_layers=num_layers, norm=nn.Linear(num_hiddens, vocab_size))

    for i in range(num_layers):
        
        # Copy the weights from the custom model's MultiHead layer to the PyTorch model
        custom_decoder.blks[i].attention1.attention.W_q.weight = nn.Parameter(pytorch_decoder.layers[i].self_attn.in_proj_weight[:num_hiddens])
        # custom_decoder.blks[i].attention1.attention.W_q.bias = nn.Parameter(pytorch_decoder.layers[i].self_attn.in_proj_bias[:num_hiddens])
        custom_decoder.blks[i].attention1.attention.W_k.weight = nn.Parameter(pytorch_decoder.layers[i].self_attn.in_proj_weight[num_hiddens:2*num_hiddens])
        # custom_decoder.blks[i].attention1.attention.W_k.bias = nn.Parameter(pytorch_decoder.layers[i].self_attn.in_proj_bias[num_hiddens:2*num_hiddens])
        custom_decoder.blks[i].attention1.attention.W_v.weight = nn.Parameter(pytorch_decoder.layers[i].self_attn.in_proj_weight[2*num_hiddens:3*num_hiddens])
        # custom_decoder.blks[i].attention1.attention.W_v.bias = nn.Parameter(pytorch_decoder.layers[i].self_attn.in_proj_bias[2*num_hiddens:])
        custom_decoder.blks[i].attention1.attention.W_o.weight = nn.Parameter(pytorch_decoder.layers[i].self_attn.out_proj.weight)
        # custom_decoder.blks[i].attention1.attention.W_o.bias = nn.Parameter(pytorch_decoder.layers[i].self_attn.out_proj.bias)

        # Copy the weights from the custom model's AddNorm layer to the PyTorch model
        pytorch_decoder.layers[i].norm1.weight = nn.Parameter(custom_decoder.blks[i].addnorm1.ln.weight)
        pytorch_decoder.layers[i].norm1.bias = nn.Parameter(custom_decoder.blks[i].addnorm1.ln.bias)

        pytorch_decoder.layers[i].norm3.weight = nn.Parameter(custom_decoder.blks[i].addnorm2.ln.weight)
        pytorch_decoder.layers[i].norm3.bias = nn.Parameter(custom_decoder.blks[i].addnorm2.ln.bias)

        # Copy the weights from the custom model's PositionWiseFFN layer to the PyTorch model
        pytorch_decoder.layers[i].linear1.weight = nn.Parameter(custom_decoder.blks[i].ffn.dense1.weight)
        pytorch_decoder.layers[i].linear1.bias = nn.Parameter(custom_decoder.blks[i].ffn.dense1.bias)
        pytorch_decoder.layers[i].linear2.weight = nn.Parameter(custom_decoder.blks[i].ffn.dense2.weight)
        pytorch_decoder.layers[i].linear2.bias = nn.Parameter(custom_decoder.blks[i].ffn.dense2.bias)

    # Copy the weights from the custom model's final Dense layer to the PyTorch model
    pytorch_decoder.norm.weight = nn.Parameter(custom_decoder.dense.weight)
    pytorch_decoder.norm.bias = nn.Parameter(custom_decoder.dense.bias)

    tgt_msk = ~torch.tril(torch.ones((seq_len, seq_len), device=X.device)).bool()
        
    pytorch_output = pytorch_decoder(custom_decoder.embedding(X).clone(), enc_outputs, tgt_mask=tgt_msk)
    my_out = custom_decoder(custom_decoder.embedding(X).clone(), custom_decoder.init_state())

    assert my_out[0].shape == pytorch_output.shape
    assert torch.allclose(my_out[0], pytorch_output, atol=1e-4)

    # Check individual blocks
    # my_Y = custom_decoder_block.addnorm1(X, custom_decoder_block.attention1(X,X,X,tgt_msk))
    # torch_Y = pytorch_decoder_layer.norm1(X+pytorch_decoder_layer._sa_block(X, tgt_msk, None, False))
    # np.testing.assert_allclose(my_Y.detach().numpy(), torch_Y.detach().numpy())
    # my_out = pytorch_decoder_layer.norm3(torch_Y + pytorch_decoder_layer._ff_block(torch_Y))
    # torch_out = custom_decoder_block.addnorm2(my_Y, custom_decoder_block.ffn(my_Y))
    # np.testing.assert_allclose(my_out.detach().numpy(), torch_out.detach().numpy())

    # # Compare the outputs
    # np.testing.assert_allclose(custom_output.detach().numpy(), pytorch_output.detach().numpy(), rtol=1e-4)
    # print("The custom TransformerDecoderBlock output matches with PyTorch's TransformerDecoderLayer output.")

