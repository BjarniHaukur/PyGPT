import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

   
class AdditiveAttention(nn.Module):  
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)  

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):  
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    def forward(self, queries, keys, values, attn_mask=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -float('inf'))
        self.attention_weights =nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class PyMultiheadAttention(nn.Module): 
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        # for the purpose of paralleslising the computation of multiple heads 
        # Shape of input X: (batch_size, no. of queries or key-value pairs,num_hiddens).

        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)# Shape of output X: (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)

        X = X.permute(0, 2, 1, 3)# Shape of output X: (batch_size, num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)

        return X.reshape(-1, X.shape[2], X.shape[3])# Shape of output: (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)

    def transpose_output(self, X):
        # to reverse the transpose_qkv operation 
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
      
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,num_hiddens / num_heads)
        B = queries.shape[0]
        L = queries.shape[1] #  no. of queries
        S = keys.shape[1]  # no. of key-value pairs
        num_hiddens = queries.shape[2]
        

        queries = self.W_q(queries).view(B, L, self.num_heads, num_hiddens // self.num_heads).transpose(1, 2) # (B, num_heads,no. of queries, num_hiddens// num_heads)
        keys = self.W_k(keys).view(B, S, self.num_heads, num_hiddens // self.num_heads).transpose(1, 2) # (B, num_heads,no. of key-value pairs, num_hiddens// num_heads)
        values = self.W_v(values).view(B, S, self.num_heads, num_hiddens // self.num_heads).transpose(1, 2) # (B, num_heads,no. of key-value pairs, num_hiddens// num_heads)
        # efficient attention using Flash Attention ( ordinarily no. of queries == no. of KV pairs==T)

        # (B, num_heads, L, num_hiddens// num_heads) x (B, num_heads, num_hiddens// num_heads, S) -> (B, num_heads, L, S)
        # (B, num_heads, L, S) x (B, num_heads, S, num_hiddens// num_heads) -> (B, num_heads, L, num_hiddens// num_heads)
        y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, L, num_hiddens) # re-assemble all head outputs side by side (B, L, num_hiddens)


        return self.W_o(y)
    
    
if __name__=="__main__":
    import numpy as np
    
    B = 5
    L = S = 3
    D = 10

    q = torch.randn(B, L, D)
    k = torch.randn(B, S, D)
    v = torch.randn(B, S, D)

    dot_prod_attention = DotProductAttention(0)
    out = dot_prod_attention(q, k, v)
    torch_out = F.scaled_dot_product_attention(q, k, v)
    
    assert torch.allclose(out, torch_out, atol=1e-6)

    n_heads = 2
    mha = PyMultiheadAttention(D, n_heads, 0)
    torch_mha = nn.MultiheadAttention(D, n_heads, bias=False, batch_first=True)

    mha.W_q.weight = torch.nn.Parameter(torch_mha.in_proj_weight[:D])
    mha.W_k.weight = torch.nn.Parameter(torch_mha.in_proj_weight[D:2*D])
    mha.W_v.weight = torch.nn.Parameter(torch_mha.in_proj_weight[2*D:3*D])
    mha.W_o.weight = torch.nn.Parameter(torch_mha.out_proj.weight)

    attn_mask = ~torch.tril(torch.ones((n_heads*B, L, S))).bool()

    out = mha(q, k, v)
    torch_out = torch_mha(q, k, v, attn_mask=attn_mask)



    assert torch.allclose(out, torch_out[0], atol=1e-6)