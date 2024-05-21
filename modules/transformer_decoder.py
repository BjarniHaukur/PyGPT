import torch
import torch.nn as nn
import torch.nn.functional as F


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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout)
    
    def forward(self, query, key, value, valid_lens):
        attn_output, _ = self.attention(query, key, value, attn_mask=self._get_attention_mask(valid_lens, query.device))
        return attn_output

    def _get_attention_mask(self, valid_lens, device):
        if valid_lens is None:
            return None
        max_len = valid_lens.max().item()
        mask = (torch.arange(max_len, device=device).expand(len(valid_lens), max_len) >= valid_lens.unsqueeze(1))
        return mask.unsqueeze(1).unsqueeze(2)  # For multi-head attention, need to add extra dimensions

class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super(TransformerDecoderBlock, self).__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens) # can get rid of this if we want , but might be useful for "starting code"
        Z = self.addnorm2(Y, Y2)
        
        return self.addnorm3(Z, self.ffn(Z)), state
