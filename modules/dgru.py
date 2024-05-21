import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from gru import PyGRUCell

class PyGRU(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int):
        super(PyGRU,self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
        self.grus = nn.Sequential(*[PyGRUCell(
            input_dim if i==0 else hidden_dim, hidden_dim)
                                    for i in range(num_layers)])
        
    def forward(self, x:Tensor, h0:Tensor=None)->tuple[Tensor, Tensor]:
        outputs = x
        if h0 is None: h0 = [None] * self.num_layers
        for i in range(self.num_layers):
            outputs, h0[i] = self.grus[i](outputs, h0[i])
        return outputs, torch.stack(h0, 0)   # outputs will be like (batch, length, hidden dims), hidden state will be like (layers, batch, hidden dims)
    
if __name__ == "__main__":
    import numpy as np
    input_dim = 10
    hidden_dim = 20
    n_layers = 3
    def copy_weights_to_torch_gru(deep_gru, torch_gru):
        for i in range(deep_gru.num_layers):
            # Copying weights for the input-hidden connections
            getattr(torch_gru, 'weight_ih_l' + str(i)).data.copy_(torch.cat([
                getattr(deep_gru.grus[i], 'W_xr').T.data,
                getattr(deep_gru.grus[i], 'W_xz').T.data,
                getattr(deep_gru.grus[i], 'W_xh').T.data
            ], dim=0))
            
            # Copying weights for the hidden-hidden connections
            getattr(torch_gru, 'weight_hh_l' + str(i)).data.copy_(torch.cat([
                getattr(deep_gru.grus[i], 'W_hr').T.data,
                getattr(deep_gru.grus[i], 'W_hz').T.data,
                getattr(deep_gru.grus[i], 'W_hh').T.data
            ], dim=0))
            
            # Copying biases
            getattr(torch_gru, 'bias_ih_l' + str(i)).data.copy_(torch.cat([
                getattr(deep_gru.grus[i], 'b_r').data,
                getattr(deep_gru.grus[i], 'b_z').data,
                getattr(deep_gru.grus[i], 'b_h').data
            ],dim=0))
                
                # No hidden-hidden bias in PyTorch's GRU, so just copy the biases for the input-hidden connections
            getattr(torch_gru, 'bias_hh_l' + str(i)).data.copy_(torch.zeros_like(torch.cat([
                getattr(deep_gru.grus[i], 'b_r').data,
                getattr(deep_gru.grus[i], 'b_z').data,
                getattr(deep_gru.grus[i], 'b_h').data
            ])))

    deep_gru = PyGRU(input_dim=input_dim, hidden_dim=hidden_dim,num_layers=n_layers)
    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    output, H = deep_gru(x)
    print("Output shape:", output.shape)  # Expected: (5, 3, 20)
    print("Last Layer Hidden state shape:", H[n_layers-1].shape)  # Expected: (5, 20)

    torch_gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
    copy_weights_to_torch_gru(deep_gru, torch_gru)

    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    out_py_gru, h_py_gru = deep_gru(x)
    out_torch_gru, h_torch_gru = torch_gru(x)

    np.testing.assert_allclose(out_py_gru.detach().numpy(), out_torch_gru.detach().numpy(), rtol=1e-4,atol=1e-5)
    np.testing.assert_allclose(h_py_gru.detach().numpy(), h_torch_gru.detach().numpy(), rtol=1e-4,atol=1e-5)