import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PyGRUCell(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int):
        super(PyGRUCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

        init_weight_matrix = lambda *shape: nn.Parameter(torch.empty(*shape) )
        triplicate = lambda: (init_weight_matrix(input_dim, hidden_dim),
                          init_weight_matrix(hidden_dim, hidden_dim),
                          nn.Parameter(torch.zeros(hidden_dim)))
        self.W_xz, self.W_hz, self.b_z = triplicate()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triplicate()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triplicate()  # Candidate hidden state

        self.init_params()

    def init_params(self):

        nn.init.xavier_normal_(self.W_xz)
        nn.init.xavier_normal_(self.W_hz)
        nn.init.zeros_(self.b_z)

        nn.init.xavier_normal_(self.W_xr)
        nn.init.xavier_normal_(self.W_hr)
        nn.init.zeros_(self.b_r)

        nn.init.xavier_normal_(self.W_xh)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.zeros_(self.b_h)

    def forward(self, x:Tensor, h_0:Tensor=None)->tuple[Tensor, Tensor]:
        B, L, _ = x.shape # batch, length, input dims
        
        if h_0 is None:
            h_0 = torch.zeros(B, self.hidden_dim, device=x.device) # batch, hidden dims

        h_t = h_0.clone()
        
        outputs = []
        for t in range(L):
            x_t = x[:,t]
            h_t_minus_1 = h_t.clone()
            z_t = torch.sigmoid(torch.matmul(x_t, self.W_xz) +  # update gate
                    torch.matmul(h_t, self.W_hz) + self.b_z)
            r_t = torch.sigmoid(torch.matmul(x_t, self.W_xr) + # reset gate
                    torch.matmul(h_t, self.W_hr) + self.b_r)
            h_candidate = torch.tanh(torch.matmul(x_t, self.W_xh) +
                        r_t*(torch.matmul(h_t, self.W_hh) + self.b_h))
            h_t = z_t * h_t_minus_1 + (1 - z_t) * h_candidate 
            outputs.append(h_t.clone())
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, h_t # outputs will be like (batch, length, hidden dims), hidden state will be like (batch, hidden dims)
    

if __name__ == "__main__":
    import numpy as np
    input_dim = 10
    hidden_dim = 20
    n_layers = 1
    def copy_weights_to_torch_gru(py_gru, torch_gru):
        # Copying weights for the input-hidden connections
        getattr(torch_gru, 'weight_ih_l' + str(0)).data.copy_(torch.cat([
            getattr(py_gru, 'W_xr').T.data,
            getattr(py_gru, 'W_xz').T.data,
            getattr(py_gru, 'W_xh').T.data
        ], dim=0))
        
        # Copying weights for the hidden-hidden connections
        getattr(torch_gru, 'weight_hh_l' + str(0)).data.copy_(torch.cat([
            getattr(py_gru, 'W_hr').T.data,
            getattr(py_gru, 'W_hz').T.data,
            getattr(py_gru, 'W_hh').T.data
        ], dim=0))
        
        # Copying biases
        getattr(torch_gru, 'bias_ih_l' + str(0)).data.copy_(torch.cat([
            getattr(py_gru, 'b_r').data,
            getattr(py_gru, 'b_z').data,
            getattr(py_gru, 'b_h').data
        ],dim=0))
            
        # No hidden-hidden bias in PyTorch's GRU, so just copy the biases for the input-hidden connections
        getattr(torch_gru, 'bias_hh_l' + str(0)).data.copy_(torch.zeros_like(torch.cat([
            getattr(py_gru, 'b_r').data,
            getattr(py_gru, 'b_z').data,
            getattr(py_gru, 'b_h').data
        ])))

    py_gru = PyGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
    torch_gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
    copy_weights_to_torch_gru(py_gru, torch_gru)

    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    h0 = torch.zeros(n_layers, 5, hidden_dim)  # Initial hidden state
    out_py_gru, h_py_gru = py_gru(x, h0[0].clone())
    h_py_gru = h_py_gru.unsqueeze(0)
    out_torch_gru, h_torch_gru = torch_gru(x, h0.clone())
   

    np.testing.assert_allclose(out_py_gru.detach().numpy(), out_torch_gru.detach().numpy(), rtol=1e-4)
    np.testing.assert_allclose(h_py_gru.detach().numpy(), h_torch_gru.detach().numpy(), rtol=1e-4)

    
        
        


        
