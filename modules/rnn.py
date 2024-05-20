import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# does not support bidirectionality since it can't be used to generate code
class PyRNN(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, n_layers:int):
        super(PyRNN, self).__init__()
        self.input_dim, self.hidden_dim, self.n_layers = input_dim, hidden_dim, n_layers

        self.WI = nn.ParameterList([nn.Parameter(torch.empty((hidden_dim, input_dim if i==0 else hidden_dim))) for i in range(n_layers)])
        self.BI = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])
        self.WH = nn.ParameterList([nn.Parameter(torch.empty((hidden_dim, hidden_dim))) for _ in range(n_layers)])
        self.BH = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])

        self.init_params()

    def init_params(self):
        for layer in range(self.n_layers):
            nn.init.xavier_normal_(self.WI[layer])
            nn.init.xavier_normal_(self.WH[layer])
            nn.init.zeros_(self.BI[layer])
            nn.init.zeros_(self.BH[layer])

    def forward(self, x:Tensor, h_0:Tensor=None)->tuple[Tensor, Tensor]:
        B, L, _ = x.shape # batch, length, dimensionality

        if h_0 is None:
            h_0 = torch.zeros(self.n_layers, B, self.hidden_dim, device=x.device)

        h_t = h_0.clone()

        output = []
        for t in range(L):
            h_t_minus_1 = h_t.clone()
            for layer in range(self.n_layers):
                x_layer = x[:, t] if layer == 0 else h_t[layer - 1] # first layers input is x, other layers receive the previous h_
                h_t[layer] = torch.tanh(
                    x_layer @ self.WI[layer].T + self.BI[layer] +
                    h_t_minus_1[layer] @ self.WH[layer].T + self.BH[layer]
                ) # don't know why but pytorch's implementation has two learnable biases... would assume that they could be combined into one
            output.append(h_t[-1].clone().detach())

        output = torch.stack(output, dim=1)
        return output, h_t


if __name__ == "__main__":
    import numpy as np

    input_dim = 10
    hidden_dim = 20
    n_layers = 2

    def copy_weights_to_torch_rnn(py_rnn, torch_rnn):
        for i in range(py_rnn.n_layers):
            getattr(torch_rnn, 'weight_ih_l' + str(i)).data.copy_(py_rnn.WI[i].data)
            getattr(torch_rnn, 'weight_hh_l' + str(i)).data.copy_(py_rnn.WH[i].data)
            getattr(torch_rnn, 'bias_ih_l' + str(i)).data.copy_(py_rnn.BI[i].data)
            getattr(torch_rnn, 'bias_hh_l' + str(i)).data.copy_(py_rnn.BH[i].data)
    
    py_rnn = PyRNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    torch_rnn = nn.RNN(input_dim,  hidden_dim, n_layers, nonlinearity='tanh', batch_first=True)
    copy_weights_to_torch_rnn(py_rnn, torch_rnn)

    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    h0 = torch.randn(n_layers, 5, hidden_dim)  # Initial hidden state
    out_py_rnn, h_py_rnn = py_rnn(x, h0)
    out_torch_rnn, h_torch_rnn = torch_rnn(x, h0)

    
    np.testing.assert_allclose(out_py_rnn.detach().numpy(), out_torch_rnn.detach().numpy(), rtol=1e-4)
    np.testing.assert_allclose(h_py_rnn.detach().numpy(), h_torch_rnn.detach().numpy(), rtol=1e-4)
    
    
