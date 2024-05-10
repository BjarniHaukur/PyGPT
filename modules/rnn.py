import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# does not support bidirectionality since it can't be used to generate code
class PyRNN(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, n_layers:int):
        super(PyRNN, self).__init__()
        self.input_dim, self.hidden_dim, self.n_layers = input_dim, hidden_dim, n_layers

        self.WI = nn.ParameterList([nn.Parameter(torch.empty((input_dim if i==0 else hidden_dim, hidden_dim))) for i in range(n_layers)])
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

        h_t_minus_1 = h_0
        h_t = h_0

        output = []
        for t in range(L):
            for layer in range(self.n_layers):
                x_layer = x[:, t] if layer == 0 else h_t[layer - 1] # first layers input is x, other layers receive the previous h_t
                h_t[layer] = torch.tanh(
                    torch.mm(x_layer, self.WI[layer]) + self.BI[layer] +
                    torch.mm(h_t_minus_1[layer], self.WH[layer]) + self.BH[layer]
                ) # don't know why but pytorch's implementation has two learnable biases... would assume that they could be combined into one
            output.append(h_t[-1])
            h_t_minus_1 = h_t

        output = torch.stack(output, dim=1)
        return output, h_t


if __name__ == "__main__":
    import numpy as np

    input_dim = 10
    hidden_dim = 20
    n_layers = 2

    py_rnn = PyRNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    output, h_n = py_rnn(x)
    print("Output shape:", output.shape)  # Expected: (5, 3, 20)
    print("Hidden state shape:", h_n.shape)  # Expected: (2, 5, 20) for each layer

    def copy_weights_to_torch_rnn(py_rnn, torch_rnn):
        for i in range(py_rnn.n_layers):
            getattr(torch_rnn, 'weight_ih_l' + str(i)).data.copy_(py_rnn.WI[i].data.T)
            getattr(torch_rnn, 'weight_hh_l' + str(i)).data.copy_(py_rnn.WH[i].data.T)
            getattr(torch_rnn, 'bias_ih_l' + str(i)).data.copy_(py_rnn.BI[i].data)
            getattr(torch_rnn, 'bias_hh_l' + str(i)).data.copy_(py_rnn.BH[i].data)

    torch_rnn = nn.RNN(input_dim, hidden_dim, n_layers, nonlinearity='tanh', batch_first=True)
    copy_weights_to_torch_rnn(py_rnn, torch_rnn)

    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    out_py_rnn, h_py_rnn = py_rnn(x)
    out_torch_rnn, h_torch_rnn = torch_rnn(x)

    np.testing.assert_allclose(out_py_rnn.detach().numpy(), out_torch_rnn.detach().numpy(), rtol=1e-4)
    np.testing.assert_allclose(h_py_rnn.detach().numpy(), h_torch_rnn.detach().numpy(), rtol=1e-4)




