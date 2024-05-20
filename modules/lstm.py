import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# does not support bidirectionality since it can't be used to generate code
class PyLSTM(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, n_layers:int):
        super(PyLSTM, self).__init__()
        self.input_dim, self.hidden_dim, self.n_layers = input_dim, hidden_dim, n_layers

        self.W_ii = nn.ParameterList([nn.Parameter(torch.empty((input_dim if i==0 else hidden_dim, hidden_dim))) for i in range(n_layers)])
        self.B_ii = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])
        self.W_hi = nn.ParameterList([nn.Parameter(torch.empty((hidden_dim, hidden_dim))) for _ in range(n_layers)])
        self.B_hi = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])

        self.W_if = nn.ParameterList([nn.Parameter(torch.empty((input_dim if i==0 else hidden_dim, hidden_dim))) for i in range(n_layers)])
        self.B_if = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])
        self.W_hf = nn.ParameterList([nn.Parameter(torch.empty((hidden_dim, hidden_dim))) for _ in range(n_layers)])
        self.B_hf = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])

        self.W_ig = nn.ParameterList([nn.Parameter(torch.empty((input_dim if i==0 else hidden_dim, hidden_dim))) for i in range(n_layers)])
        self.B_ig = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])
        self.W_hg = nn.ParameterList([nn.Parameter(torch.empty((hidden_dim, hidden_dim))) for _ in range(n_layers)])
        self.B_hg = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])

        self.W_io = nn.ParameterList([nn.Parameter(torch.empty((input_dim if i==0 else hidden_dim, hidden_dim))) for i in range(n_layers)])
        self.B_io = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])
        self.W_ho = nn.ParameterList([nn.Parameter(torch.empty((hidden_dim, hidden_dim))) for _ in range(n_layers)])
        self.B_ho = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim)) for _ in range(n_layers)])

        self.init_params()

    def init_params(self):
        for layer in range(self.n_layers):
            nn.init.xavier_normal_(self.W_ii[layer])
            nn.init.xavier_normal_(self.W_hi[layer])
            nn.init.zeros_(self.B_ii[layer])
            nn.init.zeros_(self.B_hi[layer])
            
            nn.init.xavier_normal_(self.W_if[layer])
            nn.init.xavier_normal_(self.W_hf[layer])
            nn.init.zeros_(self.B_if[layer])
            nn.init.zeros_(self.B_hf[layer])

            nn.init.xavier_normal_(self.W_ig[layer])
            nn.init.xavier_normal_(self.W_hg[layer])
            nn.init.zeros_(self.B_ig[layer])
            nn.init.zeros_(self.B_hg[layer])
            
            nn.init.xavier_normal_(self.W_io[layer])
            nn.init.xavier_normal_(self.W_ho[layer])
            nn.init.zeros_(self.B_io[layer])
            nn.init.zeros_(self.B_ho[layer])

    def forward(self, x:Tensor, h_0:Tensor=None, c_0:Tensor=None)->tuple[Tensor, Tensor, Tensor]:
        B, L, _ = x.shape # batch, length, dimensionality

        # initialize hidden state
        if h_0 is None:
            h_0 = torch.zeros(self.n_layers, B, self.hidden_dim, device=x.device)

        h_t_minus_1 = h_0
        h_t = h_0
        
        # initialize memory cell
        if c_0 is None:
            c_0 = torch.zeros(self.n_layers, B, self.hidden_dim, device=x.device)

        c_t_minus_1 = c_0
        c_t = c_0

        output = []
        for t in range(L):
            for layer in range(self.n_layers):
                x_layer = x[:, t] if layer == 0 else h_t[layer - 1] # first layers input is x, other layers receive the previous h_t
                
                i_t = torch.sigmoid(
                    torch.mm(x_layer, self.W_ii[layer]) + self.B_ii[layer] +
                    torch.mm(h_t_minus_1[layer], self.W_hi[layer]) + self.B_hi[layer]
                ) 
                
                f_t = torch.sigmoid(
                    torch.mm(x_layer, self.W_if[layer]) + self.B_if[layer] +
                    torch.mm(h_t_minus_1[layer], self.W_hf[layer]) + self.B_hf[layer]
                ) 
                
                g_t = torch.tanh(
                    torch.mm(x_layer, self.W_ig[layer]) + self.B_ig[layer] +
                    torch.mm(h_t_minus_1[layer], self.W_hg[layer]) + self.B_hg[layer]
                ) 
                
                o_t = torch.sigmoid(
                    torch.mm(x_layer, self.W_io[layer]) + self.B_io[layer] +
                    torch.mm(h_t_minus_1[layer], self.W_ho[layer]) + self.B_ho[layer]
                ) 
                
                c_t[layer] = f_t * c_t_minus_1[layer] + i_t * g_t
                
                h_t[layer] = o_t * torch.tanh(c_t[layer])

            output.append(h_t[-1].clone().detach())

            h_t_minus_1 = h_t.clone().detach()
            c_t_minus_1 = c_t.clone().detach()

        output = torch.stack(output, dim=1)
        return output, (h_t, c_t)


if __name__ == "__main__":
    import numpy as np

    input_dim = 10
    hidden_dim = 20
    n_layers = 2

    def copy_weights_to_torch_lstm(py_lstm, torch_lstm):
        for i in range(py_lstm.n_layers):
            w_ih_l = torch.vstack([py_lstm.W_ii[i].T, py_lstm.W_if[i].T, py_lstm.W_ig[i].T, py_lstm.W_io[i].T])
            getattr(torch_lstm, 'weight_ih_l' + str(i)).data.copy_(w_ih_l.data)
            w_hh_l = torch.vstack([py_lstm.W_hi[i].T, py_lstm.W_hf[i].T, py_lstm.W_hg[i].T, py_lstm.W_ho[i].T])
            getattr(torch_lstm, 'weight_hh_l' + str(i)).data.copy_(w_hh_l.data)
            b_ih_l = torch.hstack([py_lstm.B_ii[i], py_lstm.B_if[i], py_lstm.B_ig[i], py_lstm.B_io[i]])
            getattr(torch_lstm, 'bias_ih_l' + str(i)).data.copy_(b_ih_l.data)
            b_hh_l = torch.hstack([py_lstm.B_hi[i], py_lstm.B_hf[i], py_lstm.B_hg[i], py_lstm.B_ho[i]])
            getattr(torch_lstm, 'bias_hh_l' + str(i)).data.copy_(b_hh_l.data)

            assert torch.allclose(getattr(torch_lstm, 'bias_ih_l' + str(i)), b_ih_l)
            assert torch.allclose(getattr(torch_lstm, 'bias_hh_l' + str(i)), b_hh_l)
            assert torch.allclose(getattr(torch_lstm, 'weight_ih_l' + str(i)), w_ih_l)
            assert torch.allclose(getattr(torch_lstm, 'weight_hh_l' + str(i)), w_hh_l)
    
    py_lstm = PyLSTM(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    torch_lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
    copy_weights_to_torch_lstm(py_lstm, torch_lstm)

    x = torch.randn(5, 3, input_dim)  # Batch size of 5, sequence length of 3, feature size of 10
    h_0 = torch.zeros(n_layers, 5, hidden_dim, device=x.device)
    c_0 = torch.zeros(n_layers, 5, hidden_dim, device=x.device)
    out_py_lstm, (h_py_lstm, c_py_lstm) = py_lstm(x, h_0=h_0.clone(), c_0=c_0.clone())
    out_torch_lstm, (h_torch_lstm, c_torch_lstm) = torch_lstm(x, (h_0.clone(), c_0.clone()))

    RTOL = 1e-4

    np.testing.assert_allclose(out_py_lstm.detach().numpy(), out_torch_lstm.detach().numpy(), rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(h_py_lstm.detach().numpy(), h_torch_lstm.detach().numpy(), rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(c_py_lstm.detach().numpy(), c_torch_lstm.detach().numpy(), rtol=RTOL, atol=RTOL)




