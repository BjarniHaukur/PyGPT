import torch.nn as nn

class PyLSTM(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, n_layers:int, **kwargs):
        super(PyLSTM, self).__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.rnn(x) # i.e. 100% teacher forcing
        x = self.linear(x)
        return x, h