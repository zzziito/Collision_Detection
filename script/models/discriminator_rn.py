import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, output_channels=7, output_seq_len=3000):
        super(Discriminator, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_channels * output_seq_len)
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.output_seq_len = output_seq_len
 
    def forward(self, x):
        _, h_n = self.rnn(x)  # h_n is the last hidden state
        h_n = h_n[-1] if not self.rnn.bidirectional else h_n.view(h_n.size(1), -1)  # Flatten if bidirectional
        x = self.fc(h_n)
        x = x.view(-1, self.output_channels, self.output_seq_len)  # Reshape to [batch_size, 7, 3000]
        return x