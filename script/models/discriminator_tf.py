import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
 
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_encoder_layers, output_channels=7, output_seq_len=3000):
        super(Discriminator, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_channels * output_seq_len)
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.output_seq_len = output_seq_len
 
    def forward(self, x):
        x = self.input_linear(x)
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, features]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to [batch_size, seq_len, features]
        x = torch.mean(x, dim=1)  # Aggregate features
        x = self.output_linear(x)
        x = x.view(-1, self.output_channels, self.output_seq_len)  # Reshape to [batch_size, 7, 3000]
        return x