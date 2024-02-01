import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
 
class Transformer(nn.Module):
    def __init__(self, num_joints, max_seq_len, hidden_size=512, nhead=8, num_encoder_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_joints = num_joints
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.input_linear = nn.Linear(3 * num_joints, hidden_size)
        self.output_linear = nn.Linear(hidden_size, num_joints)

    def forward(self, src):
        # 입력 차원 조정: [batch_size, 3 * num_joints, max_seq_len] -> [max_seq_len, batch_size, hidden_size]
        src = src.permute(2, 0, 1)  # [max_seq_len, batch_size, 3 * num_joints]
        src = self.input_linear(src)  # [max_seq_len, batch_size, hidden_size]
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output)
        # 출력 차원 조정: [max_seq_len, batch_size, num_joints] -> [batch_size, num_joints, max_seq_len]
        output = output.permute(1, 2, 0)
        return output
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
if __name__=="__main__":

    num_joints = 7
    max_seq_len = 100
    hidden_size = 512
    nhead = 8
    num_encoder_layers = 6
    dropout = 0.1
    
    model = Transformer(num_joints=num_joints, max_seq_len=max_seq_len,
                                    hidden_size=hidden_size, nhead=nhead,
                                    num_encoder_layers=num_encoder_layers, dropout=dropout)

    if torch.cuda.is_available():
        model.cuda()

    batch_size = 4
    example_input = torch.randn(batch_size, 3 * num_joints, max_seq_len)
    
    if torch.cuda.is_available():
        example_input = example_input.cuda()

    with torch.no_grad():  
        output = model(example_input)
    
    print("Output tensor shape:", output.shape)